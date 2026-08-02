"""End-to-end GSPIM mechanism pipeline over typed frames and payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .config import ArchitectureConfig
from .dataflow import Batch, RenderOrderPlan, depth_stability_score, make_batch_plan, plan_render_orders
from .hcf import GPUFallback, HCFRecord, HCFResult, HierarchicalCompactionFabric, RecordLocation as HCFLocation, ReorgPurpose
from .model import FrameRequest, Geometry, ModelLayout
from .ppim import PPIMInterpreter, ProgramId
from .renderer import ReferenceRenderer, RenderedFrame, camera_depth
from .runtime import MemoryRegion, PIMTaskDescriptor, PIMTaskKind, Runtime, TaskOutcome
from .scheduling import PingPongBatchBuffers
from .trace import TaskTrace


@dataclass(frozen=True)
class WindowResult:
    frames: tuple[FrameRequest, ...]
    active_ids: set[int]
    stable_ids: set[int]
    unstable_ids: set[int]
    render_order_plan: RenderOrderPlan
    batches: tuple[Batch, ...]
    rendered_frames: tuple[RenderedFrame, ...]
    next_window_size: int
    trace: TaskTrace
    completion_events: Mapping[str, object]
    sorting_metadata: Mapping[int, "SortingMetadata"]
    s3_pack_results: tuple[HCFResult, ...]
    batch_pack_results: tuple[HCFResult, ...]
    live_at_completion: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class SequenceResult:
    windows: tuple[WindowResult, ...]
    trace: TaskTrace


@dataclass(frozen=True)
class SortingMetadata:
    """The only S3 payload HCF may place in stable or unstable regions."""

    active_index: int
    window_min_depth: float
    frame_depths: tuple[float, ...]


@dataclass(frozen=True)
class BatchPayload:
    """S4-selected shared attributes and frame-local fields for one primitive."""

    shared_attributes: Mapping[str, object]
    local_positions: tuple[tuple[int, ...], ...]
    frame_geometry: tuple[Geometry | None, ...]


@dataclass(frozen=True)
class PackedBatch:
    batch: Batch
    slot: int
    result: HCFResult
    gpu_fallback_records: tuple[HCFRecord, ...]
    task_ids: tuple[str, ...]


def _outcome(result, die: int, records: Mapping[int, HCFRecord]) -> TaskOutcome:
    ranges = tuple(item for owner, item in result.die_ranges.items() if owner == die)
    blocks = tuple(item for item in result.block_completions if item.die == die)
    count = sum(item.count for item in ranges)
    overflow = tuple(record_id for record_id in result.overflow_ids if records[record_id].location.die == die)
    return TaskOutcome(count, ranges, blocks, overflow)


class GSPIMPipeline:
    """Functional composition of PPIM, HCF, GPU stages, and hybrid dependencies."""

    def __init__(self, l2_capacity: int | None = None, config: ArchitectureConfig | None = None) -> None:
        self.config = config or ArchitectureConfig()
        self.l2_capacity = self.config.l2_capacity_bytes if l2_capacity is None else l2_capacity

    def _validate_layout_mapping(self, layout: ModelLayout) -> None:
        """Validate that every modeled row address fits this PPIM/HCF tile."""

        locations = [record.location for record in layout.decision_records()]
        locations.extend(
            layout.record_location(primitive_id)
            for record in layout.decision_records()
            for primitive_id in layout.binding_records(record.record_id)
        )
        for location in locations:
            if location.die >= self.config.total_dies:
                raise ValueError("physical die is outside the configured GSPIM topology")
            if location.bank >= self.config.pim_banks_per_die:
                raise ValueError("physical bank is outside the configured PIM-bank range")
            if location.slot >= self.config.block_records:
                raise ValueError("physical slot is outside the configured PPIM block width")

    def _hcf_records(self, layout: ModelLayout, primitive_ids: set[int]) -> Mapping[int, HCFRecord]:
        records: dict[int, HCFRecord] = {}
        for primitive_id in primitive_ids:
            location = layout.record_location(primitive_id)
            records[primitive_id] = HCFRecord(primitive_id, HCFLocation(primitive_id, location.die, location.bank, location.block, location.slot), layout.payload(primitive_id))
        return records

    def _active_hcf_records(
        self,
        layout: ModelLayout,
        decision_ids: set[int],
    ) -> Mapping[int, HCFRecord]:
        """Build S1 inputs without expanding anchor bindings on the host path."""

        decisions = {record.record_id: record for record in layout.decision_records()}
        records: dict[int, HCFRecord] = {}
        for decision_id in decision_ids:
            decision = decisions[decision_id]
            binding_ids = layout.binding_records(decision_id)
            if binding_ids == (decision_id,):
                location = layout.record_location(decision_id)
                records[decision_id] = HCFRecord(
                    decision_id,
                    HCFLocation(decision_id, location.die, location.bank, location.block, location.slot),
                    layout.payload(decision_id),
                )
                continue
            bound = self._hcf_records(layout, set(binding_ids))
            if any(record.location.die != decision.location.die for record in bound.values()):
                raise ValueError("HCF anchor binding expansion crosses dies")
            records.update(bound)
            records[decision_id] = HCFRecord(
                decision_id,
                HCFLocation(
                    decision_id,
                    decision.location.die,
                    decision.location.bank,
                    decision.location.block,
                    decision.location.slot,
                ),
                {"binding_start": binding_ids[0], "binding_count": len(binding_ids)},
                tuple(bound[primitive_id] for primitive_id in binding_ids),
            )
        return records

    @staticmethod
    def _metadata_records(
        layout: ModelLayout,
        depths: Mapping[int, tuple[float, ...]],
        active_indices: Mapping[int, int],
    ) -> Mapping[int, HCFRecord]:
        if set(depths) != set(active_indices):
            raise ValueError("S3 metadata must preserve every globally assigned Active Index")
        return {
            primitive_id: HCFRecord(
                primitive_id,
                HCFLocation(
                    primitive_id,
                    layout.record_location(primitive_id).die,
                    layout.record_location(primitive_id).bank,
                    layout.record_location(primitive_id).block,
                    layout.record_location(primitive_id).slot,
                ),
                SortingMetadata(active_indices[primitive_id], min(sequence), sequence),
            )
            for primitive_id, sequence in depths.items()
        }

    @staticmethod
    def _consume_hcf_result(
        result: HCFResult,
        source_records: Mapping[int, HCFRecord],
        expected_ids: set[int],
        stage: str,
    ) -> Mapping[int, object]:
        """Materialize the exact packed payloads plus explicit GPU fallback data."""

        fallback_records = tuple(
            record
            for fallback in result.gpu_fallbacks
            for record in fallback.source_records
            if record.primitive_id in fallback.overflow_ids
        )
        if set(record.primitive_id for record in fallback_records) != set(result.overflow_ids):
            raise RuntimeError(f"{stage} fallback does not preserve every overflow payload")
        if any(source_records[record.primitive_id] != record for record in fallback_records):
            raise RuntimeError(f"{stage} fallback changed an original physical payload")
        records = tuple(result.packed_records) + fallback_records
        actual_ids = {record.primitive_id for record in records}
        if actual_ids != expected_ids or len(records) != len(actual_ids):
            raise RuntimeError(f"{stage} did not deliver an exact packed-or-fallback payload set")
        return {record.primitive_id: record.payload for record in records}

    @staticmethod
    def _block_view(result: HCFResult, source_records: Mapping[int, HCFRecord], ids: set[int]) -> HCFResult:
        """Expose one physical HCF block without releasing its die-level allocation.

        HCF reserves a range for a complete die task, but S2 is allowed to consume
        each completed bank block before the final consumer releases that range.
        This view preserves the packed-or-fallback records for one block while
        retaining ownership of the parent result's live range.
        """

        packed = tuple(record for record in result.packed_records if record.primitive_id in ids)
        overflow = tuple(primitive_id for primitive_id in result.overflow_ids if primitive_id in ids)
        completions = tuple(
            completion
            for completion in result.block_completions
            if set(completion.output_ids).issubset(ids)
        )
        return HCFResult(
            result.purpose,
            tuple(record.primitive_id for record in packed),
            packed,
            result.die_ranges,
            completions,
            overflow,
            None,
            tuple(
                GPUFallback(
                    fallback.die,
                    fallback.bank,
                    fallback.block,
                    tuple(record for record in fallback.source_records if record.primitive_id in ids),
                    tuple(primitive_id for primitive_id in fallback.overflow_ids if primitive_id in ids),
                )
                for fallback in result.gpu_fallbacks
                if any(primitive_id in ids for primitive_id in fallback.overflow_ids)
            ),
        )

    @staticmethod
    def _batch_records(
        layout: ModelLayout,
        batch: Batch,
        geometry_by_frame: Sequence[Mapping[int, Geometry]],
        active_payloads: Mapping[int, Mapping[str, object]],
    ) -> Mapping[int, HCFRecord]:
        records: dict[int, HCFRecord] = {}
        for primitive_id in batch.attribute_ids:
            location = layout.record_location(primitive_id)
            positions = tuple(tuple(index for index, candidate in enumerate(rol) if candidate == primitive_id) for rol in batch.frame_rols)
            payload = BatchPayload(
                active_payloads[primitive_id],
                positions,
                tuple(frame.get(primitive_id) for frame in geometry_by_frame),
            )
            records[primitive_id] = HCFRecord(
                primitive_id,
                HCFLocation(primitive_id, location.die, location.bank, location.block, location.slot),
                payload,
            )
        return records

    def run_window(
        self,
        layout: ModelLayout,
        frames: Sequence[FrameRequest],
        window_id: int = 0,
        *,
        window_start: float,
        window_end: float,
        runtime: Runtime | None = None,
        hcf: HierarchicalCompactionFabric | None = None,
        s1_dependency: str | None = None,
        on_cross_window_ready: Callable[[str, int], None] | None = None,
    ) -> WindowResult:
        values = tuple(frames)
        if not values:
            raise ValueError("window is empty")
        start, end = float(window_start), float(window_end)
        if end <= start or any(frame.timestamp < start or frame.timestamp >= end for frame in values):
            raise ValueError("frames must belong to the explicit half-open temporal window")
        self._validate_layout_mapping(layout)
        selection = layout.select_window(start, end)
        layout.validate_window_frames(values, selection)
        program = ProgramId(selection.program_name)
        interpreter = PPIMInterpreter(self.config.fixed_fractional_bits)
        hcf = hcf or HierarchicalCompactionFabric(self.config.reserved_record_capacity_per_die)
        runtime = runtime or Runtime(self.config.total_dies)
        source = MemoryRegion(f"w{window_id}:model")
        runtime.gpu_write(source)
        runtime.writeback(source)
        decision_blocks: dict[tuple[int, int, int], list[object]] = {}
        for record in layout.decision_records():
            key = (record.location.die, record.location.bank, record.location.block)
            decision_blocks.setdefault(key, []).append(record)
        blocks_by_die: dict[int, list[tuple[int, int, tuple[object, ...]]]] = {}
        for (die, bank, block), records in sorted(decision_blocks.items()):
            blocks_by_die.setdefault(die, []).append((bank, block, tuple(records)))
        selected_decisions_by_die: dict[int, set[int]] = {}
        for die, blocks in blocks_by_die.items():
            selected_decisions = {
                record.record_id
                for _bank, _block, records in blocks
                for record in records
                if interpreter.run(program, record.fields, start, end).mask  # type: ignore[attr-defined]
            }
            selected_decisions_by_die[die] = selected_decisions
        executed_selection = set().union(*selected_decisions_by_die.values()) if selected_decisions_by_die else set()
        if executed_selection != selection.active_ids:
            raise RuntimeError("die PPIM selection does not match the layout program")

        active_source_by_die = {
            die: self._active_hcf_records(layout, decision_ids)
            for die, decision_ids in selected_decisions_by_die.items()
        }
        active_results_by_die: dict[int, HCFResult] = {}
        active_reorg_by_die: dict[int, str] = {}
        active_region_by_die: dict[int, MemoryRegion] = {}
        for die in sorted(blocks_by_die):
            block_scope = tuple(sorted({block for _bank, block, _records in blocks_by_die[die]}))
            bank_scope = tuple(sorted({bank for bank, _block, _records in blocks_by_die[die]}))
            source_region = MemoryRegion(f"w{window_id}:model-d{die}")
            mask_region = MemoryRegion(f"w{window_id}:s1-mask-d{die}")
            active_region = MemoryRegion(f"w{window_id}:active-d{die}")
            runtime.gpu_write(source_region)
            runtime.writeback(source_region)
            select_id = f"w{window_id}:s1-select-d{die}"
            runtime.submit(PIMTaskDescriptor(
                select_id,
                PIMTaskKind.SELECT,
                die,
                source_region,
                mask_region,
                program,
                block_scope,
                ReorgPurpose.ACTIVE,
                bank_scope,
                (s1_dependency,) if s1_dependency else (),
            ))
            runtime.complete(select_id, TaskOutcome(len(selected_decisions_by_die[die]), (), (), ()))
            source_records = active_source_by_die[die]
            active_result = hcf.reorganize(
                source_records,
                selected_decisions_by_die[die],
                ReorgPurpose.ACTIVE,
            )
            reorg_id = f"w{window_id}:s1-reorg-d{die}"
            runtime.submit(PIMTaskDescriptor(
                reorg_id,
                PIMTaskKind.REORG,
                die,
                mask_region,
                active_region,
                None,
                block_scope,
                ReorgPurpose.ACTIVE,
                bank_scope,
                (select_id,),
            ))
            active_results_by_die[die] = active_result
            active_reorg_by_die[die] = reorg_id
            active_region_by_die[die] = active_region

        # The Active Index is an HCF result, including explicit GPU fallback
        # suffixes, not a host-side pre-sort of source record locations.
        active_index_order = tuple(
            primitive_id
            for die in sorted(active_results_by_die)
            for primitive_id in (*active_results_by_die[die].output_ids, *active_results_by_die[die].overflow_ids)
        )
        if len(active_index_order) != len(set(active_index_order)):
            raise RuntimeError("HCF returned duplicate primitive IDs while assigning Active Index")
        active_indices = {primitive_id: index for index, primitive_id in enumerate(active_index_order)}
        active_payloads: dict[int, Mapping[str, object]] = {}
        geometry_by_frame: list[dict[int, Geometry]] = [dict() for _ in values]
        depths: dict[int, tuple[float, ...]] = {}
        scores: dict[int, float] = {}
        compacted_metadata: dict[int, SortingMetadata] = {}
        s3_results: list[HCFResult] = []
        s3_reorg_tasks: list[str] = []

        def complete_s2(
            die: int,
            bank: int,
            block: int,
            block_records: Mapping[int, HCFRecord],
            active_result: HCFResult,
            block_event: str,
        ) -> tuple[str, tuple[Mapping[int, Geometry], ...]]:
            active_block_payloads = self._consume_hcf_result(active_result, block_records, set(block_records), "S1 Active Buffer")
            if not all(isinstance(payload, Mapping) for payload in active_block_payloads.values()):
                raise RuntimeError("S1 Active Buffer contains a non-record payload")
            typed_payloads = {primitive_id: payload for primitive_id, payload in active_block_payloads.items() if isinstance(payload, Mapping)}
            duplicate = set(active_payloads) & set(typed_payloads)
            if duplicate:
                raise RuntimeError(f"multiple S1 records expanded the same primitive: {sorted(duplicate)}")
            s2_id = f"w{window_id}:s2-d{die}-bank{bank}-b{block}"
            runtime.submit_gpu_stage(s2_id, "S2", (block_event,))
            block_geometry = [layout.generate_geometry_from_payloads(typed_payloads, frame.timestamp) for frame in values]
            runtime.complete_gpu_stage(s2_id)
            for frame_index, geometry in enumerate(block_geometry):
                geometry_by_frame[frame_index].update(geometry)
            active_payloads.update(typed_payloads)
            return s2_id, tuple(block_geometry)

        def complete_s3_block(
            die: int,
            bank: int,
            block: int,
            block_geometry: Sequence[Mapping[int, Geometry]],
            s2_id: str,
        ) -> tuple[str, str]:
            """Classify and compact one physical block as soon as S2 finishes."""

            s3_gpu = f"w{window_id}:s3-gpu-d{die}-bank{bank}-b{block}"
            runtime.submit_gpu_stage(s3_gpu, "S3_GPU", (s2_id,))
            if not block_geometry:
                raise RuntimeError("S3 needs geometry for every frame in its completed S2 block")
            block_ids = set(block_geometry[0])
            if any(set(geometry) != block_ids for geometry in block_geometry[1:]):
                raise RuntimeError("S2 produced inconsistent primitive domains across the S3 window")
            block_depths: dict[int, tuple[float, ...]] = {}
            for primitive_id in sorted(block_ids):
                samples: list[float] = []
                for frame, geometry in zip(values, block_geometry):
                    depth = camera_depth(geometry[primitive_id], frame.camera)
                    if depth is None:
                        raise ValueError(f"active primitive {primitive_id} is behind the camera")
                    samples.append(depth)
                block_depths[primitive_id] = tuple(samples)
            depths.update(block_depths)
            scores.update({
                primitive_id: depth_stability_score(sequence, values[0].camera.near, values[0].camera.far)
                for primitive_id, sequence in block_depths.items()
            })
            runtime.complete_gpu_stage(s3_gpu)

            metadata_records = self._metadata_records(
                layout,
                block_depths,
                {primitive_id: active_indices[primitive_id] for primitive_id in block_depths},
            )
            stable_ids_for_block = {
                primitive_id
                for primitive_id, sequence in block_depths.items()
                if interpreter.run(
                    ProgramId.DEPTH_STABLE,
                    {"score": depth_stability_score(sequence, values[0].camera.near, values[0].camera.far)},
                    tau=self.config.tau_stability,
                ).mask
            }
            unstable_ids_for_block = set(block_depths) - stable_ids_for_block
            score_region = MemoryRegion(f"w{window_id}:scores-d{die}-bank{bank}-b{block}")
            class_mask = MemoryRegion(f"w{window_id}:s3-mask-d{die}-bank{bank}-b{block}")
            runtime.gpu_write(score_region)
            runtime.writeback(score_region)
            s3_select = f"w{window_id}:s3-select-d{die}-bank{bank}-b{block}"
            runtime.submit(PIMTaskDescriptor(
                s3_select,
                PIMTaskKind.SELECT,
                die,
                score_region,
                class_mask,
                ProgramId.DEPTH_STABLE,
                (block,),
                ReorgPurpose.STABLE,
                (bank,),
                (s3_gpu,),
            ))
            runtime.complete(s3_select, TaskOutcome(len(stable_ids_for_block), (), (), ()))
            stable_result = hcf.reorganize(metadata_records, stable_ids_for_block, ReorgPurpose.STABLE)
            unstable_result = hcf.reorganize(metadata_records, unstable_ids_for_block, ReorgPurpose.UNSTABLE)
            stable_payloads = self._consume_hcf_result(stable_result, metadata_records, stable_ids_for_block, "S3 stable metadata")
            unstable_payloads = self._consume_hcf_result(unstable_result, metadata_records, unstable_ids_for_block, "S3 unstable metadata")
            if not all(isinstance(payload, SortingMetadata) for payload in (*stable_payloads.values(), *unstable_payloads.values())):
                raise RuntimeError("S3 HCF output contains a non-metadata payload")
            stable_reorg = f"w{window_id}:s3-stable-reorg-d{die}-bank{bank}-b{block}"
            runtime.submit(PIMTaskDescriptor(
                stable_reorg,
                PIMTaskKind.REORG,
                die,
                class_mask,
                MemoryRegion(f"w{window_id}:stable-metadata-d{die}-bank{bank}-b{block}"),
                None,
                (block,),
                ReorgPurpose.STABLE,
                (bank,),
                (s3_select,),
            ))
            runtime.complete(stable_reorg, _outcome(stable_result, die, metadata_records))
            unstable_reorg = f"w{window_id}:s3-unstable-reorg-d{die}-bank{bank}-b{block}"
            runtime.submit(PIMTaskDescriptor(
                unstable_reorg,
                PIMTaskKind.REORG,
                die,
                class_mask,
                MemoryRegion(f"w{window_id}:unstable-metadata-d{die}-bank{bank}-b{block}"),
                None,
                (block,),
                ReorgPurpose.UNSTABLE,
                (bank,),
                (s3_select,),
            ))
            runtime.complete(unstable_reorg, _outcome(unstable_result, die, metadata_records))
            s3_results.extend((stable_result, unstable_result))
            compacted_metadata.update({
                primitive_id: payload
                for primitive_id, payload in {**stable_payloads, **unstable_payloads}.items()
                if isinstance(payload, SortingMetadata)
            })
            return stable_reorg, unstable_reorg

        for die, source_records in active_source_by_die.items():
            for completion in active_results_by_die[die].block_completions:
                bank, block = completion.bank, completion.block
                block_ids = set(completion.input_ids)
                block_result = self._block_view(active_results_by_die[die], source_records, block_ids)
                overflow_ids = tuple(
                    primitive_id
                    for primitive_id in completion.input_ids
                    if primitive_id not in completion.output_ids
                )
                block_event = runtime.complete_block(
                    active_reorg_by_die[die],
                    completion,
                    overflow_ids,
                )
                runtime.invalidate(active_region_by_die[die])
                runtime.gpu_acquire(active_region_by_die[die], block_event.task_id)
                if block_result.overflow_ids:
                    runtime.complete_gpu_stage(
                        f"w{window_id}:s1-gpu-fallback-d{die}-bank{bank}-b{block}",
                        "GPU_REORG",
                        (block_event.task_id,),
                    )
                s2_id, block_geometry = complete_s2(
                    die,
                    bank,
                    block,
                    {primitive_id: source_records[primitive_id] for primitive_id in block_ids},
                    block_result,
                    block_event.task_id,
                )
                s3_reorg_tasks.extend(complete_s3_block(die, bank, block, block_geometry, s2_id))
        for die, result in active_results_by_die.items():
            runtime.complete(active_reorg_by_die[die], _outcome(result, die, active_source_by_die[die]))
        for result in active_results_by_die.values():
            hcf.release(result)
        active_ids = set(active_payloads)
        selected_primitive_ids = {
            bound.primitive_id
            for decision_ids, source_records in (
                (selected_decisions_by_die[die], active_source_by_die[die])
                for die in selected_decisions_by_die
            )
            for decision_id in decision_ids
            for bound in (source_records[decision_id].bound_records or (source_records[decision_id],))
        }
        if active_ids != selected_primitive_ids or active_ids != set(active_indices):
            raise RuntimeError("S1 packed-or-fallback records do not match the window Active Index domain")

        s3_window = f"w{window_id}:s3-window"
        runtime.complete_gpu_stage(s3_window, "S3", tuple(s3_reorg_tasks))
        indexed_metadata = sorted(
            (payload.active_index, primitive_id, payload)
            for primitive_id, payload in compacted_metadata.items()
            if isinstance(payload, SortingMetadata)
        )
        if [index for index, _primitive_id, _payload in indexed_metadata] != list(range(len(active_ids))):
            raise RuntimeError("S4 received non-contiguous or duplicate Active Indices")
        compacted_depths = {
            primitive_id: payload.frame_depths
            for _index, primitive_id, payload in indexed_metadata
        }
        stable_ids = {primitive_id for primitive_id, metadata in compacted_metadata.items() if interpreter.run(ProgramId.DEPTH_STABLE, {"score": depth_stability_score(metadata.frame_depths, values[0].camera.near, values[0].camera.far)}, tau=self.config.tau_stability).mask}
        unstable_ids = set(compacted_metadata) - stable_ids
        orders = plan_render_orders(compacted_depths, stable_ids)
        attributes = {primitive_id: int(active_payloads[primitive_id]["attribute_bytes"]) for primitive_id in active_ids}
        batches = make_batch_plan(orders, attributes, self.l2_capacity, self.config.local_index_bytes, self.config.raster_intermediate_bytes)
        mean_score = sum(scores.values()) / len(scores) if scores else 1.0
        requested = len(values)
        next_window = max(self.config.min_window_size, requested - 1) if mean_score < self.config.tau_window else min(self.config.max_window_size, requested + 1)
        s4 = f"w{window_id}:s4"
        runtime.complete_gpu_stage(s4, "S4", (s3_window,))
        for result in s3_results:
            hcf.release(result)

        batch_selection = MemoryRegion(f"w{window_id}:batch-selection")
        runtime.gpu_write(batch_selection)
        runtime.writeback(batch_selection)
        batch_buffers: PingPongBatchBuffers[PackedBatch] = PingPongBatchBuffers()
        packed_by_id: dict[int, PackedBatch] = {}
        packed_results: list[HCFResult] = []
        cross_window_started = False

        def pack(batch: Batch) -> PackedBatch:
            records = self._batch_records(layout, batch, geometry_by_frame, active_payloads)
            result = hcf.reorganize(records, set(batch.attribute_ids), ReorgPurpose.BATCH)
            task_ids: list[str] = []
            for die in sorted({record.location.die for record in records.values()}):
                task_id = f"w{window_id}:s5-pack-{batch.batch_id}-d{die}"
                bank_scope = tuple(sorted({record.location.bank for record in records.values() if record.location.die == die}))
                runtime.submit(PIMTaskDescriptor(task_id, PIMTaskKind.REORG, die, batch_selection, MemoryRegion(f"w{window_id}:batch-{batch.batch_id}-d{die}"), None, (batch.batch_id,), ReorgPurpose.BATCH, bank_scope, (s4,)))
                runtime.complete(task_id, _outcome(result, die, records))
                task_ids.append(task_id)
            fallback = tuple(records[primitive_id] for primitive_id in result.overflow_ids)
            if fallback:
                fallback_task = f"w{window_id}:s5-gpu-fallback-{batch.batch_id}"
                runtime.complete_gpu_stage(fallback_task, "GPU_REORG", tuple(task_ids))
            entry = PackedBatch(batch, batch_buffers.next_slot, result, fallback, tuple(task_ids))
            slot = batch_buffers.pack(entry)
            if slot != entry.slot:
                raise AssertionError("ping-pong producer changed its reserved slot")
            packed_by_id[batch.batch_id] = entry
            packed_results.append(result)
            return entry

        if batches:
            pack(batches[0])
        elif on_cross_window_ready is not None:
            on_cross_window_ready(s3_window, next_window)
            cross_window_started = True
        renderer = ReferenceRenderer()
        render_session = renderer.begin_batches(tuple(frame.camera for frame in values))
        previous_render: str | None = None
        for batch_index, batch in enumerate(batches):
            packed = packed_by_id[batch.batch_id]
            render = f"w{window_id}:s5-render-{batch.batch_id}"
            render_dependencies = packed.task_ids if previous_render is None else packed.task_ids + (previous_render,)
            runtime.submit_gpu_stage(render, "S5", render_dependencies)
            if batch_index + 1 < len(batches):
                pack(batches[batch_index + 1])
            if not cross_window_started and on_cross_window_ready is not None:
                on_cross_window_ready(s3_window, next_window)
                cross_window_started = True
            render_session.consume(packed.batch, packed.result, packed.gpu_fallback_records)
            runtime.complete_gpu_stage(render)
            consumed = batch_buffers.consume(packed.slot)
            if consumed.batch.batch_id != batch.batch_id:
                raise RuntimeError("ping-pong buffer returned a different batch")
            hcf.release(consumed.result)
            previous_render = render
        rendered = render_session.finish()
        prefix = f"w{window_id}:"
        window_trace = TaskTrace()
        for node in runtime.trace.nodes.values():
            if node.name.startswith(prefix):
                window_trace.add(node.name, node.kind, node.dependencies)
        window_events = {task_id: event for task_id, event in runtime.events.items() if task_id.startswith(prefix)}
        window_live = {task_id: live for task_id, live in runtime.live_at_completion.items() if task_id.startswith(prefix)}
        return WindowResult(
            values,
            active_ids,
            stable_ids,
            unstable_ids,
            orders,
            batches,
            rendered,
            next_window,
            window_trace,
            window_events,
            compacted_metadata,
            tuple(s3_results),
            tuple(packed_by_id[batch.batch_id].result for batch in batches),
            window_live,
        )

    def run_sequence(
        self,
        layout: ModelLayout,
        frames: Sequence[FrameRequest],
        initial_window_size: int | None = None,
        *,
        frame_period: float,
    ) -> SequenceResult:
        values = tuple(frames)
        if not values:
            return SequenceResult((), TaskTrace())
        if frame_period <= 0.0:
            raise ValueError("frame_period must be positive to define temporal window ends")
        window_size = initial_window_size or self.config.max_window_size
        if not self.config.min_window_size <= window_size <= self.config.max_window_size:
            raise ValueError("initial_window_size is outside the paper's 2--5 range")
        runtime = Runtime(self.config.total_dies)
        hcf = HierarchicalCompactionFabric(self.config.reserved_record_capacity_per_die)
        windows: dict[int, WindowResult] = {}

        def launch(offset: int, requested_size: int, window_id: int, previous_s3: str | None) -> None:
            remaining = len(values) - offset
            if remaining < self.config.min_window_size:
                raise ValueError("sequence tail cannot form a paper-valid 2--5 frame window")
            # The final drain remains a legal paper window.  A requested five-frame
            # window over six remaining frames becomes four plus two rather than
            # emitting a one-frame tail.
            if remaining <= self.config.max_window_size:
                current_size = remaining
            else:
                current_size = requested_size
                if remaining - current_size < self.config.min_window_size:
                    current_size -= self.config.min_window_size - (remaining - current_size)
            if not self.config.min_window_size <= current_size <= self.config.max_window_size:
                raise ValueError("unable to form a paper-valid 2--5 frame window")
            current = values[offset:offset + current_size]
            if not current:
                return
            current_start = current[0].timestamp
            current_end = current[-1].timestamp + frame_period

            def launch_successor(s3_window: str, next_size: int) -> None:
                next_offset = offset + len(current)
                if next_offset < len(values):
                    launch(next_offset, next_size, window_id + 1, s3_window)

            windows[window_id] = self.run_window(
                layout,
                current,
                window_id,
                window_start=current_start,
                window_end=current_end,
                runtime=runtime,
                hcf=hcf,
                s1_dependency=previous_s3,
                on_cross_window_ready=launch_successor,
            )

        launch(0, window_size, 0, None)
        return SequenceResult(tuple(windows[index] for index in sorted(windows)), runtime.trace)
