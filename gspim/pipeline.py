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

        # Assign Active Indices in HCF output order, including GPU fallback
        # suffixes after each die's compacted prefix.
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
            raise ValueError("initial_window_size is outside the configured window range")
        runtime = Runtime(self.config.total_dies)
        hcf = HierarchicalCompactionFabric(self.config.reserved_record_capacity_per_die)
        windows: dict[int, WindowResult] = {}

        def launch(offset: int, requested_size: int, window_id: int, previous_s3: str | None) -> None:
            remaining = len(values) - offset
            if remaining < self.config.min_window_size:
                raise ValueError("sequence tail is shorter than the minimum window size")
            # Reserve a complete final window: six remaining frames with a
            # requested five-frame window become four plus two.
            if remaining <= self.config.max_window_size:
                current_size = remaining
            else:
                current_size = requested_size
                if remaining - current_size < self.config.min_window_size:
                    current_size -= self.config.min_window_size - (remaining - current_size)
            if not self.config.min_window_size <= current_size <= self.config.max_window_size:
                raise ValueError("unable to form a window within the configured bounds")
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


class CUDAPipeline:
    """Independent-frame and multi-frame comparison on the current CUDA device."""

    def __init__(self, model, cache) -> None:
        import torch
        from .renderer import CUDARenderer

        self.model = model
        self.renderer = CUDARenderer(cache, model)
        self.projection_stream = torch.cuda.Stream(device=model.xyz.device)
        self.l2_bytes = torch.cuda.get_device_properties(model.xyz.device).L2_cache_size

    def dataflow(self, times, first=None, *, execution=None):
        import torch
        from .dataflow import cuda_render_orders

        model = (self.model.select_dataflow(times, first) if hasattr(self.model, 'select_dataflow') else
                 self.model.select(times, first) if hasattr(self.model, "select") else self.model)
        generated = self.renderer.generate(model, times)
        if hasattr(model, "anchor_positions"):
            # A binding below the compositor's alpha cutoff in every frame
            # cannot affect any pixel. Compact this window's decoded outputs
            # before projection and ordering.
            active = (generated[2] >= 1 / 255).any(0).nonzero().flatten()
            generated = tuple(value.index_select(1, active) for value in generated)
            model.decoded_colors = model.decoded_colors.index_select(0, active)
        current = torch.cuda.current_stream(model.xyz.device)
        self.projection_stream.wait_stream(current)
        with torch.cuda.stream(self.projection_stream):
            projected = self.renderer.project(model, generated, True)
            if execution is not None:
                execution.remember(generated[0], projected)
        for value in generated:
            value.record_stream(self.projection_stream)
        ordering = execution.orders if execution is not None else cuda_render_orders
        render = execution.render_window if execution is not None else self._render_window
        colors_for = execution.colors if execution is not None else self.renderer.colors
        orders, adjustment, stable, depth = ordering(
            generated[0], model.view,
            float(model.config.get("stability_near", model.config["near"])),
            float(model.config.get("stability_far", model.config["far"])), self.renderer.ops,
            **({} if execution is not None else {'statistics': self.renderer.window_statistics}),
        )
        shared_tiles = len(stable) == orders.shape[1]
        current.wait_stream(self.projection_stream)
        for value in projected:
            value.record_stream(current)
        if not shared_tiles:
            visible_unstable = (projected[0] > 0).all(-1).any(0)
            visible_unstable[stable.long()] = False
            shared_tiles = not bool(visible_unstable.any())
        if shared_tiles:
            selected, crossed = (execution.visible(stable, projected[0], depth)
                                 if execution is not None else
                                 self.renderer.ops.visible_order(stable, projected[0], depth))
            if crossed or len(times) == 1:
                if len(times) > 3 or len(times) == 1:
                    colors = colors_for(model, generated[0], projected[0])
                    projected = (*projected[:4], colors, projected[5])
                    if execution is not None:
                        execution.publish(projected)
                # A high stability score does not rule out rank crossings.
                # Preserve the configured ready-window, but shorten its shared-order spans
                # and center each span's key in time to reduce order bias.
                width = len(times)
                spans = ((0, 3), (3, 5)) if width == 5 else (
                    ((0, 2), (2, 4)) if width == 4 else ((0, width),))
                image = projected[1].new_empty((width, model.height, model.width, 4))
                total_batches, maximum_footprint = 0, 0
                # Prepare the next span while the first composites. Its
                # compositor waits below so only one span composites at a time.
                if len(spans) > 1:
                    common_ready = torch.cuda.Event()
                    common_ready.record(current)
                    self.projection_stream.wait_event(common_ready)
                    for value in (*generated, *projected, orders, selected, depth, image):
                        value.record_stream(self.projection_stream)
                render_done = None
                for index, (first, last) in enumerate(spans):
                    stream = self.projection_stream if index else current
                    with torch.cuda.stream(stream):
                        span_selected, _ = self.renderer.ops.visible_order(
                            selected, projected[0][first:last], depth.new_empty(0))
                        ordered_depth = depth[first:last, span_selected]
                        key = ordered_depth.mean(0).contiguous()
                        if execution is not None:
                            shared_order = execution.span(self.renderer.ops, key, span_selected)
                        else:
                            repaired = self.renderer.ops.radix(
                                key, torch.arange(len(span_selected), dtype=torch.int32,
                                                  device=selected.device), 0, 32)[1]
                            shared_order = span_selected[repaired.long()]
                        local = tuple(value[first:last] if value.numel() else value
                                      for value in projected)
                        span_model = model
                        if hasattr(model, "frame_times"):
                            from copy import copy
                            span_model = copy(model)
                            span_model.frame_times = model.frame_times[first:last]
                        _, batches, footprint = render(
                            generated[0][first:last], local, orders[first:last],
                            shared_order, True, image[first:last], shared_order,
                            wait_for=render_done, model=span_model)
                        if index == 0 and len(spans) > 1:
                            render_done = torch.cuda.Event()
                            render_done.record(current)
                    total_batches += batches
                    maximum_footprint = max(maximum_footprint, footprint)
                if len(spans) > 1:
                    current.wait_stream(self.projection_stream)
                return image[..., :3], adjustment, total_batches, maximum_footprint
        image, batches, footprint = render(
            generated[0], projected, orders, stable, shared_tiles,
            visible_ids=selected if shared_tiles else None, model=model)
        return image, adjustment, batches, footprint

    def _render_window(self, means, projected, orders, stable, shared_tiles,
                       output=None, visible_ids=None, wait_for=None, model=None):
        import torch
        from .dataflow import cuda_batch_boundaries

        model = self.model if model is None else model
        defer_colors = shared_tiles and not hasattr(model, "anchor_positions")
        if not defer_colors:
            projected = (*projected[:4], self.renderer.colors(model, means, projected[0]), projected[5])
        unstable_count = orders.shape[1] - len(stable)
        if (hasattr(model, "anchor_positions") and not shared_tiles and
                0 < unstable_count <= max(1, orders.shape[1] // 100)):
            # Frame-owned unstable occurrences preserve every frame's ROL
            # while letting the batch compositor consume one shared sequence.
            source, owner = self.renderer.ops.shared_sequence(orders.contiguous(), stable)
            projected = tuple(value.index_select(1, source) if value.numel() else value
                              for value in projected)
            live = (owner[None] < 0) | (owner[None] ==
                    torch.arange(len(means), device=owner.device)[:, None])
            projected = (projected[0] * live[..., None], *projected[1:])
            stable = torch.arange(len(source), dtype=torch.int32, device=orders.device)
            orders = stable.long()[None].expand(len(means), -1)
            shared_tiles = True
        sh = model.sh
        source_ids = None
        if shared_tiles:
            if visible_ids is None:
                selected, _ = self.renderer.ops.visible_order(
                    stable, projected[0], projected[1].new_empty(0))
            else:
                selected = visible_ids
            # Depth has already supplied the ROL; SH and means are read from
            # their source arrays by each batch preparer.
            projected = tuple(value.new_empty(0) if field == 2 else
                              value.index_select(1, selected) if value.numel() else value
                              for field, value in enumerate(projected))
            source_ids = selected
            stable = torch.arange(len(selected), dtype=torch.int32, device=orders.device)
            orders = stable.long()[None].expand(len(means), -1)
        # Preserve exact FP32 bits while compacting stable projection fields
        # into a shared batch. The layout is lossless and only applies to the
        # Dataflow path where the ROL is already common.
        layout = (self.renderer.ops.packed_layout(
            projected[1], projected[0], projected[3], projected[5],
        ) if shared_tiles and projected[1].shape[1] else None)
        # The preparer streams SH into per-frame RGB before compositing.
        # Only the packed fields need to remain in the render working set.
        shared_bytes = 0 if defer_colors else model.sh.shape[1] * 3 * 4
        tile_records = cost_prefix = None
        if layout is not None:
            # Cull once before choosing boundaries, charging every packed
            # attribute but only tile IDs that the compositor will access.
            unassigned = orders.new_zeros((1, orders.shape[1])).expand_as(orders)
            tile_records = self.renderer._tile_records(
                model, projected, orders, unassigned, 1, True)
            cost_prefix = self.renderer.ops.tile_batch_prefix(
                layout, tile_records[1], orders.shape[1].bit_length())
        assignment, batches, footprint = cuda_batch_boundaries(
            orders, stable, projected, shared_bytes,
            self.l2_bytes, model.width, model.height, self.renderer.ops, shared_tiles, layout,
            cost_prefix=cost_prefix,
        )
        if tile_records is not None:
            tile_bits = (((model.width + 15) // 16) *
                         ((model.height + 15) // 16)).bit_length()
            keys, ids = tile_records
            keys = self.renderer.ops.tile_batch_keys(
                keys, ids, assignment[0], orders.shape[1].bit_length(), tile_bits, batches)
            tile_records = keys, ids
        image = self.renderer.window(model, projected, orders, assignment, batches, shared_tiles,
                                     means if defer_colors else None, sh, shared_tiles, source_ids, layout, output, wait_for, tile_records)
        return image, batches, footprint

    def baseline(self, time, first=None):
        model = self.model.select(time, first) if hasattr(self.model, "select") else self.model
        generated = self.renderer.generate(model, time)
        return self.renderer.frame(model, self.renderer.project(model, generated))

    def evaluate(self, images, output, frame_count: int, repeats: int = 4, warmup: int = 20) -> dict[str, object]:
        """Render the selected frames and emit one Markdown summary."""
        from pathlib import Path

        import numpy as np
        import torch
        from PIL import Image
        import hashlib
        from .renderer import cuda_image_quality

        if frame_count < 2 or frame_count > self.model.duration or repeats < 1 or warmup < 1:
            raise ValueError("invalid frame count, repeats, or warmup")
        images, output = Path(images), Path(output)
        root = Path(__file__).resolve().parents[1]
        destination = output.resolve()
        allowed = root / "outputs"
        if (destination == root or root in destination.parents) and destination != allowed and allowed not in destination.parents:
            raise ValueError("outputs inside the checkout must be under outputs/")
        targets = []
        for frame in range(frame_count):
            path = images / f"{getattr(self.model, 'image_prefix', '')}{frame:04d}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            targets.append(path)
        times = torch.arange(frame_count, device=self.model.xyz.device, dtype=torch.float32)
        windows = []
        start, size = 0, 5
        with torch.inference_mode():
            while start < frame_count:
                remaining = frame_count - start
                current = min(size, remaining)
                if remaining - current == 1:
                    current -= 1
                if current < 2:
                    raise ValueError("sequence tail cannot form a two-frame window")
                _, adjustment, _, _ = self.dataflow(times[start:start + current], start)
                windows.append((start, current))
                start += current
                size = min(5, max(2, size + adjustment))
            for index in range(warmup):
                start, size = windows[index % len(windows)]
                for frame in range(start, start + size):
                    self.baseline(times[frame:frame + 1], frame)
                self.dataflow(times[start:start + size], start)
            torch.cuda.synchronize()
            durations = {"baseline": [], "dataflow": []}
            latencies = {name: [] for name in durations}
            maximum_batches = maximum_footprint = 0
            for repeat in range(repeats):
                modes = tuple(durations) if repeat % 2 == 0 else tuple(reversed(durations))
                for mode in modes:
                    records = []
                    begin = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for start, size in windows:
                        ready = torch.cuda.Event(enable_timing=True)
                        ready.record()
                        completions = []
                        if mode == "baseline":
                            for frame in range(start, start + size):
                                image = self.baseline(times[frame:frame + 1], frame)
                                done = torch.cuda.Event(enable_timing=True)
                                done.record()
                                completions.append(done)
                        else:
                            image, _, batches, footprint = self.dataflow(times[start:start + size], start)
                            maximum_batches = max(maximum_batches, batches)
                            maximum_footprint = max(maximum_footprint, footprint)
                            done = torch.cuda.Event(enable_timing=True)
                            done.record()
                            # All outputs become available after the batch-major drain.
                            completions = [done] * size
                        records.append((ready, completions))
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    end.synchronize()
                    durations[mode].append(begin.elapsed_time(end))
                    latencies[mode].extend(ready.elapsed_time(done) for ready, events in records for done in events)
                    print(f"{self.model.name} {mode} repeat {repeat + 1}: {frame_count * 1000 / durations[mode][-1]:.3f} FPS", flush=True)
            quality = {name: [] for name in durations}
            dataflow_rgb_sha256 = []
            maximum_rgb_error = 0.0
            for start, size in windows:
                dataflow, _, _, _ = self.dataflow(times[start:start + size], start)
                for local, frame in enumerate(range(start, start + size)):
                    target = Image.open(targets[frame]).convert("RGB")
                    if target.size != (self.model.width, self.model.height):
                        if getattr(self.model, "workload", "ex4dgs") in ("explicit4d", "anchored"):
                            raise ValueError("this workload requires native half-resolution image preprocessing")
                        target = target.resize((self.model.width, self.model.height), Image.Resampling.BILINEAR)
                    target = torch.from_numpy(np.asarray(target).copy()).to(self.model.xyz.device, dtype=torch.float32) / 255
                    baseline = self.baseline(times[frame:frame + 1], frame)
                    quality["baseline"].append(cuda_image_quality(baseline, target))
                    quality["dataflow"].append(cuda_image_quality(dataflow[local], target))
                    dataflow_rgb_sha256.append(hashlib.sha256(
                        dataflow[local].detach().cpu().contiguous().numpy().tobytes()).hexdigest())
                    maximum_rgb_error = max(
                        maximum_rgb_error,
                        float((baseline - dataflow[local]).abs().max().item()),
                    )
        baseline_ms = float(np.mean(durations["baseline"]))
        dataflow_ms = float(np.mean(durations["dataflow"]))
        p95_baseline = float(np.percentile(latencies["baseline"], 95))
        p99_baseline = float(np.percentile(latencies["baseline"], 99))
        p95_dataflow = float(np.percentile(latencies["dataflow"], 95))
        p99_dataflow = float(np.percentile(latencies["dataflow"], 99))
        baseline_quality = np.mean(quality["baseline"], axis=0)
        dataflow_quality = np.mean(quality["dataflow"], axis=0)
        speedup = baseline_ms / dataflow_ms
        lines = [f"# {self.model.name}", "",
                 f"Frames: {frame_count}; repeats: {repeats}; warmup windows: {warmup}.", "",
                 "| Path | Speedup | Time (ms/frame) | P95 (ms) | P99 (ms) | PSNR | SSIM |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                 f"| baseline | 1.000000x | {baseline_ms/frame_count:.6f} | {p95_baseline:.6f} | {p99_baseline:.6f} | {baseline_quality[0]:.6f} | {baseline_quality[1]:.6f} |",
                 f"| dataflow | {speedup:.6f}x | {dataflow_ms/frame_count:.6f} | {p95_dataflow:.6f} | {p99_dataflow:.6f} | {dataflow_quality[0]:.6f} | {dataflow_quality[1]:.6f} |"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="ascii")
        return {
            "workload": getattr(self.model, "workload", "ex4dgs"),
            "scene": self.model.name,
            "frames": frame_count,
            "repeats": repeats,
            "warmup": warmup,
            "windows": windows,
            "duration_samples_ms": durations,
            "frame_latency_samples_ms": latencies,
            "baseline_ms": baseline_ms,
            "dataflow_ms": dataflow_ms,
            "baseline_fps": float(frame_count * 1000 / baseline_ms),
            "dataflow_fps": float(frame_count * 1000 / dataflow_ms),
            "p95_baseline_ms": p95_baseline,
            "p99_baseline_ms": p99_baseline,
            "p95_dataflow_ms": p95_dataflow,
            "p99_dataflow_ms": p99_dataflow,
            "psnr_baseline": float(baseline_quality[0]),
            "ssim_baseline": float(baseline_quality[1]),
            "psnr_dataflow": float(dataflow_quality[0]),
            "ssim_dataflow": float(dataflow_quality[1]),
            "max_rgb_error": maximum_rgb_error,
            "quality_samples": quality,
            "dataflow_rgb_sha256": dataflow_rgb_sha256,
        }
