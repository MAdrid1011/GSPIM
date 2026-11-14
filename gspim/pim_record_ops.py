"""GPU publication, spatial routing and block consumer operators."""
import hashlib
import torch

class SourceOps:

    def __init__(self, native, binary):
        self.native = native
        self.binary = binary

    def __getattr__(self, name):
        if name == 'ordered_tiles':
            return self.binary.ordered_tiles
        if name == 'tile_ranges':
            return self.binary.tile_ranges
        if name == 'cull_tiles':

            def cull(*args):
                args = list(args)
                args[-1] = args[2].shape[1].bit_length()
                assert args[-1] + len(args[2]) + 8 <= 32
                return self.native.cull_tiles(*args)
            return cull
        return getattr(self.native, name)

def render_window(pipeline, means, projected, orders, stable, shared_tiles, output=None, visible_ids=None, wait_for=None, model=None, prepare_fields=None):
    model = pipeline.model if model is None else model
    renderer = pipeline.renderer
    if not projected[4].numel():
        projected = (*projected[:4], renderer.colors(model, means, projected[0]), projected[5])
    if not shared_tiles:
        if output is None:
            output = projected[1].new_empty((len(means), model.height, model.width, 4))
        for frame in range(len(means)):
            local = tuple((value[frame:frame + 1] if value.numel() else value for value in projected))
            order = orders[frame].long()
            render_window(pipeline, means[frame:frame + 1], local, orders[frame:frame + 1], order, True, output[frame:frame + 1], order, wait_for, model, prepare_fields)
        footprint = sum((value.numel() * value.element_size() for value in projected))
        return (output[..., :3], len(means), footprint)
    if visible_ids is None:
        visible_ids, _ = renderer.ops.visible_order(stable, projected[0], projected[1].new_empty(0))
    orders = visible_ids.long()[None].expand(len(means), -1)
    assignment = orders.new_zeros(orders.shape)
    if prepare_fields is not None:
        prepare_fields(projected)
    image = renderer.window(model, projected, orders, assignment, 1, True, output=output, wait_for=wait_for)
    footprint = sum((value.numel() * value.element_size() for value in projected))
    return (image, 1, footprint)

class ExpandedTiles:

    def __init__(self, native, binary):
        self.native = native
        self.binary = binary
        self.capture = False
        self.cursor = 0
        self.prepared = []
        self.observations = []

    def __getattr__(self, name):
        return getattr(self.native, name)

    def reset_tiles(self, capture=False):
        self.capture = capture
        self.cursor = 0
        self.observations = []
        if capture:
            self.prepared = []

    def ordered_tiles(self, xy, radii, order, assignment, B, tw, th, ts, shared):
        assert shared and B == 1
        descriptors, publication = self.binary.rectangles(xy, radii, order, assignment, tw, th, ts)
        if self.capture:
            keys, ids = self.native.ordered_tiles(xy, radii, order, assignment, B, tw, th, ts, shared)
            self.prepared.append((keys.clone(), ids.clone(), descriptors.clone()))
        keys, ids, _ = self.prepared[self.cursor]
        self.cursor += 1
        self.observations.append((descriptors, publication, keys, ids, tw))
        return (keys, ids if getattr(self, 'physical_only', False) and (not self.capture) else ids.clone())

    def evidence(self):
        import numpy as np
        rows = []
        for (raw, timer, keys, ids, tw), (_, _, expected) in zip(self.observations, self.prepared):
            assert torch.equal(raw, expected)
            desc = raw.cpu().numpy()
            actual_keys = keys.cpu().numpy()
            actual_ids = ids.cpu().numpy()
            counts = desc[:, 6].astype(np.int64)
            offsets = np.r_[0, np.cumsum(counts)]
            assert offsets[-1] == len(actual_keys)
            for start in range(0, len(desc), 4096):
                batch = desc[start:start + 4096]
                lengths = counts[start:start + 4096]
                g = np.repeat(batch[:, 0], lengths)
                base = np.repeat(offsets[start:start + len(batch)], lengths)
                item = np.arange(offsets[start], offsets[start + len(batch)]) - base
                nx = np.repeat(batch[:, 3] - batch[:, 2], lengths)
                x = np.repeat(batch[:, 2], lengths) + item % nx
                y = np.repeat(batch[:, 4], lengths) + item // nx
                lo, hi = (offsets[start], offsets[start + len(batch)])
                assert np.array_equal(actual_ids[lo:hi], g)
                assert np.array_equal(actual_keys[lo:hi], y * tw + x)
            rows.append(dict(rectangles=raw.cpu(), publication=timer.cpu(), keys_sha256=hashlib.sha256(actual_keys.tobytes()).hexdigest(), ids_sha256=hashlib.sha256(actual_ids.tobytes()).hexdigest(), records=len(keys)))
        return rows

class SpatialExpandedTiles(ExpandedTiles):
    spatial_ownership = True

    def __init__(self, native, rectangles, router):
        super().__init__(native, rectangles)
        self.router = router
        self.physical = []
        self.directory = []
        self.routed = []
        self.current_supply = None
        self.tile_inputs = {}
        self.tile_keys = {}
        self.tile_plans = []
        self.buffers = []
        self.range_cache = []
        self.raw_indices = []
        self.capture_indices = []

    def reset_tiles(self, capture=False):
        super().reset_tiles(capture)
        self.routed = []
        self.current_supply = None
        if capture:
            self.physical = []
            self.directory = []
            self.tile_inputs.clear()
            self.tile_keys.clear()
            self.tile_plans = []
            self.buffers = []
            self.range_cache = []
            self.raw_indices = []
            self.capture_indices = []

    def pack_tiles(self, keys, ids):
        tw, th = (self.routed[-1]['tw'], self.routed[-1]['th'])
        tiles = tw * th
        counts = torch.bincount(keys.long(), minlength=tiles).int()
        tile = torch.arange(tiles, device=keys.device)
        owner = tile % tw // ((tw + 7) // 8) + 8 * (tile // tw // ((th + 3) // 4))
        capacities = torch.bincount(self.prepared[self.cursor - 1][0].long(), minlength=tiles).int()
        assert bool(torch.all(capacities >= counts))
        padded = (capacities + 7) // 8 * 8
        locations = torch.empty((tiles, 2), device=keys.device, dtype=torch.int32)
        capacity = 0
        for die in range(32):
            selected = owner == die
            values = padded[selected]
            locations[selected, 0] = die
            locations[selected, 1] = values.cumsum(0, dtype=torch.int32) - values
            capacity = max(capacity, int(values.sum()))
        physical = torch.empty((32, capacity), device=keys.device, dtype=torch.int32)
        prefix = counts.cumsum(0) - counts
        index = torch.arange(len(ids), device=ids.device) - prefix[keys.long()]
        physical[owner[keys.long()], locations[keys.long(), 1].long() + index] = ids
        counts = torch.cat((counts, counts.new_zeros(1)))
        self.tile_inputs[ids.data_ptr()] = (physical, counts, locations)
        self.tile_keys[keys.data_ptr()] = counts
        self.tile_plans.append((physical, counts, locations))

    def tile_ranges(self, keys, W, B, tiles, drop, packed):
        assert W == B == 1 and drop == 0 and packed
        if self.capture:
            self.range_cache.append(self.router.ranges(self.tile_keys[keys.data_ptr()]))
        jobs, offsets, counts, reads = self.range_cache[self.cursor - 1]
        return (jobs, offsets, counts)

    def wait_group(self):
        return

    def ordered_tiles(self, *args):
        if self.capture:
            keys, ids = super().ordered_tiles(*args)
        else:
            xy, radii, order, assignment, B, tw, th, ts, shared = args
            assert shared and B == 1
            pool = self.cursor
            metadata, bitmap, publication, raw = self.router.route_projected(xy, radii, order, assignment, tw, th, ts, *self.buffers[pool])
            keys, ids, _ = self.prepared[self.cursor]
            self.cursor += 1
            self.observations.append((raw, publication, keys, ids, tw))
        raw = self.observations[-1][0]
        tw, th = args[5:7]
        pool = self.cursor - 1
        if self.capture:
            N = len(raw)
            blocks = (N + 255) // 256
            if pool == len(self.buffers):
                self.buffers.append(None)
            if self.buffers[pool] is None or self.buffers[pool][0].shape[1] < N:
                assert self.capture, 'descriptor pool must be sized before timing'
                self.buffers[pool] = (torch.empty((32, N, 8), device=raw.device, dtype=torch.int32), torch.empty((32, blocks * 8), device=raw.device, dtype=torch.int32), torch.empty((blocks, 2), device=raw.device, dtype=torch.int64))
            metadata, bitmap, publication = self.router.route_into(raw, tw, th, *self.buffers[pool], True)
        if self.capture:
            n = len(ids)
            index = torch.arange(n, device=ids.device)
            die = keys % tw // ((tw + 7) // 8) + 8 * (keys // tw // ((th + 3) // 4))
            permutation = torch.argsort(die, stable=True)
            sorted_die = die[permutation]
            counts = torch.bincount(die.long(), minlength=32)
            offsets = counts.cumsum(0) - counts
            local = index - offsets[sorted_die.long()]
            capacity = int(counts.max())
            physical = torch.empty((32, capacity, 2), device=ids.device, dtype=torch.int32)
            physical[sorted_die.long(), local, 0] = keys[permutation]
            physical[sorted_die.long(), local, 1] = ids[permutation]
            self.raw_indices.append(permutation.int())
            mapping = torch.empty((32, capacity), device=ids.device, dtype=torch.int32)
            mapping[sorted_die.long(), local] = permutation.int()
            self.capture_indices.append(mapping)
            inverse = torch.empty(int(raw[:, 0].max()) + 1, device=ids.device, dtype=torch.int64)
            inverse[raw[:, 0].long()] = torch.arange(len(raw), device=ids.device)
            sorted_rank = inverse[ids[permutation].long()].cpu().numpy()
            records = []
            for owner, (start, count) in enumerate(zip(offsets.cpu().tolist(), counts.cpu().tolist())):
                for first in range(0, count, 256):
                    last = min(count, first + 256)
                    rank = int(sorted_rank[start + last - 1])
                    records.append((owner, first, rank // 256, last - first))
            directory = torch.tensor(records, device=ids.device, dtype=torch.int32)
            self.physical.append(physical)
            self.directory.append(directory)
        physical = self.physical[self.cursor - 1]
        directory = self.directory[self.cursor - 1]
        expansion_publication = None
        delay = torch.zeros(len(directory), device=ids.device, dtype=torch.int64)
        mask = torch.empty((*physical.shape[:2], 1), device=ids.device, dtype=torch.int32)
        reads = torch.empty_like(delay)
        self.current_supply = [physical, publication, delay, mask, reads, directory]
        self.current_supply.extend((ids.new_empty(0), self.capture_indices[self.cursor - 1]))
        self.routed.append(dict(metadata=metadata, bitmap=bitmap, publication=publication, physical=physical, mask=mask, reads=reads, directory=directory, records=len(ids), tw=tw, th=th, raw_indices=self.raw_indices[self.cursor - 1], compact_wire=True))
        if expansion_publication is not None:
            self.routed[-1]['gpu_expansion_publication'] = expansion_publication
        return (keys, ids)

    def spatial_evidence(self, sources):
        import numpy as np
        result = []
        for value, observed, prepared, expected, plan in zip(self.routed, self.observations, self.prepared, sources, self.tile_plans):
            rectangles = observed[0].cpu().numpy()
            N = len(rectangles)
            metadata = value['metadata'].cpu().numpy()
            bitmap = value['bitmap'].cpu().numpy().view(np.uint32)
            physical = value['physical'].cpu().numpy()
            mask = value['mask'].cpu().numpy()
            keys, ids = (v.cpu().numpy() for v in prepared[:2])
            prefix = np.r_[0, np.cumsum(rectangles[:, 6], dtype=np.int64)]
            live = []
            headers = []
            raw_count = 0
            for die in range(32):
                selected = np.flatnonzero(bitmap[die, np.arange(N) // 32] >> np.arange(N) % 32 & 1)
                desc = metadata[die, selected]
                live.append(desc.copy())
                headers.append(len(desc))
                cursor = 0
                expected_flags = expected['flags'].numpy()
                source_ids = expected['ids'].numpy()
                for begin in range(0, len(desc), 4096):
                    batch = desc[begin:begin + 4096]
                    g, xmin, xmax, ymin, ymax, first, nx, rank = batch.T
                    assert np.array_equal(rectangles[rank, 0], g)
                    assert np.array_equal(first, (ymin - rectangles[rank, 4]) * nx
                                          + xmin - rectangles[rank, 2])
                    lengths = (xmax - xmin) * (ymax - ymin)
                    offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
                    owner = np.repeat(np.arange(len(batch)), lengths)
                    local = np.arange(offsets[-1]) - offsets[owner]
                    x = xmin[owner] + local % (xmax - xmin)[owner]
                    y = ymin[owner] + local // (xmax - xmin)[owner]
                    original = first[owner] + (y - ymin[owner]) * nx[owner] + x - xmin[owner] + prefix[rank[owner]]
                    count = len(original)
                    pairs = physical[die, cursor:cursor + count]
                    assert np.array_equal(pairs[:, 0], y * value['tw'] + x)
                    assert np.array_equal(pairs[:, 1], g[owner])
                    if pairs.shape[1] > 2:
                        assert np.array_equal(pairs[:, 2], original)
                    assert np.array_equal(keys[original], pairs[:, 0]) and np.array_equal(ids[original], pairs[:, 1])
                    encoded = mask[die, cursor:cursor + count, 0]
                    assert np.array_equal(encoded != 0, expected_flags[original])
                    expected_ids = source_ids[original].copy()
                    expected_ids[expected_flags[original] == 0] = 0
                    assert np.array_equal(encoded, expected_ids)
                    cursor += count
                raw_count += cursor
            assert raw_count == len(keys)
            selected = expected['flags'].numpy().astype(bool)
            logical_keys = keys[selected]
            logical_ids = expected['ids'].numpy()[selected]
            order = np.argsort(logical_keys, kind='stable')
            logical_keys = logical_keys[order]
            logical_ids = logical_ids[order]
            packed, counts, locations = (v.cpu().numpy() for v in plan)
            assert len(logical_ids) == int(counts.sum())
            prefix = np.r_[0, np.cumsum(counts[:-1], dtype=np.int64)]
            for tile, count in enumerate(counts[:-1]):
                die, offset = locations[tile]
                assert np.array_equal(packed[die, offset:offset + count], logical_ids[prefix[tile]:prefix[tile + 1]])
            row = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in value.items() if k not in ('metadata', 'physical')}
            encoded = row['mask'][..., 0]
            row['mask'] = torch.stack(((encoded != 0).int(), encoded), -1)
            row.update(metadata=torch.from_numpy(np.concatenate(live)), descriptor_counts=headers, tile_ids=torch.from_numpy(packed), tile_counts=torch.from_numpy(counts), tile_locations=torch.from_numpy(locations), physical_sha256=hashlib.sha256(physical.tobytes()).hexdigest(), raw_reservations=True)
            result.append(row)
        return result

class SweptMasks:

    def __init__(self, binary, fallback, publication):
        self.binary = binary
        self.fallback = fallback
        self.publication = bool(publication)
        self.supply = []
        self.physical_only = False

    def __getattr__(self, name):
        return getattr(self.fallback, name)

    def cull_masks(self, *args):
        values = self.binary.cull_masks(*args, self.publication or bool(self.supply), self.supply, self.physical_only)
        self.last_smids = values[3] if len(values) > 3 else None
        return values[:3]

class SweptGPU:

    def __init__(self, masks, native):
        self.masks = masks
        self.native = native

    def __getattr__(self, name):
        return getattr(self.native, name)

    def cull_tiles(self, *args):
        if args[0].dtype != torch.int32 or len(args[2]) <= 1 or (not args[-1]) or (args[-1] + len(args[2]) + 8 > 32):
            return self.native.cull_tiles(*args)
        flags, encoded, _ = self.masks.binary.cull_masks(*args, False, [], False)
        return self.masks.binary.compact(args[0], encoded, flags)

class HCFNativeSupply:

    def __init__(self, native, masks):
        self.native = native
        self.masks = masks
        self.tile_results = []
        self.key_slots = {}
        self.actual_sources = []
        self.cull_stream = None
        self.cull_completions = []
        self.cull_info = {}
        self.sources = []
        self.classes = []
        self.sort_keys = []
        self.culls = []
        self.capture = False
        self.cull_cursor = 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    def reset(self, capture=False):
        if hasattr(self.native, 'reset_tiles'):
            self.native.reset_tiles(capture)
        self.capture = capture
        self.cull_cursor = 0
        self.class_cursor = self.key_cursor = 0
        self.actual_sources = []
        self.cull_completions = []
        if capture:
            self.tile_results = []
            self.key_slots = {}
            self.cull_info = {}
            self.sources = []
            self.culls = []
            self.classes = []
            self.sort_keys = []

    def classify(self, score):
        if self.capture:
            stable, unstable, delta = self.native.classify(score)
            quantized = torch.floor(score.double() * (1 << 20) + 0.5).to(torch.int64)
            mask = quantized > 1027605
            assert torch.equal(stable, mask.nonzero().flatten().to(torch.int32))
            assert torch.equal(unstable, (~mask).nonzero().flatten().to(torch.int32))
            self.classes.append((stable.clone(), unstable.clone(), delta))
        else:
            stable, unstable, expected = self.classes[self.class_cursor]
            delta = self.masks.window_delta(score)
            assert delta == expected
        self.class_cursor += 1
        return (stable, unstable, delta)

    def sorting_inputs(self, ids, keys):
        if self.capture:
            selected = keys[ids.long()].contiguous()
            self.sort_keys.append(selected.clone())
        else:
            selected = self.sort_keys[self.key_cursor]
            assert len(selected) == len(ids)
        self.key_cursor += 1
        return (ids, selected)

    def cull_tiles(self, *args):
        args = list(args)
        args[-1] = args[2].shape[1].bit_length()
        args = tuple(args)
        if self.capture:
            mask_args = list(args)
            mask_args[1] = args[1].clone()
            flags, encoded = self.cull_masks(*mask_args)
            selected = flags.bool()
            self.sources.append(dict(keys=args[0].cpu(), flags=flags.cpu(), ids=encoded.cpu()))
            keys, ids = (args[0][selected], encoded[selected])
            assert args[0].numel() < 1 << 32 and keys.dtype == torch.int32
            tw, th = args[5:7]
            die = keys % tw // ((tw + 7) // 8) + 8 * (keys // tw // ((th + 3) // 4))
            permutation = torch.argsort(die, stable=True)
            local_keys = keys[permutation].clone()
            local_ids = ids[permutation].clone()
            self.cull_info[local_keys.data_ptr()] = (keys.clone(), ids.clone())
            keys, ids = (local_keys, local_ids)
            self.culls.append((keys, ids))
            self.key_slots[keys.data_ptr()] = len(self.culls) - 1
            self.tile_results.append(None)
        else:
            flags, encoded = self.cull_masks(*args)
            keys, ids = self.culls[self.cull_cursor]
        self.cull_cursor += 1
        return (keys, ids)

    def cull_masks(self, *args):
        local_supply = getattr(self.native, 'current_supply', None)
        if hasattr(self.masks, 'supply'):
            self.masks.supply = local_supply or []
        physical_only = not self.capture
        if hasattr(self.masks, 'physical_only'):
            self.masks.physical_only = physical_only
        if self.cull_stream is None or self.capture:
            values = self.masks.cull_masks(*args)
        else:
            self.cull_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.cull_stream):
                for value in args:
                    if isinstance(value, torch.Tensor):
                        value.record_stream(self.cull_stream)
                for value in local_supply or []:
                    value.record_stream(self.cull_stream)
                values = self.masks.cull_masks(*args)
                done = torch.cuda.Event()
                done.record()
                self.cull_completions.append(done)
        self.native.routed[-1]['cull_publication'] = values[2]
        if getattr(self.masks, 'last_smids', None) is not None:
            self.native.routed[-1]['cull_smids'] = self.masks.last_smids
        if not self.capture:
            self.actual_sources.append(dict(keys=args[0], flags=values[0], ids=values[1]))
            if physical_only:
                self.actual_sources[-1].update(physical=local_supply[0], mask=local_supply[3], directory=local_supply[5], raw_indices=self.native.routed[-1]['raw_indices'])
        return values[:2]

    def validate_sources(self):
        import numpy as np
        assert len(self.actual_sources) == len(self.sources)
        for index, (actual, expected) in enumerate(zip(self.actual_sources, self.sources)):
            if 'physical' in actual:
                physical = actual['physical'].cpu().numpy()
                mask = actual['mask'].cpu().numpy()
                flags = np.empty_like(expected['flags'].numpy())
                ids = np.empty_like(expected['ids'].numpy())
                directory = actual['directory'].cpu().numpy().astype(np.int64)
                raw_indices = actual['raw_indices'].cpu().numpy() if physical.shape[2] == 2 else None
                offset = 0
                for begin in range(0, len(directory), 4096):
                    batch = directory[begin:begin + 4096]
                    counts = batch[:, 3]
                    borders = np.r_[0, np.cumsum(counts)]
                    owner = np.repeat(np.arange(len(batch)), counts)
                    local = np.arange(borders[-1]) - borders[owner]
                    dies, positions = batch[owner, 0], batch[owner, 1] + local
                    indices = (physical[dies, positions, 2] if raw_indices is None else
                               raw_indices[offset:offset + len(owner)]).astype(np.int64, copy=False)
                    encoded = mask[dies, positions, 0]
                    flags[indices] = encoded != 0 if mask.shape[2] == 1 else encoded
                    ids[indices] = encoded if mask.shape[2] == 1 else mask[dies, positions, 1]
                    offset += len(owner)
                if physical.shape[2] == 2:
                    ids[flags == 0] = expected['ids'].numpy()[flags == 0]
                actual = dict(actual, flags=flags, ids=ids)
            for key in expected:
                value = actual[key]
                value = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                reference = expected[key].numpy()
                assert np.array_equal(value, reference), (index, key, int(np.count_nonzero(value != reference)))

    def radix(self, keys, ids, begin, end):
        expected = self.cull_info.get(keys.data_ptr())
        if expected is None:
            return self.native.radix(keys, ids, begin, end)
        assert begin == 0 and end + 32 < 63
        slot = self.key_slots[keys.data_ptr()]
        if self.capture:
            result = self.native.radix(*expected, begin, end)
            self.tile_results[slot] = tuple((v.clone() for v in result))
            self.native.pack_tiles(*self.tile_results[slot])
        return self.tile_results[slot]

class DepthStatistics:

    def __init__(self, binary, native):
        self.binary = binary
        self.native = native
        self.mode = None
        self.validate = True
        self.trace_binary = None
        self.trace_rows = None
        self.generation_binary = None
        self.generation_trace_binary = None
        self.generation_enabled = False
        self.original_generate = native.generate
        self.generation_view = None
        self.generation_span = None
        self.generated = {}
        self.last_generation_publication = None
        self.last_generation_loads = None

    def generate(self, *args):
        if not self.generation_enabled or self.mode is None:
            return self.original_generate(*args)
        binary = self.generation_trace_binary if self.trace_binary is not None else self.generation_binary
        assert binary is not None and self.generation_view is not None
        result = binary.generate(*args, self.generation_view, self.generation_span, self.mode)
        key = result[0].data_ptr()
        assert key not in self.generated
        self.generated[key] = result[3:]
        self.last_generation_publication = result[6]
        self.last_generation_loads = result[7] if len(result) > 7 else None
        return result[:3]

    def run(self, means, view, span):
        generated = self.generated.pop(means.data_ptr(), None)
        if generated is not None:
            assert view.data_ptr() == self.generation_view.data_ptr() and span == self.generation_span
        if self.validate:
            expected = (means @ view[0, 2, :3] + view[0, 2, 3]).contiguous()
            stats = self.native.window_statistics(expected, span)
            modes = range(24) if self.mode is None else (self.mode,)
            for mode in modes:
                result = self.binary.run(means, view, span, mode)
                if torch.equal(result[0], expected):
                    assert all((torch.equal(a, b) for a, b in zip(result[1:], stats)))
                    self.mode = mode
                    if generated is not None:
                        assert all((torch.equal(a, b) for a, b in zip(generated[:3], result)))
                        return generated[:3]
                    return result
            raise AssertionError('No exact camera-dot implementation matches native depth')
        if generated is not None:
            if self.trace_binary is not None:
                self.trace_rows.append((*generated[:3], generated[3][:, :2]))
            return generated[:3]
        if self.trace_binary is not None:
            result = self.trace_binary.run(means, view, span, self.mode)
            self.trace_rows.append(result)
            return result[:3]
        return self.binary.run(means, view, span, self.mode)

class CrossingShortcut:

    def __init__(self, binary, native):
        self.binary = binary
        self.native = native
        self.minimum = None
        self.single_frame = False

    def defer(self, stable, count, minimum):
        self.minimum = minimum
        return len(stable) == count

    def visible(self, stable, radii, depth):
        if len(depth) == 1:
            self.single_frame = True
            return (stable.long(), True)
        self.single_frame = False
        if self.binary.witness(depth, radii):
            return (stable.long(), True)
        index = self.native.radix(self.minimum[stable.long()].contiguous(), stable.contiguous(), 0, 32)[1]
        return self.native.visible_order(index, radii, depth)

    def span(self, operators, key, selected):
        values = torch.arange(len(selected), dtype=torch.int32, device=selected.device)
        key, index = operators.radix(key, values, 0, 32)
        ids = selected[index.long()]
        if self.single_frame:
            return ids
        self.minimum.record_stream(torch.cuda.current_stream())
        output, overflow = self.binary.refine(key, ids, self.minimum)
        if overflow.item():
            keys = self.binary.keys(key, ids, self.minimum)
            order = operators.radix(keys, values, 0, 64)[1]
            output = ids[order.long()]
        return output

class SegmentedAoS:

    def __init__(self, binary):
        self.binary = binary
        self.trace = False
        self.prepared = []
        self.cursor = 0
        self.field_inputs = {}
        self.plans = {}
        self.observations = []
        self.plan_cursor = 0
        self.window_fields = []
        self.field_visibility = {}
        self.window_visibility = {}
        self.source_stream = torch.cuda.Stream(priority=-1)
        self.delivery = None
        self.spatial_source = None
        self.compacted_sources = []
        self.delivery = []
        self.producer_publications = {}
        self.producer_events = {}
        self.producer_completion = None
        self.producer_stream = torch.cuda.Stream(priority=-1)

    def reset(self):
        self.cursor = 0
        self.field_inputs = {}
        self.plans = {}
        self.observations = []
        self.plan_cursor = 0
        self.window_fields = []
        self.field_visibility = {}
        self.window_visibility = {}
        self.producer_publications = {}
        self.producer_events = {}
        self.producer_completion = None

    def begin_fields(self, projected):
        for whole, cutoff, physical, publication, ready in self.window_fields:
            stride = whole[5].stride(0) * whole[5].element_size()
            delta = projected[5].data_ptr() - whole[5].data_ptr()
            first = delta // stride
            last = first + len(projected[5])
            if delta >= 0 and delta % stride == 0 and (last <= len(whole[5])):
                self.field_inputs[projected[5].data_ptr()] = (projected, cutoff[first:last], physical[:, first:last], publication, ready)
                self.field_visibility[projected[5].data_ptr()] = self.window_visibility[whole[5].data_ptr()][first:last]
                blocks = getattr(self, 'field_blocks', {}).get(whole[5].data_ptr())
                if blocks is not None:
                    self.field_blocks[projected[5].data_ptr()] = blocks
                return
        self.source_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.source_stream):
            for value in projected:
                value.record_stream(self.source_stream)
            cutoff, physical, visibility = self.binary.emit_fields_selected(projected[1], projected[3], projected[4], projected[5], projected[0])
            self.field_visibility[projected[5].data_ptr()] = visibility
            publication = self.binary.publish(physical)
            ready = torch.cuda.Event()
            ready.record()
        self.field_inputs[projected[5].data_ptr()] = (projected, cutoff, physical, publication, ready)

    def begin_window_fields(self, projected):
        self.begin_fields(projected)
        self.window_fields.append(self.field_inputs[projected[5].data_ptr()])
        self.window_visibility[projected[5].data_ptr()] = self.field_visibility[projected[5].data_ptr()]

    def prepare_gpu(self, args):
        stored = self.prepared[self.plan_cursor][0]
        self.plan_cursor += 1
        if self.spatial_source is not None:
            physical_ids = self.compacted_sources[self.plan_cursor - 1]
        else:
            physical_ids = self.binary.emit_ids(args[5])
        publication = self.binary.publish(physical_ids)
        routed = self.spatial_source.routed[self.plan_cursor - 1]
        current = torch.cuda.current_stream()
        stream = self.producer_stream if self.producer_completion is not None else current
        if stream != current:
            stream.wait_stream(current)
            stream.wait_event(self.producer_completion)
        with torch.cuda.stream(stream):
            for value in (routed['cull_publication'], routed['directory']):
                value.record_stream(stream)
            self.producer_publications[args[3].data_ptr()] = self.binary.producer_anchors(routed['cull_publication'], routed['directory'])
            ready = torch.cuda.Event()
            ready.record()
            self.producer_events[args[3].data_ptr()] = ready
        self.plans[args[3].data_ptr()] = ((*stored, physical_ids), publication, args[4], args[9])

    def prepare(self, args):
        cutoff = self.binary.cutoffs(args[3])
        plan = self.linked_plan(args[4], args[9], args[5], len(args[0]), args[6], args[7])
        n = len(plan[1])
        arguments = (plan[5], plan[1], plan[2], n, len(args[0]), args[12])
        counts, ids = self.binary.pack_segments(*arguments)
        fields = self.binary.fields(*args[:4], cutoff)
        saved = (plan[0].clone(), plan[1][:n].clone(), plan[2][:n].clone(), plan[3].clone(), plan[4][:n].clone())
        self.prepared.append((saved, counts, ids, fields, tuple(args[0].shape)))
        delays = torch.zeros((n, 3), device=counts.device, dtype=torch.int64)
        delays[:, 2] = plan[1][:n, 3]
        self.delivery.append(delays)
        self.compacted_sources.append(plan[5].clone())

    def linked_plan(self, offsets, jobs, encoded, W, width, height):
        import numpy as np
        tw, th = ((width + 15) // 16, (height + 15) // 16)
        borders = offsets.cpu().numpy().astype(np.int64)
        lengths = np.diff(borders)
        per_task = np.repeat(lengths // 256 + 1, W * 2)
        heads = np.r_[0, np.cumsum(per_task, dtype=np.int64)[:-1]]
        tasks = np.repeat(np.arange(len(per_task)), per_task)
        tile = tasks // (W * 2)
        first = borders[tile] + (np.arange(len(tasks)) - heads[tasks]) * 256
        last = np.minimum(borders[tile + 1], first + 256)
        die = tile % tw // ((tw + 7) // 8) + 8 * (tile // tw // ((th + 3) // 4))
        links = np.where(last - first == 256, np.arange(len(tasks)) + 1, -1)
        values = np.column_stack((first, last, tasks, die)).astype(np.int32)
        order = np.argsort(values[:, 3], kind='stable')
        locations = np.empty(len(values), np.int32)
        locations[order] = np.arange(len(values))
        die_counts = np.bincount(values[:, 3], minlength=32)
        die_starts = np.r_[0, np.cumsum(die_counts)].astype(np.int32)
        routed = values[order].copy()
        routed[:, 3] = order
        dev = encoded.device
        tensors = [torch.tensor(v, device=dev, dtype=torch.int32) for v in (np.concatenate((heads, links)), values, locations, die_starts, routed)]
        return (*tensors, self.binary.emit_ids(encoded))

    def batches(self, *args):
        stored, counts, ids, fields, shape = self.prepared[self.cursor]
        self.cursor += 1
        assert tuple(args[0].shape) == shape
        key = args[3].data_ptr()
        assert key in self.field_inputs
        if key not in self.plans:
            self.prepare_gpu(args)
        plan, publication, offsets, jobs = self.plans[key]
        torch.cuda.current_stream().wait_event(self.producer_events[key])
        self.producer_publications[key].record_stream(torch.cuda.current_stream())
        torch.cuda.current_stream().wait_event(self.field_inputs[key][-1])
        for value in (*plan, *self.field_inputs[key][1:-1]):
            value.record_stream(torch.cuda.current_stream())
        trace = torch.zeros((len(counts), 13 if self.delivery is not None else 12), dtype=torch.int64, device=args[0].device) if self.trace else torch.empty(0, dtype=torch.int64, device=args[0].device)
        divisions = self.binary.divisions() if hasattr(self.binary, 'divisions') else 2
        tasks = (args[6] + 15) // 16 * ((args[7] + 15) // 16) * len(args[0]) * divisions
        task_trace = torch.zeros((tasks, 3), dtype=torch.int64, device=args[0].device) if self.trace else trace
        arguments = (plan[0], plan[2], counts, ids, fields, len(args[0]), args[6], args[7], args[12], args[0].shape[1], args[21], args[22], trace, task_trace)
        if self.delivery is not None:
            delays = self.delivery[self.cursor - 1]
            assert len(delays) == len(counts)
            self.binary.consume(*arguments, delays, self.producer_publications[key], self.field_inputs[key][3])
        else:
            self.binary.consume(*arguments)
        if self.trace:
            self.observations.append(dict(plan=plan, publication=publication, fields=self.field_inputs[key], input_ids=args[5], stored=stored, counts=counts, ids=ids, expected_fields=fields, trace=trace, task_trace=task_trace, bits=args[12], width=len(args[0]), input_offsets=offsets, input_jobs=jobs, pixels_width=args[6], pixels_height=args[7]))
        return args[21][..., :3]

    def evidence(self):
        import numpy as np
        rows = []
        for value in self.observations:
            plan = value['plan']
            n = len(value['counts'])
            current = self.linked_plan(value['input_offsets'], value['input_jobs'], value['input_ids'], value['width'], value['pixels_width'], value['pixels_height'])
            for i, p in enumerate(plan[:5]):
                actual = p[:n] if i in (1, 2, 4) else p
                assert torch.equal(actual, value['stored'][i])
                assert torch.equal(actual, current[i][:n] if i in (1, 2, 4) else current[i])
            projected, cutoff, physical, publication, _ = value['fields']
            fields = self.binary.fields(projected[1], projected[3], projected[4], projected[5], cutoff)
            ids = value['ids'].reshape(n, 256)
            counts = value['counts']
            locations = plan[2][:n]
            keep = torch.arange(256, device=ids.device)[None] < counts[:, 0, None]
            valid = ids[keep].long() & (1 << value['bits']) - 1
            frame_for_location = torch.empty(n, dtype=torch.int64, device=ids.device)
            frame_for_location[locations.long()] = (plan[1][:n, 2] // 2 % value['width']).long()
            frames = frame_for_location[:, None].expand(-1, 256)[keep]
            N = projected[1].shape[1]
            field_ids = frames * N + valid
            live = torch.unique(field_ids)
            assert torch.equal(fields[live], value['expected_fields'][live])
            raw = [v.cpu() for v in (projected[1], projected[3], projected[4], projected[5], cutoff)]
            memory = physical.cpu()
            slots = (N + 98303) // 98304 * 256
            words = memory.numpy().view(np.uint32).reshape(384, value['width'], slots * 10)
            g = np.arange(N)
            block = g // 256
            owner = block % 384
            slot = block // 384 * 256 + g % 256
            base = 0
            for source, stride in zip(raw, (2, 3, 3, 1, 1)):
                expected = source.numpy().view(np.uint32).reshape(value['width'], N, stride)
                for f in range(value['width']):
                    actual = words[owner[:, None], f, base * slots + slot[:, None] * stride + np.arange(stride)]
                    assert np.array_equal(actual, expected[f])
                base += stride
            input_ids = value['input_ids'].cpu()
            physical_ids = plan[5].cpu()
            k = np.arange(len(input_ids))
            block = k // 256
            owner = block % 32 * 12 + block // 32 % 12
            input_slots = physical_ids.numel() // 384
            offset = block // 384 * 256 + k % 256
            assert np.array_equal(physical_ids.numpy().reshape(384, input_slots)[owner, offset], input_ids.numpy())
            rows.append(dict(starts=plan[0].cpu(), descriptors=plan[1][:n].cpu(), locations=locations.cpu(), die_starts=plan[3].cpu(), routed_descriptors=plan[4][:n].cpu(), counts=counts.cpu(), ids=ids.cpu(), input_ids=input_ids, physical_ids=physical_ids, physical_fields=memory, fields=fields.cpu(), field_publication=publication.cpu(), id_publication=value['publication'].cpu(), trace=value['trace'].cpu(), task_trace=value['task_trace'].cpu(), bits=value['bits'], width=value['width'], N=N, source_xy=raw[0], source_conic=raw[1], source_rgb=raw[2], source_opacity=raw[3], source_log_opacity=raw[4], live_field_ids=live.cpu(), input_offsets=value['input_offsets'].cpu(), input_jobs=value['input_jobs'].cpu(), pixels_width=value['pixels_width'], pixels_height=value['pixels_height']))
            visibility = self.field_visibility[projected[5].data_ptr()].cpu()
            words = visibility.numpy().view(np.uint32).reshape(value['width'], -1)
            assert np.all(words[frames.cpu().numpy(), valid.cpu().numpy() // 32] >> valid.cpu().numpy() % 32 & 1)
            radii = projected[0].cpu().numpy()
            expected = (radii[..., 0] > 0) & (radii[..., 1] > 0)
            g = np.arange(N)
            assert np.array_equal((words[:, g // 32] >> g % 32 & 1).astype(bool), expected)
            rows[-1]['field_visibility'] = visibility
            blocks = getattr(self, 'field_blocks', {}).get(projected[5].data_ptr())
            if blocks is not None:
                rows[-1].update(field_block_publication=blocks.cpu(), field_block_records=64)
        return rows

class CompactFields:

    def __init__(self, native, binary):
        self.native = native
        self.binary = binary
        self.inputs = {}

    def __getattr__(self, name):
        return getattr(self.native, name)

    def fields(self, *args):
        output = self.native.fields(*args)
        self.inputs[output.data_ptr()] = self.binary.compact(output)
        return output

    def consume(self, *args):
        import torch
        source = args[4]
        compact = self.inputs[source.data_ptr()]
        assert source.shape[-1] == 12 and compact.shape[-1] == 10
        compact.record_stream(torch.cuda.current_stream())
        return self.binary.consume(*args[:4], compact, *args[5:])

    def launch_resources(self):
        return self.binary.launch_resources()

class ColorFieldBlocks:

    def __init__(self, renderer, consumer, binary):
        self.renderer = renderer
        self.consumer = consumer
        self.binary = binary
        self.original_colors = renderer.colors
        self.original_fields = consumer.begin_fields
        self.reset()

    def reset(self):
        self.projections = {}
        self.pending = {}
        self.streams = {}
        self.publications = []
        self.load_frontiers = []
        self.consumer.field_blocks = {}

    def remember(self, means, projected):
        self.projections[means.data_ptr()] = projected

    def colors(self, model, means, radii):
        projected = self.projections.pop(means.data_ptr())
        assert projected[0].data_ptr() == radii.data_ptr()
        temporal = hasattr(model, 'temporal_means')
        if hasattr(model, 'anchor_positions'):
            result = self.binary.run(model.decoded_colors, radii, projected[1], projected[3], projected[5])
        else:
            result = self.binary.run(model.xyz[None] if temporal else means, radii, model.sh, model.camera_position, projected[1], projected[3], projected[5], *([model.frame_times, model.temporal_means] if temporal else []))
        rgb, cutoff, physical, visibility, publication, loads = result
        self.pending[projected[5].data_ptr()] = (cutoff, physical, visibility, publication)
        self.publications.append(publication)
        self.load_frontiers.append(loads)
        return rgb

    def fields(self, projected):
        key = projected[5].data_ptr()
        prepared = self.pending.pop(key, None)
        if prepared is None:
            return self.original_fields(projected)
        cutoff, physical, visibility, blocks = prepared
        stream = self.streams.pop(key, torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            publication = self.consumer.binary.publish(physical)
            ready = torch.cuda.Event()
            ready.record()
        self.consumer.field_inputs[key] = (projected, cutoff, physical, publication, ready)
        self.consumer.field_visibility[key] = visibility
        self.consumer.field_blocks[key] = blocks
