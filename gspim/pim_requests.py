"""Physical PPIM/HCF traffic for captured GPU publications and input demand."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
import numpy as np
from .pim_memory import DTYPE, NONE

def address(bank, row=0, column=0):
    return ((row * 16 + bank) * 64 + column) * 32

class RequestRows:

    def __init__(self, initial=None, chunk_size=16384):
        self.chunks = [] if initial is None else [np.asarray(initial, dtype=DTYPE)]
        self.pending = []
        self.size = 0 if initial is None else len(initial)
        self.chunk_size = chunk_size

    def __len__(self):
        return self.size

    def append(self, row):
        self.pending.append(row)
        self.size += 1
        if len(self.pending) >= self.chunk_size:
            self.flush()

    def extend(self, rows):
        previous = len(self.pending)
        self.pending.extend(rows)
        self.size += len(self.pending) - previous
        if len(self.pending) >= self.chunk_size:
            self.flush()

    def extend_uniform(self, addresses, ready, kind, depends, cost, tag):
        if not len(addresses):
            return
        rows = np.empty(len(addresses), dtype=DTYPE)
        rows['address'] = addresses
        rows['ready'] = ready
        rows['kind'] = kind
        rows['depends'] = depends
        rows['cost'] = cost
        rows['tag'] = tag
        self.flush()
        self.chunks.append(rows)
        self.size += len(rows)

    def flush(self):
        if self.pending:
            self.chunks.append(np.array(self.pending, dtype=DTYPE))
            self.pending.clear()

    def finish(self):
        self.flush()
        if not self.chunks:
            return np.empty(0, dtype=DTYPE)
        result = self.chunks[0] if len(self.chunks) == 1 else np.concatenate(self.chunks)
        self.chunks.clear()
        assert len(result) == self.size
        return result

def vector_steps(keys, flags, width=16, banks=16):
    if not len(keys):
        return 0
    stride = int(keys.max()) + 1
    batches = (len(keys) + width - 1) // width
    grouped = np.arange(len(keys)) // width * stride + keys
    unique = np.unique(grouped[flags.astype(bool)])
    counts = np.bincount(unique // stride * banks + unique % stride % banks,
                         minlength=batches * banks)
    return int(np.maximum(1, counts.reshape(batches, banks).max(1)).sum())

def field_origin(capture):
    """Share one epoch without changing the whole-window delivery event."""
    return min((int(raw['field_block_publication'][:, 1].min()) if 'field_block_publication' in raw else int(raw['field_publication'][0]) for raw in capture['segmented_supply']))

def field_requests(capture, fabric=4, spans=4):
    rows = [RequestRows() for _ in range(32)]
    records = [[] for _ in range(32)]
    origin = field_origin(capture)
    checked = []
    field_records = []
    destination_words = 10 if capture['configuration'].get('compact_consumer_fields', False) else 12
    for span, raw in enumerate(capture['segmented_supply'][:spans]):
        desc = raw['descriptors'].numpy().astype(np.int64)
        source = raw['input_ids'].numpy().view(np.uint32)
        trace = raw['trace'].numpy()
        W = raw['width']
        ready = int(raw['id_publication'][0]) - origin
        N = raw['N']
        field_slots = (N + 98303) // 98304 * 256
        bitmap = raw['field_visibility'].numpy().view(np.uint32).reshape(W, -1)
        source_ids = np.arange(N)
        visibility = (bitmap[:, source_ids // 32] >> source_ids % 32 & 1).astype(bool)
        field_memory = raw['physical_fields'].numpy().view(np.uint32).reshape(384, W, field_slots * 10)
        expected = raw['fields'].numpy().view(np.uint32).reshape(W, N, 12)
        g = np.arange(N)
        owner = g // 256 % 384
        slot = g // 98304 * 256 + g % 256
        actual = np.zeros((W, N, 12), np.uint32)
        base = 0
        destination = [(0, 1), (2, 3, 4), (6, 7, 8), (5,), (9,)]
        for stride, columns in zip((2, 3, 3, 1, 1), destination):
            for f in range(W):
                value = field_memory[owner[:, None], f, base * field_slots + slot[:, None] * stride + np.arange(stride)]
                actual[f][:, columns] = value
            base += stride
        assert np.array_equal(actual, expected), 'field destination must preserve source block ownership'
        assert not np.any(expected[..., 10:]), 'destination padding must be zero'
        publications = raw.get('field_block_publication')
        assert raw['field_block_records'] == 64
        publications = publications.numpy()
        assert publications.shape == ((N + 63) // 64, 3)
        assert np.all(publications[:, 1] <= int(raw['field_publication'][0]))
        chunks = np.argsort(publications[:, 1], kind='stable')
        jobs = ((chunk, f) for chunk in chunks for f in range(W))
        for chunk, f in jobs:
            lo = int(chunk) * 64
            release = int(publications[chunk, 1]) - origin
            assert release >= 0
            hi = min(N, lo + 64)
            block = lo // 256
            die = block % 384 // 12
            bank = block % 12
            local = block // 384 * 256 + lo % 256
            selected = np.arange(hi - lo)
            selected = selected[visibility[f, lo:hi]]
            sectors = []
            base = 0
            for stride in (2, 3, 3, 1, 1):
                words = f * field_slots * 10 + base * field_slots + (local + selected[:, None]) * stride + np.arange(stride)
                sectors.extend(np.unique(words // 8))
                base += stride
            sectors = np.unique(sectors)
            first = (f * 12 * field_slots + bank * field_slots + local) * destination_words // 8
            output_sectors = first + np.unique((selected[:, None] * destination_words + np.arange(destination_words)) // 8)
            footprint = (len(sectors) + len(output_sectors) + int(True)) * 32
            stream = rows[die]
            tag = span * 2 << 24 | f * ((N + 63) // 64) + lo // 64
            dependency = NONE
            acquired = len(stream)
            stream.append((0, max(1, (release * 4 + 4) // 5), 7, dependency, footprint, tag))
            selected_ready = acquired
            bitmap_read = len(stream)
            stream.append((address(bank, 120000 + span * 1024 + f * 16 + block // 384 // 64, block // 384 % 64), 0, 2, acquired, fabric, tag))
            selected_ready = len(stream)
            stream.append((0, 0, 9, bitmap_read, 4 * (8 + (hi - lo + 3) // 4), tag))
            read_begin = len(stream)
            stream.extend(((address(bank, span * 8192 + int(b) // 64, int(b) % 64), 0, 2, selected_ready, fabric, tag) for b in sectors))
            fetched = len(stream)
            stream.append((0, 0, 5, selected_ready, read_begin, tag))
            write_begin = len(stream)
            stream.extend(((address(12 + int(b) // 64 % 4, 20000 + span * 8192 + int(b) // 256, int(b) % 64), 0, 3, fetched, fabric, tag) for b in output_sectors))
            drained = len(stream)
            stream.append((0, 0, 5, fetched, write_begin, tag))
            stream.append((0, 0, 8, drained, acquired, tag))
            field_records.append(dict(span=span, die=die, frame=f, chunk=lo // 64, done=drained, selected_records=len(selected), source_records=hi - lo, destination_words=destination_words))
        demands = trace[:, 0].copy()
        live = demands > 0
        assert np.all(demands[live] >= int(raw['id_publication'][0]))
        checked.append(dict(span=span, segments=len(desc), consumed_segments=int(live.sum()), field_source_ready_ns=int(raw['field_publication'][0]) - origin, source_ready_ns=ready, first_consumption_ns=int(demands[live].min()) - origin, ids_sha256=hashlib.sha256(source.tobytes()).hexdigest()))
    return ([v.finish() for v in rows], records, checked, field_records)

def pack_ids(stream, span, die, keys, mask, tasks, directory, cull, origin, expansion, raw, supply, previous, fabric, lane_base=None, tile_end=None):
    W, bits = (supply['width'], supply['bits'])
    variants = W * 2
    desc = supply['descriptors'].numpy()
    locations = supply['locations'].numpy()
    borders = supply['input_offsets'].numpy()
    expected = supply['ids'].numpy().view(np.uint32).reshape(-1, 256)
    expected_counts = supply['counts'].numpy()[:, 0]
    demands = supply['trace'].numpy()[:, 0]
    lookup = {}
    cursors = {}
    records = []
    local_count = 0
    tw, th = (raw['tw'], raw['th'])
    rw, rh = ((tw + 7) // 8, (th + 3) // 4)
    for segment, (lo, hi, task, _) in enumerate(desc):
        tile = int(task) // variants
        owner = tile % tw // rw + 8 * (tile // tw // rh)
        if owner != die:
            continue
        ordinal = int(lo - borders[tile]) // 256
        lookup[tile, int(task) % variants, ordinal] = int(segment)
        cursors[int(segment)] = 0
        records.append(dict(span=span, segment=int(segment), location=int(locations[segment]), nodes=[], demand_ns=int(demands[segment]) - origin if demands[segment] else None, task=int(task), frame=int(task) // 2 % W, footprint=0, field_chunks=np.unique((expected[locations[segment], :expected_counts[locations[segment]]] & (1 << bits) - 1) // 64)))
        local_count += 1
    local = {r['segment']: i for i, r in enumerate(records)}
    by_id = {r['segment']: r for r in records}
    tile_cursor = {}
    sealed = set()
    start = len(stream)
    writes = 0
    last_input = {}
    previous_acquire = tile_end
    for i, tile in enumerate(keys):
        last_input[int(tile)] = i
    previous_alu = tile_end
    empty = [r for r in records if int(r['task']) // variants not in last_input]
    for first in range(0, len(empty), 32):
        batch = empty[first:first + 32]
        tag = lane_base << 24 | 16777214
        acquired = len(stream)
        stream.append((0, 0, 7, previous_acquire, len(batch) * 32, tag))
        previous_acquire = acquired
        alu = len(stream)
        stream.append((0, 0, 9, acquired, 4 * (8 + (len(batch) + 3) // 4), tag))
        begin = len(stream)
        for r in batch:
            segment = r['segment']
            assert expected_counts[r['location']] == 0
            r['nodes'].append(len(stream))
            sealed.add(segment)
            sector = local[segment] * 33 + 32
            stream.append((address(12 + sector // 64 % 4, 300000 + span * 8192 + sector // 256, sector % 64), 0, 3, alu, fabric, tag))
        drained = len(stream)
        stream.append((0, 0, 5, alu, begin, tag))
        stream.append((0, 0, 8, drained, acquired, tag))
        for r in batch:
            r['done'] = drained
        writes += len(batch)
    raw_per = 4 if raw.get('compact_wire') else 2
    mask_per = 8 if raw.get('compact_wire') else 4
    source_task = np.zeros((len(keys) + 255) // 256, np.int32)
    for task in tasks:
        source_task[int(directory[task, 1]) // 256] = task
    blocks = []
    endings = set(last_input.values())
    for first in range(0, len(keys), 64):
        last = min(len(keys), first + 64)
        step = 64
        limit = 32 if variants <= 6 else 16
        if sum((i in endings for i in range(first, last))) > limit:
            step = limit
        blocks.extend(((lo, min(last, lo + step)) for lo in range(first, last, step)))
    for first, last in blocks:
        sectors = set()
        closed = []
        for i in range(first, last):
            tile = int(keys[i])
            if mask[i, 0]:
                encoded = int(mask[i, 1])
                ordinal = tile_cursor.get(tile, 0) // 256
                for variant in range(variants):
                    frame, half = (variant // 2, variant % 2)
                    segment = lookup[tile, variant, ordinal]
                    if encoded >> bits & 1 << frame and encoded >> bits + W + half & 85:
                        index = cursors[segment]
                        location = by_id[segment]['location']
                        assert expected[location, index] == np.uint32(encoded)
                        cursors[segment] += 1
                        if index % 8 == 7:
                            sectors.add(local[segment] * 33 + index // 8)
                tile_cursor[tile] = tile_cursor.get(tile, 0) + 1
                if tile_cursor[tile] % 256 == 0:
                    for variant in range(variants):
                        segment = lookup[tile, variant, ordinal]
                        count = cursors[segment]
                        if count % 8:
                            sectors.add(local[segment] * 33 + count // 8)
                        sectors.add(local[segment] * 33 + 32)
                        closed.append(segment)
                        sealed.add(segment)
            if last_input[tile] == i:
                ordinal = tile_cursor.get(tile, 0) // 256
                for variant in range(variants):
                    segment = lookup[tile, variant, ordinal]
                    count = cursors[segment]
                    assert segment not in sealed
                    assert count == expected_counts[by_id[segment]['location']]
                    if count % 8:
                        sectors.add(local[segment] * 33 + count // 8)
                    sectors.add(local[segment] * 33 + 32)
                    closed.append(segment)
                    sealed.add(segment)
        tag = lane_base + first // 64 % 4 << 24 | first // 64
        release = int(cull[source_task[first // 256], 1]) - origin
        raw_sectors = range(first // raw_per, (last + raw_per - 1) // raw_per)
        mask_sectors = range(first // mask_per, (last + mask_per - 1) // mask_per)
        footprint = (len(raw_sectors) + len(mask_sectors) + len(sectors)) * 32
        assert 0 < footprint <= 16384, (span, die, first, last, footprint)
        dependency = previous_acquire
        acquired = len(stream)
        stream.append((0, int(np.ceil(release / 1.25)), 7, dependency, footprint, tag))
        begin = len(stream)
        previous_acquire = acquired
        for sector in raw_sectors:
            stream.append((address(12 + sector // 64 % 4, 180000 + span * 8192 + sector // 256, sector % 64), 0, 2, acquired, fabric, tag))
        for sector in mask_sectors:
            stream.append((address(sector // 64 % 12, 224000 + span * 8192 + sector // 768, sector % 64), 0, 2, acquired, fabric, tag))
        fetched = len(stream)
        stream.append((0, 0, 5, acquired, begin, tag))
        begin = len(stream)
        stream.append((0, 0, 6, previous_alu, 0, tag))
        stream.append((0, 0, 6, fetched, 0, tag))
        joined = len(stream)
        stream.append((0, 0, 5, NONE, begin, tag))
        work = vector_steps(keys[first:last], mask[first:last, 0])
        alu = len(stream)
        stream.append((0, 0, 9, joined, 4 * (16 + work + len(closed)), tag))
        begin = len(stream)
        for sector in sorted(sectors):
            records[sector // 33]['nodes'].append(len(stream))
            stream.append((address(12 + sector // 64 % 4, 300000 + span * 8192 + sector // 256, sector % 64), 0, 3, alu, fabric, tag))
        drained = len(stream)
        stream.append((0, 0, 5, alu, begin, tag))
        stream.append((0, 0, 8, drained, acquired, tag))
        prefix = drained
        previous_alu = alu
        for segment in closed:
            by_id[segment]['done'] = prefix
        writes += len(sectors)
    pending = [r for r in records if r['segment'] not in sealed]
    assert not pending, 'every tile must close at its last possible input'
    done = len(stream)
    stream.append((0, 0, 5, previous_alu, start, lane_base << 24))
    for r in records:
        assert cursors[r['segment']] == expected_counts[r['location']]
        assert len(r['nodes']) == (cursors[r['segment']] + 7) // 8 + 1
    return (records, done, local_count * 1056, writes)

def spatial_requests(capture, spans=4, fabric=4):
    lanes = 4
    expansion_base = 2 * spans
    expansion_lanes = lanes
    packing_base = expansion_base + spans * expansion_lanes
    if packing_base + 4 * spans > 128:
        raise ValueError('independent die stream IDs exceed the channel namespace')
    traces, segments, checked, fields = field_requests(capture, fabric, spans=spans)
    for trace in traces:
        trace['kind'][trace['kind'] == 4] = 9
    rows = [RequestRows(v) for v in traces]
    origin = field_origin(capture)
    expansions = []
    histograms = []
    groups = []
    storage = []
    directory_previous = [[NONE, NONE] for _ in range(32)]
    for span, raw in enumerate(capture['spatial_expansion'][:spans]):
        assert raw.get('raw_reservations', False), 'one-pass append needs raw-count output reservations'
        raw_per_sector = 4 if raw.get('compact_wire', False) else 2
        mask_per_sector = 8 if raw.get('compact_wire', False) else 4
        directory = raw['directory'].numpy()
        publication = raw['publication'].numpy()
        cull = raw['cull_publication'].numpy()
        reads = raw['reads'].numpy()
        desc = raw['metadata'].numpy()
        headers = np.r_[0, np.cumsum(raw['descriptor_counts'])]
        counts = raw['tile_counts'].numpy()
        locations = raw['tile_locations'].numpy()
        supply = capture['segmented_supply'][span]
        W = supply['width']
        variants = W * 2
        packed = raw['tile_ids'].numpy()
        physical_mask = raw['mask'].numpy()
        dram = []
        for die in range(32):
            stream = rows[die]
            source = desc[headers[die]:headers[die + 1]]
            lengths = (source[:, 2] - source[:, 1]) * (source[:, 4] - source[:, 3])
            prefix = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
            n = int(prefix[-1])
            keys = np.empty(n, np.int32)
            for r, first, last in zip(source, prefix[:-1], prefix[1:]):
                g, xmin, xmax, ymin, ymax, original, nx, rank = r
                keys[first:last] = np.repeat(np.arange(ymin, ymax), xmax - xmin) * raw['tw'] + np.tile(np.arange(xmin, xmax), ymax - ymin)
            mask = physical_mask[die, :n]
            live = mask[:, 0].astype(bool)
            expected = mask[live, 1]
            selected_keys = keys[live]
            sort = np.argsort(selected_keys, kind='stable')
            selected_keys = selected_keys[sort]
            expected = expected[sort]
            own_tiles = np.flatnonzero(locations[:, 0] == die)
            assert len(own_tiles) <= 256
            assert 48 * len(own_tiles) <= 12288
            flat = np.concatenate([packed[die, locations[t, 1]:locations[t, 1] + counts[t]] for t in own_tiles])
            assert np.array_equal(expected, flat)
            tasks = np.flatnonzero(directory[:, 0] == die)
            block_of = np.zeros((n + 255) // 256, np.int32)
            for t in tasks:
                block_of[directory[t, 1] // 256] = t
            exp_done = {}
            prior_bitmap = -1
            expansion_start = len(stream)
            for first in range(0, n, 128):
                last = min(n, first + 128)
                a = int(np.searchsorted(prefix[1:], first, side='right'))
                b = int(np.searchsorted(prefix, last, side='left'))
                selected = source[a:b]
                last_block = int(selected[-1, 7]) // 256
                bitmap_blocks = list(range(prior_bitmap + 1, last_block + 1))
                prior_bitmap = max(prior_bitmap, last_block)
                capacity = 16384
                lane = expansion_base + span * expansion_lanes + first // 128 % lanes
                tag = lane << 24 | first // 128
                required = np.unique(np.r_[bitmap_blocks, selected[:, 7] // 256]).astype(np.int64)
                release = int(publication[required, 1].max()) - origin
                outputs = (last - first + raw_per_sector - 1) // raw_per_sector
                fixed_units = len(selected) + outputs + 1
                capacity_units = capacity // 32
                if fixed_units > capacity_units:
                    raise ValueError('expansion descriptors and outputs exceed staging capacity')
                tail_blocks = capacity_units - fixed_units
                scan_blocks = bitmap_blocks
                bitmap_blocks = scan_blocks[-tail_blocks:] if tail_blocks else []
                allocation_dependency = directory_previous[die][span % 2]
                # Stream a long bitmap prefix through bounded HCF allocations before
                # the final descriptor/output pass. The PPIM lane retains its prefix.
                prefix_blocks = scan_blocks[:len(scan_blocks) - len(bitmap_blocks)]
                for offset in range(0, len(prefix_blocks), capacity_units - 1):
                    batch = prefix_blocks[offset:offset + capacity_units - 1]
                    acquired = len(stream)
                    stream.append((0, int(np.ceil(release / 1.25)), 7,
                                   allocation_dependency, (len(batch) + 1) * 32, tag))
                    begin = len(stream)
                    for block in batch:
                        stream.append((address(block % 12, 140000 + span * 8192 +
                                      block // (12 * 64), block // 12 % 64),
                                      0, 2, acquired, fabric, tag))
                    fetched = len(stream)
                    stream.append((0, 0, 5, acquired, begin, tag))
                    alu = len(stream)
                    stream.append((0, 0, 9, fetched, 4 * (8 + len(batch)), tag))
                    released = len(stream)
                    stream.append((0, 0, 8, alu, acquired, tag))
                    allocation_dependency = released
                footprint = (fixed_units + len(bitmap_blocks)) * 32
                assert 0 < footprint <= capacity
                acquired = len(stream)
                stream.append((0, int(np.ceil(release / 1.25)), 7,
                               allocation_dependency, footprint, tag))
                begin = len(stream)
                for block in bitmap_blocks:
                    stream.append((address(block % 12, 140000 + span * 8192 + block // (12 * 64), block // 12 % 64), 0, 2, acquired, fabric, tag))
                for r in selected:
                    rank = int(r[7])
                    bank = rank // 256 % 12
                    local = rank // 3072 * 256 + rank % 256
                    stream.append((address(bank, 144000 + span * 8192 + local // 64, local % 64), 0, 2, acquired, fabric, tag))
                fetched = len(stream)
                stream.append((0, 0, 5, acquired, begin, tag))
                outputs = (last - first + 3) // 4
                outputs += vector_steps(keys[first:last], np.ones(last - first, bool), width=4)
                alu = len(stream)
                stream.append((0, 0, 9, fetched, 4 * (16 + len(selected) + outputs), tag))
                begin = len(stream)
                for sector in range(first // raw_per_sector, (last + raw_per_sector - 1) // raw_per_sector):
                    stream.append((address(12 + sector // 64 % 4, 180000 + span * 8192 + sector // 256, sector % 64), 0, 3, alu, fabric, tag))
                stream.append((address(12 + first // 256 % 4, 220000 + span * 1024 + first // (256 * 256), first // 256 // 4 % 64), 0, 3, alu, fabric, tag))
                done = len(stream)
                stream.append((0, 0, 5, alu, begin, tag))
                stream.append((0, 0, 8, done, acquired, tag))
                exp_done[first] = done
            for t in tasks:
                lo = int(directory[t, 1])
                hi = lo + int(directory[t, 3])
                nodes = [exp_done[i] for i in range(lo, hi, 128)]
                source_release = int(publication[directory[t, 2], 1]) - origin
                expansions.append(dict(span=span, die=die, task=int(t), nodes=nodes, publication=source_release, demand=int(reads[t]) - origin))
            tile_end = None
            tile_end = len(stream)
            stream.append((0, 0, 5, NONE, expansion_start, packing_base + span * 4 << 24))
            packed_records, done, output_bytes, writes = pack_ids(stream, span, die, keys, mask, tasks, directory, cull, origin, exp_done, raw, supply, directory_previous[die][span % 2], fabric, lane_base=packing_base + span * 4, tile_end=tile_end)
            segments[die].extend(packed_records)
            directory_previous[die][span % 2] = done
            dram.append(int(len(source) * 32 + n * (32 // raw_per_sector + 32 // mask_per_sector) + output_bytes + len(own_tiles) * variants * 16))
            groups.append(dict(span=span, die=die, nodes=[done], publication=max((int(cull[t, 1]) - origin for t in tasks), default=0), demand=None))
            histograms.append(dict(span=span, die=die, nodes=[done], publication=0, demand=None))
            continue
        storage.append(dict(span=span, maximum_dram_bytes_per_die=max(dram), directory_sram_bytes_per_die=114688, staging_bytes_per_die=16384, allocation_bytes=256))
    return ([v.finish() for v in rows], expansions, histograms, groups, segments, checked, fields, storage)

def block_requests(capture):
    """S3 traffic using current 256-record ownership and captured block publication."""
    supply = capture['decision_supply']
    origin = min((row['window_start'] for row in supply['windows']))
    streams = [RequestRows() for _ in range(32)]
    windows = []
    functional = []
    for window, row in enumerate(supply['windows']):
        tag_window = window << 16
        depth = row['depth'].numpy()
        score = row['score'].numpy()
        minimum = row['minimum'].numpy()
        width, count = depth.shape
        assert width <= 5
        assert np.array_equal(depth.min(0), minimum)
        quantized = np.floor(score.astype(np.float64) * (1 << 20) + 0.5).astype(np.int32)
        stable = quantized > 1027605
        assert np.array_equal(stable, score > 0.98)
        publication = row['publication'].numpy()
        assert publication.shape == ((count + 255) // 256, 2)
        assert np.all(publication[:, 0] <= publication[:, 1])
        if width == 1:
            assert np.all(stable)
            for die, stream in enumerate(streams):
                done = len(stream)
                stream.append((0, 0, 6, NONE, 0, tag_window))
                windows.append(dict(window=window, die=die, done=done, blocks=0, input_ready_ns=int(publication[:, 1].max()) - origin))
            functional.append(dict(window=window, count=count, stable=count, unstable=0, score_sha256=hashlib.sha256(score.tobytes()).hexdigest(), minimum_sha256=hashlib.sha256(minimum.tobytes()).hexdigest(), depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(), independent_depth_keys=True))
            continue
        slots = (count + 98303) // 98304 * 256
        first = 300000 + window * 8192
        source_bases = np.cumsum([0, (slots * 4 + 2047) // 2048 * 2048, (slots * 4 + 2047) // 2048 * 2048]) + first * 2048
        block_nodes = [[] for _ in range(32)]
        block_written = [[] for _ in range(32)]
        commands = []
        ck = lambda ns: (max(0, int(ns) - origin) * 4 + 4) // 5
        for die, stream in enumerate(streams):
            select = len(stream)
            stream.append((address(15, 700000, window * 4), ck(row['window_start']), 1, NONE, 0, tag_window))
            reorg = len(stream)
            stream.append((address(15, 700000, window * 4 + 1), 0, 1, select, 0, tag_window))
            commands.append(reorg)
        for block, (_, ready) in enumerate(publication):
            lo = block * 256
            hi = min(count, lo + 256)
            bank = block % 12
            die = block % 384 // 12
            local = block // 384 * 256
            stream = streams[die]
            tag = 17 + bank << 24 | tag_window
            nodes = []
            for base, stride in zip(source_bases, (4, 4, width * 4)):
                start = base + local * stride
                end = base + (local + hi - lo) * stride
                for sector in range(start // 32, (end + 31) // 32):
                    nodes.append(len(stream))
                    stream.append((address(bank, sector // 64, sector % 64), ck(ready), 1, commands[die], 0, tag))
            done = len(stream)
            stream.append((0, 0, 5, commands[die], nodes[0], tag))
            block_written[die].append(done)
            block_nodes[die].append((block, lo, hi, bank, local, done))
        barriers = []
        for die, stream in enumerate(streams):
            begin = len(stream)
            for node in block_written[die]:
                stream.append((0, 0, 6, node, 0, tag_window))
            barriers.append(len(stream))
            stream.append((0, 0, 5, commands[die], begin, tag_window))
        for die, stream in enumerate(streams):
            output = []
            bank_previous = [NONE] * 12
            pending = [[] for _ in range(12)]
            for block, lo, hi, bank, local, written in block_nodes[die]:
                for begin in range(lo, hi, 16):
                    end = min(hi, begin + 16)
                    position = local + begin - lo
                    ppim_tag = 1 + bank << 24 | tag_window
                    hcf_tag = 33 + bank << 24 | tag_window
                    dependency = written
                    if len(pending[bank]) >= 2:
                        gate = len(stream)
                        stream.append((0, 0, 6, pending[bank][-2], 0, ppim_tag))
                    else:
                        gate = dependency
                    before = len(stream)
                    for node in (dependency, bank_previous[bank], gate):
                        if node != NONE:
                            stream.append((0, 0, 6, node, 0, ppim_tag))
                    available = len(stream)
                    stream.append((0, 0, 5, dependency, before, ppim_tag))
                    read_start = len(stream)
                    for sector in range((int(source_bases[0]) + position * 4) // 32, (int(source_bases[0]) + (position + end - begin) * 4 + 31) // 32):
                        stream.append((address(bank, sector // 64, sector % 64), 0, 2, available, 4, ppim_tag))
                    fetched = len(stream)
                    stream.append((0, 0, 5, available, read_start, ppim_tag))
                    compute = len(stream)
                    stream.append((address(bank), 0, 4, fetched, 7 * 4 * (end - begin), ppim_tag))
                    bank_previous[bank] = compute
                    selected = np.arange(begin, end)
                    key_sectors = []
                    payload = 0
                    for mask, base, stride in ((stable, int(source_bases[1]), 4), (~stable, int(source_bases[2]), width * 4)):
                        ids = selected[mask[selected]]
                        positions = position + ids - begin
                        if len(ids):
                            key_sectors.extend(((base + positions[:, None] * stride + np.arange(stride)) // 32).ravel())
                        payload += len(ids) * (4 + stride)
                    sectors = np.unique(key_sectors)
                    bursts = (payload + 31) // 32
                    footprint = (len(sectors) + bursts) * 32
                    acquired = len(stream)
                    stream.append((0, 0, 7, compute, footprint, hcf_tag))
                    read_start = len(stream)
                    for sector in sectors:
                        stream.append((address(bank, int(sector) // 64, int(sector) % 64), 0, 2, acquired, 4, hcf_tag))
                    fetched = len(stream)
                    stream.append((0, 0, 5, acquired, read_start, hcf_tag))
                    compact = len(stream)
                    stream.append((0, 0, 9, fetched, 4 * (16 + (end - begin + 3) // 4), hcf_tag))
                    write_start = len(stream)
                    addresses = []
                    for burst in range(bursts):
                        offset = position // 16 * 16 + burst
                        destination = address(12 + bank % 4, 400000 + window * 8192 + bank * 256 + offset // 64, offset % 64)
                        addresses.append(destination)
                        stream.append((destination, 0, 3, compact, 4, hcf_tag))
                    done = len(stream)
                    stream.append((0, 0, 5, compact, write_start, hcf_tag))
                    released = len(stream)
                    stream.append((0, 0, 8, done, acquired, hcf_tag))
                    pending[bank].append(released)
                    output.append((done, addresses, bank))
            begin = len(stream)
            for done, _, _ in output:
                stream.append((0, 0, 6, done, 0, tag_window))
            finished = len(stream)
            stream.append((0, 0, 5, commands[die], begin, tag_window))
            status = len(stream)
            stream.append((address(15, 700000, 63), 0, 0, finished, 0, tag_window))
            read_start = len(stream)
            for _, addresses, bank in output:
                for source in addresses:
                    stream.append((source, 0, 0, status, 0, 49 + bank << 24 | tag_window))
            sort_ready = len(stream)
            stream.append((0, 0, 5, status, read_start, tag_window))
            windows.append(dict(window=window, die=die, done=sort_ready, blocks=len(block_nodes[die]), input_ready_ns=int(publication[:, 1].max()) - origin))
        functional.append(dict(window=window, count=count, stable=int(stable.sum()), unstable=int((~stable).sum()), score_sha256=hashlib.sha256(score.tobytes()).hexdigest(), minimum_sha256=hashlib.sha256(minimum.tobytes()).hexdigest(), depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest()))
    return ([stream.finish() for stream in streams], windows, dict(origin_ns=origin, windows=functional, descriptor_bits=146, completion_bits=86, block_records=256, mask_slots_per_bank=2, output_gpu_reads_included=True))

def activity_fields(capture, window):
    """Read current interpolation keys from immutable, fully allocated fields."""
    from .pim_trace import capture_layout
    windows, _ = capture_layout(capture)
    source = capture['decision_supply']
    attrs = source['attributes']
    ns, n = (source['static_count'], source['count'])
    first = windows[window]['first_frame']
    frames = np.arange(first, first + windows[window]['width'])
    keys = np.floor((frames + source['time_shift']) / source['interval']).astype(np.int64)
    columns = np.unique(np.r_[keys - 1, keys, keys + 1, keys + 2])
    K = attrs['motion'].shape[1]
    assert columns.min() >= 0 and columns.max() < K
    rotations = np.unique(np.r_[(np.r_[keys, keys + 1][:, None] * 4 + np.arange(4)).ravel(), (4 * K + keys[:, None] * 2 + np.arange(2)).ravel()])
    specs = []
    slots = (n + 6143) // 6144 * 16
    base = 500000 * 2048
    for name in ('xyz', 'displacement', 'rotation', 'scales', 'opacity', 'sh', 'opacity_center', 'opacity_variance', 'motion', 'motion_rotation'):
        values = attrs[name].numpy()
        lo = 0 if name in ('xyz', 'displacement', 'rotation', 'scales', 'opacity', 'sh') else ns
        hi = ns if name in ('xyz', 'displacement', 'rotation') else n
        assert len(values) == hi - lo and values.dtype == np.float32
        values = values.reshape(len(values), -1)
        full = values.shape[1]
        chosen = (columns[:, None] * 3 + np.arange(3)).ravel() if name == 'motion' else rotations if name == 'motion_rotation' else np.arange(full)
        assert chosen.max() < full
        specs.append(dict(name=name, lo=lo, hi=hi, values=values, base=base, full_columns=full, columns=chosen, soa=name in ('motion', 'motion_rotation')))
        base += (slots * full * 4 + 2047) // 2048 * 2048
    assert base - 500000 * 2048 < 1 << 27
    return (specs, slots)

def selected_source_requests(capture, selected_windows=None):
    from .pim_trace import capture_layout
    layout, _ = capture_layout(capture)
    source = capture['decision_supply']
    origin = min((row['window_start'] for row in source['windows']))
    streams = [RequestRows() for _ in range(32)]
    blocks, window_meta, checked, color_blocks = ([], [], [], [])
    sector_patterns = {}
    largest_partition = 0
    for inputs in source['window_inputs']:
        ids = inputs['source_ids'].numpy().astype(np.int64)
        groups = np.stack((ids // 16, np.arange(len(ids)) // 256), axis=1)
        first = np.r_[0, 1 + np.flatnonzero(np.any(groups[1:] != groups[:-1], axis=1))]
        largest_partition = max(largest_partition, int(np.bincount(ids[first] // 16 % 384, minlength=384).max()))
    destination_bank_extent = largest_partition * 8192
    selected_windows = (range(len(source['windows'])) if selected_windows is None
                        else selected_windows)
    for window in selected_windows:
        row = source['windows'][window]
        inputs = source['window_inputs'][window]
        ids = inputs['source_ids'].numpy().astype(np.int64)
        count, total = (inputs.get('selection_count', inputs['count']), inputs['source_count'])
        anchored = source['workload'] == 'anchored'
        if anchored:
            intervals = inputs['intervals'].numpy()
            times = inputs['times'].numpy() / 30
            assert intervals.shape == (total, 2)
            active = (intervals[:, 0] <= times[-1]) & (intervals[:, 1] >= times[0])
        else:
            masks = inputs['masks'].numpy()
            assert masks.shape[1] == total
            active = masks.any(0)
        assert len(ids) == count
        assert np.array_equal(ids, np.flatnonzero(active))
        attributes = inputs['attributes']
        names = tuple((name for name in attributes if name != 'sh'))
        values = {name: attributes[name].numpy().reshape(count, -1) for name in names}
        widths = {name: value.shape[1] * value.dtype.itemsize for name, value in values.items()}
        source_block_records = 16
        slots = (total + 384 * source_block_records - 1) // (384 * source_block_records) * source_block_records
        bases, base = ({}, 500000 * 2048)
        for name in (*names, *(('sh',) if not anchored else ())):
            width = attributes[name].numel() * attributes[name].element_size() // count
            bases[name] = base
            base += (slots * width + 2047) // 2048 * 2048
        start_offsets = list(map(len, streams))
        if not window:
            release = 0
        else:
            release = max((p['end_ns'] for p in source['phases'] if p['window'] == window - 1 and p['kind'] == 'window_delta')) - origin
        clock = (max(0, release) * 4 + 4) // 5
        tag_window = window << 16
        commands = []
        for die, stream in enumerate(streams):
            selected = len(stream)
            stream.append((address(15, 710000, window * 4), clock, 1, NONE, 0, tag_window))
            commands.append(len(stream))
            stream.append((address(15, 710000, window * 4 + 1), 0, 1, selected, 0, tag_window))
        decisions = {}
        for begin in range(0, total, 16):
            end = min(total, begin + 16)
            block = begin // source_block_records
            die, bank = (block % 384 // 12, block % 12)
            position = block // 384 * source_block_records + begin % source_block_records
            stream = streams[die]
            tag = 1 + bank << 24 | tag_window
            first = len(stream)
            for frame in [0] if anchored else inputs['keyframes'].tolist():
                stride = 8 if anchored else 1
                base = 480000 * 2048 + frame * ((slots + 2047) // 2048) * 2048
                for sector in range((base + position * stride) // 32, (base + (position + end - begin) * stride + 31) // 32):
                    stream.append((address(bank, sector // 64, sector % 64), 0, 2, commands[die], 4, tag))
            fetched = len(stream)
            stream.append((0, 0, 5, commands[die], first, tag))
            decisions[begin // 16] = len(stream)
            operations = (4 if anchored else len(inputs['keyframes']) + 2) * (end - begin)
            stream.append((address(bank), 0, 4, fetched, operations * 4, tag))
        producers = defaultdict(list)
        reads = defaultdict(list)
        per_die_writes = defaultdict(list)
        per_die_gpu = defaultdict(list)
        pending = defaultdict(list)
        local_rank = defaultdict(int)
        sh_width = 0 if anchored else attributes['sh'].numel() // count
        color_phase = next((p['start_ns'] for p in source['phases'] if p['window'] == window and p['kind'] == 'colors'))
        color_clock = (max(0, color_phase - origin) * 4 + 4) // 5
        sh_groups = defaultdict(lambda: ([], []))
        groups = np.stack((ids // 16, np.arange(count) // 256), axis=1)
        boundaries = np.r_[0, 1 + np.flatnonzero(np.any(groups[1:] != groups[:-1], axis=1)), count]
        for first_selected, last_selected in zip(boundaries[:-1], boundaries[1:]):
            selected = np.arange(first_selected, last_selected)
            source_fragment = ids[first_selected] // 16
            original = ids[selected]
            source_block = int(original[0] // source_block_records)
            die, bank = (source_block % 384 // 12, source_block % 12)
            positions = source_block // 384 * source_block_records + original % source_block_records
            stream = streams[die]
            hcf_tag = 33 + bank % 4 << 24 | tag_window
            gpu_tag = 49 + bank % 4 << 24 | tag_window
            selected_node = decisions[int(source_fragment)]
            before = len(stream)
            stream.append((0, 0, 6, selected_node, 0, hcf_tag))
            if len(pending[die, bank]) >= 2:
                stream.append((0, 0, 6, pending[die, bank][-2], 0, hcf_tag))
            gate = len(stream)
            stream.append((0, 0, 5, selected_node, before, hcf_tag))
            sectors = []
            local_ids = original % source_block_records
            pattern = int((1 << local_ids).sum())
            for name in names:
                width = widths[name]
                base_sector, phase = divmod(source_block // 384 * source_block_records * width, 32)
                key = width, phase, pattern
                if key not in sector_patterns:
                    offsets = phase + local_ids[:, None] * width + np.arange(width)
                    sector_patterns[key] = np.unique(offsets // 32)
                sectors.extend(bases[name] // 32 + base_sector + sector_patterns[key])
            sectors = np.unique(sectors)
            controls = 0 if anchored else 8 + len(row['depth'])
            payload = len(selected) * (sum(widths.values()) + controls)
            bursts = (payload + 31) // 32
            footprint = (len(sectors) + bursts) * 32
            assert 0 < footprint <= 16384
            acquired = len(stream)
            stream.append((0, 0, 7, gate, footprint, hcf_tag))
            first = len(stream)
            stream.extend_uniform(address(bank, sectors // 64, sectors % 64),
                                  0, 2, acquired, 4, hcf_tag)
            fetched = len(stream)
            stream.append((0, 0, 5, acquired, first, hcf_tag))
            copied = len(stream)
            stream.append((0, 0, 9, fetched, 4 * (16 + (len(selected) + 3) // 4), hcf_tag))
            destination_base = 600000 * 2048
            destination_base += (window % capture['window_slots'] * 12 + bank) * destination_bank_extent
            assert local_rank[die, bank] < largest_partition
            destination_base += local_rank[die, bank] * 8192
            local_rank[die, bank] += 1
            addresses = [address(12 + bank % 4, (destination_base // 32 + burst) // 64, (destination_base // 32 + burst) % 64) for burst in range(bursts)]
            first = len(stream)
            stream.extend_uniform(addresses, 0, 3, copied, 4, hcf_tag)
            written = len(stream)
            stream.append((0, 0, 5, copied, first, hcf_tag))
            per_die_writes[die].append(written)
            released = len(stream)
            stream.append((0, 0, 8, written, acquired, hcf_tag))
            pending[die, bank].append(released)
            first = len(stream)
            input_reads = range(first, first + len(addresses))
            stream.extend_uniform(addresses, 0, 0, written, 0, gpu_tag)
            consumed = len(stream)
            stream.append((0, 0, 5, written, first, gpu_tag))
            per_die_gpu[die].append(consumed)
            key = (int(first_selected // 256), die, bank)
            producers[key].append(written)
            reads[key].extend(input_reads)
            if not anchored:
                first_sector = (bases['sh'] + positions * sh_width * 4) // 32
                last_sector = (bases['sh'] + (positions + 1) * sh_width * 4 - 1) // 32
                lengths = last_sector - first_sector + 1
                offsets = np.arange(int(lengths.max()))
                touched = (first_sector[:, None] + offsets)[offsets < lengths[:, None]]
                logical = np.repeat(selected, lengths)
                first = len(stream)
                stream.extend_uniform(address(bank, touched // 64, touched % 64),
                                      color_clock, 0, commands[die], 0,
                                      81 + bank % 4 << 24 | tag_window)
                borders = np.r_[0, np.flatnonzero(logical[1:] // 64 != logical[:-1] // 64) + 1, len(logical)]
                for lo, hi in zip(borders[:-1], borders[1:]):
                    part, records = sh_groups[int(logical[lo]) // 64, die]
                    part.extend(range(first + int(lo), first + int(hi)))
                    records.extend(logical[lo:hi].tolist())
        for (block, die, bank), indices in reads.items():
            stream = streams[die]
            first = len(stream)
            for index in indices:
                stream.append((0, 0, 6, index, 0, 49 + bank % 4 << 24 | tag_window))
            done = len(stream)
            stream.append((0, 0, 5, commands[die], first, 49 + bank % 4 << 24 | tag_window))
            blocks.append(dict(window=window, block=block, die=die, bank=bank, done=done, lo=block * 256, hi=min(count, (block + 1) * 256), producers=sorted(set(producers[block, die, bank])), input_reads=sorted(set(indices))))
        for (block, die), (indices, records) in sh_groups.items():
            color_blocks.append(dict(window=window, block=block, die=die, indices=indices, records=records))
        for die, stream in enumerate(streams):
            first = len(stream)
            for node in per_die_writes[die]:
                stream.append((0, 0, 6, node, 0, tag_window))
            finished = len(stream)
            stream.append((0, 0, 5, commands[die], first, tag_window))
            status = len(stream)
            stream.append((address(15, 710000, 63), 0, 0, finished, 0, tag_window))
            first = len(stream)
            for node in per_die_gpu[die]:
                stream.append((0, 0, 6, node, 0, tag_window))
            complete = len(stream)
            stream.append((0, 0, 5, status, first, tag_window))
            color_done = len(stream)
            if anchored:
                for lane in range(81, 85):
                    stream.append((0, 0, 6, commands[die], 0, lane << 24 | tag_window))
                color_done = len(stream)
            stream.append((0, 0, 5, NONE, start_offsets[die], 85 << 24 | tag_window))
            window_meta.append(dict(window=window, die=die, done=complete, command=commands[die] - 1, color_done=color_done, release_ns=release))
        checked.append(dict(window=window, records=total, active=count, selected_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(), source_sha256={name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in attributes.items()}))
    return ([stream.finish() for stream in streams], blocks, window_meta, dict(origin_ns=origin, windows=checked, color_blocks=color_blocks, s1_attributes_and_gpu_input_reads_included=True, block_records=256, selected_source_ownership=True, source_block_records=16, decoded_color_fields=source.get('decoded_color_fields', False)))

def activity_requests(capture):
    """Select full-source temporal fields and publish compact immutable views."""
    if capture['decision_supply'].get('workload') in ('explicit4d', 'anchored'):
        return selected_source_requests(capture)
    from .pim_trace import capture_layout
    windows, _ = capture_layout(capture)
    source = capture['decision_supply']
    total, static = source['count'], source['static_count']
    intervals = source['temporal_intervals'].numpy()
    assert intervals.shape == (total - static, 2)
    origin = min(row['window_start'] for row in source['windows'])
    streams = [RequestRows() for _ in range(32)]
    blocks, window_meta, checked, color_blocks = [], [], [], []
    for window, (row, inputs) in enumerate(zip(source['windows'], source['window_inputs'])):
        ids = inputs['source_ids'].numpy().astype(np.int64)
        count = inputs['count']
        first, width = windows[window]['first_frame'], windows[window]['width']
        active = ((intervals[:, 0] <= np.float32((first + width - 1) / 30)) &
                  (intervals[:, 1] >= np.float32(first / 30)))
        expected = np.r_[np.arange(static), static + np.flatnonzero(active)]
        assert count == row['depth'].shape[1] and np.array_equal(ids, expected)
        fields, slots = activity_fields(capture, window)
        start_offsets = list(map(len, streams))
        release = 0 if not window else max(p['end_ns'] for p in source['phases']
            if p['window'] == window - 1 and p['kind'] == 'window_delta') - origin
        clock = (max(0, release) * 4 + 4) // 5
        tag_window = window << 16
        commands = []
        for stream in streams:
            selected = len(stream)
            stream.append((address(15, 710000, window * 4), clock, 1, NONE, 0, tag_window))
            commands.append(len(stream))
            stream.append((address(15, 710000, window * 4 + 1), 0, 1, selected, 0, tag_window))
        decisions = {}
        for begin in range(0, total, 16):
            end = min(total, begin + 16)
            fragment = begin // 16
            die, bank = fragment % 384 // 12, fragment % 12
            position = fragment // 384 * 16
            stream = streams[die]
            tag = 1 + bank << 24 | tag_window
            first_read = len(stream)
            for field in range(3):
                base = 480000 * 2048 + field * ((slots * 4 + 2047) // 2048) * 2048
                for sector in range((base + position * 4) // 32,
                                    (base + (position + end - begin) * 4 + 31) // 32):
                    stream.append((address(bank, sector // 64, sector % 64), 0, 2, commands[die], 4, tag))
            fetched = len(stream)
            stream.append((0, 0, 5, commands[die], first_read, tag))
            decisions[fragment] = len(stream)
            stream.append((address(bank), 0, 4, fetched, 7 * 4 * (end - begin), tag))
        producers, reads = defaultdict(list), defaultdict(list)
        writes, gpu_reads, pending = defaultdict(list), defaultdict(list), defaultdict(list)
        local_rank = defaultdict(int)
        hashes = {field['name']: hashlib.sha256() for field in fields}
        groups = np.stack((ids // 16, np.arange(count) // 64), axis=1)
        boundaries = np.r_[0, 1 + np.flatnonzero(np.any(groups[1:] != groups[:-1], axis=1)), count]
        descriptor_extent = ((slots // 16 + (count + 63) // 64 + 1) * 256 + 2047) // 2048 * 2048
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            original = ids[lo:hi]
            fragment = int(original[0] // 16)
            die, bank = fragment % 384 // 12, fragment % 12
            position = fragment // 384 * 16
            stream = streams[die]
            ht, gt = 33 + bank % 4 << 24 | tag_window, 49 + bank % 4 << 24 | tag_window
            selected = decisions[fragment]
            before = len(stream)
            stream.append((0, 0, 6, selected, 0, ht))
            if len(pending[die, bank]) >= 2:
                stream.append((0, 0, 6, pending[die, bank][-2], 0, ht))
            gate = len(stream)
            stream.append((0, 0, 5, selected, before, ht))
            sectors, sh_sectors = [], []
            for field in fields:
                selected_ids = original[(original >= field['lo']) & (original < field['hi'])]
                if not len(selected_ids):
                    continue
                positions = position + selected_ids % 16
                columns = field['columns']
                if field['soa']:
                    offsets = field['base'] + (columns[:, None] * slots + positions) * 4
                else:
                    offsets = field['base'] + (positions[:, None] * field['full_columns'] + columns) * 4
                touched = np.unique(offsets // 32)
                (sh_sectors if field['name'] == 'sh' else sectors).extend(touched)
                values = np.ascontiguousarray(field['values'][selected_ids - field['lo']][:, columns])
                hashes[field['name']].update(values.tobytes())
            bursts = (16 + len(original) * 8 + 31) // 32
            acquired = len(stream)
            stream.append((0, 0, 7, gate, bursts * 32, ht))
            copied = len(stream)
            stream.append((0, 0, 9, acquired, 4 * (16 + (len(original) + 3) // 4), ht))
            base = 650000 * 2048 + (window % capture['window_slots'] * 12 + bank) * descriptor_extent
            base += local_rank[die, bank] * 256
            local_rank[die, bank] += 1
            descriptors = [address(12 + bank % 4, (base // 32 + b) // 64, (base // 32 + b) % 64) for b in range(bursts)]
            first_write = len(stream)
            stream.extend_uniform(descriptors, 0, 3, copied, 4, ht)
            written = len(stream)
            stream.append((0, 0, 5, copied, first_write, ht))
            writes[die].append(written)
            released = len(stream)
            stream.append((0, 0, 8, written, acquired, ht))
            pending[die, bank].append(released)
            first_read = len(stream)
            header = written
            for descriptor in descriptors:
                index = len(stream)
                stream.append((descriptor, 0, 0, header, 0, gt))
                header = index
            sectors = np.unique(sectors)
            source_reads = len(stream)
            stream.extend_uniform(address(bank, sectors // 64, sectors % 64), 0, 0, header, 0, gt)
            consumed = len(stream)
            stream.append((0, 0, 5, header, source_reads, gt))
            gpu_reads[die].append(consumed)
            key = int(lo // 256), die, bank
            producers[key].append(written)
            reads[key].extend(range(first_read, header + 1))
            reads[key].extend(range(source_reads, consumed))
            color = next(p['start_ns'] for p in source['phases'] if p['window'] == window and p['kind'] == 'colors')
            color_clock = (max(0, color - origin) * 4 + 4) // 5
            sh_sectors = np.unique(sh_sectors)
            first_read = len(stream)
            stream.extend_uniform(address(bank, sh_sectors // 64, sh_sectors % 64),
                                  color_clock, 0, header, 0, 81 + bank % 4 << 24 | tag_window)
            assert len(sh_sectors) == len(original) * 6
            color_blocks.append(dict(window=window, block=int(lo // 64), die=die,
                indices=list(range(first_read, len(stream))), records=np.repeat(np.arange(lo, hi), 6).tolist()))
        for (block, die, bank), indices in reads.items():
            stream = streams[die]
            before = len(stream)
            for index in indices:
                stream.append((0, 0, 6, index, 0, 49 + bank % 4 << 24 | tag_window))
            done = len(stream)
            stream.append((0, 0, 5, commands[die], before, 49 + bank % 4 << 24 | tag_window))
            blocks.append(dict(window=window, block=block, die=die, bank=bank,
                done=done, lo=block * 256, hi=min(count, (block + 1) * 256),
                producers=sorted(set(producers[block, die, bank])), input_reads=indices))
        for die, stream in enumerate(streams):
            before = len(stream)
            for index in writes[die]:
                stream.append((0, 0, 6, index, 0, tag_window))
            finished = len(stream)
            stream.append((0, 0, 5, commands[die], before, tag_window))
            status = len(stream)
            stream.append((address(15, 710000, 63), 0, 0, finished, 0, tag_window))
            before = len(stream)
            for index in gpu_reads[die]:
                stream.append((0, 0, 6, index, 0, tag_window))
            complete = len(stream)
            stream.append((0, 0, 5, status, before, tag_window))
            color_done = len(stream)
            stream.append((0, 0, 5, NONE, start_offsets[die], 85 << 24 | tag_window))
            window_meta.append(dict(window=window, die=die, done=complete,
                command=commands[die] - 1, color_done=color_done, release_ns=release))
        checked.append(dict(window=window, active=count, records=total, ppim_edges=total * 7,
            selected_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
            source_sha256={name: value.hexdigest() for name, value in hashes.items()},
            fields=[dict(name=field['name'], allocated_columns=field['full_columns'],
                         selected_columns=field['columns'].tolist()) for field in fields]))
    return ([stream.finish() for stream in streams], blocks, window_meta,
        dict(origin_ns=origin, windows=checked, immutable_source_columns=True,
             block_records=256, mask_slots_per_bank=2, selected_source_ownership=True,
             s1_attributes_and_gpu_input_reads_included=True, dense_views=True,
             sh_gpu_reads_at_color_phase=True, color_blocks=color_blocks, source_block_records=16))


def merge_selected_source_requests(results):
    """Join independently built windows while preserving stream references."""
    streams = [[] for _ in range(32)]
    offsets = [0] * 32
    blocks, window_meta, checked, color_blocks = [], [], [], []
    evidence = None
    for request_streams, result_blocks, result_windows, result_evidence in results:
        bases = tuple(offsets)
        for die, stream in enumerate(request_streams):
            stream = stream.copy()
            base = bases[die]
            linked = stream['depends'] != NONE
            stream['depends'][linked] += base
            indexed_cost = (stream['kind'] == 5) | (stream['kind'] == 8)
            stream['cost'][indexed_cost] += base
            streams[die].append(stream)
            offsets[die] += len(stream)
        for block in result_blocks:
            block = dict(block)
            base = bases[block['die']]
            block['done'] += base
            block['producers'] = [value + base for value in block['producers']]
            block['input_reads'] = [value + base for value in block['input_reads']]
            blocks.append(block)
        for row in result_windows:
            row = dict(row)
            base = bases[row['die']]
            for name in ('done', 'command', 'color_done'):
                if row[name] != NONE:
                    row[name] += base
            window_meta.append(row)
        for block in result_evidence['color_blocks']:
            block = dict(block)
            base = bases[block['die']]
            block['indices'] = [value + base for value in block['indices']]
            color_blocks.append(block)
        checked.extend(result_evidence['windows'])
        common = {key: value for key, value in result_evidence.items()
                  if key not in ('windows', 'color_blocks')}
        if evidence is None:
            evidence = common
        else:
            assert common == evidence
    evidence = dict(evidence, windows=checked, color_blocks=color_blocks)
    return ([np.concatenate(parts) for parts in streams], blocks, window_meta, evidence)


_PARALLEL_CAPTURE = None


def _parallel_spatial(spans):
    return spatial_requests(_PARALLEL_CAPTURE, spans=spans)


def _parallel_block():
    return block_requests(_PARALLEL_CAPTURE)


def _parallel_activity():
    return activity_requests(_PARALLEL_CAPTURE)


def _parallel_selected_window(window):
    return selected_source_requests(_PARALLEL_CAPTURE, (window,))


def build_requests(capture):
    global _PARALLEL_CAPTURE
    spans = len(capture['segmented_supply'])
    assert 0 < spans and 10 * spans < 256
    selected = capture['decision_supply'].get('workload') in ('explicit4d', 'anchored')
    window_count = len(capture['decision_supply']['windows']) if selected else 1
    task_count = 2 + window_count
    workers = min(multiprocessing.cpu_count(), task_count)
    print(f'Building PIM requests with {workers} CPU workers', flush=True)
    _PARALLEL_CAPTURE = capture
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context('fork')) as executor:
            hcf_future = executor.submit(_parallel_spatial, spans)
            s3_future = executor.submit(_parallel_block)
            if selected:
                s1_futures = [executor.submit(_parallel_selected_window, window)
                              for window in range(window_count)]
            else:
                s1_future = executor.submit(_parallel_activity)
            hcf, expansions, _, _, segments, _, fields, _ = hcf_future.result()
            s3, windows, evidence = s3_future.result()
            if selected:
                s1, s1_blocks, s1_windows, s1_evidence = merge_selected_source_requests(
                    [future.result() for future in s1_futures])
            else:
                s1, s1_blocks, s1_windows, s1_evidence = s1_future.result()
    finally:
        _PARALLEL_CAPTURE = None
    assert s1_evidence['origin_ns'] == evidence['origin_ns']
    hcf_origin = field_origin(capture)
    shift_ns = hcf_origin - evidence['origin_ns']
    shift_ck = (shift_ns * 4 + 4) // 5
    assert shift_ns >= 0
    lanes = np.unique(np.concatenate([stream['tag'] >> 24 for stream in s3]))
    s1_lanes = np.unique(np.concatenate([stream['tag'] >> 24 for stream in s1]))
    assert 10 * spans + len(lanes) + len(s1_lanes) <= 256
    requests = []
    for die, (front, extra) in enumerate(zip(hcf, s3)):
        front = front.copy()
        released = front['ready'] > 0
        front['ready'][released] += shift_ck
        extra = extra.copy()
        offset = len(front)
        linked = extra['depends'] != NONE
        extra['depends'][linked] += offset
        extra['cost'][(extra['kind'] == 5) | (extra['kind'] == 8)] += offset
        original = extra['tag'] >> 24
        extra['tag'] = np.searchsorted(lanes, original).astype(np.uint32) + 10 * spans << 24 | extra['tag'] & 16777215
        activity = s1[die].copy()
        activity_offset = len(front) + len(extra)
        linked = activity['depends'] != NONE
        activity['depends'][linked] += activity_offset
        activity['cost'][(activity['kind'] == 5) | (activity['kind'] == 8)] += activity_offset
        activity['tag'] = np.searchsorted(s1_lanes, activity['tag'] >> 24).astype(np.uint32) + 10 * spans + len(lanes) << 24 | activity['tag'] & 16777215
        requests.append(np.concatenate((front, extra, activity)))
    for stream in requests:
        memory = stream['kind'] < 4
        indices = np.flatnonzero(memory)
        sectors = stream['address'][memory] // 32
        banks = sectors // 64 % 16
        pages = sectors // 1024
        for bank in range(16):
            selected = np.flatnonzero(banks == bank)
            unique, inverse = np.unique(pages[selected], return_inverse=True)
            assert len(unique) <= 65536
            stream['address'][indices[selected]] = ((inverse.astype(np.uint64) * 16 + bank) * 64 + sectors[selected] % 64) * 32
    data = dict(requests=requests, hcf_lengths=list(map(len, hcf)), s3_lengths=list(map(len, s3)), expansions=expansions, segments=segments, fields=fields, windows=windows, evidence=evidence, shift_ns=shift_ns, s3_events=sum(map(len, s3)))
    data.update(s1_blocks=s1_blocks, s1_windows=s1_windows, s1_evidence=s1_evidence, s1_events=sum(map(len, s1)))
    return data

def native_sh_visibility(capture, data):
    """Apply the per-window visibility predicate used by batch_colors."""
    from .pim_trace import capture_layout
    windows, _ = capture_layout(capture)
    source = capture['decision_supply']
    mapped = data['s1_evidence'].get('color_blocks')
    evidence = []
    for window in range(len(capture['decision_supply']['windows'])):
        n = source['windows'][window]['depth'].shape[1]
        masks = []
        for span in windows[window]['spans']:
            supply = capture['segmented_supply'][span]
            bitmap = supply['field_visibility'].numpy().view(np.uint32).reshape(supply['width'], -1)
            ids = np.arange(n)
            masks.append((bitmap[:, ids // 32] >> ids % 32 & 1).astype(bool))
        assert sum(map(len, masks)) == windows[window]['width']
        visible = np.concatenate(masks).any(0)
        before = after = 0
        if source.get('decoded_color_fields'):
            evidence.append(dict(window=window, records=n, visible_records=int(visible.sum()), original_source_sectors=0, native_source_sectors=0))
            continue
        attributes = source['window_inputs'][window].get('attributes', source['attributes']) if source.get('window_inputs') else source['attributes']
        sectors_per_record = (attributes['sh'].numel() // len(attributes['sh']) * 4 + 31) // 32
        for block in mapped:
            if block['window'] != window:
                continue
            die = block['die']
            stream = data['requests'][die]
            offset = data['hcf_lengths'][die] + data['s3_lengths'][die]
            selected = offset + np.asarray(block['indices'], np.int64)
            records = np.asarray(block['records'], np.int64)
            assert np.all(stream['kind'][selected] == 0)
            keep = visible[records]
            disabled = selected[~keep]
            stream['kind'][disabled] = 6
            stream['address'][disabled] = 0
            stream['ready'][disabled] = 0
            stream['cost'][disabled] = 0
            before += len(selected)
            after += int(keep.sum())
        assert before == n * sectors_per_record and after == int(visible.sum()) * sectors_per_record
        evidence.append(dict(window=window, records=n, visible_records=int(visible.sum()), original_source_sectors=before, native_source_sectors=after))
        continue
    return dict(windows=evidence, semantics='any-frame positive radii, as in original native batch_colors; no opacity pruning', scope='unique source sectors; per-color-CTA cache/memory service remains incomplete')

def bank_streams(data):
    """Keep each SH source bank independently issuable within the same window."""
    changed = 0
    streams = np.unique(np.concatenate([stream[data['hcf_lengths'][die] + data['s3_lengths'][die]:]['tag'] >> 24 for die, stream in enumerate(data['requests'])]))
    first = int(streams[-5])
    assert np.array_equal(streams[-5:], np.arange(first, first + 5))
    assert first + 12 < 256
    for die, stream in enumerate(data['requests']):
        begin = data['hcf_lengths'][die] + data['s3_lengths'][die]
        part = stream[begin:]
        lane = part['tag'] >> 24
        terminal = lane == first + 4
        part['tag'][terminal] = first + 12 << 24 | part['tag'][terminal] & 16777215
        reads = (part['kind'] == 0) & (part['ready'] > 0)
        assert np.all((lane[reads] >= first) & (lane[reads] < first + 4))
        banks = part['address'][reads] // 2048 % 16
        assert np.all(banks < 12)
        part['tag'][reads] = first + banks.astype(np.uint32) << 24 | part['tag'][reads] & 16777215
        changed += int(np.count_nonzero(reads))
    data['sh_lane_base'] = first
    return dict(independent_sh_banks=12, source_reads=changed, channel_request_window=128, die_stream_limit=256)

def color_input_blocks(capture, data):
    """Map immutable six-sector SH records before visibility removes reads."""
    result = {}
    if capture['decision_supply'].get('decoded_color_fields'):
        return {(w, block): [] for w, row in enumerate(capture['decision_supply']['windows']) for block in range((row['depth'].shape[1] + 63) // 64)}
    mapped = data['s1_evidence'].get('color_blocks')
    for block in mapped:
        die = block['die']
        offset = data['hcf_lengths'][die] + data['s3_lengths'][die]
        key = (block['window'], block['block'])
        result.setdefault(key, []).append((die, offset + np.asarray(block['indices'], np.uint64)))
    for w, row in enumerate(capture['decision_supply']['windows']):
        assert sum((window == w for window, _ in result)) == (row['depth'].shape[1] + 63) // 64
    return result
