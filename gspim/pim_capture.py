"""Build compact scheduling traces from GPU publications and physical requests."""

from collections import defaultdict
import hashlib
from itertools import repeat
from pathlib import Path

import numpy as np

from .pim_memory import FeedbackGraph, NONE
from .pim_requests import build_requests, bank_streams, color_input_blocks, native_sh_visibility



def connect_gpu(capture, data, schedule_details=None):
    from .pim_trace import capture_layout
    layout, span_layout = capture_layout(capture)
    graph = FeedbackGraph(None)
    source = capture['decision_supply']
    origin = data['evidence']['origin_ns']

    def ck(ns):
        return (max(0, int(ns)) * 4 + 4) // 5

    def at(ns):
        return ck(int(ns) - origin)

    def follow(child, parent, start, end):
        graph.edge(child, parent, ck(max(0, int(start) - int(end))))

    def memory(node, die, indices):
        graph.nodes[node]['memory'].append((die, np.asarray(indices, np.uint64)))
    phases = {(w, kind): [p for p in source['phases'] if p['window'] == w and p['kind'] == kind] for w in range(len(source['windows'])) for kind in ('generate', 'project', 'window_delta', 'crossing', 'colors', 'sort')}
    assert all(phases.values())
    inputs = {}
    bank_inputs = {}
    input_blocks = {}
    for b in data['s1_blocks']:
        offset = data['hcf_lengths'][b['die']] + data['s3_lengths'][b['die']]
        inputs.setdefault((b['window'], b['block']), []).append((b['die'], offset + b['done']))
        input_blocks.setdefault((b['window'], b['block']), []).append(b)
        bank_inputs.setdefault((b['window'], b['die'], b['bank']), []).append(offset + b['done'])
    windows = []
    generation = []
    depth = []
    ctas = []
    phase_nodes = {}
    s3_done = []
    phase_ctas = source.get('phase_ctas', {})
    color_nodes = {}
    color_loads = source.get('color_load_frontiers')
    color_inputs = data.get('color_inputs', {})
    generation_loads = source.get('generation_load_frontiers')
    source_publishers = [np.full(len(stream) - hcf - s3, -1, np.int32) for stream, hcf, s3 in zip(data['requests'], data['hcf_lengths'], data['s3_lengths'])]
    publisher_info = {}

    def source_publication(die, indices, kind, window, task, launch):
        local = indices - data['hcf_lengths'][die] - data['s3_lengths'][die]
        assert np.all(source_publishers[die][local] == -1)
        source_publishers[die][local] = launch
        info = (kind, window, task)
        assert publisher_info.setdefault(launch, info) == info
    if generation_loads is not None:
        assert 'sh_lane_base' in data
        base = data['sh_lane_base']
        for die, stream in enumerate(data['requests']):
            offset = data['hcf_lengths'][die] + data['s3_lengths'][die]
            part = stream[offset:]
            lanes = part['tag'] >> 24
            skipped = (part['kind'] == 6) & (part['ready'] == 0) & (lanes >= base) & (lanes < base + 4)
            part['tag'][skipped] = base + 12 << 24 | part['tag'][skipped] & 16777215
    if color_loads is not None:
        assert len(color_inputs)

    def gpu_phase(window, kind, phase, parent, parent_end):
        table = phase_ctas.get(kind)
        if table is None:
            node = graph.add(earliest=at(phase['start_ns']), duration=ck(phase['end_ns'] - phase['start_ns']), kind=kind)
            follow(node, parent, phase['start_ns'], parent_end)
            return node
        rows = table[window].numpy()
        assert rows.ndim == 2 and rows.shape[1] == 3 and len(rows)
        assert np.all(rows[:, 0] >= phase['start_ns']) and np.all(rows[:, 1] <= phase['end_ns'])
        assert np.all(rows[:, 0] <= rows[:, 1])
        joined = graph.add(earliest=at(phase['end_ns']), kind=kind + '_join')
        tail = ck(phase['end_ns'] - int(rows[:, 1].max()))
        order = np.lexsort((np.arange(len(rows)), rows[:, 1], rows[:, 0]))
        dispatch_rank = np.empty(len(rows), np.uint32)
        dispatch_rank[order] = np.arange(len(rows), dtype=np.uint32)
        for task, (begin, end, sm) in enumerate(rows):
            if kind == 'colors' and color_loads is not None:
                loaded = int(color_loads[window][task])
                assert begin <= loaded <= end
                launch = graph.add(earliest=at(begin), kind='color_cta_launch')
                follow(launch, parent, begin, parent_end)
                prefix = graph.add(duration=ck(loaded - begin), kind='color_native_load_interval')
                graph.edge(prefix, launch)
                node = graph.add(duration=ck(end - loaded), kind='colors')
                graph.edge(node, prefix)
                for die, indices in color_inputs[window, task]:
                    indices = indices[data['requests'][die]['kind'][indices] == 0]
                    data['requests'][die]['ready'][indices] = at(begin)
                    assert len(rows) <= 65536
                    tags = data['requests'][die]['tag']
                    tags[indices] = tags[indices] & 4294901760 | dispatch_rank[task]
                    memory(node, die, indices)
                    source_publication(die, indices, 's1_color', window, task, launch)
                color_nodes[window, task] = node
            else:
                node = graph.add(earliest=at(begin), duration=ck(end - begin), kind=kind)
                follow(node, parent, begin, parent_end)
                launch = node
            graph.edge(joined, node, tail)
            ctas.append(dict(kind=kind, sm=int(sm), start=int(begin), end=int(end), first=launch, last=node))
        return joined
    for w, row in enumerate(source['windows']):
        start = graph.add(earliest=at(row['window_start']), kind='window_start')
        windows.append(start)
        starts = []
        ends = []
        generation_rows = source['generation_ctas'][w].numpy()
        generation_parent = start
        generation_parent_end = row['window_start']
        decoder_blocks = {}
        decoded_blocks = set()
        if 'decoder_phases' in source:
            decoder = source['decoder_phases'][w]
            duration = ck(decoder['end_ns'] - decoder['start_ns'])
            decoder_sms = int(capture['gpu_resources']['consumer'][9])
            blocks = sorted({b['block'] for b in data['s1_blocks'] if b['window'] == w})
            assert blocks == list(range(len(blocks)))
            service = (duration * decoder_sms + len(blocks) - 1) // len(blocks)
            for block in blocks:
                task = graph.add(duration=service, kind='decoder')
                graph.edge(task, start)
                for meta in input_blocks[w, block]:
                    die = meta['die']
                    offset = data['hcf_lengths'][die] + data['s3_lengths'][die]
                    memory(task, die, [offset + meta['done']])
                decoder_blocks[block] = task
                ctas.append(dict(kind='decoder', sm=-1,
                                 start=decoder['start_ns'], end=decoder['end_ns'],
                                 first=task, last=task))
            generation_parent = None
            generation_parent_end = decoder['end_ns']
        order = np.lexsort((np.arange(len(generation_rows)), generation_rows[:, 1], generation_rows[:, 0]))
        ranks = np.empty(len(order), np.uint32)
        ranks[order] = np.arange(len(order), dtype=np.uint32)
        for task, (begin, end, sm) in enumerate(generation_rows):
            ns = source['static_count']
            total = source['window_inputs'][w]['generation_count'] if source.get('window_inputs') else row['depth'].shape[1]
            nd = total - ns
            if source.get('generation_layout') == 'records':
                ids = np.arange(task * 256, min(ns + nd, (task + 1) * 256))
            else:
                items = np.arange(task * 256, min(ns + layout[w]['width'] * nd, (task + 1) * 256))
                ids = np.where(items < ns, items, ns + (items - ns) % nd)
            if generation_loads is not None:
                assert source.get('generation_layout') == 'records'
                loaded = int(generation_loads[w][task])
                assert begin <= loaded <= end
                launch = graph.add(earliest=at(begin), kind='generation_cta_launch')
                selection_end = phases[w, 'generate'][0]['start_ns'] if source.get('window_inputs') else row['window_start']
                follow(launch, start, begin, selection_end)
                prefix = graph.add(duration=ck(loaded - begin), kind='generation_native_load_interval')
                graph.edge(prefix, launch)
                node = graph.add(duration=ck(end - loaded), kind='generation')
                graph.edge(node, prefix)
                for block in np.unique(ids // 256):
                    for meta in input_blocks[w, int(block)]:
                        die = meta['die']
                        offset = data['hcf_lengths'][die] + data['s3_lengths'][die]
                        memory(launch, die, offset + np.asarray(meta['producers'], np.uint64))
                        indices = offset + np.asarray(meta['input_reads'], np.uint64)
                        assert np.all(data['requests'][die]['kind'][indices] == 0)
                        assert not np.any(data['requests'][die]['ready'][indices])
                        memory(node, die, indices)
                        data['requests'][die]['ready'][indices] = at(begin)
                        assert len(generation_rows) <= 65536
                        tags = data['requests'][die]['tag']
                        tags[indices] = data['sh_lane_base'] + meta['bank'] << 24 | w << 16 | ranks[task]
                        source_publication(die, indices, 's1_generation', w, task, launch)
            else:
                node = graph.add(earliest=at(begin), duration=ck(end - begin), kind='generation')
                launch = node
            if generation_loads is None and 'decoder_phases' not in source:
                for block in np.unique(ids // 256):
                    for die, index in inputs[w, int(block)]:
                        memory(node, die, [index])
            if generation_loads is None:
                if decoder_blocks:
                    blocks = np.unique(ids // 10 // 256)
                    decoded_blocks.update(map(int, blocks))
                    gap = ck(max(0, int(begin) - int(generation_parent_end)))
                    for block in blocks:
                        graph.edge(node, decoder_blocks[int(block)], gap)
                else:
                    follow(node, generation_parent, begin, generation_parent_end)
            ctas.append(dict(kind='generation', sm=int(sm), start=int(begin), end=int(end), first=launch, last=node))
            starts.append(node)
            ends.append(int(end))
        assert not decoder_blocks or decoded_blocks == set(decoder_blocks)
        generation_end = phases[w, 'generate'][0]['end_ns']
        generated = graph.add(earliest=at(generation_end), kind='generation_join')
        tail = ck(max(0, generation_end - max(ends)))
        for node in starts:
            graph.edge(generated, node, tail)
        generation.append(generated)
        projection = phases[w, 'project'][0]
        if 'projection_finish' in phase_ctas:
            finish = next((p for p in source['phases'] if p['window'] == w and p['kind'] == 'projection_finish'))
            assert projection['start_ns'] <= finish['start_ns'] <= finish['end_ns'] <= projection['end_ns']
            project_prefix = dict(projection, end_ns=finish['start_ns'])
            projected = gpu_phase(w, 'projection', project_prefix, generated, generation_end)
            finish_tail = dict(finish, end_ns=projection['end_ns'])
            phase_nodes[w, 'project'] = gpu_phase(w, 'projection_finish', finish_tail, projected, finish['start_ns'])
        else:
            phase_nodes[w, 'project'] = gpu_phase(w, 'projection', projection, generated, generation_end)
        blocks = []
        if source.get('fused_generation_depth'):
            assert np.array_equal(row['publication'].numpy(), source['generation_ctas'][w][:, :2].numpy())
            blocks = starts
        else:
            for task, (begin, end) in enumerate(row['publication'].numpy()):
                node = graph.add(earliest=at(begin), duration=ck(end - begin), kind='depth_block')
                follow(node, generated, begin, generation_end)
                blocks.append(node)
                if 'depth_ctas' in source:
                    traced = source['depth_ctas'][w][task]
                    assert tuple(traced[:2]) == tuple((begin, end))
                    ctas.append(dict(kind='depth', sm=int(traced[2]), start=int(begin), end=int(end), first=node, last=node))
        depth.append(blocks)
        delta = phases[w, 'window_delta'][0]
        ready_delta = graph.add(earliest=at(delta['start_ns']), kind='window_delta_start')
        tail = ck(max(0, delta['start_ns'] - int(row['publication'][:, 1].max())))
        for block in blocks:
            graph.edge(ready_delta, block, tail)
        node = gpu_phase(w, 'window_delta', delta, ready_delta, delta['start_ns'])
        phase_nodes[w, 'window_delta'] = node
        s3 = graph.add(kind='s3_sort_inputs')
        for meta in data['windows']:
            if meta['window'] == w:
                memory(s3, meta['die'], [data['hcf_lengths'][meta['die']] + meta['done']])
        s3_done.append(s3)
        ordering = None
        mixed = [phase for phase in source['phases']
                 if phase['window'] == w and phase['kind'] == 'mixed_sort']
        if mixed:
            assert len(mixed) == 1
            ordering = gpu_phase(w, 'mixed_sort', mixed[0],
                                 phase_nodes[w, 'window_delta'], delta['end_ns'])
            graph.edge(ordering, s3)
        prior = phase_nodes[w, 'project']
        prior_end = projection['end_ns']
        for kind in ('crossing', 'colors'):
            phase = phases[w, kind][0]
            start_node = graph.add(earliest=at(phase['start_ns']), kind=kind + '_start')
            follow(start_node, prior, phase['start_ns'], prior_end)
            if kind == 'crossing':
                graph.edge(start_node, s3)
                if ordering is not None:
                    follow(start_node, ordering, phase['start_ns'], mixed[0]['end_ns'])
                delta = phases[w, 'window_delta'][0]
                follow(start_node, phase_nodes[w, 'window_delta'], phase['start_ns'], delta['end_ns'])
            phase_nodes[w, kind + '_start'] = start_node
            node = gpu_phase(w, kind, phase, start_node, phase['start_ns'])
            if kind == 'colors':
                join = graph.add(kind='colors_memory_join')
                graph.edge(join, node)
                for meta in data['s1_windows']:
                    if meta['window'] == w and meta['color_done'] != NONE:
                        memory(join, meta['die'], [data['hcf_lengths'][meta['die']] + data['s3_lengths'][meta['die']] + meta['color_done']])
                node = join
            phase_nodes[w, kind] = node
            prior = node
            prior_end = phase['end_ns']
        for span, phase in enumerate(phases[w, 'sort']):
            node = graph.add(earliest=at(phase['start_ns']), duration=0 if phase.get('shared_sort') else ck(phase['end_ns'] - phase['start_ns']), kind='sort')
            follow(node, phase_nodes[w, 'colors'], phase['start_ns'], phases[w, 'colors'][0]['end_ns'])
            if phase.get('shared_sort'):
                graph.edge(node, phase_nodes[w, 'sort', 0])
            graph.edge(node, s3)
            phase_nodes[w, 'sort', span] = node
    details = {}
    extra_resources = dict(generation=source['generation_resources'], generation_threads=256)
    if source.get('decoder_phases'):
        decoder_resources = list(source['generation_resources'])
        decoder_resources[:2] = (86, 32768)
        extra_resources.update(decoder=decoder_resources, decoder_threads=128)
    if 'depth_resources' in source:
        extra_resources.update(depth=source['depth_resources'], depth_threads=256)
    for kind, resource in source.get('phase_resources', {}).items():
        assert resource[4:] == source['generation_resources'][4:]
        threads = source['phase_threads']
        extra_resources.update({kind: resource, kind + '_threads': threads[kind] if isinstance(threads, dict) else threads})
    from .pim_trace import capture_gpu_graph
    graph, description = capture_gpu_graph(capture, data['requests'], data['expansions'], data['segments'], data['fields'], len(capture['segmented_supply']), graph=graph, origin_ns=origin, hcf_lengths=data['hcf_lengths'], extra_ctas=ctas, extra_resources=extra_resources, details=details, captured_resource_gaps=False)
    if schedule_details is not None:
        schedule_details.update(details)
    for key, node in details['roots'].items():
        kind, span, release = key[:3]
        w = details['span_windows'][span]
        local = span_layout[span]['local']
        if kind == 'field_source' and len(key) == 4:
            assert color_loads is not None, 'block fields need bound native SH inputs'
            publication = at(capture['segmented_supply'][span]['field_publication'][0])
            graph.nodes[node]['earliest'] = max(graph.nodes[node]['earliest'], publication)
            phase = phases[w, 'colors'][0]
            graph.edge(node, phase_nodes[w, 'colors'], max(0, publication - at(phase['end_ns'])))
            continue
        phase_kind = 'colors' if kind == 'field_source' else 'sort'
        phase = phases[w, phase_kind][0 if phase_kind == 'colors' else local]
        parent = phase_nodes[(w, phase_kind) if phase_kind == 'colors' else (w, 'sort', local)]
        graph.edge(node, parent, max(0, release - at(phase['end_ns'])))
    for die, stream in enumerate(data['requests']):
        hcf_end = data['hcf_lengths'][die]
        s3_end = hcf_end + data['s3_lengths'][die]
        publication_by_window = []
        for row, blocks in zip(source['windows'], depth):
            mapping = {}
            for block, (_, end) in zip(blocks, row['publication'].numpy()):
                mapping.setdefault(at(end), []).append(block)
            publication_by_window.append(mapping)
        def grouped(indices, keys):
            _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
            stable = np.empty(len(first), np.int64)
            stable[np.argsort(first)] = np.arange(len(first))
            groups = stable[inverse]
            order = np.argsort(groups, kind='stable')
            boundaries = np.r_[0, 1 + np.flatnonzero(
                groups[order][1:] != groups[order][:-1]), len(order)]
            return (indices[order[begin:end]]
                    for begin, end in zip(boundaries[:-1], boundaries[1:]))

        s3_indices = np.flatnonzero(stream['ready'][hcf_end:s3_end]) + hcf_end
        if len(s3_indices):
            s3_windows = (stream['tag'][s3_indices] >> 16 & 255).astype(np.uint64)
            s3_releases = stream['ready'][s3_indices].astype(np.uint64)
            starts = np.asarray([at(row['window_start'])
                                 for row in source['windows']], np.uint64)
            draining = s3_releases != starts[s3_windows]
            banks = np.where(draining,
                             stream['address'][s3_indices] // 2048 % 16 + 1, 0)
            keys = s3_releases << 16 | s3_windows << 8 | banks
            for indices in grouped(s3_indices, keys):
                index = int(indices[0])
                row = stream[index]
                w = int(row['tag']) >> 16 & 255
                release = int(row['ready'])
                drain = release != at(source['windows'][w]['window_start'])
                bank = int(row['address'] // 2048 % 16) if drain else None
                parent = ([windows[w]] if not drain
                          else publication_by_window[w][release])
                publisher = graph.add(earliest=release, kind='s3_publication')
                for p in parent:
                    graph.edge(publisher, p)
                if drain:
                    memory(publisher, die, bank_inputs.get((w, die, bank), []))
                graph.nodes[publisher]['publications'].extend(
                    zip(repeat(die), indices.tolist()))

        s1_indices = np.flatnonzero(stream['ready'][s3_end:]) + s3_end
        if len(s1_indices):
            s1_windows = (stream['tag'][s1_indices] >> 16 & 255).astype(np.uint64)
            s1_releases = stream['ready'][s1_indices].astype(np.uint64)
            launches = source_publishers[die][s1_indices - s3_end].astype(np.int64)
            source_reads = stream['kind'][s1_indices] == 0
            special = source_reads & (launches >= 0)
            keys = s1_releases << 8 | s1_windows
            keys[special] = np.uint64(1 << 63) | launches[special].astype(np.uint64)
            for indices in grouped(s1_indices, keys):
                index = int(indices[0])
                row = stream[index]
                w = int(row['tag']) >> 16 & 255
                release = int(row['ready'])
                launch = (int(source_publishers[die][index - s3_end])
                          if row['kind'] == 0 else -1)
                if launch >= 0:
                    kind, window, _ = publisher_info[launch]
                    assert window == w and release == graph.nodes[launch]['earliest']
                    parent = [launch]
                elif row['kind'] == 0:
                    if color_loads is not None:
                        raise AssertionError('source read has no captured publisher')
                    assert release == at(phases[w, 'colors'][0]['start_ns'])
                    kind = 's1'
                    parent = [phase_nodes[w, 'colors_start']]
                else:
                    assert (w > 0 and
                            release == at(phases[w - 1, 'window_delta'][0]['end_ns']))
                    kind = 's1'
                    parent = [phase_nodes[w - 1, 'window_delta']]
                publisher = graph.add(earliest=release, kind=kind + '_publication')
                for p in parent:
                    graph.edge(publisher, p)
                graph.nodes[publisher]['publications'].extend(
                    zip(repeat(die), indices.tolist()))
    description.update(generation_ctas=sum((t['kind'] == 'generation' for t in ctas)), phase_ctas={kind: sum((t['kind'] == kind for t in ctas)) for kind in phase_ctas}, depth_blocks=sum(map(len, depth)), fused_generation_depth=source.get('fused_generation_depth', False), kernel_join='latest CTA completion plus one measured tail; early CTA slack is retained', window_barrier=False, after_render=False, s3_publication_barrier=False, s3_bank_drain=True, field_publication_barrier=True, generation_input_decomposition=generation_loads is not None, stage_resources_incomplete=[kind for kind in ('projection_finish', 'window_delta', 'crossing', 'sort') if kind not in phase_ctas] + ([] if source.get('fused_generation_depth') else ['depth']) + [kind for kind in ('projection', 'colors') if kind not in phase_ctas], color_reads='per-64-record CTA dispatch releases physical SH reads; native load prefix overlaps their service; all source reads gate measured post-load arithmetic, stores and fences' if color_loads is not None else 'physical completion joined with color service; per-CTA demand not captured')
    return (graph, description)


def release_graph(bundle):
    nodes = bundle['nodes']
    starts = [i for i, node in enumerate(nodes) if node['kind'] == 'window_start']
    deltas = [i for i, node in enumerate(nodes) if node['kind'] == 'window_delta_join']
    if not deltas:
        deltas = [i for i, node in enumerate(nodes) if node['kind'] == 'window_delta']
    assert starts and len(starts) == len(deltas)
    for window in range(1, len(starts)):
        nodes[starts[window]]['parents'].append((deltas[window - 1], 0))
    native_service = sum((node['duration'] for node in nodes))
    released = []
    for node in nodes:
        released.append(node['earliest'])
        node['earliest'] = 0
    dispatch_edges = 0
    expected = dict(generation='window_start', projection='generation_join', projection_finish='projection_join', colors='colors_start', window_delta='window_delta_start')
    groups = {}
    for cta in bundle['ctas']:
        if cta['kind'] not in expected:
            continue
        node = nodes[cta['first']]
        parents = [(i, parent, gap) for i, (parent, gap) in enumerate(node['parents']) if nodes[parent]['kind'] == expected[cta['kind']] or (cta['kind'] == 'generation' and nodes[parent]['kind'] == 'decoder')]
        assert len(parents) == 1, (cta['kind'], parents)
        i, parent, gap = parents[0]
        groups.setdefault((cta['kind'], parent), []).append((cta['first'], i, gap))
    for (kind, parent), rows in groups.items():
        launch = min((gap for _, _, gap in rows))
        for child, index, gap in rows:
            nodes[child]['parents'][index] = (parent, launch)
            dispatch_edges += gap != launch
    sources = 0
    for stream in bundle['requests']:
        ready = stream['ready'] > 0
        sources += int(ready.sum())
    assert sum((node['duration'] for node in nodes)) == native_service
    result = dict(windows=len(starts), sources=sources, native_service_ck=native_service, removed_absolute_floors=len(released), largest_capture_floor_ck=max(released), dispatch_edges=dispatch_edges, retained='native durations, measured dependency delays, physical transactions, source publications, finite SM ownership and window-buffer reuse', window_policy='next S1 released by current width decision; each GPU CTA still waits for its actual physical inputs')
    bundle['ready_schedule'] = result
    return result


def record_blocks(begin, end, records, block):
    result = set()
    while begin < end:
        offset = begin % records
        stop = min(end, begin + records - offset)
        result.update(range(offset // block, (offset + stop - begin - 1) // block + 1))
        begin = stop
    return result


def frame_mappings(bundle, capture):
    nodes = bundle['nodes']
    supply = capture['decision_supply']
    starts = [i for i, node in enumerate(nodes) if node['kind'] == 'window_start']
    crossings = [i for i, node in enumerate(nodes) if node['kind'] == 'crossing']
    sorts = [i for i, node in enumerate(nodes) if node['kind'] == 'sort']
    assert len(starts) == len(crossings) == len(supply['windows'])
    ctas_by_kind = defaultdict(list)
    for cta in bundle['ctas']:
        ctas_by_kind[cta['kind']].append(cta)
    windows, field_producers, dependencies = ([], [], [])
    frame_offset = 0
    cursors = defaultdict(int)
    for w, raw in enumerate(supply['windows']):
        width, n = raw['depth'].shape
        generated = supply['window_inputs'][w]['generation_count'] if supply.get('window_inputs') else n
        count = dict(generation=(generated + 255) // 256, projection=(width * n + 255) // 256, colors=(n + 63) // 64)
        if ctas_by_kind['projection_finish']:
            count['projection_finish'] = (width * n + 255) // 256
        stages = {}
        for kind, amount in count.items():
            first = cursors[kind]
            stages[kind] = ctas_by_kind[kind][first:first + amount]
            assert len(stages[kind]) == amount
            cursors[kind] += amount
        windows.append(dict(start=starts[w], crossing=crossings[w], records=n, first_frame=frame_offset, width=width))
        frame_offset += width
        field_producers.append([cta['last'] for cta in stages['colors']])
        if not supply.get('decoder_phases'):
            for index, task in enumerate(stages['projection']):
                producers = record_blocks(index * 256, min((index + 1) * 256, width * n), n, 256)
                dependencies.append((task['first'], 'generation_join', [stages['generation'][b]['last'] for b in sorted(producers)]))
        for index, task in enumerate(stages.get('projection_finish', [])):
            dependencies.append((task['first'], 'projection_join', [stages['projection'][index]['last']]))
        for index, task in enumerate(stages['colors']):
            begin, end = (index * 64, min((index + 1) * 64, n))
            producers = set()
            for frame in range(width):
                producers.update((stages.get('projection_finish', stages['projection'])[b]['last'] for b in range((frame * n + begin) // 256, (frame * n + end - 1) // 256 + 1)))
            dependencies.append((task['first'], 'colors_start', sorted(producers)))
    assert frame_offset == capture['frames']
    assert all((cursors[kind] == len(ctas_by_kind[kind]) for kind in cursors))
    from .pim_trace import capture_layout
    _, captured_spans = capture_layout(capture)
    spans = []
    frame_events = [dict(submit=starts[w], complete=[]) for w, window in enumerate(windows) for _ in range(window['width'])]
    consumers = ctas_by_kind['consumer']
    cursor = 0
    span_offset = defaultdict(int)
    assert len(capture['segmented_supply']) == len(sorts)
    for span, raw in enumerate(capture['segmented_supply']):
        window = captured_spans[span]['window']
        width = raw['width']
        first_frame = windows[window]['first_frame'] + span_offset[window]
        spans.append(dict(window=window, sort=sorts[span], width=width, first_frame=first_frame))
        span_offset[window] += width
        tasks = raw['task_trace'].numpy()
        warps = capture['gpu_resources']['consumer_threads'] // 32
        for first in range(0, len(tasks), warps):
            cta = consumers[cursor]
            cursor += 1
            selected = tasks[first:first + warps]
            assert cta['start'] == int(selected[:, 0].min())
            assert cta['end'] == int(selected[:, 1].max())
            assert cta['sm'] == int(selected[0, 2])
            cta['span'] = span
            stores = [parent for parent, _ in nodes[cta['last']]['parents']]
            assert len(stores) == len(selected)
            for warp, store in enumerate(stores):
                assert nodes[store]['kind'] == 'consumer_store'
                frame = first_frame + (first + warp) // 2 % width
                frame_events[frame]['complete'].append(store)
    assert cursor == len(consumers)
    assert all((span_offset[w] == window['width'] for w, window in enumerate(windows)))
    assert all((frame['complete'] for frame in frame_events))
    bundle.update(windows=windows, spans=spans, field_producers=field_producers, block_dependencies=dependencies, frame_events=frame_events, quality=capture['quality'], input_sha256={'checkpoint': supply['checkpoint_sha256']}, measurements=dict(scene=capture['scene'], frames=capture['frames'], baseline_ms=capture['baseline_ms'], dataflow_ms=capture['durations_ms']['native_shared']))
    return bundle
def build_trace(capture, capture_sha256):
    """Bind measured GPU tasks to the common PPIM/HCF memory interface."""
    data = build_requests(capture)
    data['color_inputs'] = color_input_blocks(capture, data)
    visibility = native_sh_visibility(capture, data)
    bank_streams(data)
    details = {}
    graph, description = connect_gpu(capture, data, schedule_details=details)
    description['sh_visibility'] = visibility
    bundle = dict(nodes=graph.nodes, requests=data['requests'], description=description,
                  ctas=details['ctas'], resources=details['resources'],
                  resource_reuse_edges=details['resource_reuse_edges'],
                  consumer_reserve=capture['configuration']['consumer_reserve'],
                  capture_sha256=capture_sha256, source_sha256=capture['hashes'])
    release_graph(bundle)
    frame_mappings(bundle, capture)
    bundle['input_sha256'] = capture['input_sha256']
    bundle['candidate_rgb_sha256'] = capture['candidate_rgb_sha256']
    bundle['image_checks'] = dict(shared_gpu_rgb_error=capture['shared_gpu_rgb_error'],
                                  scheduling_rgb_error=capture['scheduling_rgb_error'],
                                  window_decisions=capture['decisions'])
    bundle['builder_sha256'] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('pim_capture.py', 'pim_requests.py', 'pim_trace.py')}
    return bundle
