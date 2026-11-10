"""Block publication and finite GPU admission for the hybrid schedule."""

from collections import Counter, defaultdict
import heapq

import numpy as np


def release_fields(bundle):
    """Connect each packed field source to its captured producer block."""
    nodes = bundle['nodes']
    spans = bundle['spans']
    producers = bundle['field_producers']
    changed = 0
    for node in nodes:
        if node['kind'] != 'field_source':
            continue
        parents = set()
        for die, index in node['publications']:
            tag = int(bundle['requests'][die]['tag'][index])
            span = (tag >> 24) // 2
            assert tag >> 24 == span * 2 and span < len(spans)
            blocks = producers[spans[span]['window']]
            block = (tag & 0xffffff) % len(blocks)
            parents.add(blocks[block])
        assert len(parents) == 1
        retained = [(parent, gap) for parent, gap in node['parents']
                    if nodes[parent]['kind'] != 'colors_memory_join']
        assert len(retained) < len(node['parents'])
        node['parents'] = retained + [(parents.pop(), 0)]
        changed += 1
    return changed


def field_queues(bundle):
    """Route packed field groups to twelve bank queues on each die."""
    streams = bundle['requests']
    field_lanes = set(range(0, 2 * len(bundle['spans']), 2))
    occupied = {int(lane) for stream in streams
                for lane in np.unique(stream['tag'] >> 24)} - field_lanes
    lanes = [lane for lane in range(256) if lane not in occupied][:12]
    if len(lanes) != 12:
        raise ValueError('field bank queues exceed the controller lane namespace')
    groups = 0
    for stream in streams:
        original = stream['tag'] >> 24
        indices = np.flatnonzero(np.isin(original, list(field_lanes)))
        tags = stream['tag'][indices]
        if not len(indices):
            continue
        boundaries = np.r_[0, np.flatnonzero(tags[1:] != tags[:-1]) + 1, len(indices)]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            chunk = indices[lo:hi]
            assert stream['kind'][chunk[0]] == 7 and stream['kind'][chunk[-1]] == 8
            assert np.all(tags[lo:hi] == tags[lo])
            reads = chunk[stream['kind'][chunk] == 2]
            assert len(reads)
            bank = (stream['address'][reads] // 2048) % 16
            assert np.all(bank == bank[0]) and bank[0] < 12
            span = int(original[chunk[0]]) // 2
            window = bundle['spans'][span]['window']
            assert 0 <= window < 128
            stream['tag'][chunk] = ((lanes[int(bank[0])] << 24) | (1 << 23)
                                    | (window << 16) | (int(tags[lo]) & 65535))
            groups += 1
    return dict(lanes=lanes, groups=groups)


def block_dependencies(bundle):
    """Replace stage-wide fences with the captured block producer mapping."""
    nodes = bundle['nodes']
    changed = Counter()
    for child, old_kind, parents in bundle['block_dependencies']:
        previous = [(parent, gap) for parent, gap in nodes[child]['parents']
                    if nodes[parent]['kind'] == old_kind]
        assert len(previous) == 1, (child, old_kind)
        nodes[child]['parents'].remove(previous[0])
        nodes[child]['parents'].extend((parent, previous[0][1]) for parent in sorted(set(parents)))
        changed[old_kind] += 1
    for span in bundle['spans']:
        crossing = bundle['windows'][span['window']]['crossing']
        edge = (crossing, 0)
        if edge not in nodes[span['sort']]['parents']:
            nodes[span['sort']]['parents'].append(edge)
    return dict(changed)


def prepare(bundle):
    """Queue a consumer until its sorting and physical input fences complete."""
    nodes = bundle['nodes']
    for child, parent, gap in bundle['resource_reuse_edges']:
        parents = nodes[child]['parents']
        parents.remove((parent, gap) if (parent, gap) in parents else (parent, 0))
    internal = {'consumer_cta_start', 'consumer_loop', 'compose',
                'consumer_store', 'consumer_cta_end'}
    children = defaultdict(list)
    for child, node in enumerate(nodes):
        if node['kind'] in internal:
            for parent, _ in node['parents']:
                if nodes[parent]['kind'] in internal:
                    children[parent].append(child)
    additional = 0
    work = {}
    for cta in bundle['ctas']:
        if cta['kind'] != 'consumer':
            continue
        head, tail = cta['first'], cta['last']
        nodes[head]['parents'].append((bundle['spans'][cta['span']]['sort'], 0))
        memory = defaultdict(set)
        for die, indices in nodes[head]['memory']:
            memory[die].update(map(int, indices))
        warp_work = []
        for loop in children[head]:
            cursor, sectors = loop, 0
            while cursor != tail:
                node = nodes[cursor]
                assert not node['publications']
                if node['kind'] == 'compose':
                    for die, indices in node['memory']:
                        memory[die].update(map(int, indices))
                        sectors += len(indices)
                following = children[cursor]
                assert len(following) == 1
                cursor = following[0]
            warp_work.append(sectors)
        work[head] = max(warp_work, default=0)
        previous = sum(len(indices) for _, indices in nodes[head]['memory'])
        nodes[head]['memory'] = [(die, np.asarray(sorted(indices), dtype=np.uint64))
                                for die, indices in sorted(memory.items())]
        additional += sum(len(indices) for _, indices in nodes[head]['memory']) - previous
    bundle['consumer_admission'] = dict(additional_input_fences=additional)
    return work


class Resources:
    """Own SM warps, registers, shared memory and block slots until CTA completion."""

    PRIORITY = ('window_delta', 'decoder', 'generation', 'depth', 'projection', 'projection_finish',
                'colors', 'cull', 'consumer')

    def __init__(self, bundle, work):
        self.capacity = tuple(bundle['description']['sm_capacity'])
        self.sms = int(bundle['resources']['consumer'][9])
        self.used = [[0] * 4 for _ in range(self.sms)]
        self.tasks, self.ends, self.assigned, self.live = {}, {}, {}, {}
        self.waiting = defaultdict(list)
        self.queued, self.admissions = set(), []
        self.work = work
        self.consumer_windows = {task['first']: bundle['spans'][task['span']]['window']
                                 for task in bundle['ctas'] if task['kind'] == 'consumer'}
        self.exclusive=set()
        for task in bundle['ctas']:
            kind = task['kind']
            if kind not in self.PRIORITY:
                raise ValueError(f'unknown GPU stage: {kind}')
            if task.get('all_sms'):
                self.exclusive.add(task['first']);demand=self.capacity
            elif task.get('full_sm'):
                demand=self.capacity
            else:
                resource = bundle['resources'][kind]
                threads = bundle['resources'][kind + '_threads']
                warps = (threads + resource[8] - 1) // resource[8]
                shared = resource[1] + (max(0, bundle['consumer_reserve'] - 6144)
                                         if kind == 'consumer' else 0)
                demand = (warps, ((resource[0] * resource[8] + 255) // 256) * warps,
                          (shared + 255) // 256, 1)
            assert all(value <= limit for value, limit in zip(demand, self.capacity))
            first, last = task['first'], task['last']
            self.tasks[first] = kind, demand, last
            self.ends[last] = first
        assert len(self.tasks) == len(bundle['ctas']) == len(self.ends)

    def enqueue(self, node, release):
        if node not in self.tasks or node in self.assigned:
            return False
        assert node not in self.queued
        self.queued.add(node)
        kind = self.tasks[node][0]
        priority = (self.consumer_windows[node], -self.work[node]) if kind == 'consumer' else release
        heapq.heappush(self.waiting[kind], (priority, release, node))
        return True

    def start(self, node):
        return self.assigned[node][1] if node in self.assigned else 0

    def complete(self, node, timestamp):
        if node in self.ends:
            first = self.ends[node]
            for sm in self.live.pop(first):
                for i, value in enumerate(self.tasks[first][1]):self.used[sm][i] -= value
                assert min(self.used[sm]) >= 0

    def dispatch(self, clock):
        admitted = []
        for kind in self.PRIORITY:
            queue = self.waiting[kind]
            while queue and queue[0][1] <= clock:
                _, _, node = queue[0]
                demand = self.tasks[node][1]
                available = [sm for sm, used in enumerate(self.used)
                             if all(a + b <= c for a, b, c in zip(used, demand, self.capacity))]
                if not available:
                    break
                if node in self.exclusive:
                    if len(available)!=self.sms:break
                    selected=tuple(available)
                else:selected=(min(available,key=lambda sm:(sum(self.used[sm]),sm)),)
                sm=selected[0]
                heapq.heappop(queue)
                self.live[node] = selected
                self.assigned[node] = sm, clock
                self.admissions.append((node, sm, clock))
                for sm in selected:
                    for i, value in enumerate(demand):self.used[sm][i] += value
                admitted.append((clock, node))
        return admitted

    def verify(self, times):
        assert not self.live and not any(self.waiting.values())
        assert len(self.admissions) == len(self.tasks)
        events = defaultdict(list)
        peaks = np.zeros(4, np.int64)
        for first, sm, start in self.admissions:
            _, demand, last = self.tasks[first]
            assert int(times[first, 0]) == start
            for owner in (range(self.sms) if first in self.exclusive else (sm,)):
                events[owner].extend(((start,1,demand),(int(times[last,1]),-1,demand)))
        for sm, rows in events.items():
            used = np.zeros(4, np.int64)
            for timestamp, sign, demand in sorted(rows):
                used += sign * np.asarray(demand)
                assert np.all(used >= 0) and np.all(used <= self.capacity), (sm, timestamp, used)
                peaks = np.maximum(peaks, used)
            assert not used.any()
        return dict(ctas=len(self.tasks), sms=self.sms, peaks=peaks.tolist(), capacity=self.capacity)


def validate_staging(rows, done):
    allocated = np.flatnonzero(rows['kind'] == 7)
    released = np.flatnonzero(rows['kind'] == 8)
    assert np.array_equal(np.sort(rows['cost'][released]), allocated)
    events = []
    for end in released:
        start = int(rows['cost'][end])
        size = ((int(rows['cost'][start]) + 255) // 256) * 256
        assert done[end] >= done[start] and 0 < size <= 16384
        events.extend(((int(done[start]), 1, start, size), (int(done[end]), -1, start, size)))
    current = maximum = 0
    live = set()
    for _, sign, node, size in sorted(events):
        if sign > 0:
            assert node not in live
            live.add(node)
        else:
            assert node in live
            live.remove(node)
        current += sign * size
        maximum = max(maximum, current)
        assert 0 <= current <= 16384
    assert not live and current == 0
    return dict(maximum_bytes=maximum, capacity_bytes=16384)


def serial_boundaries(bundle):
    """Drain one batch before admitting the following batch or window."""
    nodes = bundle['nodes']
    consumers = defaultdict(list)
    for cta in bundle['ctas']:
        if cta['kind'] == 'consumer':
            consumers[cta['span']].append(cta['last'])
    drains = []
    for span in range(len(bundle['spans'])):
        assert consumers[span]
        drains.append(len(nodes))
        nodes.append(dict(earliest=0, duration=0, kind='batch_drain',
                          parents=[(parent, 0) for parent in consumers[span]],
                          memory=[], publications=[]))
    for span, descriptor in enumerate(bundle['spans']):
        if span:
            nodes[descriptor['sort']]['parents'].append((drains[span - 1], 0))
    for node in nodes:
        if node['kind'] in ('field_source', 'route_source'):
            spans = set()
            for die, index in node['publications']:
                lane = int(bundle['requests'][die]['tag'][index]) >> 24
                if node['kind'] == 'field_source':
                    spans.add(lane // 2)
                else:
                    spans.add((lane - 2 * len(bundle['spans'])) // 4)
            assert len(spans) == 1
            span = spans.pop()
            if span:
                node['parents'].append((drains[span - 1], 0))
        elif node['kind'] == 's1_publication':
            windows = {((int(bundle['requests'][die]['tag'][index]) >> 16) & 255)
                       for die, index in node['publications']}
            assert len(windows) == 1
            window = windows.pop()
            edge = (bundle['windows'][window]['start'], 0)
            if edge not in node['parents']:
                node['parents'].append(edge)
    for window in range(1, len(bundle['windows'])):
        previous = max(span for span, descriptor in enumerate(bundle['spans'])
                       if descriptor['window'] == window - 1)
        nodes[bundle['windows'][window]['start']]['parents'].append((drains[previous], 0))


def frame_pipeline(bundle):
    """Overlap next-frame PIM preparation while keeping GPU frames serial."""
    ctas = bundle['ctas']
    staged_publication = any(cta['kind'] in ('decoder', 'projection_finish')
                             for cta in ctas)
    nodes = bundle['nodes']
    windows = bundle['windows']
    frames = bundle['frame_events']
    if len(windows) != len(frames) or any(window['width'] != 1 for window in windows):
        raise ValueError('frame pipeline requires one window per frame')

    drains = []
    for frame in frames:
        assert frame['complete']
        drains.append(len(nodes))
        nodes.append(dict(earliest=0, duration=0, kind='frame_drain',
                          parents=[(parent, 0) for parent in frame['complete']],
                          memory=[], publications=[]))

    span_windows = [descriptor['window'] for descriptor in bundle['spans']]
    if staged_publication:
        for cta in ctas:
            if cta['kind'] != 'consumer':
                continue
            window = span_windows[cta['span']]
            if window:
                edge = (drains[window - 1], 0)
                if edge not in nodes[cta['first']]['parents']:
                    nodes[cta['first']]['parents'].append(edge)
    else:
        for span, descriptor in enumerate(bundle['spans']):
            window = span_windows[span]
            if not window:
                continue
            edge = (drains[window - 1], 0)
            nodes[descriptor['sort']]['parents'].append(edge)
            first, last = windows[window]['start'], descriptor['sort']
            for cta in ctas:
                if not first <= cta['first'] <= last:
                    continue
                early = cta['kind'] in ('generation', 'projection', 'colors')
                if not early and edge not in nodes[cta['first']]['parents']:
                    nodes[cta['first']]['parents'].append(edge)

        for window in range(2, len(windows)):
            edge = (drains[window - 2], 0)
            if edge not in nodes[windows[window]['start']]['parents']:
                nodes[windows[window]['start']]['parents'].append(edge)

    for node in nodes:
        if node['kind'] == 's1_publication':
            publications = {((int(bundle['requests'][die]['tag'][index]) >> 16) & 255)
                            for die, index in node['publications']}
            assert len(publications) == 1
            window = publications.pop()
            edge = (windows[window]['start'], 0)
            if edge not in node['parents']:
                node['parents'].append(edge)
            continue
        if staged_publication:
            continue
        if node['kind'] not in ('field_source', 'route_source'):
            continue
        spans = set()
        for die, index in node['publications']:
            lane = int(bundle['requests'][die]['tag'][index]) >> 24
            if node['kind'] == 'field_source':
                spans.add(lane // 2)
            else:
                spans.add((lane - 2 * len(bundle['spans'])) // 4)
        assert len(spans) == 1
        window = span_windows[spans.pop()]
        if window >= 2:
            edge = (drains[window - 2], 0)
            if edge not in node['parents']:
                node['parents'].append(edge)
