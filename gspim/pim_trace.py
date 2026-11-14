"""Captured GPU dependencies and compact binary PIM scheduling traces."""

from __future__ import annotations

import io
import heapq
import hashlib
import pickle
from pathlib import Path
import zlib
import zipfile

import numpy as np

from .pim_graph import Publications, ResourceTokens, publication_groups


SCHEMA = 2


def capture_layout(capture):
    """Map captured spans to their actual windows and frame offsets."""
    windows, spans = [], []
    first = int(capture['configuration'].get('first', 0))
    for raw in capture['decision_supply']['windows']:
        width = len(raw['depth'])
        if not 1 <= width <= 5:
            raise ValueError('captured window must contain one to five frames')
        windows.append(dict(first_frame=first, width=width, records=raw['depth'].shape[1], spans=[]))
        first += width
    if first != int(capture['configuration'].get('first', 0)) + capture['frames']:
        raise ValueError('captured windows do not cover the requested frames')
    window, offset = 0, 0
    publications = {}
    for index, raw in enumerate(capture['segmented_supply']):
        if window >= len(windows):
            raise ValueError('captured spans exceed the window sequence')
        width = int(raw['width'])
        if width < 1 or offset + width > windows[window]['width']:
            raise ValueError('captured span crosses a window boundary')
        publication = int(raw['field_publication'][0])
        if publication in publications and publications[publication] != window:
            raise ValueError('a field publication belongs to multiple windows')
        publications[publication] = window
        windows[window]['spans'].append(index)
        spans.append(dict(window=window, local=len(windows[window]['spans']) - 1,
                          first_frame=windows[window]['first_frame'] + offset,
                          width=width))
        offset += width
        if offset == windows[window]['width']:
            window, offset = window + 1, 0
    if window != len(windows) or offset:
        raise ValueError('captured spans do not cover every window')
    return windows, spans


def input_identity(checkpoint, images, model, frames):
    """Bind a capture to its checkpoint, camera and ordered target image bytes."""
    def digest(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    camera = hashlib.sha256()
    for name in ('view', 'intrinsics'):
        values = getattr(model, name).detach().cpu().contiguous().numpy()
        camera.update(name.encode())
        camera.update(str((values.shape, values.dtype.str)).encode())
        camera.update(values.tobytes())
    camera.update(repr((model.width, model.height, *(model.config.get(key) for key in
        ('kernel_size', 'near', 'far', 'background')))).encode())
    prefix = getattr(model, 'image_prefix', '')
    return dict(checkpoint=digest(checkpoint), camera=camera.hexdigest(),
                images=[digest(Path(images) / f'{prefix}{frame:04d}.png') for frame in range(frames)])


class _MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f'unsupported trace metadata type: {module}.{name}')


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {_plain(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value) if isinstance(value, tuple) else [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f'unsupported trace metadata type: {type(value).__name__}')


def _kind_table(nodes):
    kinds = sorted({node["kind"] for node in nodes})
    return kinds, {kind: index for index, kind in enumerate(kinds)}


def _pack_nodes(nodes):
    kinds, kind_ids = _kind_table(nodes)
    count = len(nodes)
    earliest = np.empty(count, dtype=np.int64)
    duration = np.empty(count, dtype=np.int64)
    kind = np.empty(count, dtype=np.uint16)
    parent_offsets = np.zeros(count + 1, dtype=np.int64)
    parent_indices = []
    parent_gaps = []
    memory_offsets = np.zeros(count + 1, dtype=np.int64)
    memory_die = []
    memory_value_offsets = [0]
    memory_values = []
    publication_offsets = np.zeros(count + 1, dtype=np.int64)
    publication_die = []
    publication_request = []
    publication_count = 0
    for index, node in enumerate(nodes):
        earliest[index] = node["earliest"]
        duration[index] = node["duration"]
        kind[index] = kind_ids[node["kind"]]
        for parent, gap in node["parents"]:
            parent_indices.append(parent)
            parent_gaps.append(gap)
        parent_offsets[index + 1] = len(parent_indices)
        for die, values in node["memory"]:
            values = np.asarray(values, dtype=np.uint64).reshape(-1)
            memory_die.append(die)
            memory_values.append(values)
            memory_value_offsets.append(memory_value_offsets[-1] + len(values))
        memory_offsets[index + 1] = len(memory_die)
        for die, requests in publication_groups(node):
            publication_die.append(np.full(len(requests), die, np.int16))
            publication_request.append(requests)
            publication_count += len(requests)
        publication_offsets[index + 1] = publication_count
    return {
        "node_earliest": earliest,
        "node_duration": duration,
        "node_kind": kind,
        "node_parent_offsets": parent_offsets,
        "node_parent_indices": np.asarray(parent_indices, dtype=np.int32),
        "node_parent_gaps": np.asarray(parent_gaps, dtype=np.int64),
        "node_memory_offsets": memory_offsets,
        "node_memory_die": np.asarray(memory_die, dtype=np.int16),
        "node_memory_value_offsets": np.asarray(memory_value_offsets, dtype=np.int64),
        "node_memory_values": (np.concatenate(memory_values) if memory_values else np.empty(0, np.uint64)),
        "node_publication_offsets": publication_offsets,
        "node_publication_die": np.concatenate(publication_die) if publication_die else np.empty(0, np.int16),
        "node_publication_request": np.concatenate(publication_request) if publication_request else np.empty(0, np.uint32),
        "kind_names": kinds,
    }


def _pack_bundle(bundle):
    arrays = _pack_nodes(bundle["nodes"])
    arrays.update({f"request_{die}": np.asarray(request) for die, request in enumerate(bundle["requests"])})
    arrays["resource_reuse_edges"] = np.asarray(bundle.get("resource_reuse_edges", []), dtype=np.int64)
    metadata = {
        "schema": SCHEMA,
        "kind_names": arrays.pop("kind_names"),
        "description": bundle.get("description", {}),
        "capture_sha256": bundle.get("capture_sha256"),
        "captured_interval_ms": bundle.get("captured_interval_ms"),
        "source_sha256": bundle.get("source_sha256", {}),
        "ctas": bundle.get("ctas", []),
        "resources": bundle.get("resources", {}),
        "consumer_reserve": bundle.get("consumer_reserve", 0),
        "ready_schedule": bundle.get("ready_schedule", {}),
        "measurements": bundle.get("measurements", {}),
    }
    for key in ('windows', 'spans', 'field_producers', 'block_dependencies',
                'frame_events', 'mean_submission', 'quality', 'input_sha256',
                'image_checks', 'candidate_rgb_sha256', 'builder_sha256', 'recording_sha256'):
        if key in bundle:
            metadata[key] = bundle[key]
    metadata_blob = np.frombuffer(zlib.compress(pickle.dumps(_plain(metadata), protocol=5), 9), dtype=np.uint8)
    arrays["metadata"] = metadata_blob
    return arrays


def write_trace(bundle, destination: Path) -> None:
    """Write graph arrays and scalar metadata to one compressed binary file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for name, array in _pack_bundle(bundle).items():
            with archive.open(name+'.npy', 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(array), allow_pickle=False)
    temporary.replace(destination)


def _unpack_nodes(values):
    metadata = _MetadataReader(io.BytesIO(zlib.decompress(values["metadata"].tobytes()))).load()
    kinds = metadata["kind_names"]
    earliest = values["node_earliest"]
    duration = values["node_duration"]
    kind = values["node_kind"]
    parent_offsets = values["node_parent_offsets"]
    parent_indices = values["node_parent_indices"]
    parent_gaps = values["node_parent_gaps"]
    memory_offsets = values["node_memory_offsets"]
    memory_die = values["node_memory_die"]
    memory_value_offsets = values["node_memory_value_offsets"]
    memory_values = values["node_memory_values"]
    publication_offsets = values["node_publication_offsets"]
    publication_die = values["node_publication_die"]
    publication_request = values["node_publication_request"]
    nodes = []
    for index in range(len(earliest)):
        parents = [
            (int(parent_indices[offset]), int(parent_gaps[offset]))
            for offset in range(parent_offsets[index], parent_offsets[index + 1])
        ]
        memory = []
        for entry in range(memory_offsets[index], memory_offsets[index + 1]):
            begin, end = memory_value_offsets[entry : entry + 2]
            memory.append((
                int(memory_die[entry]),
                memory_values[begin:end].copy(),
            ))
        first, last = publication_offsets[index:index + 2]
        publications = Publications.from_columns(publication_die[first:last], publication_request[first:last])
        nodes.append({
            "earliest": int(earliest[index]),
            "duration": int(duration[index]),
            "kind": kinds[int(kind[index])],
            "parents": parents,
            "memory": memory,
            "publications": publications,
        })
    return nodes, metadata


def read_metadata(source: Path) -> dict:
    """Read provenance and frame mappings without expanding the scheduling graph."""
    with np.load(source, allow_pickle=False) as values:
        metadata = _MetadataReader(io.BytesIO(zlib.decompress(values['metadata'].tobytes()))).load()
        if metadata.get('schema') != SCHEMA:
            raise ValueError(f'unsupported PIM trace schema: {source}')
        if {name for name in values.files if name.startswith('request_')} != {f'request_{die}' for die in range(32)}:
            raise ValueError(f'PIM trace requires exactly 32 die request streams: {source}')
        return metadata


def read_trace(source: Path) -> dict:
    """Read a trace into the scheduler's bundle contract."""
    with np.load(source, allow_pickle=False) as values:
        nodes, metadata = _unpack_nodes(values)
        request_names = sorted((name for name in values.files if name.startswith("request_")),
                               key=lambda name: int(name.split("_")[-1]))
        requests = [values[name].copy() for name in request_names]
        result = dict(nodes=nodes, requests=requests)
        result.update(metadata)
        result["resource_reuse_edges"] = [tuple(row) for row in values["resource_reuse_edges"].tolist()]
        return result



def capture_gpu_graph(capture,requests,expansions,segments,fields,spans,
                   graph=None,origin_ns=None,hcf_lengths=None,extra_ctas=(),extra_resources=None,details=None,
                   captured_resource_gaps=False):
    """Connect physical supplies, recorded GPU service and finite SM reuse."""
    if graph is None or origin_ns is None:
        raise ValueError('captured GPU graph and timestamp origin are required')
    origin=int(origin_ns)
    def ck(ns):return (max(0,int(ns))*4+4)//5
    def at(ns):return ck(int(ns)-origin)
    windows={};span_windows=[]
    for supply in capture['segmented_supply'][:spans]:
        publication=int(supply['field_publication'][0])
        if publication not in windows:windows[publication]=len(windows)
        span_windows.append(windows[publication])
    roots={};window_sources=[[] for _ in windows];window_ends=[[] for _ in windows]
    culls={};ctas=list(extra_ctas)
    for record in expansions:
        span,task,die=record['span'],record['task'],record['die']
        raw=capture['spatial_expansion'][span];start,end=map(int,raw['cull_publication'][task])
        assert end>=start
        node=graph.add(earliest=at(start),duration=ck(end-start),kind='cull',
                       memory=[(die,np.asarray(record['nodes'],np.uint64))])
        culls[span,task]=node
        ctas.append(dict(kind='cull',sm=int(raw['cull_smids'][task]),start=start,end=end,first=node,last=node))
        window_ends[span_windows[span]].append(node)
    cull_tasks=[]
    for raw in capture['spatial_expansion'][:spans]:
        directory=raw['directory'].numpy()
        mapping={}
        for task,(die,first,*_) in enumerate(directory):
            mapping.setdefault((int(die),int(first)//256),[]).append(task)
        cull_tasks.append(mapping)
    for die,stream in enumerate(requests):
        if hcf_lengths is not None:stream=stream[:hcf_lengths[die]]
        for index in np.flatnonzero(stream['ready']):
            row=stream[index];assert row['kind']==7
            lane=int(row['tag'])>>24;release=int(row['ready'])
            if lane<2*spans:
                assert lane%2==0
                span=lane//2;kind='field_source'
            elif lane<6*spans:
                span=(lane-2*spans)//4;kind='route_source'
            else:
                assert lane<10*spans
                span=(lane-6*spans)//4;raw=capture['spatial_expansion'][span]
                first=(int(row['tag'])&0xffffff)*64
                tasks=cull_tasks[span].get((die,first//256),())
                assert len(tasks)==1
                task=int(tasks[0]);node=culls[span,task]
                assert at(raw['cull_publication'][task,1])<=release<=at(raw['cull_publication'][task,1])+1
                graph.publish(node,die,int(index));continue
            key=kind,span,release
            if kind=='field_source' and 'field_block_publication' in capture['segmented_supply'][span]:
                count=(capture['segmented_supply'][span]['N']+63)//64
                chunk=(int(row['tag'])&0xffffff)%count
                key=(*key,chunk)
            if key not in roots:
                roots[key]=graph.add(earliest=release,kind=kind)
                window_sources[span_windows[span]].append(roots[key])
            graph.publish(roots[key],die,int(index))
    field_nodes={}
    for record in fields:
        key=record['span'],record['frame'],record['chunk']
        assert key not in field_nodes
        field_nodes[key]=graph.add(kind='field_ready',memory=[(record['die'],np.asarray([record['done']],np.uint64))])
    segment_records={}
    for die,records in enumerate(segments):
        for record in records:segment_records[record['span'],record['segment']]=die,record
    for span,supply in enumerate(capture['segmented_supply'][:spans]):
        trace=supply['trace'].numpy();tasks=supply['task_trace'].numpy();starts=supply['starts'].numpy()
        assert trace.shape[1]==13,'capture must separate before-gate and after-gate timestamps'
        warps=capture['gpu_resources']['consumer_threads']//32
        for first in range(0,len(tasks),warps):
            selected=tasks[first:first+warps];sm=int(selected[0,2]);assert np.all(selected[:,2]==sm)
            begin=int(selected[:,0].min());end=int(selected[:,1].max());assert end>=begin
            head=graph.add(earliest=at(begin),kind='consumer_cta_start')
            tail=graph.add(kind='consumer_cta_end')
            for task in range(first,min(first+warps,len(tasks))):
                prior=head;cursor=begin;task_start,task_end=map(int,tasks[task,:2])
                assert begin<=task_start<=task_end<=end
                segment=int(starts[task]);seen=set()
                while segment>=0:
                    assert segment not in seen;seen.add(segment)
                    after,done=map(int,trace[segment,:2])
                    if not done:break
                    before=int(trace[segment,12]);assert cursor<=before<=after<=done<=task_end
                    prepare=graph.add(duration=ck(before-cursor),kind='consumer_loop')
                    graph.edge(prepare,prior)
                    die,record=segment_records[span,segment]
                    consume=graph.add(duration=ck(done-after),kind='compose',
                        memory=[(die,np.asarray(record.get('nodes',[record.get('done')]),np.uint64))])
                    graph.edge(consume,prepare)
                    for chunk in record['field_chunks']:
                        graph.edge(consume,field_nodes[span,record['frame'],int(chunk)])
                    prior=consume;cursor=done;segment=int(starts[len(tasks)+segment])
                store=graph.add(duration=ck(task_end-cursor),kind='consumer_store');graph.edge(store,prior)
                graph.edge(tail,store)
            ctas.append(dict(kind='consumer',sm=sm,start=begin,end=end,first=head,last=tail))
            window_ends[span_windows[span]].append(tail)
    resources=dict(capture['gpu_resources'])
    if extra_resources:resources.update(extra_resources)
    consumer=resources['consumer'];cull=resources['cull']
    assert consumer[4:]==cull[4:],'GPU resource properties must agree'
    capacity=(consumer[4]//consumer[8],consumer[5]//256,consumer[6]//256,consumer[7])
    dynamic_shared=max(0,capture['configuration'].get('consumer_reserve',0)-6144)
    def resource_demand(kind):
        resource=resources[kind];threads=resources[kind+'_threads']
        warps=threads//resource[8]
        registers_per_warp=(resource[0]*resource[8]+255)//256
        shared=resource[1]+(dynamic_shared if kind=='consumer' else 0)
        return warps,registers_per_warp*warps,(shared+255)//256,1
    peaks=[0,0,0,0];resource_edges=0;serialized_intervals=0;reuse_edges=[]
    sm_tasks=[[] for _ in range(consumer[9])]
    for task in ctas:
        if 0<=task['sm']<len(sm_tasks):sm_tasks[task['sm']].append(task)
    demands_by_kind={kind:resource_demand(kind)
                     for kind in {task['kind'] for tasks in sm_tasks for task in tasks}}
    for sm,tasks in enumerate(sm_tasks):
        tokens=[ResourceTokens(n) for n in capacity];active=[];occupied=[0]*4
        for task in sorted(tasks,key=lambda t:(t['start'],t['end'])):
            demands=demands_by_kind[task['kind']]
            assert all(d<=c for d,c in zip(demands,capacity))
            while active and (active[0][0]<=task['start'] or
                              any(capacity[kind]-occupied[kind]<demand for kind,demand in enumerate(demands))):
                _,_,old,allocations,old_demands=heapq.heappop(active)
                if old['end']>task['start']:serialized_intervals+=1
                for kind,indices in enumerate(allocations):
                    occupied[kind]-=old_demands[kind]
                    tokens[kind].release(indices,old['last'],old['end'])
            demands=demands_by_kind[task['kind']]
            allocations=[];parents={}
            for kind,demand in enumerate(demands):
                assert capacity[kind]-occupied[kind]>=demand,('captured SM occupancy exceeds hardware',sm,task,kind,occupied,demands)
                selected,previous=tokens[kind].allocate(demand);allocations.append(selected)
                for owner,owner_end in previous:
                    parents[owner]=max(parents.get(owner,0),ck(max(0,task['start']-owner_end)))
                occupied[kind]+=demand;peaks[kind]=max(peaks[kind],occupied[kind])
            for owner,gap in parents.items():
                # Tokens enforce availability. The captured idle period is
                # already bounded by each CTA's submission/earliest time.
                graph.edge(task['first'],owner,gap if captured_resource_gaps else 0)
                reuse_edges.append((task['first'],owner,gap));resource_edges+=1
            heapq.heappush(active,(task['end'],task['first'],task,allocations,demands))
    slots=int(capture['window_slots'])
    for window in range(slots,len(windows)):
        done=graph.add(kind='window_slot_release')
        for node in window_ends[window-slots]:graph.edge(done,node)
        for node in window_sources[window]:graph.edge(node,done)
    if details is not None:details.update(roots=roots,culls=culls,window_sources=window_sources,
        window_ends=window_ends,span_windows=span_windows,field_nodes=field_nodes,ctas=ctas,
        resource_reuse_edges=reuse_edges,resources=resources)
    return graph,dict(origin_ns=origin,sm_capacity=capacity,allocated_sm_peaks=peaks,
        sm_resource_edges=resource_edges,cull_ctas=len(culls),consumer_ctas=sum(t['kind']=='consumer' for t in ctas),
        windows=len(windows),sources=len(roots),conservatively_serialized_intervals=serialized_intervals,
        captured_resource_gaps=captured_resource_gaps)
