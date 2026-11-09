"""LPDDR5 channel sessions coupled to GPU publication and completion events."""

from pathlib import Path
import ctypes as C
import hashlib
import heapq
import os
import shlex
import subprocess
from contextlib import ExitStack

import numpy as np

NONE=0xffffffff
DTYPE=np.dtype([('address','<u8'),('ready','<u8'),('kind','<u4'),
                ('depends','<u4'),('cost','<u4'),('tag','<u4')])


def prepare_backend(upstream, cache):
    """Build pinned Ramulator and the LPDDR5 controller in the external cache."""
    from integrations.manifest import RAMULATOR

    upstream, cache = Path(upstream).resolve(), Path(cache).resolve()
    source = Path(__file__).with_suffix('.cpp')
    config = Path(__file__).with_name('lpddr5.yaml')
    if not (upstream / 'CMakeLists.txt').is_file():
        raise FileNotFoundError(f'missing Ramulator source: {upstream}; run --prepare')
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != RAMULATOR.commit:
        raise ValueError(f'Ramulator revision must be {RAMULATOR.commit}')
    compiler = shlex.split(os.environ.get('CXX', 'g++'))
    version = subprocess.check_output([*compiler, '--version'], text=True).splitlines()[0]
    digest = hashlib.sha256(source.read_bytes() + revision.encode() + version.encode()).hexdigest()[:20]
    directory = cache / 'pim' / digest
    library = directory / 'gspim_memory.so'
    if library.is_file():
        return library, config
    directory.mkdir(parents=True, exist_ok=True)
    upstream_key = hashlib.sha256((revision + version).encode()).hexdigest()[:20]
    upstream_build = cache / 'pim' / 'ramulator' / upstream_key
    commands = []
    if not (upstream_build / 'libramulator.so').is_file():
        upstream_build.mkdir(parents=True, exist_ok=True)
        (upstream_build / 'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.14)\n'
            'project(GSPIMMemory LANGUAGES CXX)\n'
            'set(RAMULATOR_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)\n'
            f'add_subdirectory([[{upstream}]] ramulator)\n'
            f'set_target_properties(ramulator PROPERTIES LIBRARY_OUTPUT_DIRECTORY [[{upstream_build}]])\n',
            encoding='utf-8')
        commands.extend([
            ['cmake', '-S', str(upstream_build), '-B', str(upstream_build / 'build'),
             '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CXX_COMPILER={compiler[0]}'],
            ['cmake', '--build', str(upstream_build / 'build'), '--target', 'ramulator', '--parallel', '2'],
        ])
    commands.append(
        [*compiler, '-std=c++20', '-O3', '-fPIC', '-shared', '-pthread',
         '-I', str(upstream / 'src'), str(source), '-L', str(upstream_build), '-lramulator',
         '-Wl,-rpath,' + str(upstream_build), '-o', str(library.with_suffix('.so.part'))])
    log = directory / 'build.log'
    print('Building LPDDR5 controller', flush=True)
    try:
        with log.open('w') as stream:
            for command in commands:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'LPDDR5 build failed; see {log}') from error
    library.with_suffix('.so.part').replace(library)
    return library, config


class MemoryBackend:
    def __init__(self,library,config):
        self.config=Path(config)
        self.lib=C.CDLL(str(Path(library).resolve()))
        self.lib.pim_joint_error.restype=C.c_char_p


class Statistics(C.Structure):
    _fields_=[(x,C.c_uint64) for x in ('cycles','commands','refreshes','pauses','checks',
                                     'local_reads','local_writes','gpu_reads','gpu_writes')]+[('wall_ms',C.c_double)]



class Session:
    """Retain one channel's arbitration, bank timing and staging across GPU events."""
    def __init__(self,backend,requests,refresh=True,jump=True,hcf_slots=2,hcf_capacity=8192,ranks=2):
        self.lib=backend.lib
        self.lib.pim_session_open.argtypes=[C.c_char_p,C.c_void_p,C.c_uint64,C.c_void_p]+[C.c_int]*5
        self.lib.pim_session_open.restype=C.c_void_p
        self.lib.pim_session_advance.argtypes=[C.c_void_p,C.c_uint64,C.POINTER(Statistics),C.POINTER(C.c_uint64)]
        self.lib.pim_session_publish.argtypes=[C.c_void_p,C.c_uint64,C.c_uint64]
        self.lib.pim_session_close.argtypes=[C.c_void_p]
        self.inputs=np.ascontiguousarray(requests,dtype=DTYPE)
        self.done=np.full(len(self.inputs),np.iinfo(np.uint64).max,np.uint64)
        self.handle=self.lib.pim_session_open(str(backend.config).encode(),self.inputs.ctypes.data,
                                         len(self.inputs),self.done.ctypes.data,
                                         refresh,jump,hcf_slots,hcf_capacity,ranks)
        self.clock=0;self.reason=None;self.stats={}
        if not self.handle:raise RuntimeError(self.lib.pim_joint_error().decode())

    def advance(self,horizon=(1<<63)-1):
        if not self.handle:raise RuntimeError('session is closed')
        stats=Statistics();clock=C.c_uint64()
        reason=self.lib.pim_session_advance(self.handle,horizon,C.byref(stats),C.byref(clock))
        if reason<0:raise RuntimeError(self.lib.pim_joint_error().decode())
        self.clock=clock.value;self.reason=reason
        self.stats={key:getattr(stats,key) for key,_ in Statistics._fields_}
        return reason

    def publish(self,index,timestamp):
        if not self.handle:raise RuntimeError('session is closed')
        if self.lib.pim_session_publish(self.handle,index,timestamp):
            raise RuntimeError(self.lib.pim_joint_error().decode())

    def publish_many(self,indices,timestamp):
        if not self.handle:raise RuntimeError('session is closed')
        values=np.ascontiguousarray(indices,dtype=np.uint64)
        if not len(values):return
        if not hasattr(self.lib,'pim_session_publish_many'):
            for index in values:self.publish(int(index),timestamp)
            return
        self.lib.pim_session_publish_many.argtypes=[C.c_void_p,C.c_void_p,C.c_uint64,C.c_uint64]
        if self.lib.pim_session_publish_many(self.handle,values.ctypes.data,len(values),timestamp):
            raise RuntimeError(self.lib.pim_joint_error().decode())

    def watch(self,groups):
        if not self.handle:raise RuntimeError('session is closed')
        offsets=np.r_[np.uint64(0),np.cumsum([len(group) for group in groups],dtype=np.uint64)]
        indices=np.concatenate(groups).astype(np.uint64) if groups else np.empty(0,np.uint64)
        self.lib.pim_session_watch.argtypes=[C.c_void_p,C.c_void_p,C.c_uint64,C.c_void_p,C.c_uint64]
        if self.lib.pim_session_watch(self.handle,offsets.ctypes.data,len(groups),indices.ctypes.data,len(indices)):
            raise RuntimeError(self.lib.pim_joint_error().decode())

    def observe(self):
        if not self.handle:raise RuntimeError('session is closed')
        self.lib.pim_session_observe.argtypes=[C.c_void_p,C.c_void_p,C.c_uint64]
        self.lib.pim_session_observe.restype=C.c_int64
        count=self.lib.pim_session_observe(self.handle,None,0)
        if count<0:raise RuntimeError(self.lib.pim_joint_error().decode())
        output=np.empty((count,2),np.uint64)
        if count and self.lib.pim_session_observe(self.handle,output.ctypes.data,count)!=count:
            raise RuntimeError(self.lib.pim_joint_error().decode())
        return output

    def close(self):
        if self.handle:self.lib.pim_session_close(self.handle);self.handle=None

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


class ChannelWorkers:
    """Advance independent channels together using persistent native workers."""
    def __init__(self, sessions, workers):
        self.sessions = sessions
        self.lib = sessions[0].lib
        self.lib.pim_channels_open.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64]
        self.lib.pim_channels_open.restype = C.c_void_p
        self.lib.pim_channels_advance.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p, C.c_void_p]
        self.lib.pim_channels_close.argtypes = [C.c_void_p]
        self.stats = (Statistics * len(sessions))()
        self.clocks = (C.c_uint64 * len(sessions))()
        self.reasons = (C.c_int * len(sessions))()
        handles = (C.c_void_p * len(sessions))(*(session.handle for session in sessions))
        self.handle = self.lib.pim_channels_open(handles, len(sessions), workers)
        if not self.handle:
            raise RuntimeError(self.lib.pim_joint_error().decode())

    def advance(self, horizon):
        if self.lib.pim_channels_advance(self.handle, horizon, self.stats, self.clocks, self.reasons):
            raise RuntimeError(self.lib.pim_joint_error().decode())
        for channel, session in enumerate(self.sessions):
            session.clock, session.reason = self.clocks[channel], self.reasons[channel]
            session.stats = {key: getattr(self.stats[channel], key) for key, _ in Statistics._fields_}
            yield channel, session.observe()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.lib.pim_channels_close(self.handle)



def channel_trace(left,right):
    """Two independent die streams share one x16 command/data interface."""
    left=np.array(left,dtype=DTYPE,copy=True);right=np.array(right,dtype=DTYPE,copy=True)
    assert not np.any(left['address']>>np.uint64(63))
    assert not np.any(right['address']>>np.uint64(63))
    offset=len(left)
    right['address']|=np.uint64(1)<<np.uint64(63)
    linked=right['depends']!=NONE
    right['depends'][linked]+=offset
    right['cost'][(right['kind']==5)|(right['kind']==8)]+=offset
    return np.concatenate((left,right))


def publication_trace(requests):
    """Expose existing absolute source releases as dependency-linked external events."""
    source=np.asarray(requests,dtype=DTYPE)
    if not len(source):return source.copy(),np.empty(0,np.uint64),np.empty(0,np.uint64),np.empty(0,np.uint64)
    gated=source['ready']>0
    extra=np.cumsum(gated,dtype=np.uint64)
    positions=np.arange(len(source),dtype=np.uint64)+extra
    if len(source)+int(extra[-1])>=NONE:raise ValueError('publication trace exceeds request index space')
    output=np.zeros(len(source)+int(extra[-1]),dtype=DTYPE)
    output[positions]=source
    dependent=source['depends']!=NONE
    output['depends'][positions[dependent]]=positions[source['depends'][dependent]]
    fences=source['kind']==5
    output['cost'][positions[fences]]=positions[source['cost'][fences]]-gated[source['cost'][fences]]
    released=source['kind']==8
    output['cost'][positions[released]]=positions[source['cost'][released]]
    events=positions[gated]-1
    output[events]=output[positions[gated]]
    output['kind'][events]=10;output['ready'][events]=source['ready'][gated];output['cost'][events]=0
    output['depends'][positions[gated]]=events
    output['ready'][positions[gated]]=0
    return output,positions,events,source['ready'][gated]



class FeedbackGraph:
    """Measured GPU service and explicit memory dependencies, advanced conservatively."""
    def __init__(self,backend):
        self.nodes=[]
        self.backend=backend

    def add(self,earliest=0,duration=0,kind='gpu',memory=()):
        if earliest<0 or duration<0:raise ValueError('negative GPU time')
        index=len(self.nodes)
        self.nodes.append(dict(earliest=int(earliest),duration=int(duration),kind=kind,
                               parents=[],memory=list(memory),publications=[]))
        return index

    def edge(self,child,parent,gap=0):
        if child==parent or gap<0:raise ValueError('invalid GPU dependency')
        self.nodes[child]['parents'].append((parent,int(gap)))

    def publish(self,node,die,request):
        self.nodes[node]['publications'].append((die,request))

    def run(self,traces,quantum=800,workers=8,max_cycles=1000000000,jump=True,exact_gpu_events=False,resources=None):
        if quantum<1 or len(traces)%2:raise ValueError('positive quantum and paired dies required')
        prepared=[];positions=[];gates=[];offsets=[]
        for die in range(0,len(traces),2):
            joined=channel_trace(traces[die],traces[die+1]);offsets.append(len(traces[die]))
            inputs,mapping,events,_=publication_trace(joined)
            prepared.append(inputs);positions.append(mapping)
            gates.append(joined['ready']>0)
        count=len(self.nodes);children=[[] for _ in self.nodes]
        remaining=np.zeros(count,np.int32);release=np.empty(count,np.int64)
        groups=[[] for _ in prepared];group_nodes=[[] for _ in prepared]
        publications=[[] for _ in self.nodes]
        for node,task in enumerate(self.nodes):
            release[node]=task['earliest']
            for parent,gap in task['parents']:
                if parent<0 or parent>=count:raise ValueError('GPU parent out of range')
                children[parent].append((node,gap));remaining[node]+=1
            per_channel={}
            for die,indices in task['memory']:
                channel=die//2;index=np.asarray(indices,np.uint64)+(offsets[channel] if die%2 else 0)
                per_channel.setdefault(channel,[]).append(positions[channel][index])
            for channel,chunks in per_channel.items():
                groups[channel].append(np.unique(np.concatenate(chunks)))
                group_nodes[channel].append(node);remaining[node]+=1
            for die,index in task['publications']:
                channel=die//2;original=index+(offsets[channel] if die%2 else 0)
                if not gates[channel][original]:raise ValueError('source publication is missing or duplicated')
                gates[channel][original]=False
                publications[node].append((channel,int(positions[channel][original])-1))
        if any(np.any(gated) for gated in gates):raise ValueError('every external source needs a GPU publisher')
        ready=[(int(release[node]),node) for node in np.flatnonzero(remaining==0)];heapq.heapify(ready)
        events=[];times=np.full((count,2),-1,np.int64);finished=0;clock=0;steps=0
        ready_groups=[np.asarray(value,np.int64) for value in group_nodes]
        def rounded(value):return ((int(value)+quantum-1)//quantum)*quantum
        def complete_dependency(node,timestamp):
            release[node]=max(release[node],timestamp);remaining[node]-=1
            if remaining[node]<0:raise AssertionError('GPU dependency completed twice')
            if not remaining[node]:heapq.heappush(ready,(int(release[node]),int(node)))
        with ExitStack() as scope:
            sessions=[scope.enter_context(Session(self.backend,inputs,jump=jump)) for inputs in prepared]
            for channel,(session,watched) in enumerate(zip(sessions,groups)):
                session.watch(watched)
                for group,timestamp in session.observe():
                    node=int(ready_groups[channel][int(group)])
                    heapq.heappush(events,(rounded(timestamp),1,node,int(timestamp)))
            channels=scope.enter_context(ChannelWorkers(sessions, workers))
            while True:
                pending=[[] for _ in sessions]
                while True:
                    while events and events[0][0]<=clock:
                        visible,kind,index,timestamp=heapq.heappop(events)
                        if kind==0:
                            if resources is not None:resources.complete(index,timestamp)
                            for child,gap in children[index]:complete_dependency(child,timestamp+gap)
                            for channel,request in publications[index]:pending[channel].append(request)
                            finished+=1
                        else:complete_dependency(index,timestamp)
                    if not ready or ready[0][0]>clock:
                        admitted=[] if resources is None else resources.dispatch(clock)
                        for admitted_start,admitted_node in admitted:
                            heapq.heappush(ready,(admitted_start,admitted_node))
                        if not admitted:break
                    _,node=heapq.heappop(ready)
                    if times[node,0]>=0:raise AssertionError('GPU task started twice')
                    if resources is not None and resources.enqueue(node,int(release[node])):continue
                    # Internal GPU dependencies retain their actual completion
                    # times. Controller publications still wait for the current
                    # polling boundary, so no memory operation is backdated.
                    start=int(release[node]) if exact_gpu_events else clock
                    if resources is not None:start=max(start,resources.start(node))
                    end=start+self.nodes[node]['duration'];times[node]=start,end
                    heapq.heappush(events,(rounded(end),0,node,end))
                for session,indices in zip(sessions,pending):
                    if indices:session.publish_many(np.sort(np.asarray(indices,np.uint64)),clock)
                if finished==count:
                    for session in sessions:
                        assert session.advance()==0,'GPU finished with unresolved memory sources'
                    break
                next_clock=clock+quantum
                if not any(pending) and all(session.reason in (0,2) for session in sessions):
                    candidate=min(ready[0][0] if ready else max_cycles,
                                  events[0][0] if events else max_cycles)
                    if candidate==max_cycles:
                        raise RuntimeError('GPU/memory dependency deadlock')
                    next_clock=max(next_clock,rounded(candidate))
                if next_clock>max_cycles:raise RuntimeError('joint feedback cycle bound')
                for channel,notified in channels.advance(next_clock):
                    for group,timestamp in notified:
                        node=int(ready_groups[channel][int(group)])
                        heapq.heappush(events,(rounded(timestamp),1,node,int(timestamp)))
                clock=next_clock;steps+=1
            outputs=[]
            for channel,session in enumerate(sessions):
                done=session.done[positions[channel]].copy();left=offsets[channel]
                for first,last,die in ((0,left,channel*2),(left,len(done),channel*2+1)):
                    local=done[first:last];source=traces[die];dependent=source['depends']!=NONE
                    assert np.all(local[dependent]>=local[source['depends'][dependent]])
                    outputs.append((local,dict(session.stats,channel=channel)))
            metadata=dict(quantum_ck=quantum,exact_gpu_events=exact_gpu_events,steps=steps,tasks=count,
                gpu_cycles=int(times[:,1].max(initial=0)),
                total_cycles=max(clock,max(s.stats['cycles'] for s in sessions)),
                task_times=times)
            if resources is not None:metadata['resource_dispatch']=resources.verify(times)
            for node,task in enumerate(self.nodes):
                assert times[node,0]>=task['earliest']
                assert times[node,1]-times[node,0]==task['duration']
                for parent,gap in task['parents']:assert times[node,0]>=times[parent,1]+gap
                for die,indices in task['memory']:
                    if len(indices):assert times[node,0]>=int(outputs[die][0][indices].max())
                for die,request in task['publications']:
                    assert outputs[die][0][request]>=rounded(times[node,1])
            for channel,session in enumerate(sessions):
                for kind,key in ((0,'gpu_reads'),(1,'gpu_writes'),(2,'local_reads'),(3,'local_writes')):
                    assert session.stats[key]==np.count_nonzero(prepared[channel]['kind']==kind)
        return outputs,metadata
