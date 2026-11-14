
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstring>
#include <array>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
using torch::Tensor;
__device__ unsigned ordered(float x) {
  unsigned bits=__float_as_uint(x);
  return bits&0x80000000u?~bits:bits^0x80000000u;
}
__global__ void crossing_witness(const float* depth,const int2* radii,int N,int W,int* found) {
  int g=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x%32;
  for(int shift=0;shift<=10;shift+=2) {
    int other=g^(1<<shift);bool cross=false;
    if(g<N&&other<N) {
      float a0=depth[g],b0=depth[other];
      for(int f=1;f<W;++f) {
        float a=depth[f*N+g],b=depth[f*N+other];
        cross|=(a0<b0&&b<a)||(b0<a0&&a<b);
      }
      if(cross) {
        bool av=false,bv=false;
        for(int f=0;f<W;++f) {
          int2 a=radii[f*N+g],b=radii[f*N+other];
          av|=a.x>0&&a.y>0;bv|=b.x>0&&b.y>0;
        }
        cross=av&&bv;
      }
    }
    if(__ballot_sync(0xffffffff,cross)) {
      if(!lane)atomicExch(found,1);
      return;
    }
  }
}
bool witness(Tensor depth,Tensor radii) {
  c10::cuda::CUDAGuard guard(depth.device());int N=depth.size(1),W=depth.size(0);
  auto found=torch::zeros({1},radii.options());auto stream=at::cuda::getCurrentCUDAStream();
  if(N)crossing_witness<<<(N+255)/256,256,0,stream>>>(depth.data_ptr<float>(),(int2*)radii.data_ptr<int>(),N,W,found.data_ptr<int>());
  int result=0;
  C10_CUDA_CHECK(cudaMemcpyAsync(&result,found.data_ptr<int>(),4,cudaMemcpyDeviceToHost,stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));return result;
}
Tensor witness_async(Tensor depth,Tensor radii) {
  c10::cuda::CUDAGuard guard(depth.device());int N=depth.size(1),W=depth.size(0);
  auto found=torch::zeros({1},radii.options());auto stream=at::cuda::getCurrentCUDAStream();
  if(N)crossing_witness<<<(N+255)/256,256,0,stream>>>(depth.data_ptr<float>(),(int2*)radii.data_ptr<int>(),N,W,found.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return found;
}

__global__ void refine_ties(const float* key,const int64_t* ids,const float* minimum,int64_t* output,int N,int* overflow) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
  unsigned value=__float_as_uint(key[i]);int lo=i,hi=i+1;
  while(lo>0&&__float_as_uint(key[lo-1])==value&&i-lo<32)--lo;
  while(hi<N&&__float_as_uint(key[hi])==value&&hi-i<=32)++hi;
  if(hi-lo>32||(lo>0&&__float_as_uint(key[lo-1])==value)||(hi<N&&__float_as_uint(key[hi])==value)) {
    atomicExch(overflow,1);output[i]=ids[i];return;
  }
  if(hi-lo==1){output[i]=ids[i];return;}
  int64_t id=ids[i];unsigned low=ordered(minimum[id]);int rank=0;
  for(int j=lo;j<hi;++j) {
    int64_t other=ids[j];unsigned before=ordered(minimum[other]);
    rank+=before<low||(before==low&&other<id);
  }
  output[lo+rank]=id;
}
std::vector<Tensor> refine(Tensor key,Tensor ids,Tensor minimum) {
  c10::cuda::CUDAGuard guard(key.device());int N=ids.numel();auto output=torch::empty_like(ids);
  auto overflow=torch::zeros({1},ids.options().dtype(torch::kInt32));
  if(N)refine_ties<<<(N+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(key.data_ptr<float>(),ids.data_ptr<int64_t>(),minimum.data_ptr<float>(),output.data_ptr<int64_t>(),N,overflow.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,overflow};
}
__global__ void make_keys(const float* mean,const int64_t* ids,const float* minimum,int64_t* keys,int N) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)
    keys[i]=((uint64_t(ordered(mean[i]))<<32)|ordered(minimum[ids[i]]))^(1ull<<63);
}
Tensor keys(Tensor mean,Tensor ids,Tensor minimum) {
  c10::cuda::CUDAGuard guard(mean.device());int N=ids.numel();auto output=torch::empty_like(ids);
  if(N)make_keys<<<(N+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(mean.data_ptr<float>(),ids.data_ptr<int64_t>(),minimum.data_ptr<float>(),output.data_ptr<int64_t>(),N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

#if CUDART_VERSION >= 13000
__global__ void set_condition(const int* overflow,cudaGraphConditionalHandle handle) {
  cudaGraphSetConditional(handle,unsigned(*overflow));
}

struct BoundKey {
 const uint64_t* bindings;
 __device__ float operator()(int i) const {return reinterpret_cast<const float*>(bindings[0])[i];}
};
struct BoundID {
 const uint64_t* bindings;
 __device__ int64_t operator()(int i) const {return reinterpret_cast<const int64_t*>(bindings[1])[i];}
};
__global__ void bind_inputs(uint64_t* bindings,uint64_t key,uint64_t ids,uint64_t minimum) {
 bindings[0]=key;bindings[1]=ids;bindings[2]=minimum;
}
__global__ void refine_bound(const float* key,const int64_t* ids,const uint64_t* bindings,int64_t* output,int N,int* overflow) {
  const float* minimum=reinterpret_cast<const float*>(bindings[2]);
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
  unsigned value=__float_as_uint(key[i]);int lo=i,hi=i+1;
  while(lo>0&&__float_as_uint(key[lo-1])==value&&i-lo<32)--lo;
  while(hi<N&&__float_as_uint(key[hi])==value&&hi-i<=32)++hi;
  if(hi-lo>32||(lo>0&&__float_as_uint(key[lo-1])==value)||(hi<N&&__float_as_uint(key[hi])==value)) {
    atomicExch(overflow,1);output[i]=ids[i];return;
  }
  if(hi-lo==1){output[i]=ids[i];return;}
  int64_t id=ids[i];unsigned low=ordered(minimum[id]);int rank=0;
  for(int j=lo;j<hi;++j) {
    int64_t other=ids[j];unsigned before=ordered(minimum[other]);
    rank+=before<low||(before==low&&other<id);
  }
  output[lo+rank]=id;
}
__global__ void keys_bound(const float* mean,const int64_t* ids,const uint64_t* bindings,int64_t* keys,int N) {
  const float* minimum=reinterpret_cast<const float*>(bindings[2]);
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)
    keys[i]=((uint64_t(ordered(mean[i]))<<32)|ordered(minimum[ids[i]]))^(1ull<<63);
}

class ConditionalSort {
  Tensor mean_in,ids_in,minimum_in,bindings,sorted_mean,sorted_ids,output,overflow,compound,unused_keys,scratch;
  cudaGraph_t graph=nullptr;
  cudaGraphExec_t executable=nullptr;
  int N;
  size_t bytes;
  struct Binding {
    cudaGraphNode_t node;
    cudaKernelNodeParams params;
    std::vector<std::vector<unsigned char>> values;
    std::vector<void*> arguments;
    std::vector<std::pair<int,int>> changed;
  };
  std::vector<Binding> rebound;
public:
  ConditionalSort(Tensor key,Tensor ids,Tensor minimum): N(ids.numel()) {
    c10::cuda::CUDAGuard guard(key.device());
    TORCH_CHECK(N>0 && N==key.numel() && key.is_contiguous() && ids.is_contiguous() && minimum.is_contiguous());
    mean_in=key;ids_in=ids;minimum_in=minimum;
    bindings=torch::empty({3},ids.options());
    sorted_mean=torch::empty_like(key);sorted_ids=torch::empty_like(ids);output=torch::empty_like(ids);
    overflow=torch::empty({1},ids.options().dtype(torch::kInt32));compound=torch::empty_like(ids);unused_keys=torch::empty_like(ids);
    size_t first=0,second=0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,first,mean_in.data_ptr<float>(),sorted_mean.data_ptr<float>(),ids_in.data_ptr<int64_t>(),sorted_ids.data_ptr<int64_t>(),N,0,32));
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,second,compound.data_ptr<int64_t>(),unused_keys.data_ptr<int64_t>(),sorted_ids.data_ptr<int64_t>(),output.data_ptr<int64_t>(),N,0,64));
    bytes=std::max(first,second);scratch=torch::empty({int64_t(bytes)},key.options().dtype(torch::kUInt8));
    cudaStream_t stream;C10_CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    C10_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed));
    cudaStreamCaptureStatus status;
    C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream,&status,nullptr,&graph));
    cudaGraphConditionalHandle handle;
    C10_CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(scratch.data_ptr(),bytes,mean_in.data_ptr<float>(),sorted_mean.data_ptr<float>(),ids_in.data_ptr<int64_t>(),sorted_ids.data_ptr<int64_t>(),N,0,32,stream));
    C10_CUDA_CHECK(cudaMemsetAsync(overflow.data_ptr<int>(),0,sizeof(int),stream));
    refine_bound<<<(N+255)/256,256,0,stream>>>(sorted_mean.data_ptr<float>(),sorted_ids.data_ptr<int64_t>(),(uint64_t*)bindings.data_ptr<int64_t>(),output.data_ptr<int64_t>(),N,overflow.data_ptr<int>());
    set_condition<<<1,1,0,stream>>>(overflow.data_ptr<int>(),handle);
    const cudaGraphNode_t* dependencies;size_t count;
    C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream,&status,nullptr,nullptr,&dependencies,nullptr,&count));
    cudaGraphNodeParams params{};params.type=cudaGraphNodeTypeConditional;
    params.conditional.handle=handle;params.conditional.type=cudaGraphCondTypeIf;params.conditional.size=1;
    cudaGraphNode_t conditional;
    C10_CUDA_CHECK(cudaGraphAddNode(&conditional,graph,dependencies,nullptr,count,&params));
    C10_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(stream,&conditional,nullptr,1,cudaStreamSetCaptureDependencies));
    C10_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
    C10_CUDA_CHECK(cudaStreamBeginCaptureToGraph(stream,params.conditional.phGraph_out[0],nullptr,nullptr,0,cudaStreamCaptureModeRelaxed));
    keys_bound<<<(N+255)/256,256,0,stream>>>(sorted_mean.data_ptr<float>(),sorted_ids.data_ptr<int64_t>(),(uint64_t*)bindings.data_ptr<int64_t>(),compound.data_ptr<int64_t>(),N);
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(scratch.data_ptr(),bytes,compound.data_ptr<int64_t>(),unused_keys.data_ptr<int64_t>(),sorted_ids.data_ptr<int64_t>(),output.data_ptr<int64_t>(),N,0,64,stream));
    cudaGraph_t body;C10_CUDA_CHECK(cudaStreamEndCapture(stream,&body));
    size_t number=0;C10_CUDA_CHECK(cudaGraphGetNodes(graph,nullptr,&number));
    std::vector<cudaGraphNode_t> nodes(number);C10_CUDA_CHECK(cudaGraphGetNodes(graph,nodes.data(),&number));
    for(auto node:nodes) {
      cudaGraphNodeType type;C10_CUDA_CHECK(cudaGraphNodeGetType(node,&type));
      if(type!=cudaGraphNodeTypeKernel)continue;
      Binding bound;bound.node=node;C10_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&bound.params));
      TORCH_CHECK(bound.params.kernelParams!=nullptr,"expected ordinary kernel arguments");
      size_t count=0;C10_CUDA_CHECK(cudaFuncGetParamCount(bound.params.func,&count));
      for(size_t i=0;i<count;++i) {
        size_t offset,size;C10_CUDA_CHECK(cudaFuncGetParamInfo(bound.params.func,i,&offset,&size));
        bound.values.emplace_back(size);
        std::memcpy(bound.values.back().data(),bound.params.kernelParams[i],size);
        if(size==sizeof(void*)) {
          void* value;std::memcpy(&value,bound.values.back().data(),sizeof(value));
          if(value==key.data_ptr())bound.changed.emplace_back(i,0);
          if(value==ids.data_ptr())bound.changed.emplace_back(i,1);
        }
      }
      if(!bound.changed.empty())rebound.push_back(std::move(bound));
    }
    for(auto& bound:rebound) {
      for(auto& value:bound.values)bound.arguments.push_back(value.data());
      bound.params.kernelParams=bound.arguments.data();
    }
    C10_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
    C10_CUDA_CHECK(cudaStreamDestroy(stream));
  }
  Tensor run(Tensor key,Tensor ids,Tensor minimum) {
    c10::cuda::CUDAGuard guard(key.device());
    TORCH_CHECK(key.sizes()==mean_in.sizes() && ids.sizes()==ids_in.sizes() && minimum.sizes()==minimum_in.sizes());
    bind_inputs<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>((uint64_t*)bindings.data_ptr<int64_t>(),reinterpret_cast<uint64_t>(key.data_ptr<float>()),reinterpret_cast<uint64_t>(ids.data_ptr<int64_t>()),reinterpret_cast<uint64_t>(minimum.data_ptr<float>()));
    for(auto& bound:rebound) {
      for(auto [index,which]:bound.changed) {
        void* value=which==0?key.data_ptr():ids.data_ptr();
        std::memcpy(bound.values[index].data(),&value,sizeof(value));
      }
      C10_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(executable,bound.node,&bound.params));
    }
    C10_CUDA_CHECK(cudaGraphLaunch(executable,at::cuda::getCurrentCUDAStream()));
    return output;
  }
  ~ConditionalSort() {
    if(executable)cudaGraphExecDestroy(executable);
    if(graph)cudaGraphDestroy(graph);
  }
};

__global__ void compact_flags(const uint64_t* b, int* flags, int N, int M, int W) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
 const int64_t* ids=reinterpret_cast<const int64_t*>(b[1]);
 const int2* radii=reinterpret_cast<const int2*>(b[3]);int g=ids[i];bool visible=false;
 for(int f=0;f<W;++f){int2 r=radii[f*M+g];visible|=r.x>0&&r.y>0;}
 flags[i]=visible;
}
__global__ void compact_mean(const uint64_t* b,const int64_t* ids,float* key,int N,int M,int W) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
 const float* depth=reinterpret_cast<const float*>(b[0]);int g=ids[i];float value=0;
 for(int f=0;f<W;++f)value+=depth[f*M+g];key[i]=__fmul_rn(value,1.f/W);
}
__global__ void bind_visible(uint64_t* b,uint64_t depth,uint64_t ids,uint64_t minimum,uint64_t radii){b[0]=depth;b[1]=ids;b[2]=minimum;b[3]=radii;}
class VisibleSort {
 Tensor depth_shape,ids_shape,minimum_shape,radii_shape,binding,flags,selected,visible,key,scratch;
 cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;int N,M,W,valid;size_t bytes=0;
 std::unique_ptr<ConditionalSort> sorter;
 struct Bound {cudaGraphNode_t node;cudaKernelNodeParams params;std::vector<std::vector<unsigned char>> values;std::vector<void*> arguments;std::vector<int> changed;};
 std::vector<Bound> rebound;
public:
 VisibleSort(Tensor depth,Tensor ids,Tensor minimum,Tensor radii):N(ids.numel()),M(depth.size(1)),W(depth.size(0)) {
  c10::cuda::CUDAGuard guard(depth.device());TORCH_CHECK(N>0&&(W==2||W==3)&&depth.is_contiguous()&&ids.is_contiguous()&&minimum.is_contiguous()&&radii.is_contiguous());
  depth_shape=depth;ids_shape=ids;minimum_shape=minimum;radii_shape=radii;
  binding=torch::empty({4},ids.options());flags=torch::empty({N},ids.options().dtype(torch::kInt32));visible=torch::empty({1},flags.options());selected=torch::empty_like(ids);
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr,bytes,ids.data_ptr<int64_t>(),flags.data_ptr<int>(),selected.data_ptr<int64_t>(),visible.data_ptr<int>(),N));
  scratch=torch::empty({int64_t(bytes)},ids.options().dtype(torch::kUInt8));
  auto current=at::cuda::getCurrentCUDAStream();
  bind_visible<<<1,1,0,current>>>((uint64_t*)binding.data_ptr<int64_t>(),reinterpret_cast<uint64_t>(depth.data_ptr<float>()),reinterpret_cast<uint64_t>(ids.data_ptr<int64_t>()),reinterpret_cast<uint64_t>(minimum.data_ptr<float>()),reinterpret_cast<uint64_t>(radii.data_ptr<int>()));
  compact_flags<<<(N+255)/256,256,0,current>>>((uint64_t*)binding.data_ptr<int64_t>(),flags.data_ptr<int>(),N,M,W);
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(scratch.data_ptr(),bytes,ids.data_ptr<int64_t>(),flags.data_ptr<int>(),selected.data_ptr<int64_t>(),visible.data_ptr<int>(),N,current));
  // Preparation specializes only the fixed-input recorder output shape.
  // Timed launches still compute visibility and compact the actual IDs.
  C10_CUDA_CHECK(cudaMemcpyAsync(&valid,visible.data_ptr<int>(),4,cudaMemcpyDeviceToHost,current));C10_CUDA_CHECK(cudaStreamSynchronize(current));
  key=torch::empty({valid},depth.options());
  if(valid)sorter=std::make_unique<ConditionalSort>(key,selected.narrow(0,0,valid),minimum);
  cudaStream_t stream;C10_CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));C10_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed));
  compact_flags<<<(N+255)/256,256,0,stream>>>((uint64_t*)binding.data_ptr<int64_t>(),flags.data_ptr<int>(),N,M,W);
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(scratch.data_ptr(),bytes,ids.data_ptr<int64_t>(),flags.data_ptr<int>(),selected.data_ptr<int64_t>(),visible.data_ptr<int>(),N,stream));
  if(valid)compact_mean<<<(valid+255)/256,256,0,stream>>>((uint64_t*)binding.data_ptr<int64_t>(),selected.data_ptr<int64_t>(),key.data_ptr<float>(),valid,M,W);
  C10_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
  size_t number=0;C10_CUDA_CHECK(cudaGraphGetNodes(graph,nullptr,&number));std::vector<cudaGraphNode_t> nodes(number);C10_CUDA_CHECK(cudaGraphGetNodes(graph,nodes.data(),&number));
  for(auto node:nodes){cudaGraphNodeType type;C10_CUDA_CHECK(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;Bound b;b.node=node;C10_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&b.params));size_t n=0;C10_CUDA_CHECK(cudaFuncGetParamCount(b.params.func,&n));
   for(size_t i=0;i<n;++i){size_t offset,size;C10_CUDA_CHECK(cudaFuncGetParamInfo(b.params.func,i,&offset,&size));b.values.emplace_back(size);std::memcpy(b.values.back().data(),b.params.kernelParams[i],size);if(size==sizeof(void*)){void* value;std::memcpy(&value,b.values.back().data(),sizeof(value));if(value==ids.data_ptr())b.changed.push_back(i);}}
   if(!b.changed.empty())rebound.push_back(std::move(b));}
  for(auto& b:rebound){for(auto& value:b.values)b.arguments.push_back(value.data());b.params.kernelParams=b.arguments.data();}
  C10_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));C10_CUDA_CHECK(cudaStreamDestroy(stream));
 }
 Tensor run(Tensor depth,Tensor ids,Tensor minimum,Tensor radii){
  c10::cuda::CUDAGuard guard(depth.device());TORCH_CHECK(depth.sizes()==depth_shape.sizes()&&ids.sizes()==ids_shape.sizes()&&minimum.sizes()==minimum_shape.sizes()&&radii.sizes()==radii_shape.sizes());
  bind_visible<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>((uint64_t*)binding.data_ptr<int64_t>(),reinterpret_cast<uint64_t>(depth.data_ptr<float>()),reinterpret_cast<uint64_t>(ids.data_ptr<int64_t>()),reinterpret_cast<uint64_t>(minimum.data_ptr<float>()),reinterpret_cast<uint64_t>(radii.data_ptr<int>()));
  for(auto& b:rebound){for(int i:b.changed){void* value=ids.data_ptr();std::memcpy(b.values[i].data(),&value,sizeof(value));}C10_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(executable,b.node,&b.params));}
  C10_CUDA_CHECK(cudaGraphLaunch(executable,at::cuda::getCurrentCUDAStream()));
  return valid?sorter->run(key,selected.narrow(0,0,valid),minimum):selected.narrow(0,0,0);
 }
 Tensor count(){return visible;}
 ~VisibleSort(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
};
#endif
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("witness",&witness);m.def("witness_async",&witness_async);m.def("refine",&refine);m.def("keys",&keys);
#if CUDART_VERSION >= 13000
 pybind11::class_<VisibleSort>(m,"VisibleSort").def(pybind11::init<Tensor,Tensor,Tensor,Tensor>()).def("run",&VisibleSort::run).def("count",&VisibleSort::count);
#endif
}
