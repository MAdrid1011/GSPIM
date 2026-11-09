
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
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
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("witness",&witness);m.def("refine",&refine);m.def("keys",&keys);}
