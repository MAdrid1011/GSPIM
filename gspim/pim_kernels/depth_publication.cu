
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
using torch::Tensor;
__device__ unsigned long long timestamp() {
  unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;
}
__global__ void fused_depth(const float3* means,const float* camera,int step,
    int N,int W,float range,int mode,float* depth,float* score,float* minimum,unsigned long long* publication) {
  if(!threadIdx.x) {
    publication[blockIdx.x*3]=timestamp();unsigned sm;
    asm volatile("mov.u32 %0, %%smid;":"=r"(sm));publication[blockIdx.x*3+2]=sm;
  }
  int g=blockIdx.x*blockDim.x+threadIdx.x;if(g<N) {
  float a=camera[0],b=camera[step],c=camera[2*step],t=camera[3*step];
  float previous=0,change=0,low=0;
  for(int f=0;f<W;++f) {
    float3 p=means[f*N+g];float x=p.x,y=p.y,z=p.z;
    int order=mode/4;
    if(order==1){float tmp=x;x=z;z=tmp;tmp=a;a=c;c=tmp;}
    if(order==2){float tmp=x;x=y;y=tmp;tmp=a;a=b;b=tmp;}
    if(order==3){float tmp=y;y=z;z=tmp;tmp=b;b=c;c=tmp;}
    if(order==4){float tmp=x;x=y;y=z;z=tmp;tmp=a;a=b;b=c;c=tmp;}
    if(order==5){float tmp=x;x=z;z=y;y=tmp;tmp=a;a=c;c=b;b=tmp;}
    float d;
    if(mode%4==0)d=__fmaf_rn(z,c,__fmaf_rn(y,b,__fmul_rn(x,a)));
    else if(mode%4==1)d=__fadd_rn(__fadd_rn(__fmul_rn(x,a),__fmul_rn(y,b)),__fmul_rn(z,c));
    else if(mode%4==2)d=__fadd_rn(__fmaf_rn(y,b,__fmul_rn(x,a)),__fmul_rn(z,c));
    else d=__fmaf_rn(z,c,__fadd_rn(__fmul_rn(x,a),__fmul_rn(y,b)));
    d=__fadd_rn(d,t);depth[f*N+g]=d;
    if(!f)low=d;
    else{change+=fabsf(d-previous);low=fminf(low,d);}
    previous=d;
    // Restore camera order for the next frame.
    a=camera[0];b=camera[step];c=camera[2*step];
  }
  minimum[g]=low;
  float mean=W>1?__fmul_rn(change,1.f/(W-1)):0.f;
  score[g]=expf(__fmul_rn(-mean,1.f/range));
  }
  __threadfence();__syncthreads();
  if(!threadIdx.x)publication[blockIdx.x*3+1]=timestamp();
}
std::vector<Tensor> run(Tensor means,Tensor view,float range,int mode) {
  c10::cuda::CUDAGuard guard(means.device());
  TORCH_CHECK(means.is_contiguous()&&means.size(2)==3&&means.size(0)>=1,"window means required");
  int N=means.size(1),W=means.size(0);
  auto depth=torch::empty({W,N},means.options()),score=torch::empty({N},means.options()),minimum=torch::empty_like(score);
  auto publication=torch::empty({(N+255)/256,3},means.options().dtype(torch::kInt64));
  if(N)fused_depth<<<(N+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
    (float3*)means.data_ptr<float>(),view.data_ptr<float>()+2*view.stride(1),view.stride(2),
    N,W,range,mode,depth.data_ptr<float>(),score.data_ptr<float>(),minimum.data_ptr<float>(),(unsigned long long*)publication.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {depth,score,minimum,publication};
}
__global__ void stamp_time(unsigned long long* out) {*out=timestamp();}
Tensor stamp(Tensor like) {
  auto out=torch::empty({1},like.options().dtype(torch::kInt64));
  stamp_time<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>((unsigned long long*)out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

std::vector<int64_t> resources() {
  cudaFuncAttributes a;cudaDeviceProp d;int id;
  C10_CUDA_CHECK(cudaGetDevice(&id));C10_CUDA_CHECK(cudaGetDeviceProperties(&d,id));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,fused_depth));
  return {a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),a.maxThreadsPerBlock,
    d.maxThreadsPerMultiProcessor,d.regsPerMultiprocessor,int64_t(d.sharedMemPerMultiprocessor),
    d.maxBlocksPerMultiProcessor,d.warpSize,d.multiProcessorCount};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("stamp",&stamp);m.def("resources",&resources);}
