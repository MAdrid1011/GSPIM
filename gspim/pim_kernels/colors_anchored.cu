
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <math_constants.h>
using torch::Tensor;
__device__ unsigned long long timestamp() {
  unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;
}

__device__ void store_wt(unsigned* address,unsigned value) {
  asm volatile("st.global.wt.u32 [%0], %1;"::"l"(address),"r"(value):"memory");
}

__global__ void decoded_color_fields(const float* decoded,const int2* radii,
    const float2* xy,const float3* conics,const float* opacity,int N,int W,
    float* colors,unsigned* physical,float* logarithm,unsigned* visibility,int slots,
    unsigned long long* publication,unsigned long long* load_frontier) {
  __shared__ unsigned long long sh_clock[192];
  sh_clock[threadIdx.x]=0;
  if(!threadIdx.x) {
    publication[blockIdx.x*3]=timestamp();unsigned sm;
    asm volatile("mov.u32 %0, %%smid;":"=r"(sm));publication[blockIdx.x*3+2]=sm;
  }
  int g=blockIdx.x*64+threadIdx.x/3,channel=threadIdx.x%3;
  if(g<N) {
    float value=decoded[g*3+channel];
    asm volatile("" :: "f"(value) : "memory");
    sh_clock[threadIdx.x]=timestamp();
    for(int frame=0;frame<W;++frame)colors[(frame*N+g)*3+channel]=value;
  }
  __syncthreads();
  if(threadIdx.x<64) {
    int g=blockIdx.x*64+threadIdx.x;
    int block=g/256,owner=block%384,slot=(block/384)*256+g%256;
    for(int f=0;f<W;++f) {
      bool valid=g<N&&radii[f*N+g].x>0&&radii[f*N+g].y>0;
      unsigned mask=__ballot_sync(0xffffffff,valid);
      if(!(threadIdx.x%32))store_wt(visibility+(f*((N+255)/256)+block)*8+(g%256)/32,mask);
      if(blockIdx.x+1==gridDim.x&&!threadIdx.x)
        for(int word=(g%256)/32+2;word<8;++word)
          store_wt(visibility+(f*((N+255)/256)+block)*8+word,0);
      if(g<N) {
        int j=f*N+g;unsigned* base=physical+(owner*W+f)*slots*10;
        float2 p=xy[j];float3 a=conics[j],c=((const float3*)colors)[j];
        float op=opacity[j],cutoff=__log2f(op);logarithm[j]=cutoff;
        store_wt(base+slot*2,__float_as_uint(p.x));store_wt(base+slot*2+1,__float_as_uint(p.y));
        store_wt(base+slots*2+slot*3,__float_as_uint(a.x));
        store_wt(base+slots*2+slot*3+1,__float_as_uint(a.y));
        store_wt(base+slots*2+slot*3+2,__float_as_uint(a.z));
        store_wt(base+slots*5+slot*3,__float_as_uint(c.x));
        store_wt(base+slots*5+slot*3+1,__float_as_uint(c.y));
        store_wt(base+slots*5+slot*3+2,__float_as_uint(c.z));
        store_wt(base+slots*8+slot,__float_as_uint(op));
        store_wt(base+slots*9+slot,__float_as_uint(cutoff));
      }
    }
    __threadfence_system();
  }
  __syncthreads();
  if(!threadIdx.x) {
    unsigned long long latest=publication[blockIdx.x*3];
    for(int t=0;t<192;++t)latest=max(latest,sh_clock[t]);
    load_frontier[blockIdx.x]=latest;
    publication[blockIdx.x*3+1]=timestamp();
  }
}

std::vector<Tensor> run(Tensor decoded,Tensor radii,Tensor xy,Tensor conics,Tensor opacity) {
  c10::cuda::CUDAGuard guard(decoded.device());int N=opacity.size(1),W=opacity.size(0);
  int slots=((N+98303)/98304)*256;
  TORCH_CHECK(decoded.is_contiguous()&&decoded.numel()==N*3,"contiguous decoded RGB required");
  auto colors=torch::empty({W,N,3},decoded.options()),logarithm=torch::empty_like(opacity);
  auto physical=torch::empty({384,W,slots*10},radii.options());
  auto visibility=torch::empty({W,(N+255)/256,8},radii.options());
  auto publication=torch::empty({(N+63)/64,3},decoded.options().dtype(torch::kInt64));
  auto load_frontier=torch::empty({(N+63)/64},decoded.options().dtype(torch::kInt64));
  if(N)decoded_color_fields<<<(N+63)/64,192,0,at::cuda::getCurrentCUDAStream()>>>(
    decoded.data_ptr<float>(),(int2*)radii.data_ptr<int>(),(float2*)xy.data_ptr<float>(),
    (float3*)conics.data_ptr<float>(),opacity.data_ptr<float>(),N,W,colors.data_ptr<float>(),
    (unsigned*)physical.data_ptr<int>(),logarithm.data_ptr<float>(),(unsigned*)visibility.data_ptr<int>(),slots,
    (unsigned long long*)publication.data_ptr<int64_t>(),(unsigned long long*)load_frontier.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {colors,logarithm,physical,visibility,publication,load_frontier};
}

std::vector<int64_t> resources() {
  cudaFuncAttributes a;cudaDeviceProp d;int id;
  C10_CUDA_CHECK(cudaGetDevice(&id));C10_CUDA_CHECK(cudaGetDeviceProperties(&d,id));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,decoded_color_fields));
  return {a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),a.maxThreadsPerBlock,
    d.maxThreadsPerMultiProcessor,d.regsPerMultiprocessor,int64_t(d.sharedMemPerMultiprocessor),
    d.maxBlocksPerMultiProcessor,d.warpSize,d.multiProcessorCount};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("resources",&resources);}
