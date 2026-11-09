
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
using torch::Tensor;
__device__ void write_word(int* p,int value) {
  asm volatile("st.global.wt.u32 [%0], %1;"::"l"(p),"r"(value):"memory");
}
__global__ void rectangles_kernel(const float2* xy,const int2* radii,const int64_t* order,
    const int64_t* assignment,int* output,unsigned long long* publication,
    int N,int S,int W,int tw,int th,int ts) {
  int i=blockIdx.x*256+threadIdx.x;
  if(!threadIdx.x)asm volatile("mov.u64 %0, %%globaltimer;":"=l"(publication[blockIdx.x*2]));
  if(i<N) {
    int g=order[i],xmin=tw,xmax=0,ymin=th,ymax=0;
    for(int f=0;f<W;++f) {
      int j=f*S+g;float2 p=xy[j];int2 r=radii[j];
      if(r.x<=0||r.y<=0)continue;
      xmin=min(xmin,max(0,min(tw,int(floorf((p.x-r.x)/ts)))));
      xmax=max(xmax,max(0,min(tw,int(ceilf((p.x+r.x)/ts)))));
      ymin=min(ymin,max(0,min(th,int(floorf((p.y-r.y)/ts)))));
      ymax=max(ymax,max(0,min(th,int(ceilf((p.y+r.y)/ts)))));
    }
    int* p=output+i*8;
    write_word(p,g);write_word(p+1,assignment[i]);write_word(p+2,xmin);write_word(p+3,xmax);
    write_word(p+4,ymin);write_word(p+5,ymax);write_word(p+6,max(0,xmax-xmin)*max(0,ymax-ymin));
    write_word(p+7,0);
  }
  __syncthreads();
  if(!threadIdx.x) {
    __threadfence_system();
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(publication[blockIdx.x*2+1]));
  }
}
std::vector<Tensor> rectangles(Tensor xy,Tensor radii,Tensor order,Tensor assignment,
    int tw,int th,int ts) {
  c10::cuda::CUDAGuard guard(xy.device());int N=order.size(1),S=xy.size(1),W=xy.size(0);
  auto output=torch::empty({N,8},radii.options());
  auto publication=torch::empty({(N+255)/256,2},radii.options().dtype(torch::kInt64));
  if(N)rectangles_kernel<<<(N+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
    (float2*)xy.data_ptr<float>(),(int2*)radii.data_ptr<int>(),order.data_ptr<int64_t>(),
    assignment.data_ptr<int64_t>(),output.data_ptr<int>(),(unsigned long long*)publication.data_ptr<int64_t>(),
    N,S,W,tw,th,ts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,publication};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("rectangles",&rectangles);}
