
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
using torch::Tensor;
__global__ void fused_depth(const float3* means,const float* camera,int step,
    int N,int W,float range,int mode,float* depth,float* score,float* minimum) {
  int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=N)return;
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
std::vector<Tensor> run(Tensor means,Tensor view,float range,int mode) {
  c10::cuda::CUDAGuard guard(means.device());
  TORCH_CHECK(means.is_contiguous()&&means.size(2)==3&&means.size(0)>=1,"window means required");
  int N=means.size(1),W=means.size(0);
  auto depth=torch::empty({W,N},means.options()),score=torch::empty({N},means.options()),minimum=torch::empty_like(score);
  if(N)fused_depth<<<(N+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
    (float3*)means.data_ptr<float>(),view.data_ptr<float>()+2*view.stride(1),view.stride(2),
    N,W,range,mode,depth.data_ptr<float>(),score.data_ptr<float>(),minimum.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {depth,score,minimum};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
