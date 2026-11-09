
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <math_constants.h>
using torch::Tensor;
__device__ unsigned long long timestamp() {
  unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;
}
__global__ void generate_4d_kernel(
    const float3 *xyz, const float4 *scales, const float4 *left,
    const float4 *right, const float *mu, const float *opacity,
    const bool *masks, const int64_t *source_ids, int source_count,
    const float *frames, int N, int W, float3 *means,float *covs,float *ops,const float* camera,int step,float range,int mode,
float* depth,float* score,float* minimum,unsigned long long* publication,unsigned long long* loaded) {

  int g=blockIdx.x*blockDim.x+threadIdx.x;
  if(!threadIdx.x) {
    publication[blockIdx.x*3]=timestamp();unsigned sm;
    asm volatile("mov.u32 %0, %%smid;":"=r"(sm));publication[blockIdx.x*3+2]=sm;
  }
  float4 l={},r={},scale={};float3 origin={};float center=0,amplitude=0;
  bool keep[5]={};
  __shared__ volatile unsigned input_hash[256];
  if(g<N) {
    l=left[g];r=right[g];scale=scales[g];origin=xyz[g];
    center=mu[g];amplitude=opacity[g];int64_t source=source_ids[g];
    unsigned hash=__float_as_uint(center)^__float_as_uint(amplitude);
    const float* words=(const float*)&l;
    for(int c=0;c<4;++c)hash^=__float_as_uint(words[c]);
    words=(const float*)&r;
    for(int c=0;c<4;++c)hash^=__float_as_uint(words[c]);
    words=(const float*)&scale;
    for(int c=0;c<4;++c)hash^=__float_as_uint(words[c]);
    hash^=__float_as_uint(origin.x)^__float_as_uint(origin.y)^__float_as_uint(origin.z);
    for(int f=0;f<W;++f) {
      int interval=int(frames[f])/20;
      keep[f]=masks[int64_t(interval)*source_count+source]||
              masks[int64_t(interval+1)*source_count+source];
      hash^=unsigned(keep[f])<<f;
    }
    input_hash[threadIdx.x]=hash;
  }
  __syncthreads();
  if(!threadIdx.x)loaded[blockIdx.x]=timestamp();
  if(g<N) {
  float L[4][4] = {{l.x, -l.y, l.z, -l.w}, {l.y, l.x, -l.w, -l.z},
                   {-l.z, l.w, l.x, -l.y}, {l.w, l.z, l.y, l.x}};
  float R[4][4] = {{r.x, -r.y, r.z, r.w}, {r.y, r.x, -r.w, r.z},
                   {-r.z, r.w, r.x, r.y}, {-r.w, -r.z, -r.y, r.x}};
  float s[4] = {scale.x, scale.y, scale.z, scale.w}, M[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float value = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) value += R[i][k]*L[k][j];
      M[i][j] = s[i]*value;
    }
  float sigma[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float value = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) value += M[k][i]*M[k][j];
      sigma[i][j] = value;
    }
  float variance = sigma[3][3];
  float velocity[3] = {sigma[0][3]/variance, sigma[1][3]/variance,
                       sigma[2][3]/variance};
  float covariance[6] = {
      sigma[0][0]-sigma[0][3]*sigma[0][3]/variance,
      sigma[0][1]-sigma[0][3]*sigma[1][3]/variance,
      sigma[0][2]-sigma[0][3]*sigma[2][3]/variance,
      sigma[1][1]-sigma[1][3]*sigma[1][3]/variance,
      sigma[1][2]-sigma[1][3]*sigma[2][3]/variance,
      sigma[2][2]-sigma[2][3]*sigma[2][3]/variance};
  for (int f = 0; f < W; ++f) {
    int index = f*N+g;
    float delta = frames[f]/30.0f-center;
    float marginal = __expf(-.5f*delta*delta/variance);
    bool selected = keep[f];
    means[index] = make_float3(origin.x+velocity[0]*delta,
        origin.y+velocity[1]*delta, origin.z+velocity[2]*delta);
#pragma unroll
    for (int k = 0; k < 6; ++k) covs[index*6+k] = covariance[k];
    ops[index] = selected && marginal > .05f ? amplitude*marginal : 0;
  }
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
std::vector<Tensor> generate_4d(Tensor xyz, Tensor scales, Tensor left,
    Tensor right, Tensor mu, Tensor opacity, Tensor masks, Tensor source_ids,
    int source_count,Tensor frames,Tensor view,float range,int mode) {
  c10::cuda::CUDAGuard guard(xyz.device());
  int N = xyz.size(0), W = frames.numel();
  auto means = torch::empty({W, N, 3}, xyz.options());
  auto covs = torch::empty({W, N, 6}, xyz.options());
  auto ops = torch::empty({W, N}, xyz.options());
  TORCH_CHECK(W>=1&&W<=5,"one to five frames required");
  auto depth=torch::empty({W,N},xyz.options()),score=torch::empty({N},xyz.options()),minimum=torch::empty_like(score);
  auto publication=torch::empty({(N+255)/256,3},xyz.options().dtype(torch::kInt64));
  auto loaded=torch::empty({(N+255)/256},xyz.options().dtype(torch::kInt64));
  if (N) generate_4d_kernel<<<(N+255)/256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (float3 *)xyz.data_ptr<float>(), (float4 *)scales.data_ptr<float>(),
      (float4 *)left.data_ptr<float>(), (float4 *)right.data_ptr<float>(),
      mu.data_ptr<float>(), opacity.data_ptr<float>(), masks.data_ptr<bool>(),
      source_ids.data_ptr<int64_t>(), source_count, frames.data_ptr<float>(),
      N, W, (float3 *)means.data_ptr<float>(), covs.data_ptr<float>(), ops.data_ptr<float>(),view.data_ptr<float>()+2*view.stride(1),view.stride(2),range,mode,
      depth.data_ptr<float>(),score.data_ptr<float>(),minimum.data_ptr<float>(),
      (unsigned long long*)publication.data_ptr<int64_t>(),(unsigned long long*)loaded.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {means,covs,ops,depth,score,minimum,publication,loaded};
}


std::vector<int64_t> resources() {
  cudaFuncAttributes a;cudaDeviceProp d;int id;
  C10_CUDA_CHECK(cudaGetDevice(&id));C10_CUDA_CHECK(cudaGetDeviceProperties(&d,id));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,generate_4d_kernel));
  return {a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),a.maxThreadsPerBlock,
    d.maxThreadsPerMultiProcessor,d.regsPerMultiprocessor,int64_t(d.sharedMemPerMultiprocessor),
    d.maxBlocksPerMultiProcessor,d.warpSize,d.multiProcessorCount};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("generate",&generate_4d);m.def("launch_resources",&resources);}
