
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <math_constants.h>
using torch::Tensor;
__device__ unsigned long long timestamp() {
  unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;
}
__global__ void generate_anchored_kernel(
    const float4 *anchors, const float4 *offsets, const float4 *log_scales,
    const float *decoded, const float *opacity, const bool *is_static,
    const float *frames, int N, int K, int W, float threshold,
    float3 *means, float *covs, float *ops,unsigned long long* publication) {
  if(!threadIdx.x) {
    publication[blockIdx.x*3]=timestamp();unsigned sm;
    asm volatile("mov.u32 %0, %%smid;":"=r"(sm));publication[blockIdx.x*3+2]=sm;
  }
  do {

  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= N) break;
  int anchor = g/K;
  const float *prediction = decoded+12*g;
  float norm_l = 0, norm_r = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    norm_l += prediction[4+i]*prediction[4+i];
    norm_r += prediction[8+i]*prediction[8+i];
  }
  norm_l = 1.f/fmaxf(sqrtf(norm_l), 1e-12f);
  norm_r = 1.f/fmaxf(sqrtf(norm_r), 1e-12f);
  float a = prediction[4]*norm_l, b = prediction[5]*norm_l;
  float c = prediction[6]*norm_l, d = prediction[7]*norm_l;
  float p = prediction[8]*norm_r, q = prediction[9]*norm_r;
  float r = prediction[10]*norm_r, s = prediction[11]*norm_r;
  float L[4][4] = {{a,-b,-c,-d}, {b,a,-d,c}, {c,d,a,-b}, {d,-c,b,a}};
  float R[4][4] = {{p,-q,-r,-s}, {q,p,s,-r}, {r,-s,p,q}, {s,r,-q,p}};
  float4 base = log_scales[g];
  float scales[4] = {expf(base.x)/(1+expf(-prediction[0])),
                     expf(base.y)/(1+expf(-prediction[1])),
                     expf(base.z)/(1+expf(-prediction[2])),
                     expf(base.w)/(1+expf(-prediction[3]))};
  float M[4][4], sigma[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float value = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) value += L[i][k]*R[k][j];
      M[i][j] = value*scales[j];
    }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float value = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) value += M[i][k]*M[j][k];
      sigma[i][j] = value;
    }
  float variance = fmaxf(sigma[3][3], 1e-12f);
  float vx = sigma[0][3]/variance, vy = sigma[1][3]/variance;
  float vz = sigma[2][3]/variance;
  float covariance[6] = {
      sigma[0][0]-sigma[0][3]*vx, sigma[0][1]-sigma[0][3]*vy,
      sigma[0][2]-sigma[0][3]*vz, sigma[1][1]-sigma[1][3]*vy,
      sigma[1][2]-sigma[1][3]*vz, sigma[2][2]-sigma[2][3]*vz};
  float4 origin = anchors[anchor], offset = offsets[g];
  origin.x += offset.x; origin.y += offset.y;
  origin.z += offset.z; origin.w += offset.w;
  bool frozen = is_static[anchor];
  for (int f = 0; f < W; ++f) {
    int index = f*N+g;
    float delta = frozen ? 0 : frames[f]/30.f-origin.w;
    float marginal = frozen ? 1.f : expf(-.5f*delta*delta/variance);
    means[index] = make_float3(origin.x+vx*delta, origin.y+vy*delta, origin.z+vz*delta);
#pragma unroll
    for (int k = 0; k < 6; ++k) covs[index*6+k] = covariance[k];
    ops[index] = marginal > threshold && opacity[g] > 0 ? opacity[g]*marginal : 0;
  }

  } while(false);
  __threadfence();__syncthreads();
  if(!threadIdx.x)publication[blockIdx.x*3+1]=timestamp();
}
std::vector<Tensor> generate_anchored(Tensor anchors, Tensor offsets,
    Tensor log_scales, Tensor decoded, Tensor opacity, Tensor is_static,
    Tensor frames, double threshold) {
  c10::cuda::CUDAGuard guard(anchors.device());
  int A = anchors.size(0), K = offsets.size(1), N = A*K, W = frames.numel();
  TORCH_CHECK(anchors.sizes() == at::IntArrayRef({A,4}) &&
      offsets.sizes() == at::IntArrayRef({A,K,4}) &&
      log_scales.sizes() == offsets.sizes() &&
      decoded.sizes() == at::IntArrayRef({A,K,12}) &&
      opacity.sizes() == at::IntArrayRef({A,K}) && is_static.numel() == A,
      "Anchored generation requires matching anchor/binding fields");
  auto means = torch::empty({W,N,3}, anchors.options());
  auto covs = torch::empty({W,N,6}, anchors.options());
  auto ops = torch::empty({W,N}, anchors.options());
  auto publication=torch::empty({(N+255)/256,3},anchors.options().dtype(torch::kInt64));
  if (N) generate_anchored_kernel<<<(N+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
      (float4 *)anchors.data_ptr<float>(), (float4 *)offsets.data_ptr<float>(),
      (float4 *)log_scales.data_ptr<float>(), decoded.data_ptr<float>(),
      opacity.data_ptr<float>(), is_static.data_ptr<bool>(), frames.data_ptr<float>(),
      N,K,W,float(threshold),(float3 *)means.data_ptr<float>(),
      covs.data_ptr<float>(),ops.data_ptr<float>(),(unsigned long long*)publication.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {means,covs,ops,publication};
}


std::vector<int64_t> resources() {
  cudaFuncAttributes a;cudaDeviceProp d;int id;
  C10_CUDA_CHECK(cudaGetDevice(&id));C10_CUDA_CHECK(cudaGetDeviceProperties(&d,id));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,generate_anchored_kernel));
  return {a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),a.maxThreadsPerBlock,
    d.maxThreadsPerMultiProcessor,d.regsPerMultiprocessor,int64_t(d.sharedMemPerMultiprocessor),
    d.maxBlocksPerMultiProcessor,d.warpSize,d.multiProcessorCount};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&generate_anchored);m.def("resources",&resources);}
