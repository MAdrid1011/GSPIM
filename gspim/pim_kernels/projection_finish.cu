
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <math_constants.h>
using torch::Tensor;
__device__ unsigned long long timestamp() {
  unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;
}
__global__ void projection_finish_kernel(const float2 *xy, int2 *radii,
                                         const float *opacity,
                                         const float *compensation,
                                         float *out_opacity, int n, int width,
                                         int height,unsigned long long* publication) {
  if(!threadIdx.x) {
    publication[blockIdx.x*3]=timestamp();unsigned sm;
    asm volatile("mov.u32 %0, %%smid;":"=r"(sm));publication[blockIdx.x*3+2]=sm;
  }
  do {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    break;
  float2 p = xy[i];
  if (fabsf((p.x - width / 2.f) * 2 / width) > 1.3f ||
      fabsf((p.y - height / 2.f) * 2 / height) > 1.3f)
    radii[i] = make_int2(0, 0);
  out_opacity[i] = opacity[i] * compensation[i];

  } while(false);
  __threadfence();__syncthreads();
  if(!threadIdx.x)publication[blockIdx.x*3+1]=timestamp();
}
std::vector<Tensor> projection_finish(Tensor xy, Tensor radii, Tensor opacity,
                         Tensor compensation, int width, int height) {
  c10::cuda::CUDAGuard guard(xy.device());
  auto output = torch::empty_like(opacity);
  int n = opacity.numel();
  auto publication=torch::empty({(n+255)/256,3},xy.options().dtype(torch::kInt64));
  if (n)
    projection_finish_kernel<<<(n + 255) / 256, 256, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
        (float2 *)xy.data_ptr<float>(), (int2 *)radii.data_ptr<int>(),
        opacity.data_ptr<float>(), compensation.data_ptr<float>(),
        output.data_ptr<float>(), n, width, height,(unsigned long long*)publication.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output,publication};
}


std::vector<int64_t> resources() {
  cudaFuncAttributes a;cudaDeviceProp d;int id;
  C10_CUDA_CHECK(cudaGetDevice(&id));C10_CUDA_CHECK(cudaGetDeviceProperties(&d,id));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,projection_finish_kernel));
  return {a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),a.maxThreadsPerBlock,
    d.maxThreadsPerMultiProcessor,d.regsPerMultiprocessor,int64_t(d.sharedMemPerMultiprocessor),
    d.maxBlocksPerMultiProcessor,d.warpSize,d.multiProcessorCount};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&projection_finish);m.def("resources",&resources);}
