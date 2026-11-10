
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
namespace at::native {
inline thread_local torch::Tensor gspim_mean_ctas;
inline thread_local std::vector<int64_t> gspim_mean_resources;
inline thread_local int64_t gspim_mean_threads;
inline thread_local int gspim_mean_cached_device=-1;
inline thread_local int64_t gspim_mean_cached_shared=-1;
}
#include "Reduce.cuh"
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
using torch::Tensor;
std::vector<Tensor> run(Tensor score) {
  c10::cuda::CUDAGuard guard(score.device());
  TORCH_CHECK(score.is_contiguous()&&score.scalar_type()==torch::kFloat32&&score.numel()>0,"contiguous float score required");
  Tensor output=torch::empty({},score.options());
  auto iter=at::native::make_reduction("mean",output,score,at::OptionalIntArrayRef{},false,torch::kFloat32,torch::kFloat32);
  TORCH_CHECK(iter.can_use_32bit_indexing(),"32-bit native reduction required");
  float factor=static_cast<float>(iter.num_output_elements())/iter.numel();
  at::native::gspim_mean_ctas=Tensor();
  at::native::gspim_gpu_reduce_kernel<float,float>(iter,at::native::MeanOps<float,float,float,float>{factor});
  auto ctas=at::native::gspim_mean_ctas;at::native::gspim_mean_ctas=Tensor();
  return {output,ctas};
}
std::vector<int64_t> resources(){return at::native::gspim_mean_resources;}
int64_t threads(){return at::native::gspim_mean_threads;}
std::tuple<int64_t,Tensor> delta(Tensor score) {
  auto values=run(score);auto stream=at::cuda::getCurrentCUDAStream();float average;
  C10_CUDA_CHECK(cudaMemcpyAsync(&average,values[0].data_ptr<float>(),4,cudaMemcpyDeviceToHost,stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  return {average>=.98f?1:-1,values[1]};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("delta",&delta);m.def("resources",&resources);m.def("threads",&threads);}
