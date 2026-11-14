#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
using torch::Tensor;
__global__ void flags_kernel(const float* intervals,const float* times,bool* flags,int N,int W) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 // Tensor scalar division uses a rounded FP32 reciprocal multiplication.
 if(i<N)flags[i]=intervals[2*i]<=__fmul_rn(times[W-1],1.f/30.f)&&intervals[2*i+1]>=__fmul_rn(times[0],1.f/30.f);
}
__global__ void active_kernel(const float* opacity,bool* flags,int N,int W) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<N){bool active=false;for(int f=0;f<W;++f)active|=opacity[f*N+i]>=1.f/255.f;flags[i]=active;}
}
Tensor selected(Tensor flags) {
 int N=flags.numel();auto ids=torch::empty({N},flags.options().dtype(torch::kInt64));
 auto count=torch::empty({},flags.options().dtype(torch::kInt32));
 if(!N)return ids;
 auto stream=at::cuda::getCurrentCUDAStream();size_t bytes=0;
 thrust::counting_iterator<int64_t> input(0);
 C10_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr,bytes,input,flags.data_ptr<bool>(),ids.data_ptr<int64_t>(),count.data_ptr<int>(),N,stream));
 auto scratch=torch::empty({int64_t(bytes)},flags.options().dtype(torch::kUInt8));
 C10_CUDA_CHECK(cub::DeviceSelect::Flagged(scratch.data_ptr(),bytes,input,flags.data_ptr<bool>(),ids.data_ptr<int64_t>(),count.data_ptr<int>(),N,stream));
 int size;C10_CUDA_CHECK(cudaMemcpyAsync(&size,count.data_ptr<int>(),sizeof(size),cudaMemcpyDeviceToHost,stream));
 C10_CUDA_CHECK(cudaStreamSynchronize(stream));return ids.narrow(0,0,size);
}
__global__ void gather_selection(const int64_t* ids,int M,
 const float* anchors,const float* offsets,const float* features,const float* scales,const bool* is_static,
 float* out_anchors,float* out_offsets,float* out_features,float* out_scales,bool* out_static) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<M*40){int j=i/40,c=i%40,source=ids[j];out_offsets[i]=offsets[source*40+c];out_scales[i]=scales[source*40+c];}
 if(i<M*32){int j=i/32,c=i%32;out_features[i]=features[ids[j]*32+c];}
 if(i<M*4){int j=i/4,c=i%4;out_anchors[i]=anchors[ids[j]*4+c];}
 if(i<M)out_static[i]=is_static[ids[i]];
}
std::vector<Tensor> select_anchors(Tensor intervals,Tensor times,Tensor anchors,Tensor offsets,Tensor features,Tensor scales,Tensor is_static) {
 c10::cuda::CUDAGuard guard(anchors.device());
 int N=anchors.size(0);TORCH_CHECK(anchors.is_contiguous()&&offsets.is_contiguous()&&features.is_contiguous()&&scales.is_contiguous()&&is_static.is_contiguous());
 TORCH_CHECK(anchors.numel()==N*4&&offsets.numel()==N*40&&features.numel()==N*32&&scales.numel()==N*40&&is_static.numel()==N);
 auto flags=torch::empty({N},anchors.options().dtype(torch::kBool));auto stream=at::cuda::getCurrentCUDAStream();
 if(N)flags_kernel<<<(N+255)/256,256,0,stream>>>(intervals.data_ptr<float>(),times.data_ptr<float>(),flags.data_ptr<bool>(),N,times.numel());
 auto ids=selected(flags);int M=ids.numel();
 auto a=torch::empty({M,4},anchors.options()),o=torch::empty({M,10,4},offsets.options()),f=torch::empty({M,32},features.options()),s=torch::empty({M,10,4},scales.options()),b=torch::empty({M},is_static.options());
 if(M)gather_selection<<<(M*40+255)/256,256,0,stream>>>(ids.data_ptr<int64_t>(),M,anchors.data_ptr<float>(),offsets.data_ptr<float>(),features.data_ptr<float>(),scales.data_ptr<float>(),is_static.data_ptr<bool>(),a.data_ptr<float>(),o.data_ptr<float>(),f.data_ptr<float>(),s.data_ptr<float>(),b.data_ptr<bool>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {ids,a,o,f,s,b};
}
__global__ void gather_generated(const int64_t* ids,int M,int N,int W,
 const float* means,const float* covariance,const float* opacity,const float* colors,
 float* out_means,float* out_covariance,float* out_opacity,float* out_colors) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<W*M*6){int j=i/6,c=i%6;out_covariance[i]=covariance[(j/M*N+ids[j%M])*6+c];}
 if(i<W*M*3){int j=i/3,c=i%3;out_means[i]=means[(j/M*N+ids[j%M])*3+c];}
 if(i<W*M){int frame=i/M;out_opacity[i]=opacity[frame*N+ids[i%M]];}
 if(i<M*3){int j=i/3,c=i%3;out_colors[i]=colors[ids[j]*3+c];}
}
std::vector<Tensor> compact(Tensor means,Tensor covariance,Tensor opacity,Tensor colors) {
 c10::cuda::CUDAGuard guard(means.device());int W=means.size(0),N=means.size(1);
 TORCH_CHECK(means.is_contiguous()&&covariance.is_contiguous()&&opacity.is_contiguous()&&colors.is_contiguous());
 auto flags=torch::empty({N},means.options().dtype(torch::kBool));auto stream=at::cuda::getCurrentCUDAStream();
 if(N)active_kernel<<<(N+255)/256,256,0,stream>>>(opacity.data_ptr<float>(),flags.data_ptr<bool>(),N,W);
 auto ids=selected(flags);int M=ids.numel();
 auto m=torch::empty({W,M,3},means.options()),c=torch::empty({W,M,6},covariance.options()),o=torch::empty({W,M},opacity.options()),rgb=torch::empty({M,3},colors.options());
 if(M)gather_generated<<<(W*M*6+255)/256,256,0,stream>>>(ids.data_ptr<int64_t>(),M,N,W,means.data_ptr<float>(),covariance.data_ptr<float>(),opacity.data_ptr<float>(),colors.data_ptr<float>(),m.data_ptr<float>(),c.data_ptr<float>(),o.data_ptr<float>(),rgb.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {m,c,o,rgb};
}

std::vector<Tensor> selected_async(Tensor flags) {
 int N=flags.numel();auto ids=torch::empty({N},flags.options().dtype(torch::kInt64));auto count=torch::zeros({},flags.options().dtype(torch::kInt32));
 if(!N)return {ids,count};
 auto stream=at::cuda::getCurrentCUDAStream();size_t bytes=0;thrust::counting_iterator<int64_t> input(0);
 C10_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr,bytes,input,flags.data_ptr<bool>(),ids.data_ptr<int64_t>(),count.data_ptr<int>(),N,stream));
 auto scratch=torch::empty({int64_t(bytes)},flags.options().dtype(torch::kUInt8));
 C10_CUDA_CHECK(cub::DeviceSelect::Flagged(scratch.data_ptr(),bytes,input,flags.data_ptr<bool>(),ids.data_ptr<int64_t>(),count.data_ptr<int>(),N,stream));
 return {ids,count};
}
__global__ void gather_selection_checked(const int64_t* ids,const int* count,int M,
 const float* anchors,const float* offsets,const float* features,const float* scales,const bool* is_static,
 float* out_anchors,float* out_offsets,float* out_features,float* out_scales,bool* out_static) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<M*40){int j=i/40,c=i%40;bool valid=j<*count;int source=valid?ids[j]:0;out_offsets[i]=valid?offsets[source*40+c]:0;out_scales[i]=valid?scales[source*40+c]:0;}
 if(i<M*32){int j=i/32,c=i%32;out_features[i]=j<*count?features[ids[j]*32+c]:0;}
 if(i<M*4){int j=i/4,c=i%4;out_anchors[i]=j<*count?anchors[ids[j]*4+c]:0;}
 if(i<M)out_static[i]=i<*count?is_static[ids[i]]:false;
}
std::vector<Tensor> select_cached(Tensor intervals,Tensor times,Tensor anchors,Tensor offsets,Tensor features,Tensor scales,Tensor is_static,int M) {
 c10::cuda::CUDAGuard guard(anchors.device());int N=anchors.size(0);TORCH_CHECK(M>=0&&M<=N);
 auto flags=torch::empty({N},anchors.options().dtype(torch::kBool));auto stream=at::cuda::getCurrentCUDAStream();
 if(N)flags_kernel<<<(N+255)/256,256,0,stream>>>(intervals.data_ptr<float>(),times.data_ptr<float>(),flags.data_ptr<bool>(),N,times.numel());
 auto selected=selected_async(flags);auto ids=selected[0].narrow(0,0,M),count=selected[1];
 auto a=torch::empty({M,4},anchors.options()),o=torch::empty({M,10,4},offsets.options()),ft=torch::empty({M,32},features.options()),sc=torch::empty({M,10,4},scales.options()),b=torch::empty({M},is_static.options());
 if(M)gather_selection_checked<<<(M*40+255)/256,256,0,stream>>>(ids.data_ptr<int64_t>(),count.data_ptr<int>(),M,anchors.data_ptr<float>(),offsets.data_ptr<float>(),features.data_ptr<float>(),scales.data_ptr<float>(),is_static.data_ptr<bool>(),a.data_ptr<float>(),o.data_ptr<float>(),ft.data_ptr<float>(),sc.data_ptr<float>(),b.data_ptr<bool>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {ids,a,o,ft,sc,b,count};
}
__global__ void gather_generated_checked(const int64_t* ids,const int* count,int M,int N,int W,
 const float* means,const float* covariance,const float* opacity,const float* colors,
 float* out_means,float* out_covariance,float* out_opacity,float* out_colors) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<W*M*6){int j=i/6,c=i%6;out_covariance[i]=j%M<*count?covariance[(j/M*N+ids[j%M])*6+c]:0;}
 if(i<W*M*3){int j=i/3,c=i%3;out_means[i]=j%M<*count?means[(j/M*N+ids[j%M])*3+c]:0;}
 if(i<W*M){int frame=i/M;out_opacity[i]=i%M<*count?opacity[frame*N+ids[i%M]]:0;}
 if(i<M*3){int j=i/3,c=i%3;out_colors[i]=j<*count?colors[ids[j]*3+c]:0;}
}
std::vector<Tensor> compact_cached(Tensor means,Tensor covariance,Tensor opacity,Tensor colors,int M) {
 c10::cuda::CUDAGuard guard(means.device());int W=means.size(0),N=means.size(1);TORCH_CHECK(M>=0&&M<=N);
 auto flags=torch::empty({N},means.options().dtype(torch::kBool));auto stream=at::cuda::getCurrentCUDAStream();
 if(N)active_kernel<<<(N+255)/256,256,0,stream>>>(opacity.data_ptr<float>(),flags.data_ptr<bool>(),N,W);
 auto selected=selected_async(flags);auto ids=selected[0],count=selected[1];
 auto m=torch::empty({W,M,3},means.options()),c=torch::empty({W,M,6},covariance.options()),o=torch::empty({W,M},opacity.options()),rgb=torch::empty({M,3},colors.options());
 if(M)gather_generated_checked<<<(W*M*6+255)/256,256,0,stream>>>(ids.data_ptr<int64_t>(),count.data_ptr<int>(),M,N,W,means.data_ptr<float>(),covariance.data_ptr<float>(),opacity.data_ptr<float>(),colors.data_ptr<float>(),m.data_ptr<float>(),c.data_ptr<float>(),o.data_ptr<float>(),rgb.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {m,c,o,rgb,count};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("select",&select_anchors);m.def("select_cached",&select_cached);m.def("compact_cached",&compact_cached);m.def("compact",&compact);}
