
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <math_constants.h>
using torch::Tensor;
__device__ unsigned long long timestamp() {
  unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;
}
#include <cooperative_groups.h>
#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"
namespace gsplat {
namespace cg=cooperative_groups;
template <typename scalar_t>
__global__ void projection_ewa_3dgs_fused_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [B, N, 3]
    const scalar_t *__restrict__ covars,   // [B, N, 6] optional
    const scalar_t *__restrict__ quats,    // [B, N, 4] optional
    const scalar_t *__restrict__ scales,   // [B, N, 3] optional
    const scalar_t *__restrict__ opacities, // [B, N] optional
    const scalar_t *__restrict__ viewmats, // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ radii,         // [B, C, N, 2]
    scalar_t *__restrict__ means2d,      // [B, C, N, 2]
    scalar_t *__restrict__ depths,       // [B, C, N]
    scalar_t *__restrict__ conics,       // [B, C, N, 3]
    scalar_t *__restrict__ compensations,
    unsigned long long* publication
) {
  if(!threadIdx.x) {
    publication[blockIdx.x*3]=timestamp();unsigned sm;
    asm volatile("mov.u32 %0, %%smid;":"=r"(sm));publication[blockIdx.x*3+2]=sm;
  }
  do {

    // parallelize over B * C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N) {
        break;
    }
    const uint32_t bid = idx / (C * N); // batch id
    const uint32_t cid = (idx / N) % C; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += bid * N * 3 + gid * 3;
    viewmats += bid * C * 16 + cid * 16;
    Ks += bid * C * 9 + cid * 9;

    // glm is column-major but input is row-major
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        break;
    }

    // transform Gaussian covariance to camera space
    mat3 covar;
    if (covars != nullptr) {
        covars += bid * N * 6 + gid * 6;
        covar = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quats += bid * N * 4 + gid * 4;
        scales += bid * N * 3 + gid * 3;
        quat_scale_to_covar_preci(
            glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
        );
    }
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // perspective projection
    mat2 covar2d;
    vec2 mean2d;

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    }

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        break;
    }

    // compute the inverse of the 2d covariance
    mat2 covar2d_inv = glm::inverse(covar2d);

    float extend = 3.33f;
    if (opacities != nullptr) {
        float opacity = opacities[bid * N + gid];
        if (compensations != nullptr) {
            // we assume compensation term will be applied later on.
            opacity *= compensation;
        }
        if (opacity < ALPHA_THRESHOLD) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            break;
        }
        // Compute opacity-aware bounding box.
        // https://arxiv.org/pdf/2402.00525 Section B.2
        extend = min(extend, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    }

    // compute tight rectangular bounding box (non differentiable)
    // https://arxiv.org/pdf/2402.00525
    float radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
    float radius_y = ceilf(extend * sqrtf(covar2d[1][1]));

    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        break;
    }

    // mask out gaussians outside the image region
    if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
        mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        break;
    }

    // write to outputs
    radii[idx * 2] = (int32_t)radius_x;
    radii[idx * 2 + 1] = (int32_t)radius_y;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }

  } while(false);
  __threadfence();__syncthreads();
  if(!threadIdx.x)publication[blockIdx.x*3+1]=timestamp();
}

std::vector<Tensor> project(Tensor means,Tensor covariance,Tensor opacity,Tensor views,Tensor intrinsics,
    int width,int height,float eps,float near_plane,float far_plane,bool compensate=true) {
  c10::cuda::CUDAGuard guard(means.device());int W=means.size(0),N=means.size(1);
  TORCH_CHECK(means.is_contiguous()&&covariance.is_contiguous()&&opacity.is_contiguous()&&
              views.is_contiguous()&&intrinsics.is_contiguous(),"contiguous projection inputs required");
  auto radii=torch::empty({W,N,2},means.options().dtype(torch::kInt32));
  auto xy=torch::empty({W,N,2},means.options()),depth=torch::empty_like(opacity);
  auto conics=torch::empty_like(means),compensation=torch::empty_like(opacity);
  auto publication=torch::empty({(W*N+255)/256,3},means.options().dtype(torch::kInt64));
  if(N)projection_ewa_3dgs_fused_fwd_kernel<float><<<(W*N+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
    W,1,N,means.data_ptr<float>(),covariance.data_ptr<float>(),nullptr,nullptr,opacity.data_ptr<float>(),
    views.data_ptr<float>(),intrinsics.data_ptr<float>(),width,height,eps,near_plane,far_plane,0,
    CameraModelType::PINHOLE,radii.data_ptr<int>(),xy.data_ptr<float>(),depth.data_ptr<float>(),
    conics.data_ptr<float>(),compensate?compensation.data_ptr<float>():nullptr,
    (unsigned long long*)publication.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {radii,xy,depth,conics,compensation,publication};
}

std::vector<int64_t> resources() {
  cudaFuncAttributes a;cudaDeviceProp d;int id;
  C10_CUDA_CHECK(cudaGetDevice(&id));C10_CUDA_CHECK(cudaGetDeviceProperties(&d,id));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,projection_ewa_3dgs_fused_fwd_kernel<float>));
  return {a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),a.maxThreadsPerBlock,
    d.maxThreadsPerMultiProcessor,d.regsPerMultiprocessor,int64_t(d.sharedMemPerMultiprocessor),
    d.maxBlocksPerMultiProcessor,d.warpSize,d.multiProcessorCount};
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&gsplat::project);m.def("resources",&gsplat::resources);}
