// CUDA operators for Gaussian generation, ordering, and rasterization.
// Projection uses the installed gsplat package.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>
#include <vector>
using torch::Tensor;

__global__ void generate_anchored_kernel(
    const float4 *anchors, const float4 *offsets, const float4 *log_scales,
    const float *decoded, const float *opacity, const bool *is_static,
    const float *frames, int N, int K, int W, float threshold,
    float3 *means, float *covs, float *ops) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= N) return;
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
  if (N) generate_anchored_kernel<<<(N+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
      (float4 *)anchors.data_ptr<float>(), (float4 *)offsets.data_ptr<float>(),
      (float4 *)log_scales.data_ptr<float>(), decoded.data_ptr<float>(),
      opacity.data_ptr<float>(), is_static.data_ptr<bool>(), frames.data_ptr<float>(),
      N,K,W,float(threshold),(float3 *)means.data_ptr<float>(),
      covs.data_ptr<float>(),ops.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {means,covs,ops};
}

__global__ void generate_4d_kernel(
    const float3 *xyz, const float4 *scales, const float4 *left,
    const float4 *right, const float *mu, const float *opacity,
    const bool *masks, const int64_t *source_ids, int source_count,
    const float *frames, int N, int W, float3 *means, float *covs, float *ops) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= N) return;
  float4 l = left[g], r = right[g], scale = scales[g];
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
  float3 origin = xyz[g];
  for (int f = 0; f < W; ++f) {
    int index = f*N+g, interval = int(frames[f])/20;
    float delta = frames[f]/30.0f-mu[g];
    float marginal = __expf(-.5f*delta*delta/variance);
    int64_t source = source_ids[g];
    bool selected = masks[int64_t(interval)*source_count+source] ||
                    masks[int64_t(interval+1)*source_count+source];
    means[index] = make_float3(origin.x+velocity[0]*delta,
        origin.y+velocity[1]*delta, origin.z+velocity[2]*delta);
#pragma unroll
    for (int k = 0; k < 6; ++k) covs[index*6+k] = covariance[k];
    ops[index] = selected && marginal > .05f ? opacity[g]*marginal : 0;
  }
}

std::vector<Tensor> generate_4d(Tensor xyz, Tensor scales, Tensor left,
    Tensor right, Tensor mu, Tensor opacity, Tensor masks, Tensor source_ids,
    int source_count, Tensor frames) {
  c10::cuda::CUDAGuard guard(xyz.device());
  int N = xyz.size(0), W = frames.numel();
  auto means = torch::empty({W, N, 3}, xyz.options());
  auto covs = torch::empty({W, N, 6}, xyz.options());
  auto ops = torch::empty({W, N}, xyz.options());
  if (N) generate_4d_kernel<<<(N+255)/256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (float3 *)xyz.data_ptr<float>(), (float4 *)scales.data_ptr<float>(),
      (float4 *)left.data_ptr<float>(), (float4 *)right.data_ptr<float>(),
      mu.data_ptr<float>(), opacity.data_ptr<float>(), masks.data_ptr<bool>(),
      source_ids.data_ptr<int64_t>(), source_count, frames.data_ptr<float>(),
      N, W, (float3 *)means.data_ptr<float>(), covs.data_ptr<float>(), ops.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {means, covs, ops};
}

template <typename Key>
std::vector<Tensor> radix_pairs(Tensor keys, Tensor values, int begin_bit,
                                int end_bit) {
  c10::cuda::CUDAGuard guard(keys.device());
  if (keys.numel() == 0)
    return {keys, values};
  auto ko = torch::empty_like(keys), vo = torch::empty_like(values);
  size_t bytes = 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys.data_ptr<Key>(),
                                  ko.data_ptr<Key>(), values.data_ptr<int>(),
                                  vo.data_ptr<int>(), keys.numel(), begin_bit,
                                  end_bit, stream);
  auto tmp =
      torch::empty({(int64_t)bytes}, keys.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      tmp.data_ptr(), bytes, keys.data_ptr<Key>(), ko.data_ptr<Key>(),
      values.data_ptr<int>(), vo.data_ptr<int>(), keys.numel(), begin_bit,
      end_bit, stream));
  return {ko, vo};
}
std::vector<Tensor> radix(Tensor keys, Tensor values, int begin_bit,
                          int end_bit) {
  if (keys.scalar_type() == torch::kFloat32)
    return radix_pairs<float>(keys, values, begin_bit, end_bit);
  if (keys.scalar_type() == torch::kInt32)
    return radix_pairs<int>(keys, values, begin_bit, end_bit);
  return radix_pairs<int64_t>(keys, values, begin_bit, end_bit);
}

__device__ __forceinline__ void rotation_pair(const float *source, float *q0, float *q1,
                                float &omega, float &den) {
  float n0 = 0, n1 = 0, dot = 0;
  for (int c = 0; c < 4; ++c) {
    q0[c] = source[c];
    q1[c] = source[4 + c];
    n0 += q0[c] * q0[c];
    n1 += q1[c] * q1[c];
  }
  for (int c = 0; c < 4; ++c) {
    q0[c] *= rsqrtf(n0);
    q1[c] *= rsqrtf(n1);
    dot += q0[c] * q1[c];
  }
  omega = fmaxf(acosf(fminf(1 - 1e-4f, fmaxf(-1 + 1e-4f, dot))), 1e-4f);
  den = fmaxf(sinf(omega), 1e-4f);
}

template <bool Prepared = false>
__global__ void generate_kernel(const float *xyz, const float *disp,
                                const float *motion, const float *rot,
                                const float *mrot, const float *scale,
                                const float *opacity, const float *center,
                                const float *var, const float *times, int ns,
                                int nd, int K, int W, float duration,
                                float interval, float shift, float varpad,
                                float *means, float *cov, float *op) {
  int item = blockIdx.x * blockDim.x + threadIdx.x, N = ns + nd;
  if (item >= ns + nd * W)
    return;
  // Static covariance and opacity are invariant across the temporal window.
  // Keep dynamic frame channels parallel and evaluate each static record once.
  int f = item < ns ? 0 : (item - ns) / nd;
  int g = item < ns ? item : ns + (item - ns) % nd;
  int idx = f * N + g;
  float t = times[f];
  float q[4];
  if (g < ns) {
    for (int frame = 0; frame < W; ++frame) {
      for (int c = 0; c < 3; ++c)
        means[3 * (frame * N + g) + c] =
            xyz[3 * g + c] + disp[3 * g + c] * (times[frame] / duration);
      op[frame * N + g] = opacity[g];
    }
    for (int c = 0; c < 4; ++c)
      q[c] = rot[4 * g + c];
  } else {
    int j = g - ns;
    float u = (t + shift) / interval;
    int k = (int)floorf(u);
    float a = u - k;
    for (int c = 0; c < 3; ++c) {
      int base = Prepared ? (k * 3 + c) * nd + j : (j * K + k) * 3 + c;
      int step = Prepared ? 3 * nd : 3;
      float y0 = motion[base - step], y1 = motion[base], y2 = motion[base + step],
            y3 = motion[base + 2 * step];
      means[3 * idx + c] = (2 * a * a * a - 3 * a * a + 1) * y1 +
                           (a * a * a - 2 * a * a + a) * (y2 - y0) * .5f +
                           (-2 * a * a * a + 3 * a * a) * y2 +
                           (a * a * a - a * a) * (y3 - y1) * .5f;
    }
    float q0[4], q1[4], omega, den;
    if (Prepared) {
      const float *pair = mrot + j;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        q0[c] = pair[(4 * k + c) * nd];
        q1[c] = pair[(4 * (k + 1) + c) * nd];
      }
      omega = pair[(4 * K + 2 * k) * nd];
      den = pair[(4 * K + 2 * k + 1) * nd];
    } else {
      rotation_pair(mrot + (j * K + k) * 4, q0, q1, omega, den);
    }
    float p0 = sinf((1 - a) * omega) / den, p1 = sinf(a * omega) / den,
          sum = fmaxf(p0 + p1, 1e-4f), norm = 0, abs_sum = 0;
    for (int c = 0; c < 4; ++c) {
      q[c] = q0[c] * (p0 / sum) + q1[c] * (p1 / sum);
      norm += q[c] * q[c];
      abs_sum += fabsf(q[c]);
    }
    for (int c = 0; c < 4; ++c)
      q[c] = abs_sum > 1e-4f ? q[c] * rsqrtf(norm) : q0[c];
    float c0 = center[2 * j], c1 = center[2 * j + 1], m = fminf(u - c0, u - c1),
          v = (u > c0 || u > c1) ? var[2 * j + 1] : var[2 * j];
    float sigma = Prepared ? v : expf(v) + varpad / interval / 2.36f;
    op[idx] = opacity[g] *
              ((c0 - u) * (c1 - u) < 0 ? 1 : expf(-m * m / (sigma * sigma)));
  }
  float w = q[0], x = q[1], y = q[2], z = q[3];
  float R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                2 * (x * z + w * y),     2 * (x * y + w * z),
                1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                2 * (x * z - w * y),     2 * (y * z + w * x),
                1 - 2 * (x * x + y * y)};
  int n = 0;
  for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      float v = 0;
      for (int k = 0; k < 3; ++k)
        v += R[3 * a + k] * R[3 * b + k] * scale[3 * g + k] * scale[3 * g + k];
      if (g < ns)
        for (int frame = 0; frame < W; ++frame)
          cov[6 * (frame * N + g) + n] = v;
      else
        cov[6 * idx + n] = v;
      ++n;
    }
}
// Keyframe constants are independent of the requested timestamp. Both the
// independent-frame and window paths consume the same initialized parameters.
// Positions, normalized rotations and pair angles use key/field/primitive order
// so a warp evaluating one timestamp reads contiguous FP32 fields.
__global__ void prepare_generation_kernel(const float *rotation,
    const float *variance, float *pairs, float *sigma, int N, int K,
    float interval, float varpad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * (K - 1))
    return;
  int j = i / (K - 1), k = i % (K - 1);
  float q0[4], q1[4], omega, den;
  rotation_pair(rotation + (j * K + k) * 4, q0, q1, omega, den);
  for (int c = 0; c < 4; ++c) {
    pairs[(4 * k + c) * N + j] = q0[c];
    if (k == K - 2)
      pairs[(4 * (k + 1) + c) * N + j] = q1[c];
  }
  pairs[(4 * K + 2 * k) * N + j] = omega;
  pairs[(4 * K + 2 * k + 1) * N + j] = den;
  if (!k)
    for (int side = 0; side < 2; ++side)
      sigma[2 * j + side] = expf(variance[2 * j + side]) + varpad / interval / 2.36f;
}
std::vector<Tensor> prepare_generation(Tensor motion, Tensor rotations, Tensor variances,
                                      float interval, float varpad) {
  c10::cuda::CUDAGuard guard(rotations.device());
  int N = rotations.size(0), K = rotations.size(1);
  TORCH_CHECK(K >= 2 && rotations.size(2) == 4 && interval > 0,
              "invalid native interpolation keyframes");
  auto pairs = torch::empty({4 * K + 2 * (K - 1), N}, rotations.options());
  auto sigma = torch::empty_like(variances);
  if (N)
    prepare_generation_kernel<<<(N * (K - 1) + 255) / 256, 256, 0,
                                 at::cuda::getCurrentCUDAStream()>>>(
        rotations.data_ptr<float>(), variances.data_ptr<float>(),
        pairs.data_ptr<float>(), sigma.data_ptr<float>(), N, K, interval, varpad);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {motion.permute({1, 2, 0}).contiguous(), pairs, sigma};
}

std::vector<Tensor> generate(Tensor xyz, Tensor disp, Tensor motion, Tensor rot,
                             Tensor mrot, Tensor scale, Tensor opacity,
                             Tensor center, Tensor var, Tensor times,
                             float duration, float interval, float shift,
                             float varpad) {
  c10::cuda::CUDAGuard guard(xyz.device());
  bool prepared = mrot.dim() == 2;
  int ns = xyz.size(0), nd = scale.size(0) - ns;
  int K = prepared ? motion.size(0) : motion.size(1);
  int W = times.numel(), N = ns + nd;
  auto means = torch::empty({W, N, 3}, xyz.options()),
       cov = torch::empty({W, N, 6}, xyz.options()),
       op = torch::empty({W, N}, xyz.options());
  auto kernel = prepared ? generate_kernel<true> : generate_kernel<false>;
  kernel<<<(ns + W * nd + 255) / 256, 256, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
      xyz.data_ptr<float>(), disp.data_ptr<float>(), motion.data_ptr<float>(),
      rot.data_ptr<float>(), mrot.data_ptr<float>(), scale.data_ptr<float>(),
      opacity.data_ptr<float>(), center.data_ptr<float>(),
      var.data_ptr<float>(), times.data_ptr<float>(), ns, nd, K, W, duration,
      interval, shift, varpad, means.data_ptr<float>(), cov.data_ptr<float>(),
      op.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {means, cov, op};
}

// Window statistics consume FP32 camera-space depths.
__global__ void window_statistics_kernel(const float *depth, int N, int W,
                                         float range, float *score,
                                         float *minimum) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= N)
    return;
  float previous = depth[g], change = 0, low = previous;
  for (int f = 1; f < W; ++f) {
    float d = depth[f * N + g];
    change += fabsf(d - previous);
    low = fminf(low, d);
    previous = d;
  }
  minimum[g] = low;
  float mean = W > 1 ? __fmul_rn(change, 1.f / (W - 1)) : 0.f;
  score[g] = expf(__fmul_rn(-mean, 1.f / range));
}
std::vector<Tensor> window_statistics(Tensor depth, float range) {
  c10::cuda::CUDAGuard guard(depth.device());
  int W = depth.size(0), N = depth.size(1);
  auto score = torch::empty({N}, depth.options()),
       minimum = torch::empty_like(score);
  if (N)
    window_statistics_kernel<<<(N + 255) / 256, 256, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
        depth.data_ptr<float>(), N, W, range,
        score.data_ptr<float>(), minimum.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {score, minimum};
}

struct StabilityFlag {
  bool stable;
  __device__ bool operator()(float value) const {
    return stable ? value > .98f : value <= .98f;
  }
};
std::tuple<Tensor, Tensor, int64_t> classify(Tensor score) {
  c10::cuda::CUDAGuard guard(score.device());
  int N = score.numel();
  auto integers = score.options().dtype(torch::kInt32);
  auto stable = torch::empty({N}, integers), unstable = torch::empty_like(stable);
  if (!N)
    return {stable, unstable, -1};
  auto counts = torch::empty({2}, integers);
  auto stream = at::cuda::getCurrentCUDAStream();
  cub::CountingInputIterator<int> ids(0);
  using Flags = cub::TransformInputIterator<bool, StabilityFlag, const float *>;
  Flags stable_flags(score.data_ptr<float>(), StabilityFlag{true});
  Flags unstable_flags(score.data_ptr<float>(), StabilityFlag{false});
  size_t bytes = 0;
  cub::DeviceSelect::Flagged(nullptr, bytes, ids, stable_flags,
                            stable.data_ptr<int>(), counts.data_ptr<int>(), N,
                            stream);
  auto temporary = torch::empty({(int64_t)bytes}, integers.dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
      temporary.data_ptr(), bytes, ids, stable_flags, stable.data_ptr<int>(),
      counts.data_ptr<int>(), N, stream));
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
      temporary.data_ptr(), bytes, ids, unstable_flags, unstable.data_ptr<int>(),
      counts.data_ptr<int>() + 1, N, stream));
  // Reduce and compare the stability threshold in FP32.
  auto mean = score.mean();
  int selected[2];
  float average;
  C10_CUDA_CHECK(cudaMemcpyAsync(selected, counts.data_ptr<int>(), 2 * sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(&average, mean.data_ptr<float>(), sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  return {stable.narrow(0, 0, selected[0]), unstable.narrow(0, 0, selected[1]),
          average >= .98f ? 1 : -1};
}

template <typename Index>
struct WindowVisible {
  const int2 *radii;
  int N, W;
  __device__ bool operator()(Index g) const {
    for (int f = 0; f < W; ++f) {
      int2 radius = radii[f * N + g];
      if (radius.x > 0 && radius.y > 0)
        return true;
    }
    return false;
  }
};
__global__ void visible_crossings(const int64_t *selected, const int *count,
                                  const float *depth, int N,
                                  int W, int *crossing) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  bool crossed = false;
  if (i + 1 < *count) {
    int64_t a = selected[i], b = selected[i + 1];
    for (int f = 0; f < W; ++f)
      crossed |= depth[f * N + b] < depth[f * N + a];
  }
  unsigned votes = __ballot_sync(0xffffffff, crossed);
  if (votes && threadIdx.x % 32 == 0)
    atomicExch(crossing, 1);
}
template <typename Index>
std::tuple<Tensor, bool> visible_order_impl(Tensor order, Tensor radii,
                                            Tensor depth) {
  c10::cuda::CUDAGuard guard(radii.device());
  int capacity = order.numel(), W = radii.size(0), N = radii.size(1);
  auto selected = torch::empty({capacity}, order.options().dtype(torch::kInt64));
  if (!capacity)
    return {selected, false};
  auto summary = torch::zeros({2}, order.options().dtype(torch::kInt32));
  auto stream = at::cuda::getCurrentCUDAStream();
  using Flags = cub::TransformInputIterator<bool, WindowVisible<Index>, const Index *>;
  Flags flags(order.data_ptr<Index>(), WindowVisible<Index>{
      (int2 *)radii.data_ptr<int>(), N, W});
  size_t bytes = 0;
  cub::DeviceSelect::Flagged(nullptr, bytes, order.data_ptr<Index>(), flags,
      selected.data_ptr<int64_t>(), summary.data_ptr<int>(), capacity, stream);
  auto temporary = torch::empty({(int64_t)bytes}, order.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(temporary.data_ptr(), bytes,
      order.data_ptr<Index>(), flags, selected.data_ptr<int64_t>(),
      summary.data_ptr<int>(), capacity, stream));
  if (depth.numel())
    visible_crossings<<<(capacity + 255) / 256, 256, 0, stream>>>(
        selected.data_ptr<int64_t>(), summary.data_ptr<int>(), depth.data_ptr<float>(),
        N, W, summary.data_ptr<int>() + 1);
  int host[2];
  C10_CUDA_CHECK(cudaMemcpyAsync(host, summary.data_ptr<int>(), sizeof(host),
                                 cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {selected.narrow(0, 0, host[0]), host[1] != 0};
}
std::tuple<Tensor, bool> visible_order(Tensor order, Tensor radii, Tensor depth) {
  if (order.scalar_type() == torch::kInt32)
    return visible_order_impl<int>(order, radii, depth);
  return visible_order_impl<int64_t>(order, radii, depth);
}

__device__ float rectangle_sigma(float2, float3, float, float, float, float);

__global__ void projection_finish_kernel(const float2 *xy, int2 *radii,
                                         const float *opacity,
                                         const float *compensation,
                                         float *out_opacity, int n, int width,
                                         int height) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float2 p = xy[i];
  if (fabsf((p.x - width / 2.f) * 2 / width) > 1.3f ||
      fabsf((p.y - height / 2.f) * 2 / height) > 1.3f)
    radii[i] = make_int2(0, 0);
  out_opacity[i] = opacity[i] * compensation[i];
}
Tensor projection_finish(Tensor xy, Tensor radii, Tensor opacity,
                         Tensor compensation, int width, int height) {
  c10::cuda::CUDAGuard guard(xy.device());
  auto output = torch::empty_like(opacity);
  int n = opacity.numel();
  if (n)
    projection_finish_kernel<<<(n + 255) / 256, 256, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
        (float2 *)xy.data_ptr<float>(), (int2 *)radii.data_ptr<int>(),
        opacity.data_ptr<float>(), compensation.data_ptr<float>(),
        output.data_ptr<float>(), n, width, height);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

// Conservative ellipse bound at the compositor's alpha threshold.
__device__ __forceinline__ bool tile_visible(float2 p, float3 conic, float opacity,
                                          int x, int y, int ts) {
  float lower = rectangle_sigma(p, conic, x * ts + .5f, x * ts + ts - .5f,
                                y * ts + .5f, y * ts + ts - .5f);
  return !(lower > __logf(255.f * opacity) + 1e-4f);
}

__global__ void ordered_tile_counts(const float2 *xy, const int2 *radii,
                                    const int64_t *order,
                                    int N, int W, int tw, int th, int ts, int *counts,
                                    bool shared) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * (shared ? 1 : W))
    return;
  int f = i / N, g = f * N + order[i];
  int2 r = radii[g];
  counts[i] = 0;
  if (shared || (r.x > 0 && r.y > 0)) {
    float2 p = xy[g];
    int xmin = max(0, min(tw, (int)floorf((p.x - r.x) / ts))),
        xmax = max(0, min(tw, (int)ceilf((p.x + r.x) / ts))),
        ymin = max(0, min(th, (int)floorf((p.y - r.y) / ts))),
        ymax = max(0, min(th, (int)ceilf((p.y + r.y) / ts)));
    if (shared) {
      xmin = tw;
      ymin = th;
      xmax = ymax = 0;
      for (int frame = 0; frame < W; ++frame) {
        float2 center = xy[frame * N + g];
        int2 radius = radii[frame * N + g];
        if (radius.x <= 0 || radius.y <= 0)
          continue;
        xmin =
            min(xmin, max(0, min(tw, (int)floorf((center.x - radius.x) / ts))));
        xmax =
            max(xmax, max(0, min(tw, (int)ceilf((center.x + radius.x) / ts))));
        ymin =
            min(ymin, max(0, min(th, (int)floorf((center.y - radius.y) / ts))));
        ymax =
            max(ymax, max(0, min(th, (int)ceilf((center.y + radius.y) / ts))));
      }
    }
    counts[i] = max(0, xmax - xmin) * max(0, ymax - ymin);
  }
}
template <typename Key>
__global__ void ordered_tile_write(const float2 *xy, const int2 *radii,
                                   const int64_t *order,
                                   const int64_t *assignment, const int *scan,
                                   int N, int W, int tw, int th, int ts,
                                   bool shared, Key *keys, int *ids) {
  constexpr int Lanes = 8;
  int lane = threadIdx.x % Lanes;
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / Lanes;
  if (i >= N * (shared ? 1 : W))
    return;
  int g = 0, image = 0, start = 0, xmin = 0, xmax = 0, ymin = 0, ymax = 0;
  if (!lane) {
    int f = i / N;
    g = f * N + order[i];
    image = shared ? assignment[i] : assignment[i] * W + f;
    start = scan[i];
    xmin = tw;
    ymin = th;
    for (int frame = 0; frame < (shared ? W : 1); ++frame) {
      int j = shared ? frame * N + g : g;
      int2 r = radii[j];
      if (r.x <= 0 || r.y <= 0)
        continue;
      float2 p = xy[j];
      xmin = min(xmin, max(0, min(tw, (int)floorf((p.x - r.x) / ts))));
      xmax = max(xmax, max(0, min(tw, (int)ceilf((p.x + r.x) / ts))));
      ymin = min(ymin, max(0, min(th, (int)floorf((p.y - r.y) / ts))));
      ymax = max(ymax, max(0, min(th, (int)ceilf((p.y + r.y) / ts))));
    }
  }
  unsigned active = __activemask();
  g = __shfl_sync(active, g, 0, Lanes);
  image = __shfl_sync(active, image, 0, Lanes);
  start = __shfl_sync(active, start, 0, Lanes);
  xmin = __shfl_sync(active, xmin, 0, Lanes);
  xmax = __shfl_sync(active, xmax, 0, Lanes);
  ymin = __shfl_sync(active, ymin, 0, Lanes);
  ymax = __shfl_sync(active, ymax, 0, Lanes);
  int nx = max(0, xmax - xmin), ny = max(0, ymax - ymin);
  int shift = 0;
  while ((1 << shift) <= tw * th)
    ++shift;
  for (int item = lane; item < nx * ny; item += Lanes) {
    int x = xmin + item % nx, y = ymin + item / nx;
    keys[start + item] = (Key(image) << shift) | (y * tw + x);
    ids[start + item] = g;
  }
}
template <typename Key>
std::vector<Tensor> ordered_tiles_impl(Tensor xy, Tensor radii, Tensor order,
                                  Tensor assignment, int tw, int th,
                                  int ts, bool shared) {
  c10::cuda::CUDAGuard guard(xy.device());
  int W = order.size(0), N = order.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  int totaln = N * (shared ? 1 : W);
  auto key_options = xy.options().dtype(sizeof(Key) == 4 ? torch::kInt32
                                                        : torch::kInt64);
  if (!totaln)
    return {torch::empty({0}, key_options),
            torch::empty({0}, xy.options().dtype(torch::kInt32))};
  auto cnt = torch::empty({totaln}, xy.options().dtype(torch::kInt32)),
       scan = torch::empty_like(cnt);
  ordered_tile_counts<<<(totaln + 255) / 256, 256, 0, stream>>>(
      (float2 *)xy.data_ptr<float>(), (int2 *)radii.data_ptr<int>(),
      order.data_ptr<int64_t>(), N, W, tw,
      th, ts, cnt.data_ptr<int>(), shared);
  size_t bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, bytes, cnt.data_ptr<int>(),
                                scan.data_ptr<int>(), totaln, stream);
  auto tmp = torch::empty({(int64_t)bytes}, cnt.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(
      cub::DeviceScan::ExclusiveSum(tmp.data_ptr(), bytes, cnt.data_ptr<int>(),
                                    scan.data_ptr<int>(), totaln, stream));
  int last[2];
  C10_CUDA_CHECK(cudaMemcpyAsync(last, scan.data_ptr<int>() + totaln - 1, 4,
                                 cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(last + 1, cnt.data_ptr<int>() + totaln - 1, 4,
                                 cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  int total = last[0] + last[1];
  auto keys = torch::empty({total}, key_options),
       ids = torch::empty({total}, cnt.options());
  ordered_tile_write<Key><<<(totaln + 31) / 32, 256, 0, stream>>>(
      (float2 *)xy.data_ptr<float>(), (int2 *)radii.data_ptr<int>(),
      order.data_ptr<int64_t>(), assignment.data_ptr<int64_t>(),
      scan.data_ptr<int>(), N, W, tw, th, ts, shared, keys.data_ptr<Key>(),
      ids.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {keys, ids};
}

std::vector<Tensor> ordered_tiles(Tensor xy, Tensor radii, Tensor order,
                                  Tensor assignment, int B, int tw, int th,
                                  int ts, bool shared) {
  int bits = 0;
  while ((int64_t(1) << bits) <= int64_t(tw) * th)
    ++bits;
  // Shared keys contain only batch/tile groups, without FP32 depth bits.
  if (shared && int64_t(B) * (int64_t(1) << bits) <= INT_MAX)
    return ordered_tiles_impl<int>(xy, radii, order, assignment, tw, th, ts, shared);
  return ordered_tiles_impl<int64_t>(xy, radii, order, assignment, tw, th, ts, shared);
}

// A tile overlapping the projected radius lies at most radius + tile size
// from its center. Compute this conservative support once per frame record.
__global__ void projected_support(const int2 *radii, const float3 *conics,
                                   const float *opacity, float4 *support,
                                   int n, int ts) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n)
    return;
  float3 a = conics[j];
  int2 r = radii[j];
  float mx = float(r.x) + ts, my = float(r.y) + ts;
  float error = 2e-6f * (fabsf(a.x) * mx * mx + fabsf(a.z) * my * my +
                          2 * fabsf(a.y) * mx * my);
  float target = fmaxf(0, __logf(255.f * opacity[j]) + error + 1e-3f);
  float product = a.x * a.z, square = a.y * a.y;
  float det = product - square - 2e-6f * (fabsf(product) + square);
  float rx = CUDART_INF_F, ry = CUDART_INF_F;
  if (det > 0 && a.x > 0 && a.z > 0) {
    rx = sqrtf(2 * target * a.z / det) * (1 + 1e-6f) + 1e-3f;
    ry = sqrtf(2 * target * a.x / det) * (1 + 1e-6f) + 1e-3f;
  }
  support[j] = make_float4(rx, ry, __logf(255.f * opacity[j]) + 1e-4f, error);
}

__device__ __forceinline__ bool tile_visible_bound(float2 p, float3 conic, float cutoff,
                                                int x, int y, int ts) {
  float lower = rectangle_sigma(p, conic, x * ts + .5f, x * ts + ts - .5f,
                                y * ts + .5f, y * ts + ts - .5f);
  return !(lower > cutoff);
}

template <bool SubtileMasks = false, typename Key = int64_t>
__global__ void cull_tile_flags(const Key *keys, int *ids, const float2 *xy,
                                const float3 *conics, const float *opacity,
                                int n, int tile_shift, int drop_bits, int tw,
                                int ts, int *flags, const int2 *radii, int N,
                                int W, int id_bits, const float4 *support) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int tile = (keys[i] >> drop_bits) & ((1 << tile_shift) - 1), g = ids[i];
  if (!id_bits) {
    flags[i] =
        tile_visible(xy[g], conics[g], opacity[g], tile % tw, tile / tw, ts);
    return;
  }
  unsigned mask = 0, regions = 0;
  int x = tile % tw, y = tile / tw;
  for (int f = 0; f < W; ++f) {
    int j = f * N + g;
    float2 p = xy[j];
    int2 r = radii[j];
    if (r.x <= 0 || r.y <= 0 || x < (int)floorf((p.x - r.x) / ts) ||
        x >= (int)ceilf((p.x + r.x) / ts) ||
        y < (int)floorf((p.y - r.y) / ts) || y >= (int)ceilf((p.y + r.y) / ts))
      continue;
    float4 bound;
    if (SubtileMasks)
      bound = support[j];
    bool visible = SubtileMasks ? tile_visible_bound(p, conics[j], bound.z, x, y, ts)
                                 : tile_visible(p, conics[j], opacity[j], x, y, ts);
    if (visible) {
      mask |= 1u << f;
      if (SubtileMasks) {
        // Complete the square along the stronger diagonal. Independent
        // interval distances lower-bound both nonnegative squared terms;
        // the projection-wide FP32 margin keeps rejection conservative.
        // Union the resulting per-frame regions, retaining the existing ID.
        float3 a = conics[j];
        bool swap = a.z > a.x;
        float pivot = swap ? a.z : a.x;
        float ratio = __fdividef(a.y, pivot);
        float other = (swap ? a.x : a.z) - a.y * ratio;
        bool valid = pivot > 0 && other >= 0 && isfinite(ratio);
#pragma unroll
        for (int part = 0; part < 8; ++part) {
          float px = x * 16 + (part % 2) * 8 + .5f;
          float py = y * 16 + (part / 2) * 4 + .5f;
          if (p.x + bound.x < px || p.x - bound.x > px + 7 ||
              p.y + bound.y < py || p.y - bound.y > py + 3)
            continue;
          float dx = p.x - (px + 3.5f), dy = p.y - (py + 1.5f);
          float u = swap ? dy : dx, v = swap ? dx : dy;
          float ru = swap ? 1.5f : 3.5f, rv = swap ? 3.5f : 1.5f;
          float d = fmaxf(0, fabsf(u + ratio * v) - ru - fabsf(ratio) * rv);
          float e = fmaxf(0, fabsf(v) - rv);
          float lower = valid ? .5f * (pivot * d * d + other * e * e) - bound.w
                              : -CUDART_INF_F;
          if (!(lower > bound.z))
            regions |= 1u << part;
        }
      }
    }
  }
  // Spare ID bits hold the eight 8x4 pixel-region bits. The host enables
  // this only when both primitive IDs and all frame bits still fit int32.
  flags[i] = mask != 0;
  ids[i] = g | (mask << id_bits);
  if (SubtileMasks)
    ids[i] |= regions << (id_bits + W);
}
template <typename Key>
__global__ void compact_tiles(const Key *keys, const int *ids,
                              const int *flags, const int *scan, int n,
                              Key *output_keys, int *output_ids) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && flags[i]) {
    output_keys[scan[i]] = keys[i];
    output_ids[scan[i]] = ids[i];
  }
}
template <typename Key>
std::vector<Tensor> cull_tiles_impl(Tensor keys, Tensor ids, Tensor xy,
                               Tensor conics, Tensor opacity, int tw, int th,
                               int ts, int drop_bits, Tensor radii,
                               int id_bits) {
  c10::cuda::CUDAGuard guard(xy.device());
  int n = ids.numel(), shift = 0;
  if (!n)
    return {keys, ids};
  while ((1 << shift) <= tw * th)
    ++shift;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto flags = torch::empty_like(ids), scan = torch::empty_like(ids);
  bool masked = id_bits > 0 && id_bits + xy.size(0) + 8 <= 32;
  Tensor support;
  if (masked) {
    support = torch::empty({xy.size(0) * xy.size(1), 4}, xy.options());
    projected_support<<<(xy.size(0) * xy.size(1) + 255) / 256, 256, 0, stream>>>(
        (int2 *)radii.data_ptr<int>(), (float3 *)conics.data_ptr<float>(),
        opacity.data_ptr<float>(), (float4 *)support.data_ptr<float>(),
        xy.size(0) * xy.size(1), ts);
  }
  auto kernel = masked ? cull_tile_flags<true, Key> : cull_tile_flags<false, Key>;
  kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      keys.data_ptr<Key>(), ids.data_ptr<int>(),
      (float2 *)xy.data_ptr<float>(), (float3 *)conics.data_ptr<float>(),
      opacity.data_ptr<float>(), n, shift, drop_bits, tw, ts,
      flags.data_ptr<int>(), (int2 *)radii.data_ptr<int>(), xy.size(1),
      xy.size(0), id_bits, masked ? (float4 *)support.data_ptr<float>() : nullptr);
  size_t bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, bytes, flags.data_ptr<int>(),
                                scan.data_ptr<int>(), n, stream);
  auto tmp = torch::empty({(int64_t)bytes}, ids.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      tmp.data_ptr(), bytes, flags.data_ptr<int>(), scan.data_ptr<int>(), n,
      stream));
  int last[2];
  C10_CUDA_CHECK(cudaMemcpyAsync(last, scan.data_ptr<int>() + n - 1, 4,
                                 cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(last + 1, flags.data_ptr<int>() + n - 1, 4,
                                 cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  auto output_keys = torch::empty({last[0] + last[1]}, keys.options());
  auto output_ids = torch::empty({last[0] + last[1]}, ids.options());
  compact_tiles<Key><<<(n + 255) / 256, 256, 0, stream>>>(
      keys.data_ptr<Key>(), ids.data_ptr<int>(), flags.data_ptr<int>(),
      scan.data_ptr<int>(), n, output_keys.data_ptr<Key>(),
      output_ids.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_keys, output_ids};
}

std::vector<Tensor> cull_tiles(Tensor keys, Tensor ids, Tensor xy,
                               Tensor conics, Tensor opacity, int tw, int th,
                               int ts, int drop_bits, Tensor radii,
                               int id_bits) {
  if (keys.scalar_type() == torch::kInt32)
    return cull_tiles_impl<int>(keys, ids, xy, conics, opacity, tw, th, ts,
                                drop_bits, radii, id_bits);
  return cull_tiles_impl<int64_t>(keys, ids, xy, conics, opacity, tw, th, ts,
                                drop_bits, radii, id_bits);
}

__global__ void shift_keys(const int64_t *input, int64_t *output, int n,
                           int bits) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    output[i] = (uint64_t)input[i] >> bits;
}

// Encode projection FP32 bit patterns; RGB stays native and frame-local.
__device__ int field_bits(const float2 *xy, const float3 *conics,
                           const float *opacity, int j, int field) {
  const float *value = field < 2 ? (const float *)xy + 2 * j + field :
                       field == 2 ? opacity + j :
                                    (const float *)conics + 3 * j + field - 3;
  return __float_as_int(*value);
}
__global__ void packed_sizes(const float2 *xy, const int2 *radii,
                             const float3 *conics, const float *opacity,
                             int N, int W, uint64_t *costs, int *descriptors) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= N)
    return;
  int base = 0;
  while (base + 1 < W && (radii[base * N + g].x <= 0 ||
                         radii[base * N + g].y <= 0))
    ++base;
  bool compact = true;
  for (int f = 1; f < W; ++f) {
    if (radii[f * N + g].x <= 0 || radii[f * N + g].y <= 0)
      continue;
    for (int field = 0; field < 6; ++field) {
      int64_t d = int64_t(field_bits(xy, conics, opacity, f * N + g, field)) -
                  field_bits(xy, conics, opacity, base * N + g, field);
      compact &= d >= -32768 && d <= 32767;
    }
  }
  unsigned words = compact ? W * 6 + 3 : W * 9;
  // One scan supplies both compact-record ranks and cumulative payload sizes.
  costs[g] = (uint64_t(words) << 32) | unsigned(compact);
  descriptors[g] = (compact ? 0x80000000u : 0u) |
                   (base << 28);
}
__global__ void packed_offsets(int *layout, const uint64_t *prefix, int N) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g > N)
    return;
  uint64_t item = prefix[g];
  layout[g] |= unsigned(item);
  layout[N + 1 + g] = unsigned(item);
  layout[2 * (N + 1) + g] = item >> 32;
}
Tensor packed_layout(Tensor xy, Tensor radii, Tensor conics, Tensor opacity) {
  c10::cuda::CUDAGuard guard(xy.device());
  int N = xy.size(1), W = xy.size(0);
  TORCH_CHECK(W >= 1 && W <= 5 && int64_t(N) * W * 9 < (1 << 28),
              "packed layout exceeds the descriptor range");
  auto integers = xy.options().dtype(torch::kInt32);
  auto layout = torch::zeros({3, N + 1}, integers);
  if (!N)
    return layout;
  auto costs = torch::empty({N}, integers.dtype(torch::kInt64));
  auto prefix = torch::zeros({N + 1}, costs.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  packed_sizes<<<(N + 255) / 256, 256, 0, stream>>>(
      (float2 *)xy.data_ptr<float>(), (int2 *)radii.data_ptr<int>(),
      (float3 *)conics.data_ptr<float>(), opacity.data_ptr<float>(), N, W,
      (uint64_t *)costs.data_ptr<int64_t>(), layout.data_ptr<int>());
  size_t bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, bytes, (uint64_t *)costs.data_ptr<int64_t>(),
                                (uint64_t *)prefix.data_ptr<int64_t>() + 1, N, stream);
  auto temporary = torch::empty({(int64_t)bytes}, integers.dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(temporary.data_ptr(), bytes,
      (uint64_t *)costs.data_ptr<int64_t>(), (uint64_t *)prefix.data_ptr<int64_t>() + 1, N, stream));
  packed_offsets<<<(N + 256) / 256, 256, 0, stream>>>(layout.data_ptr<int>(),
      (uint64_t *)prefix.data_ptr<int64_t>(), N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return layout;
}

template <bool Compact = false>
__global__ void stable_batch_costs(const int64_t *order, const float2 *xy,
                                   const int2 *radii, int64_t *costs, int N,
                                   int W, int tw, int th, int shared_bytes,
                                   bool shared_tiles, const int *layout) {
  int rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= N)
    return;
  int64_t cost = 0;
  bool any = false;
  int lx = tw, ly = th, hx = 0, hy = 0;
  int g = order[rank];
  for (int f = 0; f < W; ++f) {
    float2 p = xy[f * N + g];
    int2 r = radii[f * N + g];
    if (r.x <= 0 || r.y <= 0)
      continue;
    any = true;
    int xmin = max(0, min(tw, (int)floorf((p.x - r.x) / 16))),
        xmax = max(0, min(tw, (int)ceilf((p.x + r.x) / 16))),
        ymin = max(0, min(th, (int)floorf((p.y - r.y) / 16))),
        ymax = max(0, min(th, (int)ceilf((p.y + r.y) / 16)));
    if (!shared_tiles) {
      cost += 52;
      cost += (int64_t)(xmax - xmin) * (ymax - ymin) * 4;
    }
    lx = min(lx, xmin);
    ly = min(ly, ymin);
    hx = max(hx, xmax);
    hy = max(hy, ymax);
  }
  if (shared_tiles && any) {
    // A compact descriptor supplies the inverse ID. Radius/depth arrays and
    // frame orders are consumed before compositing.
    cost += (int64_t)(hx - lx) * (hy - ly) * 4 +
            (Compact ? (layout[2 * (N + 1) + g + 1] - layout[2 * (N + 1) + g]) * 4 : W * 36);
  }
  costs[rank] = cost + (any ? shared_bytes + (shared_tiles ? 4 : 0) : 0);
}

Tensor stable_batch_prefix(Tensor orders, Tensor xy, Tensor radii, int tw,
                           int th, int shared_bytes, bool shared_tiles, Tensor layout) {
  c10::cuda::CUDAGuard guard(xy.device());
  int W = orders.size(0), N = orders.size(1);
  auto costs = torch::empty({N}, orders.options());
  auto prefix = torch::zeros({N + 1}, orders.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (!N)
    return prefix;
  auto kernel = layout.numel() ? stable_batch_costs<true> : stable_batch_costs<false>;
  kernel<<<(N + 255) / 256, 256, 0, stream>>>(
      orders.data_ptr<int64_t>(), (float2 *)xy.data_ptr<float>(),
      (int2 *)radii.data_ptr<int>(), costs.data_ptr<int64_t>(), N, W, tw, th,
      shared_bytes, shared_tiles, layout.numel() ? layout.data_ptr<int>() : nullptr);
  size_t bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, bytes, costs.data_ptr<int64_t>(),
                                prefix.data_ptr<int64_t>() + 1, N, stream);
  auto tmp = torch::empty({(int64_t)bytes}, xy.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
      tmp.data_ptr(), bytes, costs.data_ptr<int64_t>(),
      prefix.data_ptr<int64_t>() + 1, N, stream));
  return prefix;
}
// Attributes are already in common ROL order. Charge the complete packed
// payload, including records with no surviving tile, plus each actual tile ID.
__global__ void tile_payload_cost(const int *layout, int N, int64_t *cost) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g < N)
    cost[g] = 4 + int64_t(layout[2 * (N + 1) + g + 1] -
                          layout[2 * (N + 1) + g]) * 4;
}
__global__ void tile_record_cost(const int *ids, int n, int id_bits,
                                  int64_t *cost) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int g = unsigned(ids[i]) & ((1u << id_bits) - 1);
    unsigned same = __match_any_sync(__activemask(), g);
    if ((threadIdx.x & 31) == __ffs(same) - 1)
      atomicAdd((unsigned long long *)(cost + g), __popc(same) * 4ull);
  }
}
Tensor tile_batch_prefix(Tensor layout, Tensor ids, int id_bits) {
  c10::cuda::CUDAGuard guard(layout.device());
  int N = layout.size(1) - 1;
  auto cost = torch::empty({N}, layout.options().dtype(torch::kInt64));
  auto prefix = torch::zeros({N + 1}, cost.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (!N)
    return prefix;
  tile_payload_cost<<<(N + 255) / 256, 256, 0, stream>>>(
      layout.data_ptr<int>(), N, cost.data_ptr<int64_t>());
  if (ids.numel())
    tile_record_cost<<<(ids.numel() + 255) / 256, 256, 0, stream>>>(
        ids.data_ptr<int>(), ids.numel(), id_bits, cost.data_ptr<int64_t>());
  size_t bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, bytes, cost.data_ptr<int64_t>(),
                               prefix.data_ptr<int64_t>() + 1, N, stream);
  auto temp = torch::empty({(int64_t)bytes}, layout.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
      temp.data_ptr(), bytes, cost.data_ptr<int64_t>(),
      prefix.data_ptr<int64_t>() + 1, N, stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return prefix;
}
template <typename Key>
__global__ void assign_tile_batch_keys(Key *keys, const int *ids,
                                       const int64_t *assignment, int n,
                                       int id_bits, int tile_bits) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    keys[i] |= Key(assignment[unsigned(ids[i]) & ((1u << id_bits) - 1)]) << tile_bits;
}
Tensor tile_batch_keys(Tensor keys, Tensor ids, Tensor assignment, int id_bits,
                        int tile_bits, int B) {
  c10::cuda::CUDAGuard guard(keys.device());
  bool compact = int64_t(B) * (int64_t(1) << tile_bits) <= INT_MAX;
  // Fresh unassigned keys can be updated in place; widen before writing if
  // batch and tile fields would exceed the compact key range.
  auto result = keys.to(compact ? torch::kInt32 : torch::kInt64);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (ids.numel()) {
    if (compact)
      assign_tile_batch_keys<int><<<(ids.numel() + 255) / 256, 256, 0, stream>>>(
          result.data_ptr<int>(), ids.data_ptr<int>(), assignment.data_ptr<int64_t>(),
          ids.numel(), id_bits, tile_bits);
    else
      assign_tile_batch_keys<int64_t><<<(ids.numel() + 255) / 256, 256, 0, stream>>>(
          result.data_ptr<int64_t>(), ids.data_ptr<int>(), assignment.data_ptr<int64_t>(),
          ids.numel(), id_bits, tile_bits);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

__global__ void batch_limits_kernel(const int64_t *prefix, int N,
                                      int64_t budget, int *limits,
                                      int64_t *summary) {
  int start = 0, batches = 0;
  int64_t maximum = 0;
  limits[0] = 0;
  while (start < N) {
    int low = start + 1, high = N + 1;
    int64_t base = prefix[start];
    while (low < high) {
      int middle = low + (high - low) / 2;
      if (prefix[middle] - base <= budget)
        low = middle + 1;
      else
        high = middle;
    }
    int end = low - 1;
    if (end <= start) {
      summary[0] = -1;
      summary[1] = 0;
      return;
    }
    maximum = max(maximum, prefix[end] - base);
    limits[++batches] = end;
    start = end;
  }
  summary[0] = batches;
  summary[1] = maximum;
}
__global__ void batch_assignment_kernel(const int *limits, int B,
                                          int64_t *assignment, int N, int W) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= int64_t(N) * W)
    return;
  int rank = i % N, low = 1, high = B + 1;
  while (low < high) {
    int middle = low + (high - low) / 2;
    if (limits[middle] <= rank)
      low = middle + 1;
    else
      high = middle;
  }
  assignment[i] = low - 1;
}
std::tuple<Tensor, int64_t, int64_t> bounded_batches(Tensor prefix,
                                                   int64_t budget, int W) {
  c10::cuda::CUDAGuard guard(prefix.device());
  TORCH_CHECK(budget > 0 && W > 0 && prefix.numel() > 0,
              "invalid batch budget or prefix");
  int N = prefix.numel() - 1;
  auto assignment = torch::empty({W, N}, prefix.options());
  if (!N)
    return {assignment, 1, 0};
  auto limits = torch::empty({N + 1}, prefix.options().dtype(torch::kInt32));
  auto summary = torch::empty({2}, prefix.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  batch_limits_kernel<<<1, 1, 0, stream>>>(prefix.data_ptr<int64_t>(), N,
      budget, limits.data_ptr<int>(), summary.data_ptr<int64_t>());
  int64_t host[2];
  C10_CUDA_CHECK(cudaMemcpyAsync(host, summary.data_ptr<int64_t>(), sizeof(host),
                                 cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  TORCH_CHECK_VALUE(host[0] >= 0, "one render interval exceeds L2; reduce the window");
  batch_assignment_kernel<<<(int64_t(N) * W + 255) / 256, 256, 0, stream>>>(
      limits.data_ptr<int>(), host[0], assignment.data_ptr<int64_t>(), N, W);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {assignment, host[0], host[1]};
}

__global__ void linear_jobs(int64_t *jobs, int n, int shift, int tiles) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int64_t key = jobs[i];
  int image = key >> shift, tile = key & ((1 << shift) - 1);
  jobs[i] = (int64_t)image * tiles + tile;
}
__global__ void job_boundaries(const int64_t *jobs, int *begins, int n, int B,
                               int span) {
  int bi = blockIdx.x * blockDim.x + threadIdx.x;
  if (bi > B)
    return;
  int lo = 0, hi = n;
  int64_t key = (int64_t)bi * span;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (jobs[mid] < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  begins[bi] = lo;
}
template <typename Key>
__global__ void dense_shared_ranges(const Key *keys, int n, int tiles,
                                   int shift, int total, int64_t *jobs,
                                   int *offsets) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>total)return;
  Key key=(Key(i/tiles)<<shift)|(i%tiles);
  int lo=0,hi=n;
  while(lo<hi){int mid=lo+(hi-lo)/2;if(keys[mid]<key)lo=mid+1;else hi=mid;}
  offsets[i]=lo;
  if(i<total)jobs[i]=key;
}

template <typename Key>
std::tuple<Tensor, Tensor, std::vector<int64_t>>
tile_ranges_impl(Tensor keys, int W, int B, int tiles, int drop_bits, bool packed_jobs) {
  c10::cuda::CUDAGuard guard(keys.device());
  int n = keys.numel(), shift = 0;
  while ((1 << shift) <= tiles)
    ++shift;
  auto integers = keys.options().dtype(torch::kInt32);
  if (!n)
    return {torch::empty({0}, keys.options().dtype(torch::kInt64)),
            torch::zeros({1}, integers),
            std::vector<int64_t>(B, 0)};
  auto stream = at::cuda::getCurrentCUDAStream();
  if(packed_jobs && !drop_bits && B<=4) {
    int total=B*tiles;
    auto jobs=torch::empty({total},keys.options().dtype(torch::kInt64));
    auto offsets=torch::empty({total+1},integers);
    dense_shared_ranges<<<(total+256)/256,256,0,stream>>>(
        keys.data_ptr<Key>(),n,tiles,shift,total,jobs.data_ptr<int64_t>(),offsets.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {jobs,offsets,std::vector<int64_t>(B,tiles)};
  }
  Tensor input = keys;
  if (drop_bits) {
    input = torch::empty_like(keys);
    shift_keys<<<(n + 255) / 256, 256, 0, stream>>>(
        keys.data_ptr<int64_t>(), input.data_ptr<int64_t>(), n, drop_bits);
  }
  auto jobs = torch::empty_like(keys), lengths = torch::empty({n}, integers),
       runs = torch::empty({1}, integers);
  size_t bytes = 0;
  cub::DeviceRunLengthEncode::Encode(
      nullptr, bytes, input.data_ptr<Key>(), jobs.data_ptr<Key>(),
      lengths.data_ptr<int>(), runs.data_ptr<int>(), n, stream);
  auto temporary =
      torch::empty({(int64_t)bytes}, keys.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
      temporary.data_ptr(), bytes, input.data_ptr<Key>(),
      jobs.data_ptr<Key>(), lengths.data_ptr<int>(), runs.data_ptr<int>(),
      n, stream));
  int nruns;
  C10_CUDA_CHECK(cudaMemcpyAsync(&nruns, runs.data_ptr<int>(), sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  jobs = jobs.narrow(0, 0, nruns).to(torch::kInt64);
  auto offsets = torch::empty({nruns + 1}, integers);
  bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, bytes, lengths.data_ptr<int>(),
                                offsets.data_ptr<int>(), nruns, stream);
  temporary = torch::empty({(int64_t)bytes}, temporary.options());
  C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      temporary.data_ptr(), bytes, lengths.data_ptr<int>(),
      offsets.data_ptr<int>(), nruns, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(offsets.data_ptr<int>() + nruns, &n,
                                 sizeof(int), cudaMemcpyHostToDevice, stream));
  auto begins = torch::empty({B + 1}, integers);
  // Shared masked composition reuses the RLE batch/tile encoding directly.
  if (!packed_jobs)
    linear_jobs<<<(nruns + 255) / 256, 256, 0, stream>>>(jobs.data_ptr<int64_t>(),
                                                       nruns, shift, tiles);
  job_boundaries<<<(B + 256) / 256, 256, 0, stream>>>(
      jobs.data_ptr<int64_t>(), begins.data_ptr<int>(), nruns, B,
      packed_jobs ? (1 << shift) : W * tiles);
  std::vector<int> host_begins(B + 1);
  C10_CUDA_CHECK(cudaMemcpyAsync(host_begins.data(), begins.data_ptr<int>(),
                                 (B + 1) * sizeof(int), cudaMemcpyDeviceToHost,
                                 stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<int64_t> counts(B);
  for (int bi = 0; bi < B; ++bi)
    counts[bi] = host_begins[bi + 1] - host_begins[bi];
  return {jobs.narrow(0, 0, nruns), offsets, counts};
}
std::tuple<Tensor, Tensor, std::vector<int64_t>>
tile_ranges(Tensor keys, int W, int B, int tiles, int drop_bits, bool packed_jobs) {
  TORCH_CHECK(!packed_jobs || W == 1,
              "packed jobs require one shared tile list");
  if (keys.scalar_type() == torch::kInt32)
    return tile_ranges_impl<int>(keys, W, B, tiles, drop_bits, packed_jobs);
  return tile_ranges_impl<int64_t>(keys, W, B, tiles, drop_bits, packed_jobs);
}

// Minimum conic exponent over a pixel rectangle. The unconstrained minimum is
// the Gaussian center; otherwise a convex quadratic reaches its minimum on
// a rectangle edge. All four edges are needed for rotated ellipses.
__device__ float rectangle_sigma(float2 p, float3 a, float xmin, float xmax,
                                 float ymin, float ymax) {
  float lx = xmin - p.x, hx = xmax - p.x;
  float ly = ymin - p.y, hy = ymax - p.y;
  if (lx <= 0 && hx >= 0 && ly <= 0 && hy >= 0)
    return 0;
  float result = CUDART_INF_F;
  // Reuse each edge slope; the error allowance below also covers its FP32
  // reciprocal/multiply rounding before the edge point is clamped.
  float xy_ratio = __fdividef(-a.y, a.z), yx_ratio = __fdividef(-a.y, a.x);
  for (int side = 0; side < 2; ++side) {
    float x = side ? hx : lx;
    float y = fminf(hy, fmaxf(ly, xy_ratio * x));
    result = fminf(result, .5f * (a.x * x * x + a.z * y * y) + a.y * x * y);
  }
  for (int side = 0; side < 2; ++side) {
    float y = side ? hy : ly;
    float x = fminf(hx, fmaxf(lx, yx_ratio * y));
    result = fminf(result, .5f * (a.x * x * x + a.z * y * y) + a.y * x * y);
  }
  // Rasterization evaluates the quadratic in float. Nearly singular conics
  // can cancel large terms, so account for their float error before rejecting
  // a rectangle whose mathematically exact minimum exceeds the threshold.
  float mx = fmaxf(fabsf(lx), fabsf(hx));
  float my = fmaxf(fabsf(ly), fabsf(hy));
  float error = 2e-6f * (fabsf(a.x) * mx * mx + fabsf(a.z) * my * my +
                         2 * fabsf(a.y) * mx * my);
  return fmaxf(0, result - error);
}

// Real SH basis through degree three, evaluated for the native RGB layout.
template <bool Streaming = false, bool IndexedSource = false, bool Compact = false,
          bool Temporal = false>
__global__ void batch_colors_kernel(const float3 *means, const int2 *radii,
                                    const float *sh, const float *camera,
                                    const int64_t *order, int start, int count,
                                    int N, int W, float *colors,
                                    const float2 *xy, const float3 *conics,
                                    const float *opacity, float *packed,
                                    int stride, const int64_t *source_ids,
                                    int source_count, const int *layout,
                                    const float *frames, const float *temporal_mu) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * 3)
    return;
  int rank = start + i / 3, channel = i % 3;
  int g = order ? order[rank] : rank;
  if (radii) {
    bool visible = false;
    for (int f = 0; f < W; ++f)
      visible |= radii[f * N + g].x > 0 && radii[f * N + g].y > 0;
    if (!visible) {
      for (int f = 0; f < W; ++f)
        colors[3 * (f * N + g) + channel] = .5f;
      return;
    }
  }
  constexpr int Coefficients = Temporal ? 48 : 16;
  float coefficients[Coefficients];
  int source = IndexedSource ? source_ids[g] : g;
#pragma unroll
  for (int k = 0; k < Coefficients; ++k)
    coefficients[k] = Streaming ? __ldcs(sh + source * Coefficients * 3 + k * 3 + channel)
                                : sh[source * Coefficients * 3 + k * 3 + channel];
  for (int f = 0; f < W; ++f) {
    int j = f * N + g;
    if (radii && (radii[j].x <= 0 || radii[j].y <= 0)) {
      colors[3 * j + channel] = .5f;
      continue;
    }
    int mean_id = Temporal ? source : (IndexedSource ? f * source_count + source : j);
    float3 p = Streaming ? make_float3(__ldcs(&means[mean_id].x), __ldcs(&means[mean_id].y), __ldcs(&means[mean_id].z)) : means[mean_id];
    float x = p.x - camera[0], y = p.y - camera[1], z = p.z - camera[2];
    float inv = rsqrtf(x * x + y * y + z * z);
    x *= inv;
    y *= inv;
    z *= inv;
    float xx = x * x, yy = y * y, zz = z * z;
    float basis[16] = {.28209479177387814f,
                       -.4886025119029199f * y,
                       .4886025119029199f * z,
                       -.4886025119029199f * x,
                       1.0925484305920792f * x * y,
                       -1.0925484305920792f * y * z,
                       .31539156525252005f * (2 * zz - xx - yy),
                       -1.0925484305920792f * x * z,
                       .5462742152960396f * (xx - yy),
                       -.5900435899266435f * y * (3 * xx - yy),
                       2.890611442640554f * x * y * z,
                       -.4570457994644658f * y * (4 * zz - xx - yy),
                       .3731763325901154f * z * (2 * zz - 3 * xx - 3 * yy),
                       -.4570457994644658f * x * (4 * zz - xx - yy),
                       1.445305721320277f * z * (xx - yy),
                       -.5900435899266435f * x * (xx - 3 * yy)};
    float value = 0;
#pragma unroll
    for (int k = 0; k < 16; ++k)
      value += basis[k] * coefficients[k];
    if (Temporal) {
      float first = 0, second = 0;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        first += basis[k]*coefficients[16+k];
        second += basis[k]*coefficients[32+k];
      }
      float phase = 2*CUDART_PI_F*(temporal_mu[source]-frames[f]/30.0f)/10.0f;
      value += cosf(phase)*first;
      value += cosf(2*phase)*second;
    }
    if (packed) {
      if (Compact) {
        unsigned descriptor = layout[rank];
        bool compact = descriptor >> 31;
        int base = (descriptor >> 28) & 7;
        int cstart = layout[N + 1 + start];
        int compact_count = layout[N + 1 + start + count] - cstart;
        int ci = (descriptor & 0x0fffffffu) - cstart;
        int local = rank - start;
        int ui = local - ci;
        // Each frame has a contiguous delta or raw region. Colors follow both.
        int compact_words = (3 * W + 3) * compact_count;
        int geometry_words = compact_words + (count - compact_count) * 6 * W;
        if (!i && !f)
          ((int4 *)packed)[0] = make_int4(compact_count, count, cstart, 0);
        float *dest = packed + 4;
        __stcs(dest + geometry_words + (f * count + local) * 3 + channel, fmaxf(0, value + .5f));
        if (!channel) {
          for (int field = 0; field < 6; ++field) {
            int v = field_bits(xy, conics, opacity, j, field);
            if (!compact)
              __stcs(dest + compact_words + (f * (count - compact_count) + ui) * 6 + field, __int_as_float(v));
            else {
              int b = field_bits(xy, conics, opacity, base * N + g, field);
              if (!f)
                __stcs(dest + ci * 6 + field, __int_as_float(b));
              else
                ((short *)(dest + 6 * compact_count))[(f - 1) * compact_count * 6 + ci * 6 + field] =
                    (short)(int64_t(v) - b);
            }
          }
        }
      } else if (Streaming) {
        int dest = (f * stride + i / 3) * 9;
        __stcs(packed + dest + 6 + channel, fmaxf(0, value + .5f));
        if (!channel) {
          float2 p = __ldcs(xy + j);
          __stcs(packed + dest, p.x);
          __stcs(packed + dest + 1, p.y);
          __stcs(packed + dest + 2, __ldcs(opacity + j));
          __stcs(packed + dest + 3, __ldcs(&conics[j].x));
          __stcs(packed + dest + 4, __ldcs(&conics[j].y));
          __stcs(packed + dest + 5, __ldcs(&conics[j].z));
        }
      } else {
        int dest = (f * stride + i / 3) * 9;
        packed[dest + 6 + channel] = fmaxf(0, value + .5f);
        if (!channel) {
          float2 p = xy[j];
          float3 c = conics[j];
          packed[dest] = p.x;
          packed[dest + 1] = p.y;
          packed[dest + 2] = opacity[j];
          packed[dest + 3] = c.x;
          packed[dest + 4] = c.y;
          packed[dest + 5] = c.z;
        }
      }
    } else {
      colors[3 * j + channel] = fmaxf(0, value + .5f);
    }
  }
}
// Pack RGB prepared once for the complete ready-window into each span.
template <bool Compact = false>
__global__ void pack_projected_kernel(const float2 *xy, const float3 *conics,
    const float *opacity, const float3 *colors, float *packed, int start,
    int count, int N, int W, int stride, const int64_t *order, const int *layout) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  int rank = start + i, g = order ? order[rank] : rank;
  for (int f = 0; f < W; ++f) {
    int j = f * N + g;
    float3 color = make_float3(__ldcs(&colors[j].x), __ldcs(&colors[j].y), __ldcs(&colors[j].z));
    if (Compact) {
      unsigned descriptor = layout[g];
      bool compact = descriptor >> 31;
      int base = (descriptor >> 28) & 7;
      int cstart = layout[N + 1 + start];
      int nc = layout[N + 1 + start + count] - cstart;
      int ci = (descriptor & 0x0fffffffu) - cstart, ui = i - ci;
      int geometry_words = (3 * W + 3) * nc + (count - nc) * 6 * W;
      if (!i && !f)
        ((int4 *)packed)[0] = make_int4(nc, count, cstart, 0);
      float *dest = packed + 4;
      int color_offset = geometry_words + (f * count + i) * 3;
      __stcs(dest + color_offset, color.x);
      __stcs(dest + color_offset + 1, color.y);
      __stcs(dest + color_offset + 2, color.z);
#pragma unroll
      for (int field = 0; field < 6; ++field) {
        int v = field_bits(xy, conics, opacity, j, field);
        if (!compact)
          __stcs(dest + (3 * W + 3) * nc + (f * (count - nc) + ui) * 6 + field, __int_as_float(v));
        else {
          int b = field_bits(xy, conics, opacity, base * N + g, field);
          if (!f)
            __stcs(dest + ci * 6 + field, __int_as_float(b));
          else
            ((short *)(dest + 6 * nc))[(f - 1) * nc * 6 + ci * 6 + field] =
                (short)(int64_t(v) - b);
        }
      }
    } else {
      int dest = (f * stride + i) * 9;
      float2 p = __ldcs(xy + j);
      __stcs(packed + dest, p.x);
      __stcs(packed + dest + 1, p.y);
      __stcs(packed + dest + 2, __ldcs(opacity + j));
      __stcs(packed + dest + 3, __ldcs(&conics[j].x));
      __stcs(packed + dest + 4, __ldcs(&conics[j].y));
      __stcs(packed + dest + 5, __ldcs(&conics[j].z));
      __stcs(packed + dest + 6, color.x);
      __stcs(packed + dest + 7, color.y);
      __stcs(packed + dest + 8, color.z);
    }
  }
}

Tensor batch_colors(Tensor means, Tensor radii, Tensor sh, Tensor camera) {
  c10::cuda::CUDAGuard guard(means.device());
  auto colors = torch::empty_like(means);
  int N = means.size(1), W = means.size(0);
  if (!N)
    return colors;
  batch_colors_kernel<false><<<(N * 3 + 255) / 256, 256, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
      (float3 *)means.data_ptr<float>(), (int2 *)radii.data_ptr<int>(),
      sh.data_ptr<float>(), camera.data_ptr<float>(), nullptr, 0, N, N, W,
      colors.data_ptr<float>(), nullptr, nullptr, nullptr, nullptr, 0, nullptr, 0, nullptr,
      nullptr, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return colors;
}

Tensor temporal_colors(Tensor xyz, Tensor radii, Tensor sh, Tensor camera,
                       Tensor frames, Tensor temporal_mu) {
  c10::cuda::CUDAGuard guard(xyz.device());
  int N = xyz.size(0), W = frames.numel();
  auto colors = torch::empty({W, N, 3}, xyz.options());
  if (N) batch_colors_kernel<false, false, false, true><<<(N*3+255)/256, 256, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      (float3 *)xyz.data_ptr<float>(), (int2 *)radii.data_ptr<int>(),
      sh.data_ptr<float>(), camera.data_ptr<float>(), nullptr, 0, N, N, W,
      colors.data_ptr<float>(), nullptr, nullptr, nullptr, nullptr, 0, nullptr, 0,
      nullptr, frames.data_ptr<float>(), temporal_mu.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return colors;
}

__global__ void inverse_order(const int64_t *order, int *inverse, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    inverse[order[i]] = i;
}

template <bool WarpCull, bool PackedState = false>
__global__ void
batch_kernel(const float2 *xy, const float3 *conic, const float3 *colors,
             const float *opacity, const int *offsets, const int *ids, int n,
             int W, int width, int height, int tw, int th, int bi, int B,
             const int64_t *jobs, int job_start, float3 *image, float *trans,
             bool *stopped, bool *tile_stopped, int N, int id_bits,
             const float3 *packed, const int *inverse, int pack_start,
             int pack_stride, float stop_threshold, float4 *state, bool, float) {
  int job_index = (id_bits ? blockIdx.x / W : blockIdx.x) + job_start;
  int job = jobs[job_index],
      f = id_bits ? blockIdx.x % W : (job / (tw * th)) % W,
      tile = job % (tw * th);
  if (tile_stopped[f * tw * th + tile])
    return;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int wx = (tile % tw) * 16 + (warp % 2) * 8;
  int wy = (tile / tw) * 16 + (warp / 2) * 4;
  int x = WarpCull ? wx + lane % 8
                                  : (tile % tw) * 16 + threadIdx.x % 16,
      y = WarpCull ? wy + lane / 8
                                  : (tile / tw) * 16 + threadIdx.x / 16,
      p = (f * height + y) * width + x;
  bool inside = x < width && y < height;
  float4 pixel = make_float4(0, 0, 0, -1);
  if (PackedState && inside)
    pixel = state[p];
  bool already_done = !inside || (PackedState ? pixel.w < 0 : stopped[p]);
  bool done = already_done;
  bool initialized = PackedState || !WarpCull, dirty = false;
  float T = PackedState ? fabsf(pixel.w)
                        : (!WarpCull && !already_done ? trans[p] : 0);
  float3 C = PackedState
                 ? make_float3(pixel.x, pixel.y, pixel.z)
                 : (!WarpCull && !already_done ? image[p]
                                              : make_float3(0, 0, 0));
  __shared__ float2 sxy[256];
  __shared__ float3 sco[256], scol[256];
  __shared__ float sop[256];
  {
    int lo = offsets[job_index];
    int hi = offsets[job_index + 1];
    for (int base = lo; base < hi; base += 256) {
      if (__syncthreads_count(done) == 256)
        break;
      int j = base + threadIdx.x;
      if (j < hi) {
        int g = ids[j];
        if (id_bits) {
          unsigned encoded = g;
          bool visible = ((encoded >> id_bits) & (1 << f)) != 0;
          g = (encoded & ((1u << id_bits) - 1)) + f * N;
          if (!visible) {
            sxy[threadIdx.x] = make_float2(0, 0);
            sco[threadIdx.x] = scol[threadIdx.x] = make_float3(0, 0, 0);
            sop[threadIdx.x] = 0;
          } else {
            if (packed) {
              int local = inverse[g - f * N] - pack_start + f * pack_stride;
              float3 p = packed[3 * local];
              sxy[threadIdx.x] = make_float2(p.x, p.y);
              sop[threadIdx.x] = p.z;
              sco[threadIdx.x] = packed[3 * local + 1];
              scol[threadIdx.x] = packed[3 * local + 2];
            } else {
              sxy[threadIdx.x] = xy[g];
              sco[threadIdx.x] = conic[g];
              scol[threadIdx.x] = colors[g];
              sop[threadIdx.x] = opacity[g];
            }
          }
        } else {
          sxy[threadIdx.x] = xy[g];
          sco[threadIdx.x] = conic[g];
          scol[threadIdx.x] = colors[g];
          sop[threadIdx.x] = opacity[g];
        }
      }
      __syncthreads();
      int total = min(256, hi - base);
      for (int group = 0; group < (WarpCull ? total : 1); group += 32) {
        unsigned mask = 0xffffffff;
        if (WarpCull) {
          int candidate = group + lane;
          bool intersects = false;
          if (candidate < total) {
            float threshold = __logf(255.f * sop[candidate]);
            float lower =
                rectangle_sigma(sxy[candidate], sco[candidate], wx + .5f,
                                fminf(wx + 7.5f, width - .5f), wy + .5f,
                                fminf(wy + 3.5f, height - .5f));
            // A small outward tolerance keeps the cull conservative at the
            // alpha threshold under float evaluation.
            intersects = lower <= threshold + 1e-4f;
          }
          mask = __ballot_sync(0xffffffff, intersects);
          if (__all_sync(0xffffffff, done))
            break;
          if (mask && !initialized) {
            if (!already_done) {
              T = trans[p];
              C = image[p];
            }
            initialized = true;
          }
        }
        for (int plain = 0; WarpCull ? mask != 0 : plain < total; ++plain) {
          int k = WarpCull ? group + __ffs(mask) - 1 : plain;
          if (WarpCull)
            mask &= mask - 1;
          if (done)
            continue;
          float dx = sxy[k].x - (x + .5f), dy = sxy[k].y - (y + .5f);
          float3 a = sco[k];
          float sigma = .5f * (a.x * dx * dx + a.z * dy * dy) + a.y * dx * dy;
          if (sigma < 0)
            continue;
          float alpha = fminf(.999f, sop[k] * __expf(-sigma));
          if (alpha < 1.f / 255.f)
            continue;
          if (!initialized) {
            T = trans[p];
            C = image[p];
            initialized = true;
          }
          float next = T * (1 - alpha);
          if (stop_threshold > 0 && next <= stop_threshold) {
            done = true;
            dirty = true;
            break;
          }
          float w = alpha * T;
          float3 col = scol[k];
          C.x += col.x * w;
          C.y += col.y * w;
          C.z += col.z * w;
          T = next;
          dirty = true;
        }
      }
      __syncthreads();
    }
  }
  if (__syncthreads_count(done) == 256 && threadIdx.x == 0)
    tile_stopped[f * tw * th + tile] = true;
  if (!already_done && dirty) {
    if (PackedState)
      state[p] = make_float4(C.x, C.y, C.z, done ? -T : T);
    else {
      image[p] = C;
      trans[p] = T;
      stopped[p] = done;
    }
  }
}
constexpr int kDataflowPixelsPerThread = 2;
// Several 8x4 regions per warp share candidate attributes while retaining separate
// FP32 pixel accumulators and stop decisions. The single-frame path is unchanged.
template <bool OrderedAttributes = false, bool Compact = false>
__global__ void batch_multi_pixel_kernel(
    const float2 *xy, const float3 *conic, const float3 *colors,
    const float *opacity, const int *offsets, const int *ids, int n, int W,
    int width, int height, int tw, int th, int bi, int B, const int64_t *jobs,
    int job_start, float3 *image, float *trans, bool *stopped,
    bool *tile_stopped, int N, int id_bits, const float3 *packed,
    const int *inverse, int pack_start, int pack_stride, float stop_threshold,
    float4 *state, bool padded_output, float background) {
  constexpr int Pixels = kDataflowPixelsPerThread, Threads = 64;
  int sub = blockIdx.x % 2, item = blockIdx.x / 2;
  int job_index = item / W + job_start;
  uint64_t job = jobs[job_index];
  int f = item % W, tile_x = unsigned(job), tile_y = (job >> 32) & 0x7fffffffu;
  int tile = tile_y * tw + tile_x;
  int control = (f * tw * th + tile) * 2 + sub;
  if (tile_stopped[control])
    return;
  // The second bitmap tracks initialization separately from termination.
  bool first_touch = !tile_stopped[2 * W * tw * th + control];
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int x = tile_x * 16 + warp * 8 + lane % 8;
  int y = tile_y * 16 + sub * 8 + lane / 8;
  int region = sub * 4 + warp;
  int point[Pixels];
  constexpr unsigned AllPixels = 5;
  unsigned done_bits = 0, dirty_bits = 0;
  float T[Pixels];
  float3 C[Pixels];
#pragma unroll
  for (int q = 0; q < Pixels; ++q) {
    point[q] = (tile * W + f) * 256 + (y % 16 + q * 4) * 16 + x % 16;
    bool inside = x < width && y + q * 4 < height;
    float4 a = inside ? (first_touch ? make_float4(0, 0, 0, 1)
                                    : __ldcs(state + point[q]))
                      : make_float4(0, 0, 0, -1);
    if (!inside || a.w < 0)
      done_bits |= 1u << (q * 2);
    T[q] = fabsf(a.w);
    C[q] = make_float3(a.x, a.y, a.z);
  }
  unsigned already_done = done_bits;
  struct Candidate { float4 geometry, color, extra; };
  __shared__ Candidate staged[Threads];
  int hi = offsets[job_index + 1];
  for (int base = offsets[job_index]; base < hi; base += Threads) {
    if (__syncthreads_count(done_bits == AllPixels) == Threads)
      break;
    int j = base + threadIdx.x;
    if (j < hi) {
      unsigned encoded = ids[j];
      unsigned region_mask = encoded >> (id_bits + W);
      float2 local_xy = make_float2(0, 0);
      float3 local_conic = make_float3(0, 0, 0), local_color = make_float3(0, 0, 0);
      float local_op = 0;
      int g = encoded & ((1u << id_bits) - 1);
      if (!((encoded >> id_bits) & (1u << f)) ||
          !(region_mask & (15u << (sub * 4)))) {
        // The warp mask below excludes this candidate; its attributes and
        // cutoff are therefore never consumed by the compositor.
        local_op = 0;
      } else if (packed) {
        int rank = OrderedAttributes ? g : inverse[g];
        if (Compact) {
          unsigned descriptor = __ldg(inverse + rank);
          bool compact = descriptor >> 31;
          int4 header = __ldg((const int4 *)packed);
          int nc = header.x, count = header.y;
          int ci = (descriptor & 0x0fffffffu) - header.z;
          int local = rank - pack_start, ui = local - ci;
          const float *record = (const float *)packed + 4;
          int compact_words = (3 * W + 3) * nc;
          int geometry_words = compact_words + (count - nc) * 6 * W;
          float values[6];
#pragma unroll
          for (int field = 0; field < 6; ++field) {
            int bits = __float_as_int(__ldg(record + (compact ? ci * 6 + field : compact_words + (f * (count - nc) + ui) * 6 + field)));
            if (compact && f)
              bits = unsigned(bits) + ((const short *)(record + nc * 6))[(f - 1) * nc * 6 + ci * 6 + field];
            values[field] = __int_as_float(bits);
          }
          local_xy = make_float2(values[0], values[1]);
          local_op = values[2];
          local_conic = make_float3(values[3], values[4], values[5]);
          int color = geometry_words + (f * count + local) * 3;
          local_color = make_float3(__ldg(record + color),
                                          __ldg(record + color + 1),
                                          __ldg(record + color + 2));
        } else {
          int local = rank - pack_start + f * pack_stride;
          const float *record = (const float *)packed;
          // Three float3 fields occupy nine scalar floats per record.
          float3 a = make_float3(__ldg(record + 9 * local),
                                 __ldg(record + 9 * local + 1),
                                 __ldg(record + 9 * local + 2));
          local_xy = make_float2(a.x, a.y);
          local_op = a.z;
          local_conic = packed[3 * local + 1];
          local_color = packed[3 * local + 2];
        }
      } else {
        g += f * N;
        local_xy = xy[g];
        local_conic = conic[g];
        local_color = colors[g];
        local_op = opacity[g];
      }
      // Stage log2 opacity and base-two quadratic coefficients per candidate.
      staged[threadIdx.x] = {
          make_float4(local_xy.x, local_xy.y, __fmul_rn(-.7213475204444817f, local_conic.x),
                      __fmul_rn(-1.4426950408889634f, local_conic.y)),
          make_float4(__fmul_rn(-.7213475204444817f, local_conic.z), local_op, local_color.x, local_color.y),
          make_float4(local_color.z, __log2f(local_op),
                      __uint_as_float(local_op > 0 ? region_mask : 0), 0)};
    }
    __syncthreads();
    int total = min(Threads, hi - base);
    for (int group = 0; group < total; group += 32) {
      int candidate = group + lane;
      unsigned region_bits = AllPixels << region;
      bool intersects = candidate < total &&
                        (__float_as_uint(staged[candidate].extra.z) & region_bits);
      unsigned mask = __ballot_sync(0xffffffff, intersects);
      if (__all_sync(0xffffffff, done_bits == AllPixels))
        break;
      while (mask) {
        int k = group + __ffs(mask) - 1;
        mask &= mask - 1;
        Candidate value = staged[k];
        unsigned active = ((__float_as_uint(value.extra.z) >> region) & AllPixels) & ~done_bits;
        float2 p = make_float2(value.geometry.x, value.geometry.y);
        float3 a = make_float3(value.geometry.z, value.geometry.w, value.color.x);
        float3 col = make_float3(value.color.z, value.color.w, value.extra.x);
        float log_opacity = value.extra.y;
        float dx = p.x - (x + .5f);
        float xhalf = __fmaf_rn(__fmul_rn(a.x, dx), dx, log_opacity);
        float yhalf = a.z;
        float cross = __fmul_rn(a.y, dx);
#pragma unroll
        for (int q = 0; q < Pixels; ++q) {
          if (!(active & (1u << (q * 2))))
            continue;
          float dy = p.y - (y + q * 4 + .5f);
          // Share the x contribution across the thread's two pixel rows.
          float sigma = __fmaf_rn(dy, __fmaf_rn(dy, yhalf, cross), xhalf);
          float exponent;
          asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponent) : "f"(sigma));
          float alpha = fminf(.999f, exponent);
          if (sigma > log_opacity)
            continue;
          if (alpha < 1.f / 255.f)
            continue;
          float next = __fmaf_rn(-alpha, T[q], T[q]);
          dirty_bits |= 1u << (q * 2);
          if (stop_threshold > 0 && next <= stop_threshold) {
            done_bits |= 1u << (q * 2);
            continue;
          }
          float w = alpha * T[q];
          C[q].x += col.x * w;
          C[q].y += col.y * w;
          C[q].z += col.z * w;
          T[q] = next;
        }
      }
    }
    __syncthreads();
  }
  bool finished = __syncthreads_count(done_bits == AllPixels) == Threads ||
                  (job >> 63);
  if (finished && threadIdx.x == 0)
    tile_stopped[control] = true;
#pragma unroll
  for (int q = 0; q < Pixels; ++q) {
    if (finished) {
      if (x < width && y + q * 4 < height) {
        int pixel = (f * height + y + q * 4) * width + x;
        float background_T = background*T[q];
        if (padded_output)
          __stcs((float4 *)image + pixel,
                 make_float4(C[q].x + background_T, C[q].y + background_T, C[q].z + background_T, 0));
        else
          image[pixel] = make_float3(C[q].x + background_T, C[q].y + background_T, C[q].z + background_T);
      }
    } else if (!(already_done & (1u << (q * 2))) &&
        ((dirty_bits & (1u << (q * 2))) || first_touch))
      __stcs(state + point[q], make_float4(C[q].x, C[q].y, C[q].z,
                                         done_bits & (1u << (q * 2)) ? -T[q] : T[q]));
  }
  if (first_touch && threadIdx.x == 0)
    tile_stopped[2 * W * tw * th + control] = true;
}
__global__ void initialize_pixels(float4 *state, int n) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < n)
    state[p] = make_float4(0, 0, 0, 1);
}
__global__ void last_tile_jobs(const int64_t *jobs, int n, int tile_mask, int *last) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    atomicMax(last + (jobs[i] & tile_mask), i);
}
__global__ void flag_last_tile_jobs(const int64_t *jobs, int n, int tile_mask,
                                    int tw, const int *last, int64_t *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int tile = jobs[i] & tile_mask;
    output[i] = uint64_t(unsigned(tile % tw)) |
                (uint64_t(unsigned(tile / tw)) << 32) |
                (uint64_t(last[tile] == i) << 63);
  }
}
__global__ void resolve_empty_pixels(float3 *image, int width, int height,
                                      int tw, bool padded_output, const int *last, float background) {
  int tile = blockIdx.x;
  if (last[tile] >= 0)
    return;
  int x = (tile % tw) * 16 + threadIdx.x % 16;
  int y = (tile / tw) * 16 + threadIdx.x / 16;
  if (x < width && y < height) {
    int pixel = (blockIdx.y * height + y) * width + x;
    if (padded_output)
      __stcs((float4 *)image + pixel, make_float4(background, background, background, 0));
    else
      image[pixel] = make_float3(background, background, background);
  }
}
__global__ void resolve_pixels(const float4 *state, float3 *image, int n, float background) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < n) {
    float4 a = state[p];
    float t = background*fabsf(a.w);
    image[p] = make_float3(a.x + t, a.y + t, a.z + t);
  }
}
int64_t raster_blocks(Tensor reference, bool masked) {
  c10::cuda::CUDAGuard guard(reference.device());
  int maximum = 0;
  if (masked) {
    for (auto kernel : {batch_multi_pixel_kernel<true, true>,
                        batch_multi_pixel_kernel<true>,
                        batch_multi_pixel_kernel<false>}) {
      int blocks;
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 64, 0));
      maximum = std::max(maximum, blocks);
    }
  } else {
    for (auto kernel : {batch_kernel<true, true>, batch_kernel<false, true>,
                        batch_kernel<true>, batch_kernel<false>}) {
      int blocks;
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 256, 0));
      maximum = std::max(maximum, blocks);
    }
  }
  return maximum;
}

Tensor batches(Tensor xy, Tensor c, Tensor col, Tensor op, Tensor offsets,
               Tensor ids, int width, int height, int B, Tensor jobs,
               std::vector<int64_t> counts, bool warp_cull, int id_bits,
               Tensor means, Tensor sh, Tensor camera, Tensor order,
               std::vector<int64_t> boundaries, bool ordered_attributes,
               Tensor source_ids, Tensor layout, Tensor output,
               float background, c10::optional<Tensor> frames,
               c10::optional<Tensor> temporal_mu) {
  c10::cuda::CUDAGuard guard(xy.device());
  int W = xy.size(0), tw = (width + 15) / 16, th = (height + 15) / 16;
  // Batch-major rendering revisits C/T frequently. Use one aligned load and
  // store; a negative T marks a stopped pixel without losing its magnitude.
  bool packed_state = W > 1 || id_bits > 0;
  bool masked = packed_state && id_bits > 0 && id_bits + W + 8 <= 32;
  // The caller may supply attributes already in the common ROL sequence.
  // Other callers retain the general primitive-ID to packed-rank mapping.
  bool direct_packed = masked && ordered_attributes && !boundaries.empty();
  // Aligned final stores expose an RGB view without a full-image resolve pass.
  bool padded_output = masked && (!output.numel() ||
      (output.dim() == 4 && output.size(3) == 4 && output.is_contiguous()));
  TORCH_CHECK(!output.numel() || (packed_state && output.is_contiguous() &&
              output.device() == xy.device() && output.scalar_type() == xy.scalar_type() &&
              output.sizes() == at::IntArrayRef({W, height, width, padded_output ? 4 : 3})),
              "window output must be a matching contiguous RGB or padded four-channel image");
  auto image = output.numel() ? (padded_output ? output.slice(3, 0, 3) : output)
               : padded_output ? torch::empty({W, height, width, 4}, xy.options()).slice(3, 0, 3)
               : packed_state ? torch::empty({W, height, width, 3}, xy.options())
                            : torch::zeros({W, height, width, 3}, xy.options());
  auto T = packed_state ? torch::empty({0}, xy.options())
                        : torch::ones({W, height, width}, xy.options());
  auto stop = torch::zeros({packed_state ? 0 : W * height * width},
                            xy.options().dtype(torch::kBool));
  auto tile_stop = torch::zeros({masked ? 4 * W : W, th, tw}, xy.options().dtype(torch::kBool));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (masked) {
    int tile_mask = 1;
    while (tile_mask <= tw * th) tile_mask <<= 1;
    --tile_mask;
    // Masked composition consumes packed batch/tile jobs. Reuse each word
    // for tile x/y and a last-visit sign bit, retaining the L2 footprint.
    auto last = torch::full({tw * th}, -1, ids.options());
    if (jobs.numel())
      last_tile_jobs<<<(jobs.numel() + 255) / 256, 256, 0, stream>>>(
          jobs.data_ptr<int64_t>(), jobs.numel(), tile_mask, last.data_ptr<int>());
    resolve_empty_pixels<<<dim3(tw * th, W), 256, 0, stream>>>(
        (float3 *)image.data_ptr<float>(), width, height, tw, padded_output, last.data_ptr<int>(), background);
    auto flagged_jobs = torch::empty_like(jobs);
    if (jobs.numel())
      flag_last_tile_jobs<<<(jobs.numel() + 255) / 256, 256, 0, stream>>>(
          jobs.data_ptr<int64_t>(), jobs.numel(), tile_mask, tw,
          last.data_ptr<int>(), flagged_jobs.data_ptr<int64_t>());
    jobs = flagged_jobs;
  }
  int state_pixels = W * (masked ? tw * th * 256 : height * width);
  auto state = packed_state
                   ? torch::empty({state_pixels, 4}, xy.options())
                   : torch::empty({0}, xy.options());
  if (packed_state && !masked)
    initialize_pixels<<<(state_pixels + 255) / 256, 256, 0, stream>>>(
        (float4 *)state.data_ptr<float>(), state_pixels);
  Tensor packed, inverse;
  int max_count = 0;
  TORCH_CHECK(!layout.numel() || direct_packed,
              "lossless layout requires ordered shared attribute batches");
  bool compact_layout = direct_packed && layout.numel();
  int slot_words = 0;
  auto prepare_stream = at::cuda::getStreamFromPool(false, xy.get_device());
  at::cuda::CUDAEvent inputs_ready, ready[2], consumed[2];
  if (!boundaries.empty() && xy.size(1)) {
    for (int bi = 0; bi < B; ++bi)
      max_count =
          std::max(max_count, (int)(boundaries[bi + 1] - boundaries[bi]));
    slot_words = W * max_count * 9;
    if (compact_layout)
      slot_words = (slot_words + 7) & ~3;
    packed = torch::empty({2, slot_words}, xy.options());
    if (!direct_packed) {
      inverse = torch::empty({xy.size(1)}, ids.options());
      inverse_order<<<(xy.size(1) + 255) / 256, 256, 0, stream>>>(
          order.data_ptr<int64_t>(), inverse.data_ptr<int>(), xy.size(1));
    }
    inputs_ready.record(stream);
    inputs_ready.block(prepare_stream);
  }
  int job_start = 0;
  for (int bi = 0; bi < B; ++bi) {
    if (counts[bi]) {
      if (!boundaries.empty()) {
        int begin = boundaries[bi], n = boundaries[bi + 1] - begin;
        int slot = bi % 2;
        if (bi >= 2)
          consumed[slot].block(prepare_stream);
        if (col.numel()) {
          auto prepare = compact_layout ? pack_projected_kernel<true>
                                        : pack_projected_kernel<false>;
          prepare<<<(n + 255) / 256, 256, 0, prepare_stream>>>(
              (float2 *)xy.data_ptr<float>(), (float3 *)c.data_ptr<float>(),
              op.data_ptr<float>(), (float3 *)col.data_ptr<float>(),
              packed.data_ptr<float>() + (int64_t)slot * slot_words,
              begin, n, xy.size(1), W, max_count, order.data_ptr<int64_t>(),
              compact_layout ? layout.data_ptr<int>() : nullptr);
        } else {
        auto prepare = compact_layout
                           ? (source_ids.numel() ? batch_colors_kernel<true, true, true>
                                                : batch_colors_kernel<true, false, true>)
                           : (source_ids.numel() ? batch_colors_kernel<true, true>
                                                : batch_colors_kernel<true>);
        if (frames.has_value()) {
          TORCH_CHECK(temporal_mu.has_value(), "temporal SH requires Gaussian time means");
          prepare = compact_layout
              ? (source_ids.numel() ? batch_colors_kernel<true, true, true, true>
                                   : batch_colors_kernel<true, false, true, true>)
              : (source_ids.numel() ? batch_colors_kernel<true, true, false, true>
                                   : batch_colors_kernel<true, false, false, true>);
        }
        prepare<<<(n * 3 + 255) / 256, 256, 0, prepare_stream>>>(
            (float3 *)means.data_ptr<float>(), nullptr, sh.data_ptr<float>(),
            camera.data_ptr<float>(), order.data_ptr<int64_t>(), begin, n,
            xy.size(1), W, nullptr, (float2 *)xy.data_ptr<float>(),
            (float3 *)c.data_ptr<float>(), op.data_ptr<float>(),
            packed.data_ptr<float>() + (int64_t)slot * slot_words,
            max_count, source_ids.numel() ? source_ids.data_ptr<int64_t>() : nullptr,
            means.size(1), compact_layout ? layout.data_ptr<int>() : nullptr,
            frames.has_value() ? frames->data_ptr<float>() : nullptr,
            temporal_mu.has_value() ? temporal_mu->data_ptr<float>() : nullptr);
        }
        ready[slot].record(prepare_stream);
        ready[slot].block(stream);
      }
      auto kernel = compact_layout ? batch_multi_pixel_kernel<true, true>
                  : masked ? (direct_packed ? batch_multi_pixel_kernel<true>
                                            : batch_multi_pixel_kernel<false>)
                           : (packed_state
                                  ? (warp_cull ? batch_kernel<true, true>
                                               : batch_kernel<false, true>)
                                  : (warp_cull ? batch_kernel<true> : batch_kernel<false>));
      kernel<<<counts[bi] * (id_bits ? W : 1) * (masked ? 2 : 1), masked ? 64 : 256, 0, stream>>>(
          (float2 *)xy.data_ptr<float>(), (float3 *)c.data_ptr<float>(),
          (float3 *)col.data_ptr<float>(), op.data_ptr<float>(),
          offsets.data_ptr<int>(), ids.data_ptr<int>(), ids.numel(), W, width,
          height, tw, th, bi, B, jobs.data_ptr<int64_t>(), job_start,
          (float3 *)image.data_ptr<float>(), T.data_ptr<float>(),
          stop.data_ptr<bool>(), tile_stop.data_ptr<bool>(), xy.size(1),
          id_bits,
          boundaries.empty()
              ? nullptr
              : (float3 *)(packed.data_ptr<float>() +
                           (int64_t)(bi % 2) * slot_words),
          compact_layout ? layout.data_ptr<int>() :
              (boundaries.empty() || direct_packed ? nullptr : inverse.data_ptr<int>()),
          boundaries.empty() ? 0 : boundaries[bi], max_count,
          .0001f,
          packed_state ? (float4 *)state.data_ptr<float>() : nullptr, padded_output, background);
      if (!boundaries.empty())
        consumed[bi % 2].record(stream);
    }
    job_start += counts[bi];
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (packed_state) {
    if (!masked)
      resolve_pixels<<<(W * height * width + 255) / 256, 256, 0, stream>>>(
          (float4 *)state.data_ptr<float>(), (float3 *)image.data_ptr<float>(),
          W * height * width, background);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return image;
  }
  return background == 1 ? image + T.unsqueeze(-1) : image + background*T.unsqueeze(-1);
}

__device__ bool merge_less(float a, int ai, float b, int bi) {
  return a < b || (a == b && ai < bi);
}

__global__ void stable_ranks_kernel(const int *stable, int ns, int *ranks) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ns) ranks[stable[i]] = i;
}
struct UnstablePosition {
  const int64_t *orders;
  const int *ranks;
  __device__ bool operator()(int i) const { return ranks[orders[i]] < 0; }
};
__device__ int gap_bound(const int *positions, int frame, int nu, int N,
                          int gap, bool inclusive) {
  int lo = 0, hi = nu;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    int value = positions[frame * nu + mid] % N - mid;
    if (value < gap || (inclusive && value == gap)) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
__global__ void shared_sequence_kernel(const int64_t *orders, const int *stable,
    const int *positions, int N, int W, int ns, int nu, int64_t *source, int *owner) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ns + W * nu) return;
  int gap, local = 0, frame = -1, g;
  if (i < ns) { gap = i; g = stable[i]; }
  else {
    int occurrence = i - ns;
    frame = occurrence / nu;
    local = occurrence % nu;
    gap = positions[occurrence] % N - local;
    g = orders[positions[occurrence]];
  }
  int dest = gap + local;
  for (int f = 0; f < W; ++f)
    if (f != frame)
      dest += gap_bound(positions, f, nu, N, gap, frame < 0 || f < frame);
  source[dest] = g;
  owner[dest] = frame;
}
std::vector<Tensor> shared_sequence(Tensor orders, Tensor stable) {
  c10::cuda::CUDAGuard guard(orders.device());
  int W = orders.size(0), N = orders.size(1), ns = stable.numel(), nu = N - ns;
  TORCH_CHECK(W > 1 && nu > 0 && stable.scalar_type() == torch::kInt32 &&
              orders.scalar_type() == torch::kInt64 && orders.is_contiguous(),
              "shared sequence requires contiguous frame ROLs and stable IDs");
  auto integers = stable.options();
  auto ranks = torch::full({N}, -1, integers);
  auto positions = torch::empty({W * nu}, integers), count = torch::empty({1}, integers);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (ns) stable_ranks_kernel<<<(ns+255)/256,256,0,stream>>>(
      stable.data_ptr<int>(), ns, ranks.data_ptr<int>());
  cub::CountingInputIterator<int> indices(0);
  using Flags = cub::TransformInputIterator<bool, UnstablePosition, cub::CountingInputIterator<int>>;
  Flags flags(indices, UnstablePosition{orders.data_ptr<int64_t>(),ranks.data_ptr<int>()});
  size_t bytes = 0;
  cub::DeviceSelect::Flagged(nullptr,bytes,indices,flags,positions.data_ptr<int>(),
                              count.data_ptr<int>(),W*N,stream);
  auto scratch = torch::empty({int64_t(bytes)}, integers.dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(scratch.data_ptr(),bytes,indices,flags,
      positions.data_ptr<int>(),count.data_ptr<int>(),W*N,stream));
  auto source = torch::empty({ns+W*nu},orders.options());
  auto owner = torch::empty({ns+W*nu},integers);
  shared_sequence_kernel<<<(source.numel()+255)/256,256,0,stream>>>(
      orders.data_ptr<int64_t>(),stable.data_ptr<int>(),positions.data_ptr<int>(),
      N,W,ns,nu,source.data_ptr<int64_t>(),owner.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {source,owner};
}
// Each thread owns four consecutive merge-path diagonals. Find its co-rank
// once, then merge its short disjoint segment without scatter writes.
__global__ void merge_kernel(const int *stable, const int *unstable,
                             const float *minimum, const float *depth, int ns,
                             int nu, int N, int W, int64_t *output) {
  int segments = (N + 3) / 4;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= W * segments)
    return;
  int f = i / segments, diagonal = (i % segments) * 4;
  const int *u = unstable + f * nu;
  int lo = max(0, diagonal - nu), hi = min(diagonal, ns);
  while (lo < hi) {
    int a = (lo + hi) / 2, b = diagonal - a;
    if (b > 0 && a < ns &&
        merge_less(minimum[stable[a]], stable[a], depth[f * N + u[b - 1]],
                   u[b - 1]))
      lo = a + 1;
    else
      hi = a;
  }
  int a = lo, b = diagonal - lo;
  for (int j = diagonal; j < min(diagonal + 4, N); ++j) {
    bool take_stable =
        b == nu || (a < ns && merge_less(minimum[stable[a]], stable[a],
                                         depth[f * N + u[b]], u[b]));
    output[f * N + j] = take_stable ? stable[a++] : u[b++];
  }
}
Tensor merge(Tensor stable, Tensor unstable, Tensor minimum, Tensor depth) {
  c10::cuda::CUDAGuard guard(depth.device());
  int W = depth.size(0), N = depth.size(1), ns = stable.numel(),
      nu = unstable.size(1);
  auto result = torch::empty({W, N}, depth.options().dtype(torch::kInt64));
  if (!N)
    return result;
  merge_kernel<<<(W * ((N + 3) / 4) + 255) / 256, 256, 0,
                 at::cuda::getCurrentCUDAStream()>>>(
      stable.data_ptr<int>(), unstable.data_ptr<int>(),
      minimum.data_ptr<float>(), depth.data_ptr<float>(), ns, nu, N, W,
      result.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shared_sequence", &shared_sequence);
  m.def("generate_anchored", &generate_anchored);
  m.def("merge", &merge);
  m.def("visible_order", &visible_order);
  m.def("generate", &generate);
  m.def("generate_4d", &generate_4d);
  m.def("prepare_generation", &prepare_generation);
  m.def("window_statistics", &window_statistics);
  m.def("classify", &classify);
  m.def("packed_layout", &packed_layout);
  m.def("ordered_tiles", &ordered_tiles);
  m.def("batches", &batches, pybind11::arg("xy"), pybind11::arg("c"),
        pybind11::arg("col"), pybind11::arg("op"), pybind11::arg("offsets"),
        pybind11::arg("ids"), pybind11::arg("width"), pybind11::arg("height"),
        pybind11::arg("B"), pybind11::arg("jobs"), pybind11::arg("counts"),
        pybind11::arg("warp_cull"), pybind11::arg("id_bits"), pybind11::arg("means"),
        pybind11::arg("sh"), pybind11::arg("camera"), pybind11::arg("order"),
        pybind11::arg("boundaries"), pybind11::arg("ordered_attributes"),
        pybind11::arg("source_ids"), pybind11::arg("layout"), pybind11::arg("output"),
        pybind11::arg("background") = 1.0f,
        pybind11::arg("frames") = pybind11::none(),
        pybind11::arg("temporal_mu") = pybind11::none());
  m.def("raster_blocks", &raster_blocks);
  m.def("tile_ranges", &tile_ranges, pybind11::arg("keys"), pybind11::arg("W"),
        pybind11::arg("B"), pybind11::arg("tiles"), pybind11::arg("drop_bits"),
        pybind11::arg("packed_jobs") = false);
  m.def("stable_batch_prefix", &stable_batch_prefix);
  m.def("tile_batch_prefix", &tile_batch_prefix);
  m.def("tile_batch_keys", &tile_batch_keys);
  m.def("bounded_batches", &bounded_batches);
  m.def("cull_tiles", &cull_tiles);
  m.def("batch_colors", &batch_colors);
  m.def("temporal_colors", &temporal_colors);
  m.def("projection_finish", &projection_finish);
  m.def("radix", &radix);
}
