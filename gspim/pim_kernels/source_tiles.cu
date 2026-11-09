#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
using torch::Tensor;
__global__ void ordered_tile_counts(const float2 *xy, const int2 *radii,
                                    const int64_t *order,
                                    int N, int S, int W, int tw, int th, int ts, int *counts,
                                    bool shared) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * (shared ? 1 : W))
    return;
  int f = i / N, g = f * S + order[i];
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
        float2 center = xy[frame * S + g];
        int2 radius = radii[frame * S + g];
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
                                   int N, int S, int W, int tw, int th, int ts,
                                   bool shared, Key *keys, int *ids) {
  constexpr int Lanes = 8;
  int lane = threadIdx.x % Lanes;
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / Lanes;
  if (i >= N * (shared ? 1 : W))
    return;
  int g = 0, image = 0, start = 0, xmin = 0, xmax = 0, ymin = 0, ymax = 0;
  if (!lane) {
    int f = i / N;
    g = f * S + order[i];
    image = shared ? assignment[i] : assignment[i] * W + f;
    start = scan[i];
    xmin = tw;
    ymin = th;
    for (int frame = 0; frame < (shared ? W : 1); ++frame) {
      int j = shared ? frame * S + g : g;
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
  int W = order.size(0), N = order.size(1), S = xy.size(1);
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
      order.data_ptr<int64_t>(), N, S, W, tw,
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
      scan.data_ptr<int>(), N, S, W, tw, th, ts, shared, keys.data_ptr<Key>(),
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



__global__ void dense_offsets(const int* keys,int* offsets,int64_t* jobs,int n,int tiles) {
  int tile=blockIdx.x*blockDim.x+threadIdx.x;if(tile>tiles)return;
  int lo=0,hi=n;
  while(lo<hi){int mid=lo+(hi-lo)/2;if(keys[mid]<tile)lo=mid+1;else hi=mid;}
  offsets[tile]=lo;if(tile<tiles)jobs[tile]=tile;
}
std::tuple<Tensor,Tensor,std::vector<int64_t>> tile_ranges(Tensor keys,int W,int B,int tiles,int drop,bool packed) {
  TORCH_CHECK(keys.scalar_type()==torch::kInt32&&W==1&&B==1&&drop==0&&packed,"shared dense tile path required");
  c10::cuda::CUDAGuard guard(keys.device());auto offsets=torch::empty({tiles+1},keys.options());
  auto jobs=torch::empty({tiles},keys.options().dtype(torch::kInt64));
  dense_offsets<<<(tiles+256)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(keys.data_ptr<int>(),
    offsets.data_ptr<int>(),jobs.data_ptr<int64_t>(),keys.numel(),tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {jobs,offsets,{tiles}};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("tile_ranges",&tile_ranges);m.def("ordered_tiles",&ordered_tiles);}
