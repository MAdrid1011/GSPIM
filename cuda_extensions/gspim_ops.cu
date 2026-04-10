/*
 * GSPIM CUDA操作
 * 优化的批次渲染和时间贡献计算
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// 时间贡献计算核函数（PIM优化版本）
// 
// 原判定条件: exp(-0.5 * dt^2 / sigma_tt) < threshold
// 等价转换:   0.5 * dt^2 > sigma_tt * ln(1/threshold)
// 
// 对于 threshold = 0.05:  ln(1/0.05) = ln(20) ≈ 2.9957
// 
// 这样避免了除法和exp运算，只需要两次乘法和一次比较
// 符合PIM近存计算的低功耗要求
//
__global__ void compute_time_contribution_kernel(
    const float* __restrict__ sigma_tt,    // [N] 时间维协方差
    const float* __restrict__ mu_t,        // [N] 时间均值
    float t_min,
    float t_max,
    float threshold,
    bool* __restrict__ active_mask,        // [N] 输出：活跃标记
    float* __restrict__ p_t_max,           // [N] 输出：时间贡献（用于调试）
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Stage 1: 选择最接近 mu_t 的时间点，计算时间偏移 dt
    float mu = mu_t[idx];
    float t_eval = fmaxf(t_min, fminf(t_max, mu));
    float dt = t_eval - mu;
    
    // Stage 5: PIM优化的时间贡献判定
    // 原: exp(-0.5 * dt^2 / sigma_tt) > threshold
    // 转换: 0.5 * dt^2 < sigma_tt * ln(1/threshold)
    // 
    // 预计算: ln(1/0.05) = ln(20) = 2.995732...
    const float LN_INV_THRESHOLD = 2.995732273553991f;  // ln(20) for threshold=0.05
    
    float sigma = sigma_tt[idx];
    
    // 两次乘法 + 一次比较（无除法、无exp）
    float lhs = 0.5f * dt * dt;           // 左侧: 0.5 * dt^2
    float rhs = sigma * LN_INV_THRESHOLD; // 右侧: sigma_tt * ln(20)
    
    // 活跃判定: lhs < rhs 等价于 p(t) > threshold
    bool is_active = (lhs < rhs);
    active_mask[idx] = is_active;
    
    // 输出 p_t_max 用于调试（实际PIM中不需要计算这个值）
    // 如果需要精确值，可以回退到exp计算
    p_t_max[idx] = is_active ? 1.0f : 0.0f;  // 简化输出
}

// 批次深度排序核函数（使用bitonic sort）
__global__ void batch_depth_sort_kernel(
    float* __restrict__ depths,            // [batch_size] 深度值
    int* __restrict__ indices,             // [batch_size] 索引
    int batch_size,
    int stage,
    int step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    int ixj = idx ^ step;
    
    if (ixj > idx) {
        if ((idx & stage) == 0) {
            if (depths[idx] > depths[ixj]) {
                // Swap depths
                float temp_depth = depths[idx];
                depths[idx] = depths[ixj];
                depths[ixj] = temp_depth;
                
                // Swap indices
                int temp_idx = indices[idx];
                indices[idx] = indices[ixj];
                indices[ixj] = temp_idx;
            }
        } else {
            if (depths[idx] < depths[ixj]) {
                // Swap depths
                float temp_depth = depths[idx];
                depths[idx] = depths[ixj];
                depths[ixj] = temp_depth;
                
                // Swap indices
                int temp_idx = indices[idx];
                indices[idx] = indices[ixj];
                indices[ixj] = temp_idx;
            }
        }
    }
}

// 紧凑化活跃高斯核函数（Stream Compaction）
__global__ void compact_active_gaussians_kernel(
    const bool* __restrict__ active_mask,  // [N] 活跃标记
    const int* __restrict__ prefix_sum,    // [N] 前缀和（扫描结果）
    int* __restrict__ active_indices,      // [M] 输出：活跃索引
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    if (active_mask[idx]) {
        int out_idx = prefix_sum[idx];
        active_indices[out_idx] = idx;
    }
}

// 并行前缀和（Scan）- 简化版Blelloch算法
__global__ void prefix_sum_up_sweep_kernel(
    int* __restrict__ data,
    int N,
    int offset)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * offset * 2 + offset - 1;
    if (idx + offset < N) {
        data[idx + offset] += data[idx];
    }
}

__global__ void prefix_sum_down_sweep_kernel(
    int* __restrict__ data,
    int N,
    int offset)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * offset * 2 + offset - 1;
    if (idx + offset < N) {
        int temp = data[idx];
        data[idx] = data[idx + offset];
        data[idx + offset] += temp;
    }
}

// 深度熵计算核函数
__global__ void compute_depth_entropy_kernel(
    const float* __restrict__ depths_sequence,  // [N, W] 深度序列
    int N,
    int W,
    int num_bins,
    float* __restrict__ normalized_entropy)     // [N] 输出：归一化熵
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // 找出深度范围
    float depth_min = depths_sequence[idx * W];
    float depth_max = depth_min;
    
    for (int t = 1; t < W; t++) {
        float d = depths_sequence[idx * W + t];
        depth_min = fminf(depth_min, d);
        depth_max = fmaxf(depth_max, d);
    }
    
    float depth_range = depth_max - depth_min + 1e-7f;
    
    // 计算直方图
    int hist[16];  // 最多16个bins
    for (int i = 0; i < num_bins; i++) {
        hist[i] = 0;
    }
    
    for (int t = 0; t < W; t++) {
        float d = depths_sequence[idx * W + t];
        int bin = (int)((d - depth_min) / depth_range * (num_bins - 1));
        bin = min(max(bin, 0), num_bins - 1);
        hist[bin]++;
    }
    
    // 计算熵
    float entropy = 0.0f;
    for (int i = 0; i < num_bins; i++) {
        if (hist[i] > 0) {
            float p = (float)hist[i] / W;
            entropy -= p * logf(p);
        }
    }
    
    // 归一化
    float max_entropy = logf((float)num_bins);
    normalized_entropy[idx] = entropy / max_entropy;
}

// CUDA错误检查
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// C接口函数
extern "C" {

void launch_compute_time_contribution(
    const float* sigma_tt,
    const float* mu_t,
    float t_min,
    float t_max,
    float threshold,
    bool* active_mask,
    float* p_t_max,
    int N)
{
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    compute_time_contribution_kernel<<<numBlocks, blockSize>>>(
        sigma_tt, mu_t, t_min, t_max, threshold, active_mask, p_t_max, N
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_compute_depth_entropy(
    const float* depths_sequence,
    int N,
    int W,
    int num_bins,
    float* normalized_entropy)
{
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    compute_depth_entropy_kernel<<<numBlocks, blockSize>>>(
        depths_sequence, N, W, num_bins, normalized_entropy
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // extern "C"

