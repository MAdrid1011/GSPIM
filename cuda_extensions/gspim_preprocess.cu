/*
 * GSPIM Multi-Frame Preprocessing CUDA Kernel
 * 
 * 融合 Stage 2-3：
 * - Stage 2: 完整的4D到3D投影
 *   1. 计算完整的4D协方差矩阵 Σ^(4)
 *   2. 条件化3D均值: μ_{xyz|t} = μ_{1:3} + Σ_{1:3,4} Σ_{4,4}^{-1} (t - μ_t)
 *   3. 条件化3D协方差: Σ_{xyz|t} = Σ_{1:3,1:3} - Σ_{1:3,4} Σ_{4,4}^{-1} Σ_{4,1:3}
 *   4. 多帧深度序列计算
 * - Stage 3: Depth Stability Analysis (相邻帧深度差异)
 * 
 * 实现说明：
 * - 使用GLM库进行矩阵运算（与原版4DGS一致）
 * - GLM高度优化，列主序存储，自动SIMD
 */

 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <torch/extension.h>
 #include <cmath>
 
 #define GLM_FORCE_CUDA
 #include <glm/glm.hpp>
 
 // ====================================================================
 // Stage 2: Multi-Frame 4D Projection & Depth Sequence Computation
 // ====================================================================
 
 // 4D协方差计算（使用GLM库，与原版4DGS完全一致）
 __device__ void compute_full_4d_covariance_glm(
     const glm::vec4& rot_l,    // rotation_l (w, x, y, z)
     const glm::vec4& rot_r,    // rotation_r (w, x, y, z)  
     const glm::vec4& scale,    // (sx, sy, sz, st)
     float mod,
     float& cov_t_out,          // Σ_{4,4}
     glm::vec3& cov_12_out,     // Σ_{1:3,4}
     float cov_3d_out[6])       // Σ_{1:3,1:3} 上三角存储
 {
     // 构建尺度矩阵 S（对角矩阵）
     glm::mat4 S = glm::mat4(1.0f);
     S[0][0] = mod * scale.x;
     S[1][1] = mod * scale.y;
     S[2][2] = mod * scale.z;
     S[3][3] = mod * scale.w;
     
     // 四元数 → 4D旋转矩阵（与原版4DGS完全一致）
     // 注意：PyTorch四元数顺序为 (w, x, y, z)
     // 对应到glm::vec4的 (.x, .y, .z, .w)
     float a = rot_l.x;  // w
     float b = rot_l.y;  // x
     float c = rot_l.z;  // y
     float d = rot_l.w;  // z
     
     float p = rot_r.x;  // w
     float q = rot_r.y;  // x
     float r = rot_r.z;  // y
     float s = rot_r.w;  // z
     
     // 左四元数矩阵（列主序，与原版完全一致）
     glm::mat4 M_l = glm::mat4(
          a,  b, -c,  d,
         -b,  a,  d,  c,
          c, -d,  a,  b,
         -d, -c, -b,  a
     );
     
     // 右四元数矩阵（列主序，与原版完全一致）
     glm::mat4 M_r = glm::mat4(
         p,  q, -r, -s,
        -q,  p,  s, -r,
         r, -s,  p, -q,
         s,  r,  q,  p
     );
     
     // GLM自动优化的矩阵乘法（编译器会优化）
     glm::mat4 R = M_r * M_l;
     glm::mat4 M = S * R;
     glm::mat4 Sigma = glm::transpose(M) * M;
     
     // 提取需要的协方差元素
     cov_t_out = Sigma[3][3];
     cov_12_out = glm::vec3(Sigma[0][3], Sigma[1][3], Sigma[2][3]);
     
     // 提取3D协方差（上三角）
     cov_3d_out[0] = Sigma[0][0];
     cov_3d_out[1] = Sigma[0][1];
     cov_3d_out[2] = Sigma[0][2];
     cov_3d_out[3] = Sigma[1][1];
     cov_3d_out[4] = Sigma[1][2];
     cov_3d_out[5] = Sigma[2][2];
 }
 
 __global__ void compute_4d_projection_kernel(
     const int M,              // 活跃高斯数量
     const int W,              // 窗口大小（帧数）
     const float* means3D,     // [M, 3] 4D均值的空间部分
     const float* mu_t,        // [M, 1] 4D均值的时间部分
     const float* scales_xyzt, // [M, 4]
     const float* rotation_l,  // [M, 4]
     const float* rotation_r,  // [M, 4]
     const float* timestamps,  // [W]
     const float* view_matrix, // [4, 4] camera world_view_transform
     float* depths_out,        // [M, W] 深度输出
     float* means3D_cond_out,  // [M, W, 3] 条件化3D均值输出
     float* cov3D_cond_out)    // [M, 6] 条件化3D协方差输出（上三角）
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= M) return;
     
     // 优化1: 使用shared memory缓存timestamps和view_matrix
     __shared__ float shared_timestamps[16];  // 最多支持16帧
     __shared__ float shared_view[16];
     
     if (threadIdx.x < W && W <= 16) {
         shared_timestamps[threadIdx.x] = timestamps[threadIdx.x];
     }
     if (threadIdx.x < 16) {
         shared_view[threadIdx.x] = view_matrix[threadIdx.x];
     }
     __syncthreads();
     
     // 优化2: 向量化内存访问（使用float4）
     // 4D参数可以用float4一次读取，减少内存事务
     const float4* scales_ptr = reinterpret_cast<const float4*>(scales_xyzt);
     const float4* rot_l_ptr = reinterpret_cast<const float4*>(rotation_l);
     const float4* rot_r_ptr = reinterpret_cast<const float4*>(rotation_r);
     
     float4 scale_vec = scales_ptr[idx];
     float4 rot_l_vec = rot_l_ptr[idx];
     float4 rot_r_vec = rot_r_ptr[idx];
     
     // 加载3D均值和时间（标量访问）
     float mean_x = means3D[idx * 3 + 0];
     float mean_y = means3D[idx * 3 + 1];
     float mean_z = means3D[idx * 3 + 2];
     float t_mean = mu_t[idx];
     
     // 提取尺度
     float sx = scale_vec.x;
     float sy = scale_vec.y;
     float sz = scale_vec.z;
     float st = scale_vec.w;
     
     // 提取四元数（PyTorch顺序：w, x, y, z）
     float lw = rot_l_vec.x;
     float lx = rot_l_vec.y;
     float ly = rot_l_vec.z;
     float lz = rot_l_vec.w;
     
     float rw = rot_r_vec.x;
     float rx = rot_r_vec.y;
     float ry = rot_r_vec.z;
     float rz = rot_r_vec.w;
     
     // 构建GLM向量（直接使用读取的float4）
     glm::vec4 rot_l_glm = glm::vec4(lw, lx, ly, lz);
     glm::vec4 rot_r_glm = glm::vec4(rw, rx, ry, rz);
     glm::vec4 scale_glm = glm::vec4(sx, sy, sz, st);
     
     // 计算完整的4D协方差矩阵（使用GLM，与原版一致）
     float cov_t;
     glm::vec3 cov_12;
     float cov_3d[6];
     
     compute_full_4d_covariance_glm(
         rot_l_glm, rot_r_glm, scale_glm,
         1.0f,  // scale_modifier
         cov_t, cov_12, cov_3d
     );
     
     // 数值稳定性
     cov_t = fmaxf(cov_t, 1e-7f);
     
     // 计算条件化3D协方差（与原版4DGS一致）
     // Σ_{xyz|t} = Σ_{1:3,1:3} - outerProduct(cov_12, cov_12) / cov_t
     glm::mat3 cov11 = glm::mat3(cov_3d[0], cov_3d[1], cov_3d[2],
                                  cov_3d[1], cov_3d[3], cov_3d[4],
                                  cov_3d[2], cov_3d[4], cov_3d[5]);
     glm::mat3 cov3D_condition = cov11 - glm::outerProduct(cov_12, cov_12) / cov_t;
     
     // 存储条件化协方差（上三角）
     cov3D_cond_out[idx * 6 + 0] = cov3D_condition[0][0];
     cov3D_cond_out[idx * 6 + 1] = cov3D_condition[0][1];
     cov3D_cond_out[idx * 6 + 2] = cov3D_condition[0][2];
     cov3D_cond_out[idx * 6 + 3] = cov3D_condition[1][1];
     cov3D_cond_out[idx * 6 + 4] = cov3D_condition[1][2];
     cov3D_cond_out[idx * 6 + 5] = cov3D_condition[2][2];
     
     // 计算条件化均值偏移
     glm::vec3 offset_coef = cov_12 / cov_t;
     
     float view_col[3];
     view_col[0] = shared_view[2];   // M[0][2]
     view_col[1] = shared_view[6];   // M[1][2]
     view_col[2] = shared_view[10];  // M[2][2]
     float view_offset = shared_view[14];  // M[3][2]
     
     #pragma unroll
     for (int t = 0; t < W; t++) {
         float timestamp_t = (W <= 16) ? shared_timestamps[t] : timestamps[t];
         float dt = timestamp_t - t_mean;
         
         glm::vec3 delta_mean = offset_coef * dt;
         glm::vec3 mean_cond = glm::vec3(mean_x, mean_y, mean_z) + delta_mean;
         
         int mean_offset = idx * W * 3 + t * 3;
         means3D_cond_out[mean_offset + 0] = mean_cond.x;
         means3D_cond_out[mean_offset + 1] = mean_cond.y;
         means3D_cond_out[mean_offset + 2] = mean_cond.z;
         
         float depth = mean_cond.x * view_col[0] + mean_cond.y * view_col[1] +
                       mean_cond.z * view_col[2] + view_offset;
         depths_out[idx * W + t] = depth;
     }
 }
 
 // ====================================================================
 // Stage 3: Depth Stability Analysis (相邻帧深度差异)
 // ====================================================================
 
 __global__ void compute_depth_stability_kernel(
     const int M,              // 高斯数量
     const int W,              // 窗口大小
     const float* depths,      // [M, W]
     float* stability_out,     // [M] 稳定性分数输出
     bool* stable_mask_out)    // [M] 稳定性标记（占位）
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= M) return;
     
     // 加载当前高斯的深度序列
     float depths_local[32];  // 假设W <= 32
     for (int t = 0; t < W; t++) {
         depths_local[t] = depths[idx * W + t];
     }
     
     // 计算相邻帧深度差的平方和
     // sum_of_squared_diffs = (d1-d2)² + (d2-d3)² + (d3-d4)² + ...
     float sum_squared_diff = 0.0f;
     for (int t = 0; t < W - 1; t++) {
         float diff = depths_local[t] - depths_local[t + 1];
         sum_squared_diff += abs(diff);
     }
     
     // 稳定性分数 = exp(-sum_squared_diff)
     // 分数越接近1，说明深度变化越小，越稳定
     float stability_score = expf(-sum_squared_diff);
     
     stability_out[idx] = stability_score;
     
     // stable_mask在Python中计算（使用动态阈值）
     stable_mask_out[idx] = false;
 }
 
 // ====================================================================
 // Python接口封装
 // ====================================================================
 
 std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> gspim_preprocess_multiframe(
     torch::Tensor means3D,        // [M, 3]
     torch::Tensor mu_t,           // [M, 1]
     torch::Tensor scales_xyzt,    // [M, 4]
     torch::Tensor rotation_l,     // [M, 4]
     torch::Tensor rotation_r,     // [M, 4]
     torch::Tensor timestamps,     // [W]
     torch::Tensor view_matrix)    // [4, 4]
 {
     int M = means3D.size(0);
     int W = timestamps.size(0);
     
     // 分配输出张量
     auto options = torch::TensorOptions().dtype(torch::kFloat32).device(means3D.device());
     auto depths_out = torch::empty({M, W}, options);
     auto means3D_cond_out = torch::empty({M, W, 3}, options);  // 条件化3D均值
     auto cov3D_cond_out = torch::empty({M, 6}, options);       // 条件化3D协方差（上三角）
     auto stability_out = torch::empty({M}, options);           // 稳定性分数
     auto stable_mask_out = torch::empty({M}, torch::TensorOptions().dtype(torch::kBool).device(means3D.device()));
     
     // CUDA配置 - 优化线程块大小
     // 使用256线程（平衡寄存器使用和occupancy）
     const int threads = 256;
     const int blocks = (M + threads - 1) / threads;
     
     // Stage 2: 完整的4D到3D投影（包括深度、条件化均值、条件化协方差）
     compute_4d_projection_kernel<<<blocks, threads>>>(
         M, W,
         means3D.data_ptr<float>(),
         mu_t.data_ptr<float>(),
         scales_xyzt.data_ptr<float>(),
         rotation_l.data_ptr<float>(),
         rotation_r.data_ptr<float>(),
         timestamps.data_ptr<float>(),
         view_matrix.data_ptr<float>(),
         depths_out.data_ptr<float>(),
         means3D_cond_out.data_ptr<float>(),
         cov3D_cond_out.data_ptr<float>()
     );
     
     // Stage 3: 计算深度稳定性分数
     compute_depth_stability_kernel<<<blocks, threads>>>(
         M, W,
         depths_out.data_ptr<float>(),
         stability_out.data_ptr<float>(),
         stable_mask_out.data_ptr<bool>()
     );
     
     // 等待CUDA完成
     cudaDeviceSynchronize();
     
     return std::make_tuple(depths_out, means3D_cond_out, cov3D_cond_out, stability_out, stable_mask_out);
 }
 
 // PyBind11绑定
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
     m.def("gspim_preprocess_multiframe", &gspim_preprocess_multiframe, 
           "GSPIM Multi-Frame Preprocessing (CUDA)");
 }
 
 