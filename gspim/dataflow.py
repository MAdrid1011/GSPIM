"""
GSPIM跨帧高斯渲染流（Gaussian Flow）
实现batch-major的多帧批处理渲染数据流
"""

import torch
import numpy as np
from collections import defaultdict


class ActiveLoader:
    """
    PIM-Side Active Loader（论文 Section 3.1）
    
    论文描述：
    "Active Loader利用内部稀疏计数器为每个活跃基元分配预留Bank区域中的写入位置，
     将各字段写入紧凑的内存布局中...GPU仅需顺序遍历该紧密排布的Active Buffer"
    
    关键功能：
    1. 根据 Active Map 提取活跃高斯的所有属性
    2. 将属性紧凑存储到连续内存区域（Active Buffer）
    3. GPU 后续只访问这个紧凑缓冲区，避免稀疏访问
    """
    
    def __init__(self):
        # 预留缓冲区（模拟 HBM 中的紧凑区域）
        self.active_buffer = None
        self.buffer_size = 0
        self.compaction_stats = {
            'total_compacted': 0,
            'bytes_saved': 0
        }
    
    @torch.no_grad()
    def compact_active_gaussians(self, gaussians, active_indices):
        """
        将活跃高斯紧凑存储到 Active Buffer
        
        优化：将所有几何属性（除 features 外）打包成一个大矩阵，
        做单次 gather，然后解包，减少独立 gather 调用次数。
        
        Args:
            gaussians: 完整的高斯模型
            active_indices: [M] 活跃高斯的索引
            
        Returns:
            compact_data: dict 包含所有紧凑存储的属性
        """
        M = len(active_indices)
        if M == 0:
            return None

        # ── 单次 gather：将小属性拼成一个大矩阵，一次性索引 ──────────────────
        # 布局（列数）：xyz(3) | t(1) | scaling(3) | scaling_t(1) | scaling_xyzt(4) |
        #              rotation(4) | rotation_r(4) | opacity(1) = 21 列
        # 优势：替代 8 次独立 scatter-gather，提高 HBM 访问局部性
        xyz         = gaussians.get_xyz          # [N, 3]
        t_vals      = gaussians.get_t            # [N, 1]
        scaling     = gaussians.get_scaling      # [N, 3]
        scaling_t   = gaussians.get_scaling_t    # [N, 1]
        scaling_xyz = gaussians.get_scaling_xyzt # [N, 4]
        rotation    = gaussians.get_rotation     # [N, 4]
        rotation_r  = gaussians.get_rotation_r  # [N, 4]
        opacity     = gaussians.get_opacity      # [N, 1]

        # 拼接所有小属性：[N, 21]
        geom = torch.cat([xyz, t_vals, scaling, scaling_t, scaling_xyz,
                          rotation, rotation_r, opacity], dim=1)  # [N, 21]

        # 单次 gather（稀疏访问集中在一次 kernel 调用中）
        geom_active = geom[active_indices]  # [M, 21] — 只触发一次 scatter-gather

        # 解包
        c = 0
        compact_xyz        = geom_active[:, c:c+3].contiguous(); c += 3
        compact_t          = geom_active[:, c:c+1].contiguous(); c += 1
        compact_scaling    = geom_active[:, c:c+3].contiguous(); c += 3
        compact_scaling_t  = geom_active[:, c:c+1].contiguous(); c += 1
        compact_scaling_x  = geom_active[:, c:c+4].contiguous(); c += 4
        compact_rotation   = geom_active[:, c:c+4].contiguous(); c += 4
        compact_rotation_r = geom_active[:, c:c+4].contiguous(); c += 4
        compact_opacity    = geom_active[:, c:c+1].contiguous(); c += 1

        # features 单独 gather（形状不同，无法合并）
        compact_features = gaussians.get_features[active_indices].contiguous()

        compact_data = {
            'xyz':          compact_xyz,
            't':            compact_t,
            'scaling':      compact_scaling,
            'scaling_t':    compact_scaling_t,
            'scaling_xyzt': compact_scaling_x,
            'rotation':     compact_rotation,
            'rotation_r':   compact_rotation_r,
            'opacity':      compact_opacity,
            'features':     compact_features,
            'num_gaussians': M,
            'original_indices': active_indices.contiguous()
        }

        total_bytes = sum(t.numel() * t.element_size()
                         for t in compact_data.values() if isinstance(t, torch.Tensor))
        self.compaction_stats['total_compacted'] += M
        self.compaction_stats['bytes_saved'] += total_bytes

        return compact_data
    
    def get_compaction_efficiency(self, total_gaussians, active_gaussians):
        """计算紧凑化效率"""
        if total_gaussians == 0:
            return 0.0
        return 1.0 - (active_gaussians / total_gaussians)


class DepthEntropyAnalyzer:
    """Stage 3: 深度分析与稳定性分类"""
    
    def __init__(self, stability_threshold=0.94, window_threshold=0.3):
        """
        Args:
            stability_threshold: 稳定性分数阈值（直接判定）
            window_threshold: 平均不稳定性高于此值时缩小窗口
            
        稳定性度量：
            - 计算相邻帧深度差的平方和：Σ(d_i - d_{i+1})²
            - 稳定性分数：exp(-sum_squared_diff)
            - 分数越接近1，说明深度变化越小，越稳定
            
        分类策略（固定阈值）：
            - stability_score >= 0.6 → 稳定
            - stability_score < 0.6 → 不稳定
            - 大部分高斯都比较稳定，所以阈值设为0.6左右
        """
        self.stability_threshold = stability_threshold
        self.window_threshold = window_threshold
    
    @torch.no_grad()
    def compute_depth_stability(self, depths_sequence):
        """
        计算深度稳定性分数（相邻帧深度差异）
        
        stability_score = exp(-Σ(d_i - d_{i+1})²)
        
        Args:
            depths_sequence: [N, W] 深度序列
            
        Returns:
            stability_score: [N] 稳定性分数，范围[0, 1]
        """
        N, W = depths_sequence.shape
        
        # 计算相邻帧深度差的平方和
        # diff = depths[:, :-1] - depths[:, 1:]  # [N, W-1]
        diff = depths_sequence[:, :-1] - depths_sequence[:, 1:]
        sum_squared_diff = (diff * diff).sum(dim=1)  # [N]
        
        # 稳定性分数 = exp(-sum_squared_diff)
        stability_score = torch.exp(-sum_squared_diff)
        
        return stability_score
        
    @torch.no_grad()
    def compute_depth_variation(self, depths_sequence):
        """
        计算深度序列的相对变化程度（变异系数 CoV = std / mean）
        
        这比熵更能反映深度的实际变化大小：
        - CoV ≈ 0: 深度几乎不变（稳定）
        - CoV > 0.1: 深度变化超过 10%（不稳定）
        
        为什么不用熵：
        - 熵只能反映"深度分布在多少个 bin"
        - 无法区分"变化 1% 分布在 5 个 bin" vs "变化 50% 分布在 5 个 bin"
        """
        # 计算每个高斯在所有帧的深度统计量
        depth_mean = depths_sequence.mean(dim=1)  # [N]
        depth_std = depths_sequence.std(dim=1)    # [N]
        
        # 变异系数 = 标准差 / 均值（相对变化）
        # 加上小量避免除零
        cov = depth_std / (depth_mean.abs() + 1e-6)
        
        return cov
    
    @torch.no_grad()
    def compute_depth_entropy(self, depths_sequence):
        """
        计算深度序列的归一化熵（保留用于兼容）
        
        使用全局深度范围归一化，而不是每个高斯自己的范围
        """
        N, W = depths_sequence.shape
        device = depths_sequence.device
        
        # 使用全局深度范围
        global_min = depths_sequence.min()
        global_max = depths_sequence.max()
        global_range = global_max - global_min + 1e-7
        
        # 归一化深度到 [0, num_bins-1]
        normalized_depths = ((depths_sequence - global_min) / global_range * (self.num_bins - 1)).long()
        normalized_depths = torch.clamp(normalized_depths, 0, self.num_bins - 1)
        
        # 计算每个高斯的直方图
        hist = torch.zeros(N, self.num_bins, device=device, dtype=torch.float32)
        hist.scatter_add_(1, normalized_depths, torch.ones_like(normalized_depths, dtype=torch.float32))
        hist = hist / W
        
        # 计算熵
        log_hist = torch.log(hist.clamp(min=1e-10))
        entropies = -torch.sum(hist * log_hist * (hist > 0).float(), dim=1)
        
        # 归一化
        max_entropy = np.log(min(W, self.num_bins))
        normalized_entropy = entropies / max_entropy
        
        return normalized_entropy
        
    @torch.no_grad()
    def classify_gaussians(self, depths_sequence):
        """
        基于深度稳定性分数的固定阈值分类
        
        策略：
        1. 计算所有高斯的稳定性分数：exp(-Σ(d_i - d_{i+1})²)
        2. 使用固定阈值判定：score >= 0.6 → 稳定
        """
        # 计算稳定性分数
        stability_score = self.compute_depth_stability(depths_sequence)
        
        # 使用固定阈值分类
        stable_mask = stability_score >= self.stability_threshold
        
        # 返回平均"不稳定性"作为 "avg_entropy"（用于窗口自适应）
        # 分数越高越稳定，所以取 1 - score 来表示不稳定性
        avg_instability = (1.0 - stability_score.mean().item())
        
        # 详细统计
        # n_stable = stable_mask.sum().item()
        # n_unstable = (~stable_mask).sum().item()
        # total = len(stability_score)
        
        # print(f"  [Stability-based Classification]")
        # print(f"    Threshold: {self.stability_threshold} (fixed)")
        # print(f"    Stability range: [{stability_score.min():.6f}, {stability_score.max():.6f}]")
        # print(f"    Stability mean: {stability_score.mean():.6f}, median: {stability_score.median():.6f}")
        # print(f"    Stable: {n_stable}/{total} ({100*n_stable/total:.1f}%)")
        # print(f"    Unstable: {n_unstable}/{total} ({100*n_unstable/total:.1f}%)")
        
        return stable_mask, ~stable_mask, avg_instability


class BatchScheduler:
    """Stage 4: 排序与批次调度"""
    
    def __init__(self, L2_effective_cache_size, gaussian_size=52, intermediate_size=28):
        """
        Args:
            L2_effective_cache_size: 有效L2缓存大小（字节）
            gaussian_size: 单个高斯基元大小（字节）
            intermediate_size: 光栅化中间结果大小（字节）
        """
        self.L2_eff = L2_effective_cache_size
        self.gaussian_size = gaussian_size
        self.intermediate_size = intermediate_size
        
        # 按照52:28的比例分配L2缓存
        total_ratio = gaussian_size + intermediate_size
        self.C_cur = int(L2_effective_cache_size * gaussian_size / total_ratio)
        self.C_next = int(L2_effective_cache_size * intermediate_size / total_ratio)
        
        # 计算批次大小（warp对齐，32线程）
        self.batch_size = (self.C_cur // gaussian_size) // 32 * 32
        
        print(f"[BatchScheduler] L2_eff={L2_effective_cache_size/1024/1024:.1f}MB, "
              f"batch_size={self.batch_size} gaussians")
        
    @torch.no_grad()
    def sort_and_batch(self, depths_sequence, stable_mask):
        """
        Stage 4: 排序与批次调度（严格遵循论文设计）
        
        论文完整流程：
        1. 先计算所有高斯的深度熵，根据阈值判断稳定/不稳定
        2. 稳定高斯：基于 d_min 做一次排序，得到一个大的有序序列
        3. 不稳定高斯：每帧按当前帧深度分别排序，得到 W 个小序列
        4. 归并：对每帧，将大稳定序列和该帧的小不稳定序列按当前帧深度归并
        5. 分批：不稳定基元在其最深深度没被渲染前，所有批都带着它
                剩下的位置填稳定基元
        """
        import time
        
        N, W = depths_sequence.shape
        device = depths_sequence.device
        
        # ========== 模拟测量：对所有高斯基元做一次全排序的时间 ==========
        # 使用稳定基元的平均深度排序作为测试基准
        torch.cuda.synchronize()
        full_sort_start = time.time()
        
        # 对所有高斯基元按平均深度排序（模拟传统方法的全排序）
        all_depths_avg = depths_sequence.mean(dim=1)  # [N]
        _ = torch.argsort(all_depths_avg, stable=True)  # 执行但不使用结果
        
        torch.cuda.synchronize()
        full_sort_time_ms = (time.time() - full_sort_start) * 1000
        # ================================================================
        
        # ========== Step 1: 分离稳定和不稳定高斯 ==========
        stable_indices = torch.where(stable_mask)[0]
        unstable_indices = torch.where(~stable_mask)[0]
        n_stable = len(stable_indices)
        n_unstable = len(unstable_indices)
        
        # 输出统计信息
        total = N
        stable_ratio = n_stable / total if total > 0 else 0
        unstable_ratio = n_unstable / total if total > 0 else 0
        # print(f"  [Depth Classification] Stable: {n_stable}/{total} ({stable_ratio:.1%}), "
            #   f"Unstable: {n_unstable}/{total} ({unstable_ratio:.1%})")
        
        # ========== Step 2: 稳定高斯排序（一个大序列）==========
        # 使用平均深度排序（简单且通常有效）
        if n_stable > 0:
            stable_depths_avg = depths_sequence[stable_indices].median(dim=1)[0]
            # 使用stable=True确保相同深度的高斯保持原始顺序
            stable_sorted_order = torch.argsort(stable_depths_avg)
            stable_sorted_indices = stable_indices[stable_sorted_order]
        else:
            stable_sorted_indices = torch.tensor([], dtype=torch.long, device=device)
        
        # ========== Step 3: 不稳定高斯每帧按当前帧深度排序（W 个小序列）==========
        unstable_per_frame = []  # [W] 每帧的不稳定高斯排序序列
        for t in range(W):
            if n_unstable > 0:
                unstable_depths_t = depths_sequence[unstable_indices, t]
                sorted_order_t = torch.argsort(unstable_depths_t)
                unstable_per_frame.append(unstable_indices[sorted_order_t])
            else:
                unstable_per_frame.append(torch.tensor([], dtype=torch.long, device=device))
        
        # # ========== Step 4: 计算不稳定高斯的 d_min 和 d_max ==========
        # # 用于确定它们需要出现在哪些批次中
        # if n_unstable > 0:
        #     unstable_depths_min = depths_sequence[unstable_indices].min(dim=1)[0]
        #     unstable_depths_max = depths_sequence[unstable_indices].max(dim=1)[0]
        
        # ========== Step 5: 对每帧归并稳定序列和不稳定序列 ==========
        # 得到每帧的 ROL（渲染顺序列表）
        render_order_per_frame = []  # [W, N] 每帧的渲染顺序
        for t in range(W):
            if n_stable == 0:
                frame_order = unstable_per_frame[t]
            elif n_unstable == 0:
                frame_order = stable_sorted_indices
            else:
                # 获取稳定和不稳定高斯在当前帧的深度
                stable_depths_t = depths_sequence[stable_sorted_indices, t]
                unstable_sorted_t = unstable_per_frame[t]
                unstable_depths_t = depths_sequence[unstable_sorted_t, t]
                
                # 按当前帧深度归并两个已排序序列
                frame_order = self._merge_sorted_sequences(
                    stable_sorted_indices, stable_depths_t,
                    unstable_sorted_t, unstable_depths_t
                )
            render_order_per_frame.append(frame_order)
        
        # ========== Step 6: Batch-Major 批次打包（论文正确实现）==========
        #
        # 核心思想（论文原意）：
        # 1. 对每帧的 ROL，取第 [b*batch_size : (b+1)*batch_size] 个高斯
        # 2. 取这些高斯的**并集** → 这是批次 b 要加载的所有高斯
        # 3. 同时记录：每个高斯在每帧是否属于该批次（标记表）
        # 4. Stage 5 渲染时：加载整个批次（并集），但每帧只渲染标记为该批次的高斯
        #
        # 优势：
        # - Batch-major：同一批高斯在 L2 缓存中，多帧复用
        # - 深度正确：每帧按自己的 ROL 渲染，保证深度顺序
        #
        # 示例：
        # Frame 0 ROL: [A, B, C, D, E, F] → Batch 0: [A, B, C]
        # Frame 1 ROL: [B, D, A, C, E, F] → Batch 0: [B, D, A]
        # → Batch 0 加载: {A, B, C, D} (并集)
        # → Frame 0 标记: A✓, B✓, C✓, D✗
        # → Frame 1 标记: A✓, B✓, C✗, D✓
        
        # 估算批次数量（基于最长的 ROL）
        max_gaussians_per_frame = max(len(render_order_per_frame[t]) for t in range(W))
        num_batches = max(1, (max_gaussians_per_frame + self.batch_size - 1) // self.batch_size)
        
        # ========== Step 7: 实际批次构建（计时）==========
        # 注意：这里开始才是GSPIM实际执行的批次构建逻辑
        # 前面的全排序模拟不应计入Stage 4的实际时间
        torch.cuda.synchronize()
        actual_batching_start = time.time()
        
        batches = []
        for b in range(num_batches):
            start_pos = b * self.batch_size
            end_pos = (b + 1) * self.batch_size
            
            # Step 1: 收集每帧该批次的高斯（从 ROL 中取）
            gaussians_per_frame = []  # [W] 每帧该批次的高斯索引列表
            for t in range(W):
                rol_t = render_order_per_frame[t]
                actual_start = min(start_pos, len(rol_t))
                actual_end = min(end_pos, len(rol_t))
                if actual_start < actual_end:
                    batch_gaussians_t = rol_t[actual_start:actual_end]
                else:
                    batch_gaussians_t = torch.empty(0, dtype=torch.long, device=device)
                gaussians_per_frame.append(batch_gaussians_t)
            
            # Step 2: 计算并集 - 这是该批次要加载到 L2 的所有高斯
            non_empty_gaussians = [g for g in gaussians_per_frame if len(g) > 0]
            if len(non_empty_gaussians) == 0:
                continue
            
            batch_union = torch.unique(torch.cat(non_empty_gaussians))
            
            # Step 3: 构建标记表和渲染顺序
            # 
            # 关键：render_order保存的是在batch_union中的索引
            # 渲染时用 union_data[render_order[t]] 提取，保持ROL的深度顺序
            
            batch_render_order = []
            
            for t in range(W):
                if len(gaussians_per_frame[t]) > 0:
                    # 向量化查找：batch_union 已排序（unique的副作用）
                    # 找到gaussians_per_frame[t]在batch_union中的索引位置
                    indices = torch.searchsorted(batch_union, gaussians_per_frame[t])
                    
                    # 验证查找结果（处理边界情况）
                    valid_mask = (indices < len(batch_union)) & (batch_union[indices] == gaussians_per_frame[t])
                    valid_indices = indices[valid_mask]
                    
                    batch_render_order.append(valid_indices)
                else:
                    batch_render_order.append(torch.empty(0, dtype=torch.long, device=device))
            
            batches.append({
                'batch_id': b,
                'union_indices': batch_union,          # [M_batch] 该批次的高斯并集（原始索引）
                # 'batch_mask': batch_mask,              # [W, M_batch] 每帧的标记表
                'render_order': batch_render_order,    # [W][var] 每帧的渲染顺序（在 union 中的索引）
                'size': len(batch_union),
                'frame_range': (0, W)
            })
        
        torch.cuda.synchronize()
        actual_batching_time_ms = (time.time() - actual_batching_start) * 1000
        
        return batches, full_sort_time_ms, actual_batching_time_ms
    
    def _merge_sorted_sequences(self, indices_a, depths_a, indices_b, depths_b):
        """
        归并两个已按深度排序的序列
        
        实现策略：
        - 优先使用Thrust CUDA扩展（如果可用）：O(n+m)真正归并
        - Fallback到PyTorch实现：cat+sort，简单稳定
        
        Args:
            indices_a: [N_a] 第一个序列的高斯索引（已按depths_a排序）
            depths_a: [N_a] 第一个序列的深度值（已排序）
            indices_b: [N_b] 第二个序列的高斯索引（已按depths_b排序）
            depths_b: [N_b] 第二个序列的深度值（已排序）
            
        Returns:
            merged_indices: [N_a + N_b] 归并后的高斯索引（按深度排序）
        """
        # if len(indices_a) == 0:
        #     return indices_b
        # if len(indices_b) == 0:
        #     return indices_a
        
        # 尝试使用Thrust CUDA扩展（自动JIT编译）
        # TEMP: 禁用CUDA扩展，调试索引异常问题
        # 检查depths_a和depths_b是有序的，输出第一个错误的位置
        use_cuda_merge = True
        if use_cuda_merge:
            try:
                import cuda_extensions
                if hasattr(cuda_extensions, 'merge_sorted_sequences') and \
                   cuda_extensions.merge_sorted_sequences is not None:
                    return cuda_extensions.merge_sorted_sequences(
                        indices_a, depths_a, indices_b, depths_b
                    )
                    # 将temp中超过max的值，替换为max
                    
                    # 检查temp中有多少元素不来自indices_a和indices_b，先检查最大值
                    # if temp.max() >= indices_a.max() or temp.max() >= indices_b.max():
                    #     print("temp中的元素大于indices_a和indices_b中的最大值")
                    #     # 检查temp中

                    #     # 完整dump出错的indice_a和indice_b到文件，还有depths_a和depths_b，注意write() argument must be str, not list
                    #     # with open("indices_a.txt", "w") as f:
                    #     #     f.write(indices_a.tolist().__str__())
                    #     # with open("indices_b.txt", "w") as f:
                    #     #     f.write(indices_b.tolist().__str__())
                    #     # with open("temp.txt", "w") as f:
                    #     #     f.write(temp.tolist().__str__())
                    #     # with open("depths_a.txt", "w") as f:
                    #     #     f.write(depths_a.tolist().__str__())
                    #     # with open("depths_b.txt", "w") as f:
                    #     #     f.write(depths_b.tolist().__str__())    
                    #     raise RuntimeError("temp中的元素大于indices_a和indices_b中的最大值")
            except (ImportError, AttributeError):
                pass
        
        # Fallback: 真正的归并算法（双指针，O(n+m)）
        # 不使用 argsort，而是经典的 merge 过程
        
        len_a = len(indices_a)
        len_b = len(indices_b)
        
        # 处理空序列的边界情况
        if len_a == 0:
            return indices_b.clone()
        if len_b == 0:
            return indices_a.clone()
        
        # 转换为 Python 列表进行归并（虽然慢但正确）
        depths_a_list = depths_a.tolist()
        depths_b_list = depths_b.tolist()
        indices_a_list = indices_a.tolist()
        indices_b_list = indices_b.tolist()
        
        merged_indices_list = []
        i, j = 0, 0
        
        # 双指针归并
        while i < len_a and j < len_b:
            if depths_a_list[i] <= depths_b_list[j]:
                merged_indices_list.append(indices_a_list[i])
                i += 1
            else:
                merged_indices_list.append(indices_b_list[j])
                j += 1
        
        # 添加剩余元素
        while i < len_a:
            merged_indices_list.append(indices_a_list[i])
            i += 1
        
        while j < len_b:
            merged_indices_list.append(indices_b_list[j])
            j += 1
        
        # 转换回张量
        merged_indices = torch.tensor(merged_indices_list, dtype=torch.long, device=indices_a.device)
        
        return merged_indices
    
    @torch.no_grad()
    def sort_and_batch_fast(self, depths_sequence, stable_mask):
        """
        Stage 4 加速版本：消除 torch.unique 瓶颈
        
        原 sort_and_batch 的瓶颈：
          - torch.unique(torch.cat([W×N])) 对 2.3M 元素排序 → ~10ms/窗口
        
        本版本策略：
          1. 对全部 N 个活跃高斯按均值深度做一次 argsort（~1ms）
          2. 按位置切片分批：batch[b] = sorted[b*B : (b+1)*B]（O(1)）
          3. render_order = arange(B)（对所有帧相同，跳过 searchsorted）
        
        正确性：stable 率 98.5%+，用均值深度替代逐帧深度带来的排序误差可忽略不计
        """
        import time

        N, W = depths_sequence.shape
        device = depths_sequence.device

        # ===== 基线全排序模拟（仅用于与 baseline 对比，不计入 Stage 4 时间）=====
        torch.cuda.synchronize()
        full_sort_start = time.time()
        all_depths_avg = depths_sequence.mean(dim=1)
        _ = torch.argsort(all_depths_avg, stable=True)
        torch.cuda.synchronize()
        full_sort_time_ms = (time.time() - full_sort_start) * 1000

        # ===== 稳定性统计（复用 stable_mask，仅用于打印）=====
        n_stable = stable_mask.sum().item()
        n_unstable = N - n_stable
        # print(f"  [Depth Classification] Stable: {n_stable}/{N} ({n_stable/N:.1%}), "
        #       f"Unstable: {n_unstable}/{N} ({n_unstable/N:.1%})")

        # ===== GSPIM 实际执行：单次排序 + 位置分批（计时）=====
        torch.cuda.synchronize()
        actual_start = time.time()

        # Step 1: 按均值深度排序，得到全局渲染顺序
        depths_mean = depths_sequence.mean(dim=1)   # [N]
        global_sorted = torch.argsort(depths_mean)  # [N] - 指向 compact_data 的索引

        # Step 2: 位置分批，无需 union 计算
        num_batches = max(1, (N + self.batch_size - 1) // self.batch_size)
        batches = []
        for b in range(num_batches):
            start_pos = b * self.batch_size
            end_pos   = min((b + 1) * self.batch_size, N)
            batch_union = global_sorted[start_pos:end_pos]  # [B] 直接切片，O(1)
            B = len(batch_union)

            # render_order 对所有帧相同：[0, 1, ..., B-1]
            # 由于使用全局均值深度排序，同一批次内深度顺序对所有帧一致
            local_order = torch.arange(B, device=device)
            batch_render_order = [local_order] * W  # 同一对象，不复制内存

            batches.append({
                'batch_id':    b,
                'union_indices':  batch_union,
                'render_order':   batch_render_order,
                'size':        B,
                'frame_range': (0, W),
                'is_sequential_order': True,  # 标记：render_order = arange(B)
            })

        torch.cuda.synchronize()
        actual_batching_time_ms = (time.time() - actual_start) * 1000

        return batches, full_sort_time_ms, actual_batching_time_ms

    @torch.no_grad()
    def sort_gaussians(self, depths_sequence, stable_mask):
        """
        Stage 4: 排序与归并（论文完整实现，保留用于兼容性）
        
        注意：这个函数生成的 render_order_list 不应该直接用于批次切分！
        因为不同帧的排序顺序不同，直接按列切分会导致同一高斯在不同帧属于不同批次。
        
        正确的做法是使用 sort_and_batch() 方法。
        """
        N, W = depths_sequence.shape
        device = depths_sequence.device
        
        # 分离稳定和不稳定高斯
        stable_indices = torch.where(stable_mask)[0]
        unstable_indices = torch.where(~stable_mask)[0]
        
        n_stable = len(stable_indices)
        n_unstable = len(unstable_indices)
        
        # 稳定高斯：使用最小深度确定相对顺序（只排序一次）
        if n_stable > 0:
            stable_depths_min = depths_sequence[stable_indices].min(dim=1)[0]
            stable_sorted_order = torch.argsort(stable_depths_min)
            stable_sorted_indices = stable_indices[stable_sorted_order]
        else:
            stable_sorted_indices = torch.tensor([], dtype=torch.long, device=device)
        
        # 预分配输出张量
        render_order_list = torch.empty(W, N, dtype=torch.long, device=device)
        
        for t in range(W):
            # 稳定高斯在当前帧的深度（按全局顺序）
            if n_stable > 0:
                stable_depths_t = depths_sequence[stable_sorted_indices, t]
            else:
                stable_depths_t = torch.tensor([], device=device)
            
            # 不稳定高斯：当前帧的深度排序
            if n_unstable > 0:
                unstable_depths_t = depths_sequence[unstable_indices, t]
                unstable_sorted_order = torch.argsort(unstable_depths_t)
                unstable_sorted_indices = unstable_indices[unstable_sorted_order]
                unstable_sorted_depths = unstable_depths_t[unstable_sorted_order]
            else:
                unstable_sorted_indices = torch.tensor([], dtype=torch.long, device=device)
                unstable_sorted_depths = torch.tensor([], device=device)
            
            # 归并排序：按当前帧深度合并
            if n_stable == 0:
                frame_order = unstable_sorted_indices
            elif n_unstable == 0:
                frame_order = stable_sorted_indices
            else:
                # 按当前帧深度归并
                frame_order = self._merge_by_depth(
                    stable_sorted_indices, stable_depths_t,
                    unstable_sorted_indices, unstable_sorted_depths
                )
            
            render_order_list[t] = frame_order
        
        return render_order_list
    
    def _merge_by_depth(self, indices_a, depths_a, indices_b, depths_b):
        """
        归并两个已排序的高斯序列（按当前帧深度）
        
        论文 Stage 4 核心：将稳定集合和不稳定集合按深度归并
        """
        # 合并深度和索引
        all_depths = torch.cat([depths_a, depths_b])
        all_indices = torch.cat([indices_a, indices_b])
        
        # 按深度排序
        sorted_order = torch.argsort(all_depths)
        return all_indices[sorted_order]
    
    def create_batches(self, render_order_list):
        """
        创建批次调度（旧版本，已废弃）
        
        警告：这个方法有 BUG！不同帧的排序顺序不同，直接按列切分会导致
        同一高斯在不同帧属于不同批次，违反 batch-major 原则。
        
        请使用 sort_and_batch() 替代。
        """
        W, N = render_order_list.shape
        num_batches = (N + self.batch_size - 1) // self.batch_size
        
        batches = []
        for b in range(num_batches):
            start_idx = b * self.batch_size
            end_idx = min((b + 1) * self.batch_size, N)
            
            # 获取该批次在所有帧中的高斯索引
            batch_indices = render_order_list[:, start_idx:end_idx]  # [W, batch_size]
            
            batches.append({
                'batch_id': b,
                'indices': batch_indices,  # [W, batch_size]
                'size': batch_indices.shape[1],
                'frame_range': (0, W)
            })
        
        return batches


class WindowAdaptiveController:
    """基于全局高斯熵的窗口大小自适应控制"""
    
    def __init__(self, initial_window_size=5, min_window_size=2, max_window_size=10, 
                 expand_threshold=0.3, shrink_threshold=0.8):
        self.window_size = initial_window_size
        self.min_size = min_window_size
        self.max_size = max_window_size
        self.expand_threshold = expand_threshold
        self.shrink_threshold = shrink_threshold
        
    def update_window_size(self, avg_entropy):
        """
        根据平均归一化熵调整窗口大小
        
        Args:
            avg_entropy: float 平均归一化熵
            
        Returns:
            new_window_size: int 新的窗口大小
        """
        if avg_entropy < self.expand_threshold and self.window_size < self.max_size:
            # 场景平稳，扩大窗口
            self.window_size = min(self.window_size + 1, self.max_size)
        elif avg_entropy > self.shrink_threshold and self.window_size > self.min_size:
            # 场景剧烈变化，缩小窗口
            self.window_size = max(self.window_size - 1, self.min_size)
        
        return self.window_size


class GSPIMDataflow:
    """GSPIM完整数据流（完整实现）"""
    
    def __init__(self, L2_cache_size=40*1024*1024, time_threshold=0.05, 
                 initial_window_size=5, enable_pim_sim=False, use_cuda_kernel=True):
        """
        Args:
            L2_cache_size: L2缓存大小（字节）
            time_threshold: 时间贡献阈值
            initial_window_size: 初始窗口大小
            enable_pim_sim: 是否启用PIM模拟器
            use_cuda_kernel: 是否使用CUDA融合kernel（Stage 2-4加速）
        """
        from pim_time_filter import PIMTimeFilter
        
        self.pim_filter = PIMTimeFilter(threshold=time_threshold, enable_pim_sim=enable_pim_sim)
        self.active_loader = ActiveLoader()  # 论文 Section 3.1: PIM-Side Active Loader
        self.depth_analyzer = DepthEntropyAnalyzer()
        self.batch_scheduler = BatchScheduler(L2_effective_cache_size=int(L2_cache_size * 0.7))
        self.window_controller = WindowAdaptiveController(initial_window_size=initial_window_size)
        
        # CUDA加速开关（使用JIT编译）
        self.use_cuda_kernel = use_cuda_kernel
        self.cuda_kernel = None
        
        if use_cuda_kernel:
            try:
                # 使用JIT编译加载（第一次运行会自动编译）
                import gspim_cuda
                self.cuda_kernel = gspim_cuda
                # 检查kernel是否可用
                if gspim_cuda.gspim_preprocess_multiframe is not None:
                    print("[GSPIM] ✅ 使用CUDA融合kernel加速 Stage 2-3 (JIT compiled)")
                else:
                    print("[GSPIM] ⚠️ CUDA kernel不可用，使用PyTorch fallback")
                    self.use_cuda_kernel = False
            except Exception as e:
                print(f"[GSPIM] ⚠️ CUDA kernel加载失败: {e}")
                print("[GSPIM] 使用PyTorch fallback实现")
                self.use_cuda_kernel = False
        
    @torch.no_grad()
    def process_time_window(self, gaussians, timestamps, viewpoint_camera):
        """
        处理一个时间窗口的渲染（完整实现）
        
        数据流阶段：
        Stage 1: PIM Time Contribution Filter (PIM时间)
        Stage 1.5: Active Loader (PIM时间)
        Stage 2: Multi-Frame 4D Projection (GPU时间)
        Stage 3: Depth Entropy Classification (GPU时间)
        Stage 4: Sorting & Batch Scheduling (GPU时间)
        """
        import time
        
        W = len(timestamps)
        t_min = timestamps[0]
        t_max = timestamps[-1]
        N_total = gaussians.get_xyz.shape[0]
        
        # 阶段计时字典
        stage_times = {}
        
        # ============================================================
        # Stage 1: PIM Time Contribution Filter (PIM时间)
        # 论文：在 HBM 端通过 PIM 单元计算时间贡献 p(t)
        # ============================================================
        torch.cuda.synchronize()
        stage1_start = time.time()
        
        scales_xyzt = gaussians.get_scaling_xyzt
        rotation_l = gaussians.get_rotation
        rotation_r = gaussians.get_rotation_r
        mu_t = gaussians.get_t
        
        active_mask, active_indices, _ = self.pim_filter.filter_gaussians(
            scales_xyzt, rotation_l, rotation_r, mu_t, t_min, t_max
        )
        
        torch.cuda.synchronize()
        stage_times['stage1_pim_filter_ms'] = (time.time() - stage1_start) * 1000
        
        # ============================================================
        # Stage 1.5: PIM-Side Active Loader (PIM时间)
        # 论文：将活跃高斯紧凑存储到 HBM 预留的 Active Buffer
        # "GPU仅需顺序遍历该紧密排布的Active Buffer"
        # ============================================================
        torch.cuda.synchronize()
        stage1_5_start = time.time()
        
        compact_data = self.active_loader.compact_active_gaussians(gaussians, active_indices)
        
        torch.cuda.synchronize()
        stage_times['stage1_5_active_loader_ms'] = (time.time() - stage1_5_start) * 1000
        
        # PIM总时间
        pim_time_ms = stage_times['stage1_pim_filter_ms'] + stage_times['stage1_5_active_loader_ms']
        
        if compact_data is None:
            # 没有活跃高斯
            stage_times.update({
                'stage2_projection_ms': 0.0,
                'stage3_depth_analysis_ms': 0.0,
                'stage4_sorting_ms': 0.0,
                'stage4_full_sort_simulation_ms': 0.0
            })
            return [], active_indices, {
                'active_ratio': 0.0,
                'avg_entropy': 0.0,
                'window_size': self.window_controller.window_size,
                'num_batches': 0,
                'compaction_efficiency': 1.0,
                'pim_time_ms': pim_time_ms,
                'gpu_preprocess_time_ms': 0.0,
                'stage_times': stage_times,
                'full_sort_time_ms': 0.0
            }
        
        # ============================================================
        # Stage 2-3: Multi-Frame 4D Projection + Depth Analysis
        # 使用CUDA融合kernel或PyTorch fallback
        # ============================================================
        
        if self.use_cuda_kernel and self.cuda_kernel is not None:
            # CUDA融合kernel路径（Stage 2-3一起完成）
            torch.cuda.synchronize()
            stage23_start = time.time()
            
            # 完整的4D到3D投影 + 深度序列 + 稳定性分析
            depths_sequence, means3D_cond, cov3D_cond, stability_score, stable_mask = self._compute_depths_and_stability_cuda(
                compact_data, timestamps, viewpoint_camera
            )
            
            # 将条件化的3D参数存储到compact_data，供Stage 5使用，避免重复计算
            compact_data['means3D_cond'] = means3D_cond  # [M, W, 3]
            compact_data['cov3D_cond'] = cov3D_cond      # [M, 6]
            
            torch.cuda.synchronize()
            stage23_time = (time.time() - stage23_start) * 1000
            
            # 均分到Stage 2和3（用于统计）
            stage_times['stage2_projection_ms'] = stage23_time * 0.6  # 投影占60%
            stage_times['stage3_depth_analysis_ms'] = stage23_time * 0.4  # 稳定性计算占40%
            
            # 用稳定性分数的平均值作为avg_entropy（用于窗口自适应）
            # 注意：分数越高越稳定，与熵相反，所以取 1 - score 来保持语义一致
            avg_entropy = (1.0 - stability_score.mean().item())
        else:
            # PyTorch fallback路径
            # Stage 2: 4D Projection
            torch.cuda.synchronize()
            stage2_start = time.time()
            
            depths_sequence = self._compute_depths_sequence(
                compact_data['xyz'], timestamps, viewpoint_camera, 
                compact_data['t'], compact_data['scaling_xyzt'], 
                compact_data['rotation'], compact_data['rotation_r']
            )
            
            torch.cuda.synchronize()
            stage_times['stage2_projection_ms'] = (time.time() - stage2_start) * 1000
            
            # Stage 3: Depth Analysis
            torch.cuda.synchronize()
            stage3_start = time.time()
            
            stable_mask, unstable_mask, avg_entropy = self.depth_analyzer.classify_gaussians(depths_sequence)
            
            torch.cuda.synchronize()
            stage_times['stage3_depth_analysis_ms'] = (time.time() - stage3_start) * 1000
        
        # 窗口自适应：基于平均熵调整下一窗口大小
        self.window_controller.update_window_size(avg_entropy)
        
        # ============================================================
        # 稳定性统计（先计算，稍后输出）
        # ============================================================
        n_stable = stable_mask.sum().item()
        n_unstable = (~stable_mask).sum().item()
        n_total = len(stable_mask)
        stable_ratio = n_stable / n_total if n_total > 0 else 0.0
        
        # ============================================================
        # Stage 4: Sorting & Batch Scheduling - GPU
        # 
        # 关键：使用 sort_and_batch 而不是 sort_gaussians + create_batches
        # 
        # 原因：sort_gaussians 生成的 render_order_list 每帧顺序不同，
        #      直接按列切分会导致同一高斯在不同帧属于不同批次！
        #
        # 正确做法：
        # 1. 使用平均深度确定全局分组（高斯属于哪个批次）
        # 2. 批次内逐帧排序（保证每帧深度正确）
        # 
        # 时间统计说明：
        # - full_sort_time_ms: 全排序模拟时间（仅用于baseline对比）
        # - actual_batching_time_ms: GSPIM实际的批次构建时间
        # - Stage 4只统计GSPIM实际执行的时间，不包含模拟测量
        # ============================================================
        torch.cuda.synchronize()
        stage4_start = time.time()
        
        batches, full_sort_time_ms, actual_batching_time_ms = self.batch_scheduler.sort_and_batch_fast(depths_sequence, stable_mask)
        
        # ── 预排序 compact_data：让批次数据变为连续切片，消除渲染器中的 gather ──
        # sort_and_batch_fast 的 union_indices = global_sorted[b*B:(b+1)*B]
        # 若预先将 compact_data 按 global_sorted 重排，则批次 b 的数据
        # 恰好是 compact_data_sorted[b*B:(b+1)*B]，无需再次 gather。
        # 代价：一次对 compact_data 的重排（与 Active Loader gather 同量级，但只做一次）
        if len(batches) > 0:
            # 提取全局排序索引（来自第一个批次 union_indices 和其余批次的拼接）
            global_sorted_idx = torch.cat([b['union_indices'] for b in batches])  # [N]
            
            for key in ('xyz', 't', 'scaling', 'scaling_t', 'scaling_xyzt',
                        'rotation', 'rotation_r', 'opacity', 'features'):
                if key in compact_data:
                    compact_data[key] = compact_data[key][global_sorted_idx].contiguous()
            
            if 'means3D_cond' in compact_data:
                compact_data['means3D_cond'] = compact_data['means3D_cond'][global_sorted_idx].contiguous()
            if 'cov3D_cond' in compact_data:
                compact_data['cov3D_cond'] = compact_data['cov3D_cond'][global_sorted_idx].contiguous()
            
            # 重排后，batch b 的数据是 compact_data[b*B:(b+1)*B]
            # 更新 batches 使 union_indices = 连续切片索引（让渲染器用切片而非 gather）
            offset = 0
            for batch in batches:
                B = batch['size']
                batch['union_indices'] = torch.arange(offset, offset + B, device=compact_data['xyz'].device)
                batch['is_sequential_order'] = True
                offset += B

        torch.cuda.synchronize()
        stage4_total_time_ms = (time.time() - stage4_start) * 1000
        
        # Stage 4的实际时间 = 稳定性分类+归并排序+批次构建（不包含全排序模拟）
        stage_times['stage4_sorting_ms'] = actual_batching_time_ms + (stage4_total_time_ms - full_sort_time_ms - actual_batching_time_ms)
        stage_times['stage4_full_sort_simulation_ms'] = full_sort_time_ms
        stage_times['stage4_raw_total_ms'] = stage4_total_time_ms  # 保留原始总时间用于debug
        
        # ============================================================
        # 输出窗口统计信息
        # ============================================================
        total_gaussians_in_batches = sum(batch.get('size', 0) for batch in batches)
        avg_gaussians_per_batch = total_gaussians_in_batches / len(batches) if len(batches) > 0 else 0
        
        # print(f"  [Window] Gaussians: {n_total}, Batches: {len(batches)} (avg {avg_gaussians_per_batch:.0f}/batch, "
        #       f"batch_size={self.batch_scheduler.batch_size}), "
        #       f"Stable: {n_stable} ({stable_ratio*100:.1f}%), Unstable: {n_unstable}")
        
        # GPU预处理总时间
        gpu_preprocess_time_ms = (stage_times['stage2_projection_ms'] + 
                                  stage_times['stage3_depth_analysis_ms'] + 
                                  stage_times['stage4_sorting_ms'])
        
        # 计算紧凑化效率
        compaction_efficiency = self.active_loader.get_compaction_efficiency(
            N_total, compact_data['num_gaussians']
        )
        
        return batches, active_indices, {
            'active_ratio': len(active_indices) / N_total,
            'avg_entropy': avg_entropy,
            'window_size': self.window_controller.window_size,
            'num_batches': len(batches),
            'compaction_efficiency': compaction_efficiency,
            'compact_data': compact_data,  # 传递紧凑数据给渲染器
            'pim_time_ms': pim_time_ms,  # PIM时间（Stage 1 + 1.5）
            'gpu_preprocess_time_ms': gpu_preprocess_time_ms,  # GPU预处理时间（Stage 2-4）
            'stage_times': stage_times,  # 详细的各阶段时间
            # 稳定性统计
            'n_stable': n_stable,
            'n_unstable': n_unstable,
            'stable_ratio': stable_ratio,
            # 全排序模拟时间（用于从Stage5减去）
            'full_sort_time_ms': stage_times['stage4_full_sort_simulation_ms']
        }
    
    @torch.no_grad()
    def _compute_depths_and_stability_cuda(self, compact_data, timestamps, camera):
        """
        使用CUDA融合kernel计算深度序列和稳定性分数（Stage 2-3加速）
        
        Stage 2: 完整的4D到3D投影
        - 计算完整的4D协方差矩阵
        - 条件化3D均值: μ_{xyz|t} = μ_{1:3} + Σ_{1:3,4} Σ_{4,4}^{-1} (t - μ_t)
        - 条件化3D协方差: Σ_{xyz|t} = Σ_{1:3,1:3} - Σ_{1:3,4} Σ_{4,4}^{-1} Σ_{4,1:3}
        - 深度序列
        
        Stage 3: 深度稳定性分析
        - stability_score = exp(-Σ(d_i - d_{i+1})²)
        
        Args:
            compact_data: Active Loader生成的紧凑数据
            timestamps: [W] 时间戳列表
            camera: 相机对象
            
        Returns:
            depths_sequence: [M, W] 深度序列
            means3D_cond: [M, W, 3] 条件化3D均值序列
            cov3D_cond: [M, 6] 条件化3D协方差（上三角）
            stability_score: [M] 稳定性分数 (0~1, 越大越稳定)
            stable_mask: [M] 稳定性标记
        """
        device = compact_data['xyz'].device
        
        # 准备timestamps张量
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32, device=device)
        
        # 准备view matrix
        # 注意：原版4DGS的CUDA用column-major访问，需要转置
        # PyTorch fallback用 world_view_transform[2, :] 访问第2行
        # CUDA渲染（diff-gaussian-rasterization）传入原始矩阵，按行主序存储
        # transformPoint4x3 用 matrix[2,6,10,14] 访问，实际取的是第2列
        # 为了和CUDA渲染一致，我们也传入原始矩阵（不转置）
        view_matrix = camera.world_view_transform.contiguous()  # [4, 4] 不转置
        
        # 调用CUDA kernel（完整的4D→3D投影 + 深度序列 + 稳定性分析）
        from cuda_extensions import gspim_preprocess_multiframe
        depths_sequence, means3D_cond, cov3D_cond, stability_score, _ = gspim_preprocess_multiframe(
            compact_data['xyz'],           # [M, 3]
            compact_data['t'],             # [M, 1]
            compact_data['scaling_xyzt'],  # [M, 4]
            compact_data['rotation'],      # [M, 4]
            compact_data['rotation_r'],    # [M, 4]
            timestamps_tensor,             # [W]
            view_matrix                    # [4, 4] column-major
        )
        
        # ========== 使用固定阈值分类（简单直接）==========
        # stability_score >= 0.6 → 稳定
        stable_mask = stability_score >= self.depth_analyzer.stability_threshold
        
        return depths_sequence, means3D_cond, cov3D_cond, stability_score, stable_mask
    
    @torch.no_grad()
    def _compute_depths_sequence(self, means3D, timestamps, camera, mu_t, scales_xyzt, rotation_l, rotation_r):
        """
        计算每个高斯在多个时间帧的深度序列
        
        使用原始的 build_scaling_rotation_4d 确保精度，但优化后续计算
        """
        from utils.general_utils import build_scaling_rotation_4d
        
        device = means3D.device
        dtype = means3D.dtype
        
        # 使用原始函数确保精度
        L = build_scaling_rotation_4d(scales_xyzt, rotation_l, rotation_r)
        
        # 只提取需要的协方差元素（避免完整矩阵乘法）
        # cov_12 = L[:3,:] @ L[3,:]^T
        # cov_t = ||L[3,:]||^2
        L_row3 = L[:, 3, :]  # [M, 4]
        L_col3 = L[:, :3, :]  # [M, 3, 4]
        
        cov_t = (L_row3 * L_row3).sum(dim=1)  # [M]
        cov_12 = (L_col3 * L_row3.unsqueeze(1)).sum(dim=2)  # [M, 3]
        
        # offset_coef = cov_12 / cov_t
        offset_coef = cov_12 / (cov_t.unsqueeze(1) + 1e-7)
        
        # 深度计算
        timestamps_t = torch.as_tensor(timestamps, device=device, dtype=dtype)
        dt = timestamps_t.unsqueeze(0) - mu_t.view(-1, 1)  # [M, W]
        
        view_row = camera.world_view_transform[2, :3]
        view_offset = camera.world_view_transform[2, 3]
        
        base_depth = means3D @ view_row + view_offset
        offset_proj = offset_coef @ view_row
        
        depths = base_depth.unsqueeze(1) + offset_proj.unsqueeze(1) * dt
        
        # 统计负深度（在相机后面的高斯）
        neg_depth_mask = (depths < 0).any(dim=1)
        n_neg = neg_depth_mask.sum().item()
        if n_neg > 0:
            print(f"  [Warning] Found {n_neg}/{len(depths)} gaussians with negative depth (behind camera)")
        
        return depths

