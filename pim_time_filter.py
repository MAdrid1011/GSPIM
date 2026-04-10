"""
GSPIM PIM时间贡献筛选模拟器
在HBM近存端计算时间贡献并生成Active Map

优化版本：使用纯 PyTorch 向量化操作，避免 Python 循环
"""

import torch


class PIMTimeFilter:
    """PIM时间贡献筛选器（高度优化版本）"""
    
    # 预计算常量
    LN_INV_THRESHOLD = 2.995732273553991  # ln(20) for threshold=0.05
    
    def __init__(self, threshold=0.05, enable_pim_sim=False):
        self.threshold = threshold
        self.enable_pim_sim = enable_pim_sim
        self.pim_ops_count = 0
    
    @torch.no_grad()
    def filter_gaussians(self, scales, rotation_l, rotation_r, mu_t, t_min, t_max):
        """
        完整的PIM时间贡献筛选（高度优化的单函数版本）
        
        优化策略：
        1. 融合所有计算到单个函数，避免中间张量
        2. 使用 torch.no_grad() 禁用 autograd
        3. 原地操作减少内存分配
        4. 使用 PIM 优化公式：0.5 * dt^2 < sigma_tt * ln(20)
        
        Args:
            scales: [N, 4] 四维尺度 (sx, sy, sz, st)
            rotation_l: [N, 4] 左四元数 (w, x, y, z)
            rotation_r: [N, 4] 右四元数 (w, x, y, z)
            mu_t: [N, 1] 时间均值
            t_min: float 时间窗口最小值
            t_max: float 时间窗口最大值
            
        Returns:
            active_mask: [N] bool tensor
            active_indices: [M] 活跃高斯索引
            p_t_max: [N] 活跃标记
        """
        # ========== Stage 1-4: 计算 sigma_tt ==========
        # 提取四元数分量（使用切片避免额外分配）
        lw = rotation_l[:, 0]
        lx = rotation_l[:, 1]
        ly = rotation_l[:, 2]
        lz = rotation_l[:, 3]
        rw = rotation_r[:, 0]
        rx = rotation_r[:, 1]
        ry = rotation_r[:, 2]
        rz = rotation_r[:, 3]
        
        # 计算 4D 旋转矩阵第3行 (只需要这一行来计算 sigma_tt)
        # R = M_r * M_l，R[3,:] = M_r[3,:] @ M_l
        # 
        # 四元数映射（PyTorch: w,x,y,z → CUDA: a=x,b=y,c=z,d=w）
        a, b, c, d = lx, ly, lz, lw
        p, q, r, s = rx, ry, rz, rw
        
        # M_r的第3行: [s, r, q, p]
        # M_l的列：
        #   col 0: [ a, -b,  c, -d]
        #   col 1: [ b,  a, -d, -c]
        #   col 2: [-c,  d,  a, -b]
        #   col 3: [ d,  c,  b,  a]
        
        R30 = s*a - r*b + q*c - p*d
        R31 = s*b + r*a - q*d - p*c
        R32 = -s*c + r*d + q*a - p*b
        R33 = s*d + r*c - q*b + p*a
        
        # sigma_tt = ||L[3,:]||^2 = sum((R[3,i] * s_i)^2)
        # 融合计算，避免创建中间张量
        sigma_tt = (
            (R30 * scales[:, 0]).square() +
            (R31 * scales[:, 1]).square() +
            (R32 * scales[:, 2]).square() +
            (R33 * scales[:, 3]).square()
        )
        
        # ========== Stage 5: 时间贡献判定 ==========
        # 选择最接近 mu_t 的时间点
        mu_t_flat = mu_t.view(-1)
        dt = mu_t_flat.clamp(t_min, t_max) - mu_t_flat
        
        # PIM 优化判定：0.5 * dt^2 < sigma_tt * ln(20)
        # 等价于 exp(-0.5 * dt^2 / sigma_tt) > 0.05
        active_mask = (0.5 * dt.square()) < (sigma_tt * self.LN_INV_THRESHOLD)
        
        # 获取活跃索引
        active_indices = active_mask.nonzero(as_tuple=False).view(-1)
        
        # 返回值（p_t_max 用于兼容性）
        p_t_max = active_mask.float()
        
        self.pim_ops_count += scales.shape[0] * 45
        
        return active_mask, active_indices, p_t_max
    
    def get_statistics(self):
        """获取PIM操作统计信息"""
        return {'pim_ops_count': self.pim_ops_count}
    
    def reset_statistics(self):
        """重置统计信息"""
        self.pim_ops_count = 0

