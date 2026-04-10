"""
GSPIM CUDA Extensions - JIT编译加载
包含所有CUDA扩展的自动编译加载逻辑
"""

import torch
import os
from torch.utils.cpp_extension import load

# 初始化为None（确保变量始终存在）
gspim_preprocess_multiframe = None
merge_sorted_sequences = None

# 获取当前目录
cuda_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. GSPIM Preprocess (Stage 2-3融合kernel)
# ============================================================
try:
    # GLM header: prefer submodules/, fall back to embedded diff-gaussian-rasterization/
    parent_dir = os.path.dirname(cuda_dir)
    glm_include = os.path.join(parent_dir, "submodules", "diff-gaussian-rasterization", "third_party", "glm")
    if not os.path.isdir(glm_include):
        glm_include = os.path.join(parent_dir, "diff-gaussian-rasterization", "third_party", "glm")
    
    _gspim_preprocess = load(
        name='gspim_preprocess',
        sources=[os.path.join(cuda_dir, "gspim_preprocess.cu")],
        extra_cflags=[f'-I{glm_include}'],
        extra_cuda_cflags=[
            f'-I{glm_include}',
            '-O3',
            '--use_fast_math',
            '-Xptxas', '-O3',
            '--expt-relaxed-constexpr'
        ],
        verbose=True  # 显示编译详情，方便调试
    )
    gspim_preprocess_multiframe = _gspim_preprocess.gspim_preprocess_multiframe
    print("[GSPIM] ✅ Stage 2-3 CUDA kernel 加载成功")
except Exception as e:
    print(f"[GSPIM] ⚠️  Stage 2-3 CUDA kernel 加载失败: {e}")
    print(f"[GSPIM]    将使用PyTorch fallback")
    gspim_preprocess_multiframe = None

# ============================================================
# 2. Merge Sorted (Thrust归并)
# ============================================================
try:
    _merge_sorted = load(
        name='merge_sorted_thrust',
        sources=[os.path.join(cuda_dir, "merge_sorted.cu")],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '--expt-relaxed-constexpr'
        ],
        verbose=False
    )
    merge_sorted_sequences = _merge_sorted.merge_sorted_sequences
    print("[GSPIM] ✅ Thrust归并 CUDA kernel 加载成功")
except Exception as e:
    print(f"[GSPIM] ⚠️  Thrust归并 CUDA kernel 加载失败: {e}")
    print(f"[GSPIM]    将使用PyTorch fallback (功能相同，稍慢)")
    merge_sorted_sequences = None

# 导出所有函数（即使是None也导出，避免ImportError）
__all__ = ['gspim_preprocess_multiframe', 'merge_sorted_sequences']
