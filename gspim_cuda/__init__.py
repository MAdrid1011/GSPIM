"""
GSPIM CUDA Extension - 重定向到cuda_extensions
为了兼容性保留此模块，实际加载在cuda_extensions中完成
"""

# 从统一的cuda_extensions模块导入
try:
    from cuda_extensions import gspim_preprocess_multiframe
except ImportError as e:
    print(f"[GSPIM] ⚠️ CUDA kernel加载失败: {e}")
    print("[GSPIM] 使用PyTorch fallback实现")
    gspim_preprocess_multiframe = None

__all__ = ['gspim_preprocess_multiframe']

