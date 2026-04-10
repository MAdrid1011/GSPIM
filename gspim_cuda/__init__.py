"""
gspim_cuda — JIT CUDA kernel loader for GSPIM preprocessing.

Wraps cuda_extensions and provides a graceful PyTorch fallback when
the compiled kernel is unavailable.
"""

try:
    from cuda_extensions import gspim_preprocess_multiframe
except ImportError as e:
    print(f"[gspim_cuda] CUDA kernel unavailable ({e}); using PyTorch fallback.")
    gspim_preprocess_multiframe = None

__all__ = ['gspim_preprocess_multiframe']
