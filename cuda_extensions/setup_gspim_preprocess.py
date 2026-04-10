"""
Setup script for GSPIM CUDA preprocessing extension
编译 Stage 2-4 的融合CUDA kernel
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# CUDA编译选项
extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '-Xptxas', '-O3',
        '--expt-relaxed-constexpr',
        '-gencode=arch=compute_70,code=sm_70',  # V100
        '-gencode=arch=compute_75,code=sm_75',  # T4, RTX 20xx
        '-gencode=arch=compute_80,code=sm_80',  # A100
        '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
        '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
        '-gencode=arch=compute_90,code=sm_90',  # H100
    ]
}

setup(
    name='gspim_preprocess',
    ext_modules=[
        CUDAExtension(
            name='gspim_preprocess',
            sources=['gspim_preprocess.cu'],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

