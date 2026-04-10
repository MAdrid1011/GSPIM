"""
GSPIM CUDA扩展构建脚本
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='gspim_cuda_ops',
    ext_modules=[
        CUDAExtension(
            name='gspim_cuda_ops',
            sources=['gspim_ops.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-arch=sm_70',  # 适配Volta及以上架构
                    '--extended-lambda',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

