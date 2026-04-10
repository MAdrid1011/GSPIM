# GSPIM CUDA Extensions

> **注意**: 此模块为兼容性保留，所有CUDA扩展现已统一在 `cuda_extensions/` 中管理。

## 📦 新的统一架构

所有CUDA扩展现在在 `cuda_extensions/` 中统一管理：

```
cuda_extensions/
├── __init__.py              # 统一的JIT编译加载器
├── gspim_preprocess.cu      # Stage 2-3融合kernel
└── merge_sorted.cu          # Thrust归并kernel
```

## 🚀 使用方式（无变化）

```bash
# 直接运行，自动编译
python benchmark_gspim.py --model_path /path/to/model
```

第一次运行时会自动编译所有CUDA扩展（1-3分钟），后续运行使用缓存（秒级）。

## 🔧 可用的CUDA扩展

1. **gspim_preprocess_multiframe** - Stage 2-3融合kernel
   - 4D→3D投影 + 深度序列 + 稳定性分析
   
2. **merge_sorted_sequences** - Thrust归并（新增）
   - O(n+m)真正归并，适合大数据量

## 📝 添加新的CUDA扩展

在 `cuda_extensions/__init__.py` 中添加：

```python
_new_kernel = load(
    name='new_kernel',
    sources=[os.path.join(cuda_dir, "new_kernel.cu")],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)
```

## 🗑️ 清除缓存

如果遇到编译问题：

```bash
rm -rf ~/.cache/torch_extensions/
```

