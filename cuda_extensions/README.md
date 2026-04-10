# GSPIM CUDA Extensions

所有GSPIM的CUDA扩展统一在此管理，使用JIT编译自动加载。

## 🚀 特点

- ✅ **自动编译** - 第一次运行时自动编译，无需手动安装
- ✅ **智能缓存** - 编译结果缓存到 `~/.cache/torch_extensions/`
- ✅ **增量编译** - 只在源文件修改时重新编译
- ✅ **Fallback机制** - CUDA不可用时自动回退到PyTorch实现

## 📦 包含的CUDA扩展

### 1. gspim_preprocess.cu
**Stage 2-3融合kernel**

功能：
- 4D→3D条件化投影（多帧并行）
- 深度序列计算
- 稳定性分析

性能：相比PyTorch实现快3-5倍

### 2. merge_sorted.cu
**Thrust归并kernel**（新增）

功能：
- 归并两个已排序序列
- 使用Thrust的 `merge_by_key`
- O(n+m)真正归并

性能：
- 100,000 + 5,000 元素：~0.5ms
- 相比PyTorch sort快3倍

## 🔧 使用方式

### 用户视角（无需任何操作）

直接运行程序：

```bash
python benchmark_gspim.py --model_path /path/to/model
```

### 第一次运行

```
[GSPIM] ✅ Stage 2-3 CUDA kernel 加载成功
[GSPIM] ✅ Thrust归并 CUDA kernel 加载成功
```

第一次会编译所有kernel（1-3分钟），后续使用缓存（秒级）。

### 编译失败时

```
[GSPIM] ⚠️  Thrust归并 CUDA kernel 加载失败: ...
[GSPIM]    将使用PyTorch fallback (功能相同，稍慢)
```

自动回退到PyTorch实现，不影响功能。

## 📝 开发者：添加新的CUDA扩展

### 1. 创建CUDA源文件

```bash
# cuda_extensions/my_kernel.cu
#include <torch/extension.h>

torch::Tensor my_function(torch::Tensor input) {
    // CUDA实现
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_function", &my_function, "My CUDA function");
}
```

### 2. 在 __init__.py 中注册

```python
# cuda_extensions/__init__.py

try:
    _my_kernel = load(
        name='my_kernel',
        sources=[os.path.join(cuda_dir, "my_kernel.cu")],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=False
    )
    my_function = _my_kernel.my_function
    print("[GSPIM] ✅ My kernel 加载成功")
except Exception as e:
    print(f"[GSPIM] ⚠️  My kernel 加载失败: {e}")
    my_function = None
```

### 3. 在代码中使用

```python
from cuda_extensions import my_function

if my_function is not None:
    result = my_function(input_tensor)
else:
    # PyTorch fallback
    result = pytorch_implementation(input_tensor)
```

## 🔍 工作原理

```python
from torch.utils.cpp_extension import load

_C = load(
    name='kernel_name',           # 编译后的模块名
    sources=['path/to/kernel.cu'], # CUDA源文件
    extra_cuda_cflags=['-O3'],    # 编译选项
    verbose=False
)
```

PyTorch的load函数会：

1. 计算源文件的哈希值
2. 检查缓存目录 `~/.cache/torch_extensions/py{ver}_cu{cuda}/kernel_name/`
3. 如果缓存存在且哈希匹配 → 直接加载 .so 文件
4. 否则 → 调用nvcc编译 → 缓存结果 → 加载

## 🗑️ 清除缓存

如果遇到奇怪的编译问题：

```bash
# 清除所有torch扩展缓存
rm -rf ~/.cache/torch_extensions/

# 或只清除GSPIM相关
rm -rf ~/.cache/torch_extensions/*/gspim_*
rm -rf ~/.cache/torch_extensions/*/merge_sorted*
```

下次运行会重新编译。

## 🐛 故障排除

### 编译失败

检查CUDA和PyTorch版本是否匹配：

```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

### 找不到nvcc

设置CUDA路径：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 链接错误

可能是CUDA版本不匹配，尝试：

```bash
# 清除缓存
rm -rf ~/.cache/torch_extensions/

# 重新运行
python your_script.py
```

## 📊 性能对比

| Kernel | PyTorch实现 | CUDA实现 | 加速比 |
|--------|------------|----------|--------|
| Stage 2-3投影 | ~3.0ms | ~0.8ms | 3.8x |
| Thrust归并 | ~1.5ms | ~0.5ms | 3.0x |

（测试场景：100,000个活跃高斯，窗口5帧）

## 📚 参考资料

- [PyTorch C++/CUDA Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CUDA Thrust Library](https://docs.nvidia.com/cuda/thrust/index.html)
- [torch.utils.cpp_extension.load](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load)
