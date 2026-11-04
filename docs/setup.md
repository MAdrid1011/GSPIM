# Setup

## CUDA environment

Use Linux, Python 3.10 through 3.12, an NVIDIA CUDA GPU, the CUDA toolkit with `nvcc`,
and a host C++ compiler supported by that toolkit. The CUDA dependencies are
PyTorch 2.5.0, gsplat 1.5.3, NumPy 2.0.1, Pillow 12.1.1, plyfile, and Ninja.
Pillow and NumPy versions preserve the image conversion described in [data](data.md).

Install FFmpeg and Git using the system package manager. For example, on Ubuntu:

```bash
sudo apt-get install ffmpeg git ninja-build
```

Install PyTorch using a wheel compatible with the installed toolkit. For CUDA 12.1:

```bash
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[cuda]'
```

If the toolkit or compiler is outside the default search path, set `CUDA_HOME`,
`CC`, and `CXX` before running. A CUDA 12.1 toolkit under `/usr` with GCC 11 uses:

```bash
export CUDA_HOME=/usr CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
```

The first render builds `gspim/renderer.cu` and, when necessary, gsplat extensions.
Compilation uses `~/.cache/gspim/` and defaults to two parallel jobs. `MAX_JOBS`
can override the compilation concurrency. An optional `--cache` selects another
cache directory outside the checkout.

## Model construction dependencies

`python -m gspim.demo --prepare` also creates pinned upstream source checkouts in
`third_party/`. Training functions use the original 4DGS rasterizer and utilities,
and fused-ssim for the loss. Install their additional Python dependencies after
the CUDA environment is ready:

```bash
python -m pip install --no-build-isolation -e '.[training]'
```

The available functions and checkpoint paths are described in [workloads](workloads.md).

## Design checks

Install and run the Python design tests:

```bash
python -m pip install -e '.[test]'
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider
```

RTL checks require JDK 17 or 21, sbt, and Verilator. From `hardware/`, run
`./verify-rtl.sh`. Generated RTL and hardware build directories are ignored by Git.
