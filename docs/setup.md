# AGX Orin setup

## Environment

The reproduction environment runs on NVIDIA Jetson AGX Orin 64 GB in MAXN mode.

| Component | Version |
| --- | --- |
| Platform | Jetson AGX Orin 64 GB, aarch64, compute capability 8.7 |
| System | Ubuntu 24.04, L4T R39.2 |
| CUDA toolkit | 13.2 |
| Python | 3.12 |
| PyTorch | 2.14.0+cu132 |
| gsplat | 1.5.3 |
| NumPy / Pillow | 2.0.1 / 12.1.1 |
| Host compiler | GCC / G++ 13.3 |

Install the Jetson system and CUDA toolkit using
[NVIDIA's Jetson software distribution](https://developer.nvidia.com/embedded/jetpack).
The commands below use the toolkit at `/usr/local/cuda` and an existing Conda
installation.

## Python and build tools

From the repository root:

```bash
sudo apt-get install build-essential cmake ninja-build ffmpeg git zstd
conda create -n gspim --override-channels -c conda-forge python=3.12 pip -y
conda activate gspim
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -e '.[cuda]'
```

The PyTorch CUDA 13.2 index supplies the Python 3.12 aarch64 wheel used by this
AGX Orin environment. Installing it first lets the project installation use the
same CUDA-enabled build. NumPy and Pillow retain the image conversion conventions
in [Data preparation](data.md).

CMake builds the Ramulator2 backend with C++20. PyTorch and Ninja build the CUDA
extensions on first use. The renderer includes the CUDA 13 iterator interface;
all builds run automatically through the main entry. CUDA compilation defaults
to two parallel jobs, configurable through `MAX_JOBS`.

## External storage

Plan for 100 GB of SSD space for inputs, the Python environment, build caches,
traces, and temporary captures. Prepared datasets occupy approximately 48 GB,
and extracted checkpoints approximately 4 GB. Captures use both host memory and
temporary storage during conversion; the environment above has 64 GB of unified
memory.

In a fresh checkout, place the data and model directories on an SSD:

```bash
export GSPIM_STORAGE=/mnt/ssd/gspim
mkdir -p "$GSPIM_STORAGE"/{datasets,checkpoints,cache/downloads,cache/tmp,results}
ln -s "$GSPIM_STORAGE/datasets" datasets
ln -s "$GSPIM_STORAGE/checkpoints" checkpoints
export TMPDIR="$GSPIM_STORAGE/cache/tmp"
```

Extract the checkpoint archives into `$GSPIM_STORAGE`; each archive already
contains a `checkpoints/` prefix. Place the two requested scene archives in
`$GSPIM_STORAGE/cache/downloads/`. The [data guide](data.md) lists all input files.

The Conda environment can also live on the SSD: replace `-n gspim` with
`--prefix /mnt/ssd/gspim/env` when creating it, then activate that path.

## Prepare and run

For the standard storage layout:

```bash
python -m gspim.demo --prepare
python -m gspim.demo
```

For the SSD layout above:

```bash
python -m gspim.demo --prepare --cache "$GSPIM_STORAGE/cache"
python -m gspim.demo --cache "$GSPIM_STORAGE/cache" --out "$GSPIM_STORAGE/results"
```

Preparation initializes the pinned Ramulator2 and Golden-Ratio sources,
converts the datasets, and verifies the supplied checkpoints. The first evaluation
compiles extensions, records GPU work, converts captures into compact traces,
and builds PIM schedules. Later evaluations reuse these caches and collect fresh
GPU measurements and quality metrics. The terminal displays the active stage
throughout execution.

## Source organization

The training modules contain model construction and export functions, with
optional dependencies listed under `training` in `pyproject.toml`. Chisel source
and hardware design checks are retained in `hardware/`. The main evaluation
entry loads the released checkpoints and uses the GPU/PIM scheduling model.
