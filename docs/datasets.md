# Datasets Guide

GSPIM evaluates on the same dynamic scene benchmarks used in the original
4DGS paper.  This page covers how to download and prepare each dataset.

---

## Supported datasets

| Dataset                     | Scenes | Resolution | Frames | Download size |
|-----------------------------|--------|------------|--------|---------------|
| Neural 3D Video (N3V)       | 6      | 2704×2028  | 300    | ~100 GB       |
| Plenoptic Video              | 6      | 1920×1080  | 150    | ~30 GB        |
| HyperNeRF                   | many   | variable   | 50–200 | ~20 GB each   |
| D-NeRF (synthetic)          | 8      | 400×400    | 100    | ~2 GB         |

---

## Neural 3D Video (N3V)

Download from the [official project page](https://neural-3d-video.github.io/):

```bash
# 6 scenes: cook_spinach, cut_roasted_beef, flame_salmon,
#           flame_steak, sear_steak, coffee_martini
wget https://github.com/facebookresearch/Neural-3D-Video/releases/download/data/<scene>.tar.gz
tar -xzf <scene>.tar.gz -C data/N3V/
```

### Directory structure expected by GSPIM

```
data/N3V/<scene>/
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── images/
│   ├── cam00/
│   │   ├── 000000.png
│   │   └── ...
│   └── cam01/
│       └── ...
└── colmap_output.txt
```

### Training command

```bash
python train.py \
    -s data/N3V/flame_salmon \
    -m output/N3V/flame_salmon \
    --eval
```

---

## Plenoptic Video

Download from the [Plenoptic Video project](https://video-nerf.github.io/):

```bash
wget https://storage.googleapis.com/plenoptic-video/<scene>.zip
unzip <scene>.zip -d data/plenoptic/
```

### Training command

```bash
python train.py \
    -s data/plenoptic/<scene> \
    -m output/plenoptic/<scene> \
    --eval
```

---

## HyperNeRF

Download from [hypernerf.github.io](https://hypernerf.github.io/):

```bash
# Example: 3D printer scene
wget https://storage.googleapis.com/hypernerf/3d-printer.zip
unzip 3d-printer.zip -d data/hypernerf/
```

HyperNeRF scenes use the **NeRF-style JSON** format:

```
data/hypernerf/3d-printer/
├── camera/
├── rgb/
├── dataset.json
└── metadata.json
```

### Training command

```bash
python train.py \
    -s data/hypernerf/3d-printer \
    -m output/hypernerf/3d-printer \
    --eval
```

---

## D-NeRF (synthetic)

Download from the [D-NeRF repository](https://github.com/albertpumarola/D-NeRF):

```bash
wget https://github.com/albertpumarola/D-NeRF/releases/download/data/data.zip
unzip data.zip -d data/dnerf/
```

D-NeRF uses the NeRF-synthetic JSON format with `transforms_train.json` and
`transforms_test.json`.

### Training command

```bash
python train.py \
    -s data/dnerf/hook \
    -m output/dnerf/hook \
    --eval \
    --sh_degree 1       # synthetic scenes work well with lower SH
```

---

## Custom scenes

### Using COLMAP

1. Collect multi-view video frames (at least 3 cameras recommended).
2. Run COLMAP sparse reconstruction:

```bash
colmap automatic_reconstructor \
    --workspace_path data/custom/colmap \
    --image_path data/custom/images
```

3. Train:

```bash
python train.py \
    -s data/custom \
    -m output/custom \
    --eval
```

### Using VGGT (optional point-cloud bootstrap)

If you have access to the VGGT model (`submodules/vggt/`), you can use it
to generate an initial point cloud from a monocular video:

```bash
python submodules/vggt/scripts/bootstrap_pointcloud.py \
    --video data/custom/video.mp4 \
    --output data/custom/sparse/0
```

Then proceed with the standard training command above.

---

## Data format reference

| Format           | `cameras` source | Image source | Timestamp source        |
|-----------------|-----------------|-------------|------------------------|
| COLMAP           | `cameras.bin`   | `images/`   | filename or JSON        |
| NeRF-synthetic   | `transforms_*.json` | JSON paths | `"time"` field in JSON |
| HyperNeRF        | `camera/*.json` | `rgb/`      | `dataset.json`          |

---

## Benchmark evaluation

After training, render and evaluate with:

```bash
python benchmark_gspim.py \
    -m output/N3V/flame_salmon \
    --eval_quality \
    --window_size 5
```

Results are printed to console and saved to `output/N3V/flame_salmon/metrics.json`.
