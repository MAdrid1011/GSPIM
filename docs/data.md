# Data preparation

## Sources and files

The four Neural 3D Video scenes are `coffee_martini`, `cook_spinach`,
`flame_steak`, and `sear_steak`. Dance3 and Actor1_4 use the official
Mobile-Stage and ENeRF-Outdoor camera packages. The preparation routine
downloads the Neural 3D Video v1.0 release archives, checks ZIP CRCs,
extracts the required members, and verifies their SHA256 values against
[`integrations/manifest.py`](../integrations/manifest.py).

```bash
python -m gspim.demo --prepare
```

Downloads are staged in `~/.cache/gspim/downloads/`. Completed scene archives are
removed after extraction. Original videos, camera poses, and processed images
remain as regular files in the checkout:

```text
datasets/n3v/<scene>/
  poses_bounds.npy
  cam00.mp4, cam01.mp4, ...
  cam00/0000.png, ..., 0299.png
  images/cam00_0000.png, ..., camXX_0299.png
```

Preparation verifies existing original files and processes up to four camera videos
concurrently, skipping completed image sequences.
A repeated invocation fills missing images without replacing complete images.
Delete a corrupt file before preparing again. Checkpoints are distributed as the
GSPIM checkpoint bundle rather than as part of a dataset download. A missing
checkpoint is reported with its required destination; see
[workloads](workloads.md).

Obtain Dance3 from [Mobile-Stage](https://docs.google.com/forms/d/e/1FAIpQLSeEbjuTV7w0lfryl-9FPX1VteuPGkbqjDvxXebY02Tm6BMejQ/viewform?usp=sf_link)
and Actor1_4 from [ENeRF-Outdoor](https://docs.google.com/forms/d/e/1FAIpQLSeFAgcnJEDYEaco4vA2QsD-bM8V8CRFuiR0QWbi_3uXmXtUKg/viewform?usp=sf_link).
Place their scene archives, `dance3_libx265.tar.gz` and
`actor1_4_libx265.tar.gz`, in `~/.cache/gspim/downloads/`, then use the same
preparation command. Preparation verifies the scene archive hashes, extracts
`videos_libx265/` and `optimized/intri.yml` / `optimized/extri.yml`, and writes
`poses_bounds.npy` and both image layouts. Existing extracted videos and
calibration can also be placed directly in `datasets/n3v/<scene>/`.

## Camera ordering and split

The Neural 3D Video and Dance3 scenes contain 300 frames at 30 frames per
second; Actor1_4 contains 200 frames. Sort the actual video filenames to match
rows of `poses_bounds.npy`. Coffee has eighteen cameras:
`00, 01, 02, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 16, 18, 19, 20`.
The other Neural 3D Video scenes contain cameras `00` through `20`. Official
camera lists are taken from their converted calibration files. Camera `00` is
the held-out view for Neural 3D Video. The official packages hold out the first
sorted calibration camera: `04` for Dance3 and `00` for Actor1_4. The prepared
`cam00/` directory and `images/cam00_*.png` names refer to that held-out view;
the original camera IDs also remain in `images/`. Training, pruning scores,
and contribution masks use only the other calibration cameras.

The Ex4DGS adapter reads the camera from the model archive. Explicit4D uses the
pinned original 4DGS normalization: align the average up direction, center the
camera rays, and scale the mean camera distance to four. Anchored uses the original
COLMAP coordinate frame. Its initialization matches source camera positions and
rotations to the pose rows geometrically.

## Image conversion

FFmpeg decodes original videos into full-resolution RGB24 frames. Both output
forms have half the original width and height, yielding 1352 by 1014 pixels.
They preserve separate conversion conventions:

- Ex4DGS held-out images use Pillow bilinear resizing and live in `cam00/`.
- Explicit4D and Anchored images use the original loader's float divide/multiply
  and byte cast, followed by Pillow's default RGB bicubic resizing. The conversion
  table is `(arange(256) / 255.0 * 255.0).astype(byte).view(uint8)`. All camera
  images live in `images/` with the actual camera ID in each filename.

The official-camera conversion applies the supplied pinhole distortion parameters
with an alpha-zero optimal camera matrix, then area-resizes 1920 by 1080 frames
to 960 by 540. The first sorted calibration camera supplies identical image pixels
to both held-out layouts; original camera IDs remain in `images/`.

Model loading and image decoding happen outside timed rendering. The renderer
uses the existing half-resolution image values directly for quality metrics.
