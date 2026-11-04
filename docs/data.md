# Data preparation

## Sources and files

The four scenes are `coffee_martini`, `cook_spinach`, `flame_steak`, and
`sear_steak` from [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video).
The preparation routine downloads their v1.0 release archives, checks ZIP CRCs,
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
Delete a corrupt file before preparing again. A missing checkpoint with no public
URL is reported with its destination; see [workloads](workloads.md).

## Camera ordering and split

Each scene contains 300 frames at 30 frames per second. Sort the actual video
filenames to match rows of `poses_bounds.npy`. Coffee has eighteen cameras:
`00, 01, 02, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 16, 18, 19, 20`.
The other scenes contain cameras `00` through `20`. Camera `00` is the held-out
view; training, pruning scores, and contribution masks use the other cameras.

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

Model loading and image decoding happen outside timed rendering. The renderer
uses the existing half-resolution image values directly for quality metrics.
