# Third-party notices

GSPIM code is covered by [LICENSE](LICENSE). Data, weights, upstream sources, and
installed dependencies retain their respective licenses and notices.

- [Ex4DGS](https://github.com/juno181/Ex4DGS), Junoh Lee et al., NeurIPS 2024:
  revision `7fac64997164fde64732c80261e56adf66ca14df` supplies interpolation
  conventions; v0.1 releases supply the model archives. `LICENSE.txt` specifies
  CC BY-NC 4.0.
- [Original 4D Gaussian Splatting](https://github.com/LogosIntelligenceLab/4d-gaussian-splatting),
  Zeyu Yang et al., ICLR 2024: revision
  `121e360ce23d6e9ca7734072393414c844dfaea2` supplies checkpoint conventions,
  camera normalization, model training methods, and the differentiable rasterizer.
  The root license is MIT, copyright 2024 Fudan Zhang Vision Group. Inherited
  Gaussian Splatting components use `LICENSE_gaussian_splatting.md`; preserve
  those terms when using or redistributing the rasterizer and derived material.
- [4DGS-1K](https://papers.neurips.cc/paper_files/paper/2025/hash/6c39d1a7eecfd98570d74bf7efec1be7-Abstract-Conference.html):
  the implementation uses its pruning and contribution-mask algorithms on original
  4DGS checkpoints. Volume weighting follows
  [LightGaussian](https://github.com/VITA-Group/LightGaussian).
- [Anchored 4DGS](https://doi.org/10.1145/3757377.3763898): the anchor model follows
  the paper and appendix, with decoder parameters recorded in checkpoints.
  [Scaffold-GS](https://github.com/city-super/Scaffold-GS) revision
  `59c833b56bbbf510f3f64d40f81721995caced66` supplies the cited growth and pruning
  conventions and carries its upstream research license.
- [fused-ssim](https://github.com/rahul-goel/fused-ssim) revision
  `a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38` supplies the differentiable SSIM
  loss under its MIT license.
- [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video) v1.0
  supplies the videos and camera poses. Follow the upstream dataset terms and
  attribution requirements.
- PyTorch and gsplat provide tensor operations and CUDA rendering components;
  CUDA provides CUB. Their installed packages retain their own licenses.

The preparation routine creates pinned source checkouts under the Git-ignored
`third_party/` directory. Model construction functions load upstream utilities
from those checkouts. Checkpoint formats and provenance are described in
[workloads](docs/workloads.md).
