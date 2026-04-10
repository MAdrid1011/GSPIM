#!/usr/bin/env python3
"""
GSPIM渲染脚本
使用GSPIM架构渲染4D高斯泼溅视频
"""

import torch
import os
import sys
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from PIL import Image

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from gspim_renderer import GSPIMRenderer
from utils.general_utils import safe_state


def render_gspim(model_path, iteration, pipeline_params, skip_train=False, skip_test=False,
                 output_path=None, L2_cache_size=40*1024*1024, time_threshold=0.05,
                 initial_window_size=5, enable_pim_sim=False, fps=30):
    """
    使用GSPIM渲染4DGS场景
    
    Args:
        model_path: 模型路径
        iteration: 迭代次数
        pipeline_params: 渲染管线参数
        skip_train: 跳过训练集
        skip_test: 跳过测试集
        output_path: 输出路径
        L2_cache_size: L2缓存大小
        time_threshold: 时间贡献阈值
        initial_window_size: 初始窗口大小
        enable_pim_sim: 是否启用PIM模拟器
        fps: 视频帧率
    """
    
    # 加载模型
    print(f"Loading model from: {model_path}")
    
    # 查找checkpoint
    if iteration == -1:
        # 查找最佳checkpoint
        checkpoint_path = Path(model_path) / "output" / "chkpnt_best.pth"
        if not checkpoint_path.exists():
            # 尝试其他可能的路径
            for name in os.listdir(model_path):
                if name.startswith("output"):
                    candidate = Path(model_path) / name / "chkpnt_best.pth"
                    if candidate.exists():
                        checkpoint_path = candidate
                        break
    else:
        checkpoint_path = Path(model_path) / f"chkpnt{iteration}.pth"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Available files in {model_path}:")
        if os.path.exists(model_path):
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith('.pth'):
                        print(f"  {os.path.join(root, file)}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # 创建高斯模型
    gaussians = GaussianModel(sh_degree=3, gaussian_dim=4, time_duration=[-0.5, 0.5], 
                              rot_4d=True, force_sh_3d=False, sh_degree_t=2)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_params, iteration = checkpoint
    gaussians.restore(model_params, None)
    
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
    # 创建场景（用于获取相机）
    dataset = ModelParams()
    dataset.source_path = model_path
    dataset.model_path = model_path
    dataset.sh_degree = 3
    dataset.images = "images"
    dataset.resolution = -1
    dataset.white_background = False
    dataset.data_device = "cuda"
    
    try:
        scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
    except:
        print("Warning: Could not load scene, will use checkpoint only")
        scene = None
    
    # 创建GSPIM渲染器
    print(f"Initializing GSPIM Renderer...")
    print(f"  L2 Cache Size: {L2_cache_size / 1024 / 1024:.1f} MB")
    print(f"  Time Threshold: {time_threshold}")
    print(f"  Initial Window Size: {initial_window_size}")
    print(f"  Enable PIM Simulator: {enable_pim_sim}")
    
    renderer = GSPIMRenderer(
        L2_cache_size=L2_cache_size,
        time_threshold=time_threshold,
        initial_window_size=initial_window_size,
        enable_pim_sim=enable_pim_sim
    )
    
    # 准备输出目录
    if output_path is None:
        output_path = Path(model_path) / "gspim_render"
    else:
        output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output path: {output_path}")
    
    # 背景颜色
    bg_color = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], 
                           dtype=torch.float32, device="cuda")
    
    # 渲染测试集
    if not skip_test and scene is not None:
        print("\n" + "="*80)
        print("Rendering Test Set with GSPIM")
        print("="*80)
        
        test_cameras = scene.getTestCameras()
        render_set(test_cameras, gaussians, renderer, pipeline_params, 
                  bg_color, output_path / "test", fps)
    
    # 渲染训练集
    if not skip_train and scene is not None:
        print("\n" + "="*80)
        print("Rendering Train Set with GSPIM")
        print("="*80)
        
        train_cameras = scene.getTrainCameras()
        render_set(train_cameras, gaussians, renderer, pipeline_params, 
                  bg_color, output_path / "train", fps)
    
    print("\n" + "="*80)
    print("Rendering Complete!")
    print("="*80)


def render_set(cameras, gaussians, renderer, pipe, bg_color, output_path, fps=30):
    """
    渲染一个相机集合
    
    Args:
        cameras: 相机列表
        gaussians: 高斯模型
        renderer: GSPIM渲染器
        pipe: 渲染管线参数
        bg_color: 背景颜色
        output_path: 输出路径
        fps: 视频帧率
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Total cameras: {len(cameras)}")
    
    # 按时间戳排序相机
    cameras = sorted(cameras, key=lambda x: x.timestamp if hasattr(x, 'timestamp') else 0)
    
    # 使用GSPIM渲染整个序列
    print("Rendering with GSPIM architecture...")
    
    with torch.no_grad():
        rendered_images, statistics = renderer.render_video_sequence(
            gaussians, cameras, pipe, bg_color, scaling_modifier=1.0
        )
    
    # 保存图像
    print("\nSaving rendered images...")
    for idx, result in enumerate(tqdm(rendered_images)):
        # 转换为numpy
        image = result['render'].detach().cpu().clamp(0, 1)
        image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # 保存
        Image.fromarray(image).save(output_path / f"{idx:05d}.png")
    
    # 保存深度图（可选）
    depth_path = output_path / "depth"
    depth_path.mkdir(exist_ok=True)
    for idx, result in enumerate(tqdm(rendered_images, desc="Saving depth")):
        if 'depth' in result and result['depth'] is not None:
            depth = result['depth'].detach().cpu().squeeze()
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-7)
            depth_image = (depth_normalized.numpy() * 255).astype(np.uint8)
            Image.fromarray(depth_image).save(depth_path / f"{idx:05d}.png")
    
    # 打印统计信息
    print("\n" + "="*80)
    print("GSPIM Rendering Statistics")
    print("="*80)
    print(f"Total Frames: {statistics['total_frames']}")
    print(f"Average Active Gaussians per Frame: {statistics['avg_active_gaussians']:.0f}")
    print(f"Average Window Size: {statistics['avg_window_size']:.2f}")
    print(f"Average Batch Count: {statistics['avg_batch_count']:.2f}")
    print("="*80)
    
    # 创建视频（使用ffmpeg）
    try:
        import subprocess
        video_path = output_path / "video.mp4"
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', str(output_path / '%05d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            str(video_path)
        ]
        print(f"\nCreating video: {video_path}")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Video saved: {video_path}")
    except Exception as e:
        print(f"Could not create video: {e}")
        print(f"You can manually create video from: {output_path}/*.png")


def main():
    parser = ArgumentParser(description="GSPIM Rendering Script")
    
    # 基本参数
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load (-1 for best)")
    parser.add_argument("--skip_train", action="store_true", help="Skip rendering training set")
    parser.add_argument("--skip_test", action="store_true", help="Skip rendering test set")
    parser.add_argument("--output_path", type=str, default=None, help="Output path")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    
    # GSPIM参数
    parser.add_argument("--l2_cache_size", type=int, default=40, help="L2 cache size in MB")
    parser.add_argument("--time_threshold", type=float, default=0.05, help="Time contribution threshold")
    parser.add_argument("--window_size", type=int, default=5, help="Initial window size")
    parser.add_argument("--enable_pim_sim", action="store_true", help="Enable PIM simulator")
    
    # 渲染管线参数
    parser.add_argument("--compute_cov3D_python", action="store_true", help="Compute covariance in Python")
    parser.add_argument("--convert_SHs_python", action="store_true", help="Convert SHs in Python")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # 创建渲染管线参数
    pipe = PipelineParams()
    pipe.compute_cov3D_python = args.compute_cov3D_python or True  # GSPIM需要
    pipe.convert_SHs_python = args.convert_SHs_python
    pipe.debug = args.debug
    pipe.env_map_res = 0
    pipe.eval_shfs_4d = True
    
    # 渲染
    render_gspim(
        model_path=args.model_path,
        iteration=args.iteration,
        pipeline_params=pipe,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        output_path=args.output_path,
        L2_cache_size=args.l2_cache_size * 1024 * 1024,
        time_threshold=args.time_threshold,
        initial_window_size=args.window_size,
        enable_pim_sim=args.enable_pim_sim,
        fps=args.fps
    )


if __name__ == "__main__":
    main()

