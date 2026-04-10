#!/usr/bin/env python3
"""
GSPIM性能基准测试脚本
比较GSPIM与baseline的渲染性能和质量
"""

import torch
import time
import argparse
from pathlib import Path
import json
import sys
import os

from scene import Scene, GaussianModel
from gaussian_renderer import render
from gspim.renderer import GSPIMRenderer, print_stage_timing_statistics
from arguments import ModelParams, PipelineParams
from gspim.profiler import GSPIMProfiler
from utils.image_utils import psnr
from utils.loss_utils import ssim


def save_rendered_video(rendered_images, output_path, fps=30):
    """保存渲染结果为视频和图像序列"""
    import cv2
    import numpy as np
    from torchvision.utils import save_image
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存图像序列
    frames_dir = output_path / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    frame_list = []
    for idx, img in enumerate(rendered_images):
        if isinstance(img, dict):
            img = img['render']
        
        # 保存单帧图像
        img_clamped = torch.clamp(img, 0, 1)
        save_image(img_clamped, frames_dir / f"frame_{idx:04d}.png")
        
        # 转换为OpenCV格式用于视频
        img_np = (img_clamped.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        frame_list.append(img_bgr)
    
    # 保存视频
    if frame_list:
        h, w = frame_list[0].shape[:2]
        video_path = output_path / "video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        for frame in frame_list:
            out.write(frame)
        out.release()
        print(f"  Saved video to {video_path}")
        print(f"  Saved {len(frame_list)} frames to {frames_dir}")


def load_gt_image(camera, bg_color):
    """加载ground truth图像（支持懒加载模式）"""
    from utils.general_utils import PILtoTorch
    from PIL import Image
    import numpy as np
    
    if camera.meta_only:
        # 懒加载模式：从文件读取
        with Image.open(camera.image_path) as image_load:
            im_data = np.array(image_load.convert("RGBA"))
        bg = np.array([0, 0, 0])  # 假设黑色背景
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image_pil = Image.fromarray(np.array(arr*255.0, dtype=np.uint8))
        gt_img = PILtoTorch(image_pil, camera.resolution)[:3, ...].clamp(0.0, 1.0)
    else:
        # 预加载模式：直接使用
        gt_img = camera.image
    
    return gt_img.cuda()


def preload_gt_images_parallel(cameras, bg_color, num_workers=8):
    """并行预加载所有 GT 图像到 GPU（解决 PSNR 计算慢的问题）"""
    from concurrent.futures import ThreadPoolExecutor
    from utils.general_utils import PILtoTorch
    from PIL import Image
    import numpy as np
    
    def load_single_gt(camera):
        """加载单个 GT 图像（CPU 部分）"""
        try:
            if camera.meta_only:
                with Image.open(camera.image_path) as image_load:
                    im_data = np.array(image_load.convert("RGBA"))
                bg = np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image_pil = Image.fromarray(np.array(arr*255.0, dtype=np.uint8))
                gt_img = PILtoTorch(image_pil, camera.resolution)[:3, ...].clamp(0.0, 1.0)
            else:
                gt_img = camera.image
            return gt_img
        except Exception as e:
            return None
    
    # 并行加载 GT 图像（磁盘 IO 并行）
    gt_images = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        gt_images = list(executor.map(load_single_gt, cameras))
    
    # 批量上传到 GPU
    gt_images_gpu = []
    for gt in gt_images:
        if gt is not None:
            gt_images_gpu.append(gt.cuda())
        else:
            gt_images_gpu.append(None)
    
    return gt_images_gpu


def compute_metrics(rendered_images, cameras, bg_color, batch_size=32):
    """计算渲染质量指标（PSNR, SSIM）- 优化版本"""
    psnr_list = []
    ssim_list = []
    
    num_images = len(rendered_images)
    
    # 预加载所有 GT 图像（并行 IO，一次性完成）
    print("  Preloading GT images...", end='\r')
    gt_images_gpu = preload_gt_images_parallel(cameras, bg_color)
    
    # 批量计算指标
    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        batch_renders = []
        batch_gts = []
        
        for idx in range(batch_start, batch_end):
            rendered = rendered_images[idx]
            gt_img = gt_images_gpu[idx]
            
            if gt_img is None:
                continue
            
            # 获取渲染图像
            if isinstance(rendered, dict):
                render_img = rendered['render']
            else:
                render_img = rendered
            
            # 确保在 GPU 上
            if not render_img.is_cuda:
                render_img = render_img.cuda()
            
            # 确保维度一致
            if render_img.shape != gt_img.shape:
                continue
            
            batch_renders.append(render_img)
            batch_gts.append(gt_img)
        
        if not batch_renders:
            continue
        
        # 批量堆叠
        render_batch = torch.stack(batch_renders)  # [B, 3, H, W]
        gt_batch = torch.stack(batch_gts)  # [B, 3, H, W]
        
        # 批量计算PSNR (GPU)
        mse_batch = ((render_batch - gt_batch) ** 2).view(render_batch.shape[0], -1).mean(dim=1)
        psnr_batch = 20 * torch.log10(1.0 / torch.sqrt(mse_batch + 1e-10))
        psnr_list.extend(psnr_batch.cpu().tolist())
        
        # 批量计算SSIM (GPU)
        ssim_batch = ssim(render_batch, gt_batch)
        if ssim_batch.numel() == 1:
            ssim_list.append(ssim_batch.item())
        else:
            ssim_list.extend(ssim_batch.cpu().tolist())
        
        print(f"  Computed metrics for {batch_end}/{num_images} frames", end='\r')
    
    print()  # 换行
    
    # 释放 GT 图像显存
    del gt_images_gpu
    torch.cuda.empty_cache()
    
    return {
        'psnr_mean': sum(psnr_list) / len(psnr_list) if psnr_list else 0,
        'ssim_mean': sum(ssim_list) / len(ssim_list) if ssim_list else 0,
        'psnr_list': psnr_list,
        'ssim_list': ssim_list
    }


def benchmark_baseline(gaussians, cameras, pipe, bg_color, compute_quality=False, label="Baseline"):
    """基准渲染（逐帧）"""
    print(f"\n[{label}] Frame-by-frame rendering...")
    print(f"  (渲染全部 {gaussians.get_xyz.shape[0]:,} 个高斯)")
    
    profiler = GSPIMProfiler(enable_cuda_profiler=True)
    rendered_images = []
    
    # 确保GPU准备好
    torch.cuda.synchronize()
    total_start = time.time()
    
    for idx, camera in enumerate(cameras):
        start = profiler.start_event(f'frame_{idx}')
        
        with torch.no_grad():
            camera_cuda = camera.cuda()
            result = render(camera_cuda, gaussians, pipe, bg_color, scaling_modifier=1.0)
            if compute_quality:
                rendered_images.append(result['render'].clone())
        
        profiler.end_event(f'frame_{idx}', start)
        
        if idx % 10 == 0:
            print(f"  Rendered {idx}/{len(cameras)} frames", end='\r')
    
    # 等待所有GPU操作完成
    torch.cuda.synchronize()
    total_time = (time.time() - total_start) * 1000  # ms
    
    print(f"\n  Total time: {total_time:.2f} ms")
    print(f"  Average per frame: {total_time/len(cameras):.2f} ms")
    print(f"  FPS: {1000*len(cameras)/total_time:.2f}")
    
    result = {
        'total_time_ms': total_time,
        'avg_frame_time_ms': total_time / len(cameras),
        'fps': 1000 * len(cameras) / total_time,
        'profiler': profiler
    }
    
    if compute_quality:
        result['rendered_images'] = rendered_images
    
    return result


def benchmark_gspim(gaussians, cameras, pipe, bg_color, window_size=5, 
                   l2_cache_size=40*1024*1024, time_threshold=0.05, compute_quality=False,
                   use_cuda_kernel=True):
    """GSPIM渲染 - 使用时间贡献过滤，只渲染活跃高斯"""
    print("\n[GSPIM] Multi-frame batch rendering...")
    print(f"  (使用时间过滤，预计只渲染 ~{time_threshold*100:.0f}% 活跃高斯)")
    if use_cuda_kernel:
        print(f"  (Stage 2-4: 尝试使用CUDA融合kernel)")
    
    renderer = GSPIMRenderer(
        L2_cache_size=l2_cache_size,
        time_threshold=time_threshold,
        initial_window_size=window_size,
        enable_pim_sim=False,
        enable_profiling=True,
        use_cuda_kernel=use_cuda_kernel
    )
    
    # 确保GPU准备好
    torch.cuda.synchronize()
    total_start = time.time()
    
    with torch.no_grad():
        rendered_images, statistics = renderer.render_video_sequence(
            gaussians, cameras, pipe, bg_color, scaling_modifier=1.0
        )
    
    # 等待所有GPU操作完成
    torch.cuda.synchronize()
    total_time = (time.time() - total_start) * 1000  # ms
    
    print(f"\n  Total time: {total_time:.2f} ms")
    print(f"  Average per frame: {total_time/len(cameras):.2f} ms")
    print(f"  FPS: {1000*len(cameras)/total_time:.2f}")
    print(f"  Active Ratio: {statistics['avg_active_gaussians']:.1f} / {gaussians.get_xyz.shape[0]} ({statistics['avg_active_gaussians']/gaussians.get_xyz.shape[0]*100:.1f}%)")
    
    result = {
        'total_time_ms': total_time,
        'avg_frame_time_ms': total_time / len(cameras),
        'fps': 1000 * len(cameras) / total_time,
        'statistics': statistics,
        'renderer': renderer
    }
    
    if compute_quality:
        result['rendered_images'] = rendered_images
    
    return result


def print_comparison(baseline_result, gspim_result):
    """打印性能比较"""
    print("\n" + "="*80)
    print("Performance Comparison")
    print("="*80)
    
    speedup = baseline_result['total_time_ms'] / gspim_result['total_time_ms']
    fps_improvement = (gspim_result['fps'] - baseline_result['fps']) / baseline_result['fps'] * 100
    
    print(f"\n{'Metric':<30} {'Baseline':<20} {'GSPIM':<20} {'Improvement':<20}")
    print("-" * 80)
    print(f"{'Total Time (ms)':<30} {baseline_result['total_time_ms']:<20.2f} {gspim_result['total_time_ms']:<20.2f} {speedup:<20.2f}x")
    print(f"{'Avg Frame Time (ms)':<30} {baseline_result['avg_frame_time_ms']:<20.2f} {gspim_result['avg_frame_time_ms']:<20.2f}")
    print(f"{'FPS':<30} {baseline_result['fps']:<20.2f} {gspim_result['fps']:<20.2f} {fps_improvement:<20.1f}%")
    
    if 'statistics' in gspim_result:
        stats = gspim_result['statistics']
        if 'bandwidth' in stats:
            bw = stats['bandwidth']
            print(f"\n{'Bandwidth Savings':<30} {'-':<20} {'-':<20} {bw['savings_percent']:<20.1f}%")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="GSPIM Benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--source_path", type=str, default=None, help="Path to source data (if different from model_path)")
    parser.add_argument("--num_frames", type=int, default=50, help="Number of frames to benchmark")
    parser.add_argument("--window_size", type=int, default=5, help="GSPIM window size")
    parser.add_argument("--l2_cache_size", type=int, default=50, help="L2 cache size in MB")
    parser.add_argument("--time_threshold", type=float, default=0.05, help="Time contribution threshold")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline benchmark")
    parser.add_argument("--quality", action="store_true", help="Compute PSNR/SSIM quality metrics")
    parser.add_argument("--save_video", action="store_true", help="Save rendered videos for comparison")
    parser.add_argument("--video_output", type=str, default="render_comparison", help="Output directory for videos")
    parser.add_argument("--no_cuda_kernel", action="store_true", help="禁用CUDA融合kernel，使用PyTorch fallback")
    
    args = parser.parse_args()
    
    print("="*80)
    print("GSPIM Benchmark")
    print("="*80)
    print(f"Model Path: {args.model_path}")
    print(f"Frames: {args.num_frames}")
    print(f"Window Size: {args.window_size}")
    print(f"L2 Cache: {args.l2_cache_size} MB")
    print(f"Time Threshold: {args.time_threshold}")
    print("="*80)
    
    # 加载模型检查点
    print("\nLoading model...")
    checkpoint_path = Path(args.model_path) / "chkpnt_best.pth"
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_params, saved_iteration = checkpoint
    
    # 先创建高斯模型对象（稍后在Scene之后恢复权重）
    gaussians = GaussianModel(sh_degree=3, gaussian_dim=4, time_duration=[-0.5, 0.5], 
                              rot_4d=True, force_sh_3d=False, sh_degree_t=2)
    
    # 确定数据源路径
    if args.source_path:
        source_path = args.source_path
    else:
        # 尝试自动检测：从 output/X/Y 推断为 data/X/Y
        model_path = Path(args.model_path)
        if model_path.parts[0] == 'output' and len(model_path.parts) >= 3:
            source_path = str(Path('data') / model_path.parts[1] / model_path.parts[2])
            print(f"Auto-detected source path: {source_path}")
        else:
            source_path = args.model_path
    
    # 检查source_path是否存在必要的文件
    if not Path(source_path).exists():
        print(f"\nError: Source path does not exist: {source_path}")
        print(f"Please specify --source_path explicitly")
        print(f"\nExample: python benchmark_gspim.py --model_path output/N3V/coffee_martini --source_path data/N3V/coffee_martini")
        return
    
    if not (Path(source_path) / "transforms_train.json").exists():
        print(f"\nError: transforms_train.json not found in {source_path}")
        print(f"Please check your source path")
        return
    
    print(f"\nData Source: {source_path}")
    
    # 加载场景
    dummy_parser = argparse.ArgumentParser()
    dataset = ModelParams(dummy_parser)
    dataset_args = argparse.Namespace()
    dataset_args.source_path = source_path
    dataset_args.model_path = args.model_path
    dataset_args.sh_degree = 3
    dataset_args.white_background = False
    dataset_args.images = "images"
    dataset_args.resolution = -1
    dataset_args.data_device = "cuda"
    dataset_args.eval = True
    dataset_args.extension = ".png"
    dataset_args.num_extra_pts = 0
    dataset_args.loaded_pth = ""
    dataset_args.frame_ratio = 1
    dataset_args.dataloader = True  # 懒加载图像，加速启动
    dataset = dataset.extract(dataset_args)
    
    try:
        scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
        cameras = scene.getTestCameras()
        # CameraDataset 对象需要通过 viewpoint_stack 属性获取相机列表
        if hasattr(cameras, 'viewpoint_stack'):
            cameras = cameras.viewpoint_stack
    except Exception as e:
        print(f"\nError: Could not load scene cameras")
        print(f"Details: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 在Scene之后恢复高斯权重（Scene会用点云初始化，这里覆盖为训练好的模型）
    gaussians.restore(model_params, None)
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians from checkpoint")
    
    # 限制帧数
    cameras = cameras[:args.num_frames]
    cameras = sorted(cameras, key=lambda x: x.timestamp if hasattr(x, 'timestamp') else 0)
    
    print(f"Using {len(cameras)} cameras")
    
    # 准备渲染
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # Pipeline配置 - 原版4DGS默认配置 (与 arguments/__init__.py 完全一致)
    pipe_parser = argparse.ArgumentParser()
    pp = PipelineParams(pipe_parser)
    pipe_args, _ = pipe_parser.parse_known_args([])
    pipe = pp.extract(pipe_args)
    
    print(f"\nPipeline Config (原版4DGS默认):")
    print(f"  compute_cov3D_python: {pipe.compute_cov3D_python}")
    print(f"  convert_SHs_python:   {pipe.convert_SHs_python}")
    print(f"  debug:                {pipe.debug}")
    
    # Warm-up
    print("\nWarming up GPU...")
    with torch.no_grad():
        camera_cuda = cameras[0].cuda()
        _ = render(camera_cuda, gaussians, pipe, bg_color)
    torch.cuda.synchronize()
    
    # 是否需要保存渲染图像
    need_images = args.quality or args.save_video

    # Baseline: 原版4DGS (纯CUDA)
    baseline_result = benchmark_baseline(gaussians, cameras, pipe, bg_color, 
                                         compute_quality=need_images, label="4DGS Baseline")
    
    # GSPIM benchmark
    # PIM 时间过滤在 Python 中模拟，GPU 渲染用 CUDA kernel
    gspim_result = benchmark_gspim(
        gaussians, cameras, pipe, bg_color,
        window_size=args.window_size,
        l2_cache_size=args.l2_cache_size * 1024 * 1024,
        time_threshold=args.time_threshold,
        compute_quality=need_images,
        use_cuda_kernel=(not args.no_cuda_kernel)
    )

    # 获取 GSPIM 的 PIM/GPU 并行时间分析
    timing_stats = gspim_result['statistics'].get('timing', None)
    
    # 计算并行时间下的 GSPIM 性能
    if timing_stats:
        gspim_parallel_time = timing_stats['parallel_total_ms']
        gspim_parallel_fps = len(cameras) / (gspim_parallel_time / 1000)
        gspim_parallel_avg = gspim_parallel_time / len(cameras)
    else:
        gspim_parallel_time = gspim_result['total_time_ms']
        gspim_parallel_fps = gspim_result['fps']
        gspim_parallel_avg = gspim_result['avg_frame_time_ms']
    
    # 比较结果
    print("\n" + "="*80)
    print("Performance Comparison")
    print("="*80)
    print(f"\n{'Method':<35} {'Total Time (ms)':<15} {'Avg (ms/frame)':<15} {'FPS':<10}")
    print("-" * 75)
    print(f"{'4DGS Baseline (CUDA)':<35} {baseline_result['total_time_ms']:<15.2f} {baseline_result['avg_frame_time_ms']:<15.2f} {baseline_result['fps']:<10.2f}")
    print(f"{'GSPIM (软件串行)':<35} {gspim_result['total_time_ms']:<15.2f} {gspim_result['avg_frame_time_ms']:<15.2f} {gspim_result['fps']:<10.2f}")
    print(f"{'GSPIM (PIM||GPU 并行)':<35} {gspim_parallel_time:<15.2f} {gspim_parallel_avg:<15.2f} {gspim_parallel_fps:<10.2f}")
    print("-" * 75)
    
    # 计算加速比
    speedup = baseline_result['total_time_ms'] / gspim_parallel_time
    print(f"\n{'Speedup (GSPIM vs Baseline):':<35} {speedup:.2f}x")
    
    # PIM/GPU 并行分析
    if timing_stats:
        print(f"\n{'PIM/GPU Parallel Analysis:'}")
        print(f"  窗口数量:          {timing_stats['num_windows']}")
        print(f"  平均 PIM 时间:     {timing_stats['avg_pim_time_ms']:.2f} ms/窗口 (时间过滤)")
        print(f"  平均 GPU 时间:     {timing_stats['avg_gpu_time_ms']:.2f} ms/窗口 (渲染{args.window_size}帧)")
        print(f"  串行总时间:        {timing_stats['serial_total_ms']:.2f} ms (PIM + GPU)")
        print(f"  并行总时间:        {timing_stats['parallel_total_ms']:.2f} ms (PIM || GPU)")
        print(f"  流水线收益:        {timing_stats['parallelism_benefit']:.2f}x")
        
        # 关键洞察
        avg_pim = timing_stats['avg_pim_time_ms']
        avg_gpu = timing_stats['avg_gpu_time_ms']
        if avg_pim < avg_gpu:
            hidden_pim = (1 - avg_pim / avg_gpu) * 100
            print(f"\n  ✓ PIM 时间 ({avg_pim:.1f}ms) < GPU 时间 ({avg_gpu:.1f}ms)")
            print(f"  ✓ PIM 计算被 GPU 渲染隐藏 {hidden_pim:.1f}%")
        else:
            print(f"\n  ! PIM 时间 ({avg_pim:.1f}ms) > GPU 时间 ({avg_gpu:.1f}ms)")
            print(f"  ! PIM 成为瓶颈，需要优化时间过滤")
    
    print("="*80)
    
    # 输出详细的阶段时间统计（传入真实的baseline FPS）
    print_stage_timing_statistics(gspim_result['statistics'], baseline_fps=baseline_result['fps'])
    
    # 计算质量指标（PSNR/SSIM）
    if args.quality:
        print("\n" + "="*80)
        print("Quality Comparison (PSNR / SSIM)")
        print("="*80)
        
        if baseline_result and 'rendered_images' in baseline_result:
            print("\nComputing Baseline quality metrics...")
            baseline_metrics = compute_metrics(baseline_result['rendered_images'], cameras, bg_color)
            print(f"  Baseline PSNR: {baseline_metrics['psnr_mean']:.2f} dB")
            print(f"  Baseline SSIM: {baseline_metrics['ssim_mean']:.4f}")
        
        if 'rendered_images' in gspim_result:
            print("\nComputing GSPIM quality metrics...")
            gspim_metrics = compute_metrics(gspim_result['rendered_images'], cameras, bg_color)
            print(f"  GSPIM PSNR: {gspim_metrics['psnr_mean']:.2f} dB")
            print(f"  GSPIM SSIM: {gspim_metrics['ssim_mean']:.4f}")
        
        if baseline_result and 'rendered_images' in baseline_result and 'rendered_images' in gspim_result:
            psnr_diff = gspim_metrics['psnr_mean'] - baseline_metrics['psnr_mean']
            ssim_diff = gspim_metrics['ssim_mean'] - baseline_metrics['ssim_mean']
            print(f"\n  PSNR Difference: {psnr_diff:+.2f} dB")
            print(f"  SSIM Difference: {ssim_diff:+.4f}")
        
            # 分析逐帧差异
            import numpy as np
            baseline_psnrs = np.array(baseline_metrics['psnr_list'])
            gspim_psnrs = np.array(gspim_metrics['psnr_list'])
            psnr_diffs = gspim_psnrs - baseline_psnrs
            
            print(f"\n  [Per-frame PSNR Analysis]")
            print(f"    Mean diff: {psnr_diffs.mean():.3f} dB")
            print(f"    Std diff:  {psnr_diffs.std():.3f} dB")
            print(f"    Min diff:  {psnr_diffs.min():.3f} dB (frame {psnr_diffs.argmin()})")
            print(f"    Max diff:  {psnr_diffs.max():.3f} dB (frame {psnr_diffs.argmax()})")
            
            # 找出差异最大的10帧
            worst_indices = np.argsort(psnr_diffs)[:min(10, len(psnr_diffs))]
            print(f"    Worst {len(worst_indices)} frames: {worst_indices.tolist()}")
            print(f"    Their PSNR diffs: {[f'{d:.2f}' for d in psnr_diffs[worst_indices]]}")
        
        print("="*80)
    
    # 保存渲染视频
    if args.save_video:
        print("\n" + "="*80)
        print("Saving Rendered Videos")
        print("="*80)
        
        video_dir = Path(args.video_output)
        
        if baseline_result and 'rendered_images' in baseline_result:
            print("\nSaving Baseline video...")
            save_rendered_video(baseline_result['rendered_images'], video_dir / "baseline", fps=30)
        
        if 'rendered_images' in gspim_result:
            print("\nSaving GSPIM video...")
            save_rendered_video(gspim_result['rendered_images'], video_dir / "gspim", fps=30)
        
        print(f"\nVideos saved to {video_dir}")
        print("="*80)
    
    # 保存结果
    if args.output:
        output_data = {
            'model_path': args.model_path,
            'num_frames': len(cameras),
            'num_gaussians': gaussians.get_xyz.shape[0],
            'baseline': baseline_result if baseline_result else None,
            'gspim': {
                'total_time_ms': gspim_result['total_time_ms'],
                'avg_frame_time_ms': gspim_result['avg_frame_time_ms'],
                'fps': gspim_result['fps'],
                'statistics': gspim_result['statistics']
            }
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    # 打印详细性能分析
    if gspim_result['renderer'].profiler:
        print("\nDetailed Performance Analysis:")
        gspim_result['renderer'].profiler.print_summary()


if __name__ == "__main__":
    main()

