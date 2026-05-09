#!/usr/bin/env python3
"""Fast FPS + CUDA-kernel profiling for sorted vs sort-free 4DGS checkpoints.

Run from the repository root, or pass --repo-root. This script intentionally does
not compute PSNR/SSIM/LPIPS; it only renders checkpoint cameras as fast as the
GPU can execute them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _add_repo_to_path(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _make_pipe(args: argparse.Namespace, *, sort_free: bool) -> SimpleNamespace:
    return SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        use_usplat=False,
        sort_free_render=bool(sort_free),
        sort_free_fast_profile=bool(args.sort_free_fast_profile and sort_free),
        temporal_mask_threshold=float(args.temporal_mask_threshold),
        temporal_mask_mode=str(args.temporal_mask_mode),
        temporal_mask_keyframes=0,
        temporal_mask_window=int(args.temporal_mask_window),
        random_dropout_prob=0.0,
        env_map_res=0,
        env_optimize_until=0,
        env_optimize_from=0,
        eval_shfs_4d=False,
    )


def _load_model_and_cameras(ckpt_path: Path, args: argparse.Namespace):
    import torch
    from scene.gaussian_model import GaussianModel, coerce_time_duration
    from utils.checkpoint_utils import load_checkpoint
    from utils.mobile_compression import cameras_from_checkpoint_scene

    ckpt = load_checkpoint(str(ckpt_path), map_location="cuda")
    gaussian_kwargs = dict(ckpt["run_config"].get("gaussian_kwargs", {}))
    gaussian_kwargs["time_duration"] = coerce_time_duration(
        gaussian_kwargs.get("time_duration", [-0.5, 0.5])
    )
    gaussians = GaussianModel(**gaussian_kwargs)
    gaussians.restore(ckpt["gaussians"], training_args=None)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    if hasattr(gaussians, "active_sh_degree_t"):
        gaussians.active_sh_degree_t = gaussians.max_sh_degree_t
    if gaussians.mobilegs_opacity_phi_nn is not None:
        gaussians.mobilegs_opacity_phi_nn.eval()

    scene_meta = ckpt.get("scene", {})
    cameras = cameras_from_checkpoint_scene(scene_meta, split=args.split, device="cuda")
    split_used = args.split
    if not cameras:
        cameras = cameras_from_checkpoint_scene(scene_meta, split="train", device="cuda")
        split_used = "train"
    if not cameras:
        raise RuntimeError(f"{ckpt_path} has no embedded cameras for split={args.split} or train")

    if args.fixed_camera:
        cameras = [cameras[int(args.camera_index) % len(cameras)]]

    background = torch.tensor(
        [1.0, 1.0, 1.0] if scene_meta.get("white_background", False) else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    return ckpt, gaussians, cameras, split_used, background


def _render_loop(render_fn, cameras, gaussians, pipe, background, frames: int):
    for i in range(frames):
        render_fn(cameras[i % len(cameras)], gaussians, pipe, background)


def _profile_torch(label: str, render_fn, cameras, gaussians, pipe, background, frames: int, trace_dir: Path):
    import torch
    from torch.profiler import ProfilerActivity, profile

    trace_dir.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    trace_path = trace_dir / f"{label}_torch_trace.json"
    with torch.inference_mode():
        with profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False) as prof:
            for i in range(frames):
                torch.cuda.nvtx.range_push(f"{label}_frame_{i}")
                render_fn(cameras[i % len(cameras)], gaussians, pipe, background)
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    return {"trace": str(trace_path), "table": table}


def _benchmark_one(label: str, ckpt_path: Path, sort_free: bool, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from gaussian_renderer import render

    ckpt, gaussians, cameras, split_used, background = _load_model_and_cameras(ckpt_path, args)
    pipe = _make_pipe(args, sort_free=sort_free)

    if sort_free and gaussians.mobilegs_opacity_phi_nn is None:
        raise RuntimeError(
            f"{label}: sort-free rendering was requested but the checkpoint has no Mobile-GS "
            "opacity/phi MLP. Use a checkpoint trained/exported with --sort_free_render, "
            "or benchmark it as --sorted-ckpt instead."
        )

    # Max-throughput inference settings. Rendering kernels dominate; this mostly
    # ensures no training graph or debug synchronization leaks into the timing.
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    warmup = int(args.warmup)
    repeats = int(args.repeats)
    with torch.inference_mode():
        _render_loop(render, cameras, gaussians, pipe, background, warmup)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        torch.cuda.nvtx.range_push(f"{label}_timed_loop")
        start_event.record()
        _render_loop(render, cameras, gaussians, pipe, background, repeats)
        end_event.record()
        torch.cuda.nvtx.range_pop()
        end_event.synchronize()
        wall_seconds = time.perf_counter() - wall_start

    gpu_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
    result = {
        "label": label,
        "checkpoint": str(ckpt_path),
        "sort_free_render": bool(sort_free),
        "requires_mobilegs": bool(ckpt.get("requires_mobilegs", False)),
        "split_used": split_used,
        "num_cameras": int(len(cameras)),
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "frames": repeats,
        "warmup": warmup,
        "gpu_seconds_cuda_events": gpu_seconds,
        "gpu_fps_cuda_events": repeats / max(gpu_seconds, 1e-9),
        "wall_seconds": wall_seconds,
        "wall_fps": repeats / max(wall_seconds, 1e-9),
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
    }

    if args.profile_torch:
        prof = _profile_torch(
            label,
            render,
            cameras,
            gaussians,
            pipe,
            background,
            int(args.profile_frames),
            Path(args.trace_dir),
        )
        result["torch_trace"] = prof["trace"]
        result["torch_profiler_top_cuda"] = prof["table"]
        print(f"\n=== {label}: top CUDA ops/kernels ===")
        print(prof["table"])

    return result


def _write_outputs(results: list[dict[str, Any]], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    fields = [
        "label",
        "checkpoint",
        "sort_free_render",
        "num_gaussians",
        "num_cameras",
        "frames",
        "gpu_fps_cuda_events",
        "wall_fps",
        "gpu_seconds_cuda_events",
        "wall_seconds",
        "peak_allocated_vram_mb",
        "device",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sorted and sort-free 4DGS checkpoints at max render throughput.")
    parser.add_argument("--repo-root", default=".", help="4D-Gaussian-Splattering repository root")
    parser.add_argument("--sorted-ckpt", required=True, help="Self-contained checkpoint to render with sorted alpha blending")
    parser.add_argument("--sort-free-ckpt", required=True, help="Self-contained checkpoint to render with sort-free Mobile-GS path")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--fixed-camera", action="store_true", help="Use one camera repeatedly to minimize Python-side camera variation overhead")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--temporal-mask-mode", choices=["marginal", "visibility"], default="marginal")
    parser.add_argument("--temporal-mask-window", type=int, default=1)
    parser.add_argument("--temporal-mask-threshold", type=float, default=0.05)
    parser.add_argument(
        "--sort-free-fast-profile",
        action="store_true",
        help="For sort-free only, bypass the Mobile-GS MLP and exact 4D covariance to profile the no-sort CUDA raster kernel itself",
    )
    parser.add_argument("--profile-torch", action="store_true", help="Collect a PyTorch CUDA profiler table and Chrome traces")
    parser.add_argument("--profile-frames", type=int, default=30)
    parser.add_argument("--trace-dir", default="fps_profile_traces")
    parser.add_argument("--output-json", default="fps_profile_results.json")
    parser.add_argument("--output-csv", default="fps_profile_results.csv")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _add_repo_to_path(repo_root)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark must be run on a machine with an NVIDIA GPU.")

    results = []
    results.append(_benchmark_one("sorted", Path(args.sorted_ckpt).resolve(), False, args))
    results.append(_benchmark_one("sort_free", Path(args.sort_free_ckpt).resolve(), True, args))

    _write_outputs(results, Path(args.output_json), Path(args.output_csv))
    print("\n=== FPS summary ===")
    for row in results:
        print(
            f"{row['label']:>9}: {row['gpu_fps_cuda_events']:.2f} FPS GPU-event, "
            f"{row['wall_fps']:.2f} FPS wall, {row['num_gaussians']} gaussians, "
            f"peak {row['peak_allocated_vram_mb']:.1f} MiB"
        )
    print(f"\nwrote {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
