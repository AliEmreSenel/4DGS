#!/usr/bin/env python3
import argparse
import json
from types import SimpleNamespace

import torch

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.checkpoint_utils import load_checkpoint
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, lpips as lpips_metric
from utils.mobile_compression import (
    benchmark_renderer,
    cameras_from_checkpoint_scene,
    human_bytes,
    load_mobile_payload,
    restore_mobile_payload,
    serialized_size,
)


def make_pipe(args, use_visibility=None):
    if use_visibility is None:
        use_visibility = bool(args.use_visibility_filter)
    return SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        use_usplat=False,
        sort_free_render=True,
        temporal_mask_threshold=float(args.temporal_mask_threshold),
        temporal_mask_mode="visibility" if use_visibility else "marginal",
        temporal_mask_keyframes=0,
        temporal_mask_window=int(args.temporal_mask_window),
        random_dropout_prob=0.0,
        env_map_res=0,
        env_optimize_until=0,
        env_optimize_from=0,
        eval_shfs_4d=False,
    )


def load_raw_model(ckpt, device="cuda"):
    kwargs = dict(ckpt["run_config"].get("gaussian_kwargs", {}))
    gm = GaussianModel(**kwargs)
    gm.restore(ckpt["gaussians"], training_args=None)
    if gm.mobilegs_opacity_phi_nn is not None:
        gm.mobilegs_opacity_phi_nn.eval()
    return gm


def main():
    parser = argparse.ArgumentParser(description="Evaluate storage and FPS after Mobile-GS quantization.")
    parser.add_argument("--ckpt-path", required=True, help="Original checkpoint, used for scene/camera metadata")
    parser.add_argument("--mobile-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--temporal-mask-window", type=int, default=1)
    parser.add_argument("--temporal-mask-threshold", type=float, default=0.05)
    parser.add_argument("--use-visibility-filter", action="store_true")
    parser.add_argument("--quality-samples", type=int, default=0, help="Compare mobile render to raw checkpoint render on N cameras")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rendering benchmark")

    ckpt = load_checkpoint(args.ckpt_path, map_location=args.device)
    payload = load_mobile_payload(args.mobile_path, map_location="cpu")
    mobile = restore_mobile_payload(payload, training_args=None, device=args.device)
    has_visibility_filter = payload.get("temporal_visibility_filter") is not None
    pipe = make_pipe(args, use_visibility=bool(args.use_visibility_filter or has_visibility_filter))
    background = torch.tensor(
        [1.0, 1.0, 1.0] if ckpt.get("scene", {}).get("white_background", False) else [0.0, 0.0, 0.0],
        device=args.device,
    )
    cameras = cameras_from_checkpoint_scene(ckpt.get("scene", {}), split=args.split, device=args.device)
    if not cameras:
        cameras = cameras_from_checkpoint_scene(ckpt.get("scene", {}), split="train", device=args.device)

    fps = benchmark_renderer(mobile, cameras, pipe, background, render, warmup=args.warmup, repeats=args.repeats)
    summary = {
        "mobile_path": args.mobile_path,
        "mobile_serialized_bytes": serialized_size(payload),
        "mobile_file_bytes": __import__("os").path.getsize(args.mobile_path),
        "num_points": int(mobile.get_xyz.shape[0]),
        "split": args.split,
        "fps": fps,
        "temporal_mask_mode": pipe.temporal_mask_mode,
        "temporal_visibility_filter_present": bool(has_visibility_filter),
    }

    if args.quality_samples and args.quality_samples > 0:
        raw = load_raw_model(ckpt, device=args.device)
        raw_pipe = SimpleNamespace(**vars(pipe))
        raw_pipe.sort_free_render = bool(ckpt.get("requires_mobilegs", False))
        raw_pipe.temporal_mask_mode = "marginal"
        q = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0}
        with torch.inference_mode():
            for cam in cameras[: args.quality_samples]:
                ref = torch.clamp(render(cam, raw, raw_pipe, background)["render"], 0.0, 1.0)
                out = torch.clamp(render(cam, mobile, pipe, background)["render"], 0.0, 1.0)
                q["l1"] += float(l1_loss(out, ref).item())
                q["psnr"] += float(psnr(out, ref).mean().item())
                q["ssim"] += float(ssim(out, ref).mean().item())
                q["lpips"] += float(lpips_metric(out[None].cpu(), ref[None].cpu()).item())
                q["count"] += 1
        if q["count"]:
            for k in ("l1", "psnr", "ssim", "lpips"):
                q[k] /= q["count"]
        summary["quality_vs_raw_checkpoint"] = q

    print("Mobile-GS quantized benchmark")
    print("  payload:", human_bytes(summary["mobile_file_bytes"]))
    print("  points:", summary["num_points"])
    print("  fps:", f"{summary['fps']['fps']:.2f}", f"({summary['fps']['frames']} frames)")
    if "quality_vs_raw_checkpoint" in summary:
        print("  quality vs raw:", summary["quality_vs_raw_checkpoint"])

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
