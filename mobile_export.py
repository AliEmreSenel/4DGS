#!/usr/bin/env python3
import argparse
import json
import os
from types import SimpleNamespace

import torch

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.checkpoint_utils import load_checkpoint
from utils.mobile_compression import (
    build_temporal_visibility_filter,
    cameras_from_checkpoint_scene,
    capture_mobile_payload,
    human_bytes,
    save_mobile_payload,
    serialized_size,
    tensor_storage_bytes,
)


def load_gaussians(ckpt_path, device="cuda"):
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    kwargs = dict(ckpt["run_config"].get("gaussian_kwargs", {}))
    gm = GaussianModel(**kwargs)
    gm.restore(ckpt["gaussians"], training_args=None)
    if gm.mobilegs_opacity_phi_nn is not None:
        gm.mobilegs_opacity_phi_nn.eval()
    return ckpt, gm, kwargs


def make_pipe(args):
    return SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        use_usplat=False,
        sort_free_render=True,
        temporal_mask_threshold=float(args.temporal_mask_threshold),
        temporal_mask_mode="marginal",
        temporal_mask_keyframes=0,
        temporal_mask_window=int(args.temporal_mask_window),
        random_dropout_prob=0.0,
        env_map_res=0,
        env_optimize_until=0,
        env_optimize_from=0,
        eval_shfs_4d=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Export a Mobile-GS/NVQ compressed payload.")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output", required=True, help="Output .mobile.pt path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--first-order-sh", action="store_true", default=True)
    parser.add_argument("--keep-full-sh", action="store_true", help="Disable first-order SH slicing")
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--kmeans-iters", type=int, default=16)
    parser.add_argument("--uniform-bits", type=int, default=8)
    parser.add_argument("--build-visibility-filter", action="store_true")
    parser.add_argument("--temporal-keyframes", type=int, default=32)
    parser.add_argument("--temporal-mask-window", type=int, default=1)
    parser.add_argument("--temporal-mask-threshold", type=float, default=0.05)
    parser.add_argument("--views-per-keyframe", type=int, default=0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for export-time visibility-mask rendering.")

    ckpt, gaussians, kwargs = load_gaussians(args.ckpt_path, device=args.device)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if ckpt.get("scene", {}).get("white_background", False) else [0.0, 0.0, 0.0],
        device=args.device,
    )

    temporal_filter = None
    if args.build_visibility_filter:
        cameras = cameras_from_checkpoint_scene(ckpt.get("scene", {}), split="train", device=args.device)
        pipe = make_pipe(args)
        temporal_filter = build_temporal_visibility_filter(
            gaussians,
            cameras,
            pipe,
            background,
            render,
            keyframes=args.temporal_keyframes,
            views_per_keyframe=args.views_per_keyframe,
        )

    payload = capture_mobile_payload(
        gaussians,
        first_order_sh=not args.keep_full_sh,
        codebook_size=args.codebook_size,
        block_size=args.block_size,
        kmeans_iters=args.kmeans_iters,
        uniform_bits=args.uniform_bits,
        include_mlp=True,
        temporal_visibility_filter=temporal_filter,
    )
    save_mobile_payload(payload, args.output)

    raw_gaussian_size = tensor_storage_bytes({
        "xyz": gaussians.get_xyz,
        "features_dc": gaussians._features_dc,
        "features_rest": gaussians._features_rest,
        "scaling": gaussians._scaling,
        "opacity": gaussians._opacity,
        "t": gaussians._t,
        "scaling_t": gaussians._scaling_t,
        "rotation": gaussians._rotation,
        "rotation_r": gaussians._rotation_r,
    })
    serialized_payload = serialized_size(payload)
    out_size = os.path.getsize(args.output)
    summary = {
        "input_checkpoint": args.ckpt_path,
        "output": args.output,
        "num_points": int(gaussians.get_xyz.shape[0]),
        "raw_checkpoint_bytes": os.path.getsize(args.ckpt_path),
        "raw_gaussian_tensor_bytes": raw_gaussian_size,
        "mobile_payload_serialized_bytes": serialized_payload,
        "mobile_file_bytes": out_size,
        "compression_vs_raw_gaussian_tensors": raw_gaussian_size / max(out_size, 1),
        "first_order_sh": not args.keep_full_sh,
        "nvq": {
            "codebook_size": args.codebook_size,
            "block_size": args.block_size,
            "kmeans_iters": args.kmeans_iters,
            "uniform_bits": args.uniform_bits,
        },
        "temporal_visibility_filter": temporal_filter is not None,
    }
    manifest_path = args.output + ".json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("Mobile-GS export complete")
    print("  output:", args.output)
    print("  manifest:", manifest_path)
    print("  raw checkpoint:", human_bytes(summary["raw_checkpoint_bytes"]))
    print("  raw Gaussian tensors:", human_bytes(raw_gaussian_size))
    print("  mobile payload:", human_bytes(out_size))
    print("  tensor compression ratio:", f"{summary['compression_vs_raw_gaussian_tensors']:.2f}x")


if __name__ == "__main__":
    main()
