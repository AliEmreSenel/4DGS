#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import torch


def human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def empty_like_cpu_float() -> torch.Tensor:
    return torch.empty(0, dtype=torch.float32)


def slim_one(src: Path, dst: Path, overwrite: bool = False) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} exists; pass --overwrite")

    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict) or ckpt.get("format") != "4dgs-self-contained-v1":
        raise ValueError(f"{src} is not a self-contained 4DGS checkpoint")

    g = list(ckpt["gaussians"])
    if len(g) != 20:
        raise ValueError(f"Unsupported gaussians tuple length: {len(g)}")

    # Gaussian tuple layout from scene/gaussian_model.py::capture()
    # Keep rendering-critical tensors:
    # xyz, features, scaling, rotation, opacity, time, time scale, 4D rotation, env map,
    # active SH degrees, MobileGS MLP model if present.
    #
    # Remove training-only / densification-only state:
    g[7] = empty_like_cpu_float()    # max_radii2D
    g[8] = empty_like_cpu_float()    # xyz_gradient_accum
    g[9] = empty_like_cpu_float()    # t_gradient_accum
    g[10] = empty_like_cpu_float()   # denom
    g[11] = None                     # optimizer state_dict

    # Keep MobileGS model weights for sort-free rendering, drop only its optimizer.
    mobile = g[19]
    if isinstance(mobile, dict):
        mobile = dict(mobile)
        mobile["optimizer"] = None
        g[19] = mobile

    ckpt["gaussians"] = tuple(g)
    ckpt["render_only"] = True
    ckpt["slim_note"] = (
        "Optimizer and densification statistics removed. "
        "Rendering/evaluation should work; training resume will not."
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst, pickle_protocol=4)

    before = src.stat().st_size
    after = dst.stat().st_size
    print(f"{src}")
    print(f"  -> {dst}")
    print(f"  size: {human(before)} -> {human(after)}  ({100 * (1 - after / before):.1f}% smaller)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="checkpoint file or folder")
    ap.add_argument("--glob", default="chkpnt*.pth", help="used when input is a folder")
    ap.add_argument("--suffix", default=".render", help="inserted before .pth")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    paths = [args.input] if args.input.is_file() else sorted(args.input.rglob(args.glob))
    if not paths:
        raise SystemExit("No checkpoints found")

    for src in paths:
        if src.name.endswith(f"{args.suffix}.pth"):
            continue
        dst = src.with_name(src.stem + args.suffix + src.suffix)
        slim_one(src, dst, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
