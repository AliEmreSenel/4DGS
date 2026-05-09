#!/usr/bin/env python3
"""Universal Gaussian checkpoint compression, evaluation, and visual diagnostics.

This script intentionally lives outside the training/rendering source tree.  It
loads a self-contained checkpoint produced by this repository, applies optional
post-training compression/pruning, restores the compressed payload, then measures
what changed with the same renderer path used by the checkpoint unless explicitly
overridden.

It is designed for all experiment rows/ablations in this repo: sorted 4DGS,
Mobile/SortFree rows, USplat rows, Dropout/ESS rows, isotropic/anisotropic rows,
3D-SH/4D-SH rows, and pruning ablations.  No fallback imports or monkey patches
are used; missing CUDA extensions, codecs, LPIPS, or render dependencies should
fail directly so the environment can be fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel, coerce_time_duration
from utils.checkpoint_utils import load_checkpoint
from utils.gpcc_utils import compress_gpcc, decompress_gpcc
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, lpips as lpips_metric, ssim
from utils.mobile_compression import (
    attach_temporal_visibility_filter,
    benchmark_renderer,
    build_temporal_visibility_filter,
    cameras_from_checkpoint_scene,
    human_bytes,
    nvq_decode_tensor,
    nvq_encode_tensor,
    serialized_size,
    tensor_storage_bytes,
)

UNIVERSAL_FORMAT = "universal-gaussian-compressed-v1"
PIPE_DEFAULTS = {
    "convert_SHs_python": False,
    "compute_cov3D_python": False,
    "debug": False,
    "use_usplat": False,
    "sort_free_render": False,
    "temporal_mask_threshold": 0.05,
    "temporal_mask_mode": "marginal",
    "temporal_mask_keyframes": 0,
    "temporal_mask_window": 2,
    "random_dropout_prob": 0.0,
    "env_map_res": 0,
    "env_optimize_until": 1000000000,
    "env_optimize_from": 0,
    "eval_shfs_4d": False,
}


def log_progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def pct(part: float | int | None, total: float | int | None) -> float | None:
    if part is None or total is None:
        return None
    total = float(total)
    if total <= 0.0:
        return None
    return 100.0 * float(part) / total


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def reduction(before: float | int | None, after: float | int | None) -> dict[str, float | None]:
    if before is None or after is None:
        return {"ratio": None, "percent": None}
    before = float(before)
    after = float(after)
    if before <= 0.0 or after <= 0.0:
        return {"ratio": None, "percent": None}
    return {"ratio": before / after, "percent": 100.0 * (1.0 - after / before)}


def load_gaussians_from_checkpoint(path: str | Path, device: str):
    checkpoint = load_checkpoint(path, map_location=device)
    kwargs = dict(checkpoint.get("run_config", {}).get("gaussian_kwargs", {}))
    kwargs["time_duration"] = coerce_time_duration(kwargs.get("time_duration", [-0.5, 0.5]))
    gaussians = GaussianModel(**kwargs)
    gaussians.restore(checkpoint["gaussians"], training_args=None)
    gaussians.active_sh_degree = int(gaussians.max_sh_degree)
    if hasattr(gaussians, "active_sh_degree_t"):
        gaussians.active_sh_degree_t = int(gaussians.max_sh_degree_t)
    if gaussians.mobilegs_opacity_phi_nn is not None:
        gaussians.mobilegs_opacity_phi_nn.eval()
    return checkpoint, gaussians, kwargs


def gaussian_tensor_size(gaussians: GaussianModel) -> int:
    return tensor_storage_bytes(capture_uncompressed_tensor_dict(gaussians, include_mlp=True))


def capture_uncompressed_tensor_dict(gaussians: GaussianModel, include_mlp: bool = True) -> dict[str, Any]:
    data: dict[str, Any] = {
        "xyz": gaussians.get_xyz,
        "features_dc": gaussians._features_dc,
        "features_rest": gaussians._features_rest,
        "scaling": gaussians._scaling,
        "opacity": gaussians._opacity,
        "t": gaussians._t,
        "scaling_t": gaussians._scaling_t,
        "rotation": gaussians._rotation,
        "rotation_r": gaussians._rotation_r,
        "env_map": gaussians.env_map,
    }
    if include_mlp and gaussians.mobilegs_opacity_phi_nn is not None:
        data["mobilegs_opacity_phi_nn"] = gaussians.mobilegs_opacity_phi_nn.state_dict()
    return data


def run_args(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    run_config = checkpoint.get("run_config", {})
    args = run_config.get("args", {}) if isinstance(run_config, Mapping) else {}
    return dict(args) if isinstance(args, Mapping) else {}


def make_pipe(
    checkpoint: Mapping[str, Any],
    *,
    render_mode: str,
    disable_dropout: bool,
    temporal_filter: bool,
    temporal_keyframes: int,
    temporal_window: int,
    temporal_threshold: float,
) -> SimpleNamespace:
    args = run_args(checkpoint)
    pipe = dict(PIPE_DEFAULTS)
    for key in pipe.keys():
        if key in args:
            pipe[key] = args[key]
    pipe["debug"] = False
    if disable_dropout:
        pipe["random_dropout_prob"] = 0.0
    if render_mode == "sorted":
        pipe["sort_free_render"] = False
    elif render_mode == "sort_free":
        pipe["sort_free_render"] = True
    elif render_mode != "checkpoint":
        raise ValueError(f"Unknown render mode: {render_mode}")
    if temporal_filter:
        pipe["temporal_mask_keyframes"] = int(temporal_keyframes)
        pipe["temporal_mask_window"] = int(temporal_window)
        pipe["temporal_mask_threshold"] = float(temporal_threshold)
        pipe["temporal_mask_mode"] = "visibility"
    return SimpleNamespace(**pipe)


def selected_render_mode(checkpoint: Mapping[str, Any], mode: str) -> str:
    if mode != "checkpoint":
        return mode
    return "sort_free" if bool(run_args(checkpoint).get("sort_free_render", False)) else "sorted"


def available_cameras(checkpoint: Mapping[str, Any], split: str, device: str):
    scene_meta = checkpoint.get("scene", {})
    if split == "all":
        train = cameras_from_checkpoint_scene(scene_meta, split="train", device=device)
        test = cameras_from_checkpoint_scene(scene_meta, split="test", device=device)
        cams = train + test
        if not cams:
            raise ValueError("The checkpoint scene metadata does not contain train/test cameras.")
        return cams, "all"
    cams = cameras_from_checkpoint_scene(scene_meta, split=split, device=device)
    split_used = split
    if not cams:
        other = "test" if split == "train" else "train"
        cams = cameras_from_checkpoint_scene(scene_meta, split=other, device=device)
        split_used = other
    if not cams:
        raise ValueError("The checkpoint scene metadata does not contain train/test cameras.")
    return cams, split_used


def background_from_checkpoint(checkpoint: Mapping[str, Any], device: str) -> torch.Tensor:
    scene_meta = checkpoint.get("scene", {})
    white = bool(scene_meta.get("white_background", False))
    return torch.tensor([1.0, 1.0, 1.0] if white else [0.0, 0.0, 0.0], dtype=torch.float32, device=device)


def quantize_xyz_u16(xyz: torch.Tensor):
    xyz_cpu = xyz.detach().float().cpu().contiguous()
    xyz_min = xyz_cpu.amin(dim=0)
    xyz_max = xyz_cpu.amax(dim=0)
    extent = (xyz_max - xyz_min).clamp_min(1e-8)
    q = torch.round((xyz_cpu - xyz_min) * (65535.0 / extent)).to(torch.int32)
    return q, {"min": xyz_min, "max": xyz_max}


def dequantize_xyz_u16(xyz_q: torch.Tensor, meta: Mapping[str, torch.Tensor], device: str) -> torch.Tensor:
    xyz_q = xyz_q.to(device=device, dtype=torch.float32)
    xyz_min = meta["min"].to(device=device, dtype=torch.float32)
    xyz_max = meta["max"].to(device=device, dtype=torch.float32)
    extent = (xyz_max - xyz_min).clamp_min(1e-8)
    return xyz_q * (extent / 65535.0) + xyz_min


def uniform_encode(tensor: torch.Tensor, bits: int):
    x = tensor.detach().float().cpu().contiguous()
    shape = tuple(x.shape)
    if x.numel() == 0:
        return {"codec": "uniform", "shape": list(shape), "bits": int(bits), "min": torch.empty(0), "step": torch.empty(0), "q": torch.empty(0, dtype=torch.int32)}
    flat = x.reshape(shape[0], -1)
    x_min = flat.amin(dim=0, keepdim=True)
    x_max = flat.amax(dim=0, keepdim=True)
    qmax = float((1 << int(bits)) - 1)
    step = ((x_max - x_min) / qmax).clamp_min(1e-8)
    q_float = torch.round((flat - x_min) / step).clamp(0, qmax)
    if bits <= 8:
        q = q_float.to(torch.uint8)
    else:
        q = q_float.to(torch.int32)
    return {"codec": "uniform", "shape": list(shape), "bits": int(bits), "min": x_min.half(), "step": step.half(), "q": q.contiguous()}


def uniform_decode(pack: Mapping[str, Any], device: str) -> torch.Tensor:
    shape = tuple(pack["shape"])
    if len(shape) == 0 or int(np.prod(shape)) == 0:
        return torch.empty(shape, device=device)
    q = pack["q"].to(device=device, dtype=torch.float32)
    x_min = pack["min"].to(device=device, dtype=torch.float32)
    step = pack["step"].to(device=device, dtype=torch.float32)
    return (q * step + x_min).reshape(shape).contiguous()


def float_encode(tensor: torch.Tensor, dtype: str):
    if dtype == "float16":
        return {"codec": "float16", "value": tensor.detach().cpu().half(), "shape": list(tensor.shape)}
    if dtype == "float32":
        return {"codec": "float32", "value": tensor.detach().cpu().float(), "shape": list(tensor.shape)}
    raise ValueError(f"Unsupported float codec: {dtype}")


def decode_attr(pack: Mapping[str, Any], device: str) -> torch.Tensor:
    codec = str(pack.get("codec"))
    if codec == "uniform":
        return uniform_decode(pack, device=device)
    if codec == "nvq":
        return nvq_decode_tensor(pack, device=device)
    if codec in ("float16", "float32"):
        return pack["value"].to(device=device, dtype=torch.float32).contiguous()
    raise ValueError(f"Unsupported attribute codec: {codec}")


def gaussian_attribute_tensors(gaussians: GaussianModel) -> dict[str, torch.Tensor]:
    attrs = {
        "features_dc": gaussians._features_dc.detach(),
        "features_rest": gaussians._features_rest.detach(),
        "scaling": gaussians._scaling.detach(),
        "opacity": gaussians._opacity.detach(),
        "t": gaussians._t.detach(),
        "scaling_t": gaussians._scaling_t.detach(),
    }
    if not gaussians.isotropic_gaussians:
        attrs["rotation"] = gaussians._rotation.detach()
    if gaussians.rot_4d and not gaussians.isotropic_gaussians:
        attrs["rotation_r"] = gaussians._rotation_r.detach()
    return attrs


def encode_attr(name: str, tensor: torch.Tensor, *, codec: str, uniform_bits: int, codebook_size: int, block_size: int, kmeans_iters: int, seed: int):
    if codec == "uniform":
        return uniform_encode(tensor, uniform_bits)
    if codec == "mobilegs":
        # Mobile-GS style postprocess: GPCC for xyz is handled separately;
        # dense attributes use sub-vector codebooks; opacity uses compact
        # min/max quantization because it is a scalar and is less sensitive.
        if name == "opacity":
            return uniform_encode(tensor, uniform_bits)
        if tensor.numel() == 0:
            return float_encode(tensor, "float16")
        return nvq_encode_tensor(tensor, codebook_size=codebook_size, block_size=block_size, iters=kmeans_iters, seed=seed)
    if codec == "nvq":
        if tensor.numel() == 0:
            return float_encode(tensor, "float16")
        return nvq_encode_tensor(tensor, codebook_size=codebook_size, block_size=block_size, iters=kmeans_iters, seed=seed)
    if codec in ("float16", "float32"):
        return float_encode(tensor, codec)
    raise ValueError(f"Unsupported codec: {codec}")


def model_meta(gaussians: GaussianModel) -> dict[str, Any]:
    return {
        "gaussian_dim": int(gaussians.gaussian_dim),
        "rot_4d": bool(gaussians.rot_4d),
        "isotropic_gaussians": bool(gaussians.isotropic_gaussians),
        "force_sh_3d": bool(gaussians.force_sh_3d),
        "max_sh_degree": int(gaussians.max_sh_degree),
        "active_sh_degree": int(gaussians.active_sh_degree),
        "max_sh_degree_t": int(gaussians.max_sh_degree_t),
        "active_sh_degree_t": int(gaussians.active_sh_degree_t),
        "time_duration": coerce_time_duration(gaussians.time_duration),
        "prefilter_var": float(gaussians.prefilter_var),
        "spatial_lr_scale": float(gaussians.spatial_lr_scale),
        "num_points": int(gaussians.get_xyz.shape[0]),
    }


def maybe_reduce_sh_degree(gaussians: GaussianModel, cap: int) -> None:
    if cap < 0:
        return
    cap = int(cap)
    if cap > int(gaussians.max_sh_degree):
        return
    keep_channels = (cap + 1) ** 2 if gaussians.force_sh_3d or gaussians.max_sh_degree_t == 0 else (cap + 1) ** 2 * (gaussians.max_sh_degree_t + 1)
    keep_rest = max(0, keep_channels - 1)
    f_rest = gaussians._features_rest.detach()
    if f_rest.shape[1] >= keep_rest:
        new_rest = f_rest[:, :keep_rest, :].contiguous()
    else:
        pad = torch.zeros((f_rest.shape[0], keep_rest - f_rest.shape[1], f_rest.shape[2]), device=f_rest.device, dtype=f_rest.dtype)
        new_rest = torch.cat([f_rest, pad], dim=1).contiguous()
    gaussians._features_rest = nn.Parameter(new_rest.requires_grad_(False))
    gaussians.max_sh_degree = cap
    gaussians.active_sh_degree = min(cap, int(gaussians.active_sh_degree))


def capture_universal_payload(
    gaussians: GaussianModel,
    *,
    codec: str,
    uniform_bits: int,
    codebook_size: int,
    block_size: int,
    kmeans_iters: int,
    include_mlp: bool,
    temporal_visibility_filter: Mapping[str, Any] | None,
    seed: int,
) -> Mapping[str, Any]:
    xyz_q, xyz_meta = quantize_xyz_u16(gaussians.get_xyz)
    log_progress("  GPCC encoding xyz")
    xyz_bitstream = compress_gpcc(xyz_q.cpu())
    log_progress(f"  GPCC xyz bitstream: {human_bytes(len(xyz_bitstream))}")
    payload: dict[str, Any] = {
        "format": UNIVERSAL_FORMAT,
        "meta": dict(model_meta(gaussians), xyz_quant=xyz_meta),
        "codec": {
            "attr_codec": codec,
            "uniform_bits": int(uniform_bits),
            "codebook_size": int(codebook_size),
            "block_size": int(block_size),
            "kmeans_iters": int(kmeans_iters),
        },
        # Keep Gaussian row order unchanged so every ablation/method can be
        # compared per primitive after restore.  GPCC still compresses the
        # quantized positions; no method-specific reordering is required.
        "xyz": xyz_bitstream,
        "attr": {},
        "temporal_visibility_filter": temporal_visibility_filter,
    }
    attr_tensors = gaussian_attribute_tensors(gaussians)
    for attr_i, (name, tensor) in enumerate(attr_tensors.items(), 1):
        log_progress(f"  encoding attribute {attr_i}/{len(attr_tensors)}: {name} shape={list(tensor.shape)}")
        payload["attr"][name] = encode_attr(
            name,
            tensor,
            codec=codec,
            uniform_bits=uniform_bits,
            codebook_size=codebook_size,
            block_size=block_size,
            kmeans_iters=kmeans_iters,
            seed=seed + len(payload["attr"]) * 1009,
        )
    if gaussians.env_map is not None and torch.is_tensor(gaussians.env_map) and gaussians.env_map.numel() > 0:
        payload["env_map"] = gaussians.env_map.detach().cpu().half()
    if include_mlp and gaussians.mobilegs_opacity_phi_nn is not None:
        payload["mobilegs_opacity_phi_nn"] = {k: v.detach().cpu().half() for k, v in gaussians.mobilegs_opacity_phi_nn.state_dict().items()}
    return payload


def restore_universal_payload(payload: Mapping[str, Any], *, device: str, training_args=None) -> GaussianModel:
    if payload.get("format") != UNIVERSAL_FORMAT:
        raise ValueError(f"Unsupported payload format: {payload.get('format')}")
    meta = payload["meta"]
    gm = GaussianModel(
        sh_degree=int(meta["max_sh_degree"]),
        gaussian_dim=int(meta["gaussian_dim"]),
        time_duration=coerce_time_duration(meta["time_duration"]),
        rot_4d=bool(meta["rot_4d"]),
        force_sh_3d=bool(meta["force_sh_3d"]),
        sh_degree_t=int(meta.get("max_sh_degree_t", 0)),
        prefilter_var=float(meta.get("prefilter_var", -1.0)),
        isotropic_gaussians=bool(meta.get("isotropic_gaussians", False)),
    )
    gm.active_sh_degree = int(meta.get("active_sh_degree", gm.max_sh_degree))
    gm.active_sh_degree_t = int(meta.get("active_sh_degree_t", 0))
    gm.spatial_lr_scale = float(meta.get("spatial_lr_scale", 1.0))

    xyz_q = decompress_gpcc(payload["xyz"]).to(device=device, dtype=torch.int32)
    xyz = dequantize_xyz_u16(xyz_q, meta["xyz_quant"], device=device)
    attrs = {name: decode_attr(pack, device=device) for name, pack in payload["attr"].items()}
    n = int(xyz.shape[0])
    if n != int(meta["num_points"]):
        raise ValueError(f"Decoded point count mismatch: {n} vs {meta['num_points']}")
    for name, tensor in attrs.items():
        if tensor.shape[0] != n:
            raise ValueError(f"Attribute {name} has {tensor.shape[0]} rows, expected {n}")

    requires_grad = training_args is not None
    gm._xyz = nn.Parameter(xyz.contiguous().requires_grad_(requires_grad))
    gm._features_dc = nn.Parameter(attrs["features_dc"].contiguous().requires_grad_(requires_grad))
    gm._features_rest = nn.Parameter(attrs["features_rest"].contiguous().requires_grad_(requires_grad))
    gm._scaling = nn.Parameter(attrs["scaling"].contiguous().requires_grad_(requires_grad))
    gm._opacity = nn.Parameter(attrs["opacity"].contiguous().requires_grad_(requires_grad))
    gm._t = nn.Parameter(attrs["t"].contiguous().requires_grad_(requires_grad))
    gm._scaling_t = nn.Parameter(attrs["scaling_t"].contiguous().requires_grad_(requires_grad))
    if gm.isotropic_gaussians:
        gm._rotation = torch.empty((0, 4), device=device, dtype=xyz.dtype)
        gm._rotation_r = torch.empty((0, 4), device=device, dtype=xyz.dtype)
    else:
        gm._rotation = nn.Parameter(attrs["rotation"].contiguous().requires_grad_(requires_grad))
        if gm.rot_4d:
            gm._rotation_r = nn.Parameter(attrs["rotation_r"].contiguous().requires_grad_(requires_grad))
        else:
            gm._rotation_r = torch.empty((0, 4), device=device, dtype=xyz.dtype)

    env_map = payload.get("env_map")
    gm.env_map = torch.empty(0, device=device) if env_map is None else env_map.to(device=device, dtype=torch.float32)
    if payload.get("mobilegs_opacity_phi_nn") is not None:
        gm.mobilegs_opacity_phi_nn = gm._build_mobilegs_opacity_phi_nn()
        gm.mobilegs_opacity_phi_nn.load_state_dict({k: v.to(device=device, dtype=torch.float32) for k, v in payload["mobilegs_opacity_phi_nn"].items()})
        gm.mobilegs_opacity_phi_nn.eval()

    gm.max_radii2D = torch.zeros((n,), device=device)
    gm.xyz_gradient_accum = torch.zeros((n, 1), device=device)
    gm.t_gradient_accum = torch.zeros((n, 1), device=device)
    gm.denom = torch.zeros((n, 1), device=device)
    gm.optimizer = None
    attach_temporal_visibility_filter(gm, payload.get("temporal_visibility_filter"), device=device)
    if training_args is not None:
        gm.training_setup(training_args)
    return gm


def save_payload(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), str(path))


def load_payload(path: str | Path, map_location: str | torch.device = "cpu") -> Mapping[str, Any]:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != UNIVERSAL_FORMAT:
        raise ValueError(f"{path} is not a {UNIVERSAL_FORMAT} payload")
    return payload


def generic_contribution_scores(gaussians: GaussianModel) -> torch.Tensor:
    opacity = gaussians.get_opacity.detach().reshape(-1).float()
    scale = gaussians.get_scaling.detach().float()
    scale_max = scale.max(dim=1).values if scale.ndim == 2 else scale.reshape(-1)
    if gaussians.get_scaling_t.numel() > 0:
        temporal = gaussians.get_scaling_t.detach().reshape(-1).float()
    else:
        temporal = torch.ones_like(opacity)
    opacity = torch.nan_to_num(opacity, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    scale_max = torch.nan_to_num(scale_max, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    temporal = torch.nan_to_num(temporal, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    opacity_n = opacity / opacity.amax().clamp_min(1e-12)
    scale_n = scale_max / scale_max.amax().clamp_min(1e-12)
    temporal_n = temporal / temporal.amax().clamp_min(1e-12)
    return opacity_n * scale_n * (0.5 + 0.5 * temporal_n)


def apply_gaussian_keep_mask(gaussians: GaussianModel, keep_mask: torch.Tensor) -> int:
    keep_mask = keep_mask.to(device=gaussians.get_xyz.device, dtype=torch.bool).reshape(-1)
    n0 = int(gaussians.get_xyz.shape[0])
    if keep_mask.numel() != n0:
        raise ValueError(f"keep_mask has {keep_mask.numel()} entries, expected {n0}")
    if n0 == 0:
        return 0
    if not bool(keep_mask.any()):
        keep_mask[torch.argmax(gaussians.get_opacity.reshape(-1))] = True

    def filter_tensor(value: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(value) or value.numel() == 0:
            return value
        return value.detach()[keep_mask].contiguous()

    gaussians._xyz = nn.Parameter(filter_tensor(gaussians._xyz).requires_grad_(False))
    gaussians._features_dc = nn.Parameter(filter_tensor(gaussians._features_dc).requires_grad_(False))
    gaussians._features_rest = nn.Parameter(filter_tensor(gaussians._features_rest).requires_grad_(False))
    gaussians._scaling = nn.Parameter(filter_tensor(gaussians._scaling).requires_grad_(False))
    gaussians._opacity = nn.Parameter(filter_tensor(gaussians._opacity).requires_grad_(False))
    if not gaussians.isotropic_gaussians:
        gaussians._rotation = nn.Parameter(filter_tensor(gaussians._rotation).requires_grad_(False))
    if gaussians.gaussian_dim == 4:
        gaussians._t = nn.Parameter(filter_tensor(gaussians._t).requires_grad_(False))
        gaussians._scaling_t = nn.Parameter(filter_tensor(gaussians._scaling_t).requires_grad_(False))
        if gaussians.rot_4d and not gaussians.isotropic_gaussians:
            gaussians._rotation_r = nn.Parameter(filter_tensor(gaussians._rotation_r).requires_grad_(False))
    n1 = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    gaussians.max_radii2D = torch.zeros((n1,), device=device)
    gaussians.xyz_gradient_accum = torch.zeros((n1, 1), device=device)
    gaussians.denom = torch.zeros((n1, 1), device=device)
    gaussians.t_gradient_accum = torch.zeros((n1, 1), device=device)
    gaussians.optimizer = None
    for cache_name in ("_temporal_visibility_mask_cache", "_temporal_active_mask_cache"):
        if hasattr(gaussians, cache_name):
            delattr(gaussians, cache_name)
    return n0 - n1


def target_keep_count(gaussians: GaussianModel, *, target_gaussians: int, target_size_mb: float, min_points: int, raw_tensor_bytes: int | None = None) -> int:
    n = int(gaussians.get_xyz.shape[0])
    min_points = max(1, int(min_points))
    keep = n
    if int(target_gaussians) > 0:
        keep = min(keep, int(target_gaussians))
    if float(target_size_mb) > 0.0:
        # This is an intentionally simple pre-encode estimate.  Exact size is
        # reported after GPCC/codebook serialization below.
        if raw_tensor_bytes is None:
            raw_tensor_bytes = gaussian_tensor_size(gaussians)
        per_point = float(raw_tensor_bytes) / float(max(n, 1))
        estimated_keep = int((float(target_size_mb) * 1024.0 * 1024.0) / max(per_point, 1.0))
        keep = min(keep, estimated_keep)
    return max(min_points, min(n, keep))


def predict_prune_indices(gaussians: GaussianModel, *, target_gaussians: int, target_size_mb: float, min_points: int, raw_tensor_bytes: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n = int(gaussians.get_xyz.shape[0])
    scores = generic_contribution_scores(gaussians).detach().float().cpu().numpy()
    keep_count = target_keep_count(gaussians, target_gaussians=target_gaussians, target_size_mb=target_size_mb, min_points=min_points, raw_tensor_bytes=raw_tensor_bytes)
    if n <= keep_count:
        return np.arange(n, dtype=np.int64), np.zeros((0,), dtype=np.int64), scores, keep_count
    order = np.argsort(scores, kind="stable")
    pruned = order[: n - keep_count].astype(np.int64)
    kept = order[n - keep_count :].astype(np.int64)
    kept.sort()
    return kept, pruned, scores, keep_count


def prune_generic_contribution(gaussians: GaussianModel, *, target_gaussians: int, target_size_mb: float, min_points: int, raw_tensor_bytes: int | None = None) -> dict[str, Any]:
    n = int(gaussians.get_xyz.shape[0])
    kept_idx, pruned_idx, scores, keep_count = predict_prune_indices(gaussians, target_gaussians=target_gaussians, target_size_mb=target_size_mb, min_points=min_points, raw_tensor_bytes=raw_tensor_bytes)
    if len(pruned_idx) == 0:
        return {
            "enabled": False,
            "strategy": "mobilegs_contribution_target",
            "removed": 0,
            "kept": n,
            "target_gaussians": int(target_gaussians),
            "target_size_mb": float(target_size_mb),
            "requested_keep_count": int(keep_count),
            "fraction_removed": 0.0,
            "score_threshold": None,
        }
    keep = torch.zeros((n,), device=gaussians.get_xyz.device, dtype=torch.bool)
    keep[torch.from_numpy(kept_idx).to(device=gaussians.get_xyz.device)] = True
    removed = apply_gaussian_keep_mask(gaussians, keep)
    return {
        "enabled": True,
        "strategy": "mobilegs_contribution_target",
        "removed": int(removed),
        "kept": int(gaussians.get_xyz.shape[0]),
        "target_gaussians": int(target_gaussians),
        "target_size_mb": float(target_size_mb),
        "requested_keep_count": int(keep_count),
        "fraction_removed": float(removed) / float(max(n, 1)),
        "score_threshold": float(scores[pruned_idx[-1]]) if len(pruned_idx) else None,
    }


def tensor_to_image_array(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().float().cpu().clamp(0.0, 1.0)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.permute(1, 2, 0)
    arr = (arr.numpy() * 255.0).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def image01(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().float().cpu().clamp(0.0, 1.0)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.permute(1, 2, 0)
    out = arr.numpy()
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    return out


def save_image(path: Path, img: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_image_array(img)).save(path)


def save_render_comparison_plot(path: Path, ref: torch.Tensor, out: torch.Tensor, *, l1_value: float, psnr_value: float, ssim_value: float, diff_scale: float, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ref_np = image01(ref)
    out_np = image01(out)
    diff = (out - ref).detach().float().cpu()
    diff_rgb = image01((diff.abs() * float(diff_scale)).clamp(0.0, 1.0))
    err_scalar = diff.abs().mean(dim=0).numpy() if diff.ndim == 3 else diff.abs().numpy()
    signed_luma = diff.mean(dim=0).numpy() if diff.ndim == 3 else diff.numpy()
    hist_vals = diff.abs().reshape(-1).numpy()
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    fig.suptitle(f"{title}\nPSNR {psnr_value:.2f} dB | SSIM {ssim_value:.4f} | L1 {l1_value:.6f}", fontsize=12)
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(ref_np)
    ax.set_title("Before compression")
    ax.axis("off")
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(out_np)
    ax.set_title("After compression")
    ax.axis("off")
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(diff_rgb)
    ax.set_title(f"RGB |after-before| x{diff_scale:g}")
    ax.axis("off")
    ax = fig.add_subplot(2, 3, 4)
    im = ax.imshow(err_scalar, cmap="magma")
    ax.set_title("Mean absolute error heatmap")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = fig.add_subplot(2, 3, 5)
    vmax = float(np.percentile(np.abs(signed_luma), 99.0)) if signed_luma.size else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(signed_luma, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title("Signed luminance difference")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = fig.add_subplot(2, 3, 6)
    ax.hist(hist_vals, bins=80, log=True)
    ax.set_title("Absolute error distribution")
    ax.set_xlabel("|after - before|")
    ax.set_ylabel("pixels (log)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_per_view_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def describe_object(obj: Any) -> str:
    if isinstance(obj, torch.nn.Parameter):
        obj = obj.data
    if torch.is_tensor(obj):
        return f"tensor shape={list(obj.shape)} dtype={obj.dtype}"
    if isinstance(obj, np.ndarray):
        return f"ndarray shape={list(obj.shape)} dtype={obj.dtype}"
    if isinstance(obj, (bytes, bytearray)):
        return f"bytes len={len(obj)}"
    if isinstance(obj, Mapping):
        return f"mapping keys={len(obj)}"
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__} len={len(obj)}"
    return type(obj).__name__


def direct_storage_bytes(obj: Any) -> int:
    if isinstance(obj, torch.nn.Parameter):
        obj = obj.data
    if torch.is_tensor(obj):
        return int(obj.untyped_storage().nbytes())
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, (bytes, bytearray)):
        return int(len(obj))
    return 0


def append_size_report_rows(rows: list[dict[str, Any]], obj: Any, *, name: str, total_bytes: int, depth: int, max_depth: int) -> None:
    recursive_bytes = tensor_storage_bytes(obj)
    serialized_bytes = serialized_size(obj) if depth <= 1 or direct_storage_bytes(obj) > 0 else None
    rows.append({
        "path": name,
        "type": describe_object(obj),
        "direct_storage_bytes": direct_storage_bytes(obj),
        "recursive_tensor_or_bytes": recursive_bytes,
        "serialized_bytes": serialized_bytes,
        "percent_of_file": pct(serialized_bytes if serialized_bytes is not None else recursive_bytes, total_bytes),
    })
    if depth >= max_depth:
        return
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            append_size_report_rows(rows, v, name=f"{name}.{k}", total_bytes=total_bytes, depth=depth + 1, max_depth=max_depth)
    elif isinstance(obj, (list, tuple)) and len(obj) <= 512:
        for i, v in enumerate(obj):
            append_size_report_rows(rows, v, name=f"{name}[{i}]", total_bytes=total_bytes, depth=depth + 1, max_depth=max_depth)


def write_size_report_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["path", "type", "direct_storage_bytes", "recursive_tensor_or_bytes", "serialized_bytes", "percent_of_file"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def checkpoint_size_report(checkpoint: Mapping[str, Any], checkpoint_path: str | Path, out_dir: Path) -> dict[str, Any]:
    total = int(os.path.getsize(checkpoint_path))
    rows: list[dict[str, Any]] = []
    append_size_report_rows(rows, checkpoint, name="checkpoint", total_bytes=total, depth=0, max_depth=4)
    csv_path = out_dir / "checkpoint_size_report.csv"
    write_size_report_csv(csv_path, rows)
    top_level = []
    for row in rows:
        if row["path"].count(".") == 1 and "[" not in row["path"]:
            top_level.append(row)
    return {
        "file_bytes": total,
        "file_human": human_bytes(total),
        "csv": str(csv_path),
        "top_level": top_level,
    }


def payload_size_report(payload: Mapping[str, Any], payload_path: str | Path, out_dir: Path) -> dict[str, Any]:
    total = int(os.path.getsize(payload_path))
    rows: list[dict[str, Any]] = []
    append_size_report_rows(rows, payload, name="payload", total_bytes=total, depth=0, max_depth=5)
    csv_path = out_dir / "payload_size_report.csv"
    write_size_report_csv(csv_path, rows)

    components = {
        "gpcc_xyz_bitstream_bytes": len(payload.get("xyz", b"")) if isinstance(payload.get("xyz"), (bytes, bytearray)) else 0,
        "temporal_filter_bytes": tensor_storage_bytes(payload.get("temporal_visibility_filter")),
        "env_map_bytes": tensor_storage_bytes(payload.get("env_map")),
        "mlp_bytes": tensor_storage_bytes(payload.get("mobilegs_opacity_phi_nn")),
        "attributes_total_bytes": tensor_storage_bytes(payload.get("attr", {})),
    }
    attr_components: dict[str, dict[str, Any]] = {}
    for name, pack in payload.get("attr", {}).items():
        info: dict[str, Any] = {"codec": pack.get("codec"), "bytes": tensor_storage_bytes(pack)}
        if pack.get("codec") == "nvq":
            info["codebooks_bytes"] = tensor_storage_bytes(pack.get("codebooks", []))
            info["indices_bytes"] = tensor_storage_bytes(pack.get("indices"))
            info["num_codebooks"] = len(pack.get("codebooks", []))
        elif pack.get("codec") == "uniform":
            info["quantized_values_bytes"] = tensor_storage_bytes(pack.get("q"))
            info["min_step_bytes"] = tensor_storage_bytes(pack.get("min")) + tensor_storage_bytes(pack.get("step"))
            info["bits"] = int(pack.get("bits", 0))
        attr_components[name] = info
    return {
        "file_bytes": total,
        "file_human": human_bytes(total),
        "csv": str(csv_path),
        "components": components,
        "attributes": attr_components,
    }


def print_top_size_report(report: Mapping[str, Any], *, title: str) -> None:
    log_progress(title)
    log_progress(f"  file: {report.get('file_human')} ({report.get('file_bytes')} bytes)")
    for row in report.get("top_level", [])[:20]:
        bytes_ = int(row.get("serialized_bytes") or row.get("recursive_tensor_or_bytes", 0) or 0)
        percent = row.get("percent_of_file")
        if bytes_ > 0 and percent is not None:
            log_progress(f"  {row['path']}: {human_bytes(bytes_)} ({percent:.2f}% of checkpoint report basis)")


def save_quality_summary_plots(out_dir: Path, per_view: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    paths: dict[str, str] = {}
    if not per_view:
        return paths
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.arange(len(per_view))
    fields = {
        "PSNR vs before (dB)": np.asarray([float(v.get("psnr", np.nan)) for v in per_view]),
        "SSIM vs before": np.asarray([float(v.get("ssim", np.nan)) for v in per_view]),
        "L1 loss": np.asarray([float(v.get("l1", np.nan)) for v in per_view]),
        "95th percentile abs error": np.asarray([float(v.get("abs_error_p95", np.nan)) for v in per_view]),
    }
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    for i, (title, values) in enumerate(fields.items(), 1):
        ax = fig.add_subplot(2, 2, i)
        ax.plot(idx, values, marker="o")
        ax.set_title(title)
        ax.set_xlabel("sample")
    quality_path = out_dir / "per_view_quality_plot.png"
    fig.savefig(quality_path, dpi=170)
    plt.close(fig)
    paths["per_view_quality_plot"] = str(quality_path)

    all_mean = np.asarray([float(v.get("abs_error_mean", np.nan)) for v in per_view])
    all_max = np.asarray([float(v.get("abs_error_max", np.nan)) for v in per_view])
    fig = plt.figure(figsize=(9, 5), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(all_mean[~np.isnan(all_mean)], bins=30, alpha=0.7, label="mean")
    ax.hist(all_max[~np.isnan(all_max)], bins=30, alpha=0.5, label="max")
    ax.set_title("Per-view error statistic distribution")
    ax.set_xlabel("absolute error")
    ax.set_ylabel("views")
    ax.legend()
    hist_path = out_dir / "per_view_error_histogram.png"
    fig.savefig(hist_path, dpi=170)
    plt.close(fig)
    paths["per_view_error_histogram"] = str(hist_path)
    return paths


def gaussian_snapshot(gaussians: GaussianModel) -> dict[str, np.ndarray]:
    xyz = gaussians.get_xyz.detach().float().cpu().numpy()
    opacity = gaussians.get_opacity.detach().float().reshape(-1).cpu().numpy()
    scale = gaussians.get_scaling.detach().float().cpu()
    if scale.ndim == 2:
        scale_max = scale.max(dim=1).values.cpu().numpy()
        scale_mean = scale.mean(dim=1).cpu().numpy()
    else:
        scale_max = scale.reshape(-1).cpu().numpy()
        scale_mean = scale_max.copy()
    score = generic_contribution_scores(gaussians).detach().float().cpu().numpy()
    temporal_scale = gaussians.get_scaling_t.detach().float().reshape(-1).cpu().numpy() if gaussians.get_scaling_t.numel() else np.zeros((xyz.shape[0],), dtype=np.float32)
    return {"xyz": xyz, "opacity": opacity, "scale_max": scale_max, "scale_mean": scale_mean, "score": score, "temporal_scale": temporal_scale}


def subsample_indices(n: int, limit: int, seed: int, weights: np.ndarray | None = None) -> np.ndarray:
    n = int(n)
    limit = int(limit)
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if limit <= 0 or n <= limit:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] == n and np.isfinite(w).all() and float(w.sum()) > 0.0:
            return np.sort(rng.choice(n, size=limit, replace=False, p=w / w.sum()).astype(np.int64))
    return np.sort(rng.choice(n, size=limit, replace=False).astype(np.int64))


def equalize_axes_3d(ax, pts: np.ndarray) -> None:
    if pts.size == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * 0.5, 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def save_gaussian_spatial_plots(out_dir: Path, snap: Mapping[str, np.ndarray], kept_idx: np.ndarray, pruned_idx: np.ndarray, *, plot_max_points: int, detail_quantile: float, seed: int) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    xyz = np.asarray(snap["xyz"], dtype=np.float32)
    score = np.asarray(snap["score"], dtype=np.float32)
    opacity = np.asarray(snap["opacity"], dtype=np.float32)
    scale_max = np.asarray(snap["scale_max"], dtype=np.float32)
    temporal_scale = np.asarray(snap["temporal_scale"], dtype=np.float32)
    n = xyz.shape[0]
    detail = score / np.maximum(scale_max, 1e-8)
    finite_detail = detail[np.isfinite(detail)]
    detail_thr = float(np.quantile(finite_detail, float(detail_quantile))) if finite_detail.size else 0.0
    detail_mask = detail >= detail_thr

    keep_set = np.zeros((n,), dtype=bool)
    keep_set[np.asarray(kept_idx, dtype=np.int64)] = True
    prune_set = np.zeros((n,), dtype=bool)
    prune_set[np.asarray(pruned_idx, dtype=np.int64)] = True
    keep_plot = np.nonzero(keep_set)[0]
    prune_plot = np.nonzero(prune_set)[0]
    keep_plot = keep_plot[subsample_indices(len(keep_plot), max(1, plot_max_points // 2), seed=seed, weights=score[keep_plot] + 1e-8)] if len(keep_plot) else keep_plot
    prune_plot = prune_plot[subsample_indices(len(prune_plot), max(1, plot_max_points // 2), seed=seed + 1)] if len(prune_plot) else prune_plot

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    if len(keep_plot):
        ax.scatter(xyz[keep_plot, 0], xyz[keep_plot, 1], xyz[keep_plot, 2], s=2, alpha=0.35, label="kept")
    if len(prune_plot):
        ax.scatter(xyz[prune_plot, 0], xyz[prune_plot, 1], xyz[prune_plot, 2], s=2, alpha=0.35, label="removed")
    ax.set_title("3D Gaussian locations: kept vs removed")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")
    equalize_axes_3d(ax, xyz)
    path = out_dir / "gaussians_3d_kept_removed.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["gaussians_3d_kept_removed"] = str(path)

    all_plot = subsample_indices(n, plot_max_points, seed=seed + 2, weights=score + 1e-8)
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    sc = ax.scatter(xyz[all_plot, 0], xyz[all_plot, 1], xyz[all_plot, 2], c=score[all_plot], s=2, alpha=0.55, cmap="viridis")
    ax.set_title("3D Gaussians colored by generic contribution score")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    equalize_axes_3d(ax, xyz)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="contribution score")
    path = out_dir / "gaussians_3d_contribution_score.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["gaussians_3d_contribution_score"] = str(path)

    detail_idx = np.nonzero(detail_mask)[0]
    other_idx = np.nonzero(~detail_mask)[0]
    detail_idx = detail_idx[subsample_indices(len(detail_idx), max(1, plot_max_points // 2), seed=seed + 3, weights=detail[detail_idx] + 1e-8)] if len(detail_idx) else detail_idx
    other_idx = other_idx[subsample_indices(len(other_idx), max(1, plot_max_points // 2), seed=seed + 4)] if len(other_idx) else other_idx
    pairs = [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    for i, (a, b, name) in enumerate(pairs, 1):
        ax = fig.add_subplot(1, 3, i)
        if len(other_idx):
            ax.scatter(xyz[other_idx, a], xyz[other_idx, b], s=1, alpha=0.12, label="other")
        if len(detail_idx):
            ax.scatter(xyz[detail_idx, a], xyz[detail_idx, b], s=2, alpha=0.75, label="detail proxy")
        ax.set_title(f"{name} projection")
        ax.set_xlabel("xyz"[a])
        ax.set_ylabel("xyz"[b])
        ax.set_aspect("equal", adjustable="box")
        if i == 1:
            ax.legend(loc="best")
    fig.suptitle(f"Detail proxy: contribution / max-scale, top {(1.0 - detail_quantile) * 100.0:.1f}%")
    path = out_dir / "gaussians_detail_proxy_projections.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["gaussians_detail_proxy_projections"] = str(path)

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    values = [opacity, scale_max, score, detail, temporal_scale]
    titles = ["Opacity", "Max spatial scale", "Contribution score", "Detail proxy", "Temporal scale"]
    for i, (vals, title) in enumerate(zip(values, titles), 1):
        ax = fig.add_subplot(2, 3, i)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=80, log=True)
        ax.set_title(title)
        ax.set_ylabel("count (log)")
    ax = fig.add_subplot(2, 3, 6)
    if len(scale_max) and len(score):
        sample = subsample_indices(n, plot_max_points, seed=seed + 5, weights=score + 1e-8)
        ax.scatter(scale_max[sample], score[sample], s=2, alpha=0.35)
    ax.set_xlabel("max spatial scale")
    ax.set_ylabel("contribution score")
    ax.set_title("Scale vs contribution")
    path = out_dir / "gaussian_attribute_histograms.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["gaussian_attribute_histograms"] = str(path)
    paths["detail_proxy_threshold"] = str(detail_thr)
    return paths


def tensor_flat_mse_per_gaussian(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    before = before.detach().float()
    after = after.detach().float()
    if before.shape != after.shape or before.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)
    return (before.reshape(before.shape[0], -1) - after.reshape(after.shape[0], -1)).pow(2).mean(dim=1).detach().cpu()


def attribute_error_report(before: GaussianModel, after: GaussianModel) -> dict[str, Any]:
    report: dict[str, Any] = {}
    before_attrs = dict(gaussian_attribute_tensors(before), xyz=before.get_xyz.detach())
    after_attrs = dict(gaussian_attribute_tensors(after), xyz=after.get_xyz.detach())
    for name, b in before_attrs.items():
        a = after_attrs.get(name)
        if a is None or a.shape != b.shape:
            report[name] = {"available": False, "reason": "missing_or_shape_mismatch", "before_shape": list(b.shape), "after_shape": list(a.shape) if a is not None else None}
            continue
        diff = (a.detach().float() - b.detach().float()).reshape(-1)
        report[name] = {
            "available": True,
            "shape": list(b.shape),
            "mse": float((diff * diff).mean().detach().cpu().item()) if diff.numel() else 0.0,
            "mae": float(diff.abs().mean().detach().cpu().item()) if diff.numel() else 0.0,
            "max_abs": float(diff.abs().max().detach().cpu().item()) if diff.numel() else 0.0,
        }
    return report


def save_attribute_error_3d_plot(out_dir: Path, before: GaussianModel, after: GaussianModel, *, plot_max_points: int, seed: int) -> dict[str, str]:
    paths: dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    if int(before.get_xyz.shape[0]) != int(after.get_xyz.shape[0]):
        return paths
    xyz = before.get_xyz.detach().float().cpu().numpy()
    n = xyz.shape[0]
    per = torch.zeros((n,), dtype=torch.float32)
    count = 0
    for name, b in gaussian_attribute_tensors(before).items():
        a = gaussian_attribute_tensors(after).get(name)
        if a is None or a.shape != b.shape:
            continue
        e = tensor_flat_mse_per_gaussian(b.detach().cpu(), a.detach().cpu())
        if e.numel() == n:
            per += e
            count += 1
    if count == 0:
        return paths
    per = (per / float(count)).numpy()
    sample = subsample_indices(n, plot_max_points, seed=seed, weights=per + 1e-12)
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    sc = ax.scatter(xyz[sample, 0], xyz[sample, 1], xyz[sample, 2], c=per[sample], s=2, alpha=0.55, cmap="inferno")
    ax.set_title("3D Gaussians colored by mean attribute compression MSE")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    equalize_axes_3d(ax, xyz)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="attribute MSE")
    path = out_dir / "gaussians_3d_attribute_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["gaussians_3d_attribute_error"] = str(path)
    return paths


def render_quality_and_plots(*, raw: GaussianModel, compressed: GaussianModel, checkpoint: Mapping[str, Any], cameras: Sequence[Any], background: torch.Tensor, out_dir: Path, render_mode: str, temporal_filter: bool, temporal_keyframes: int, temporal_window: int, temporal_threshold: float, eval_samples: int, save_renders: bool, save_difference_plots: bool, diff_scale: float, disable_dropout: bool) -> dict[str, Any]:
    if eval_samples <= 0:
        return {"enabled": False}
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe_raw = make_pipe(checkpoint, render_mode=render_mode, disable_dropout=disable_dropout, temporal_filter=False, temporal_keyframes=0, temporal_window=temporal_window, temporal_threshold=temporal_threshold)
    pipe_cmp = make_pipe(checkpoint, render_mode=render_mode, disable_dropout=disable_dropout, temporal_filter=temporal_filter, temporal_keyframes=temporal_keyframes, temporal_window=temporal_window, temporal_threshold=temporal_threshold)
    samples = list(cameras[: min(len(cameras), int(eval_samples))])
    per_view: list[dict[str, Any]] = []
    with torch.inference_mode():
        for idx, cam in enumerate(samples):
            log_progress(f"quality render {idx + 1}/{len(samples)}: {getattr(cam, 'image_name', f'camera_{idx}')}")
            raw_img = render(cam, raw, pipe_raw, background)["render"].clamp(0, 1)
            cmp_img = render(cam, compressed, pipe_cmp, background)["render"].clamp(0, 1)
            diff = (cmp_img - raw_img).detach()
            abs_err = diff.abs()
            row = {
                "index": idx,
                "uid": int(getattr(cam, "uid", idx)),
                "image_name": str(getattr(cam, "image_name", f"camera_{idx}")),
                "timestamp": float(getattr(cam, "timestamp", 0.0)),
                "psnr": float(psnr(cmp_img.unsqueeze(0), raw_img.unsqueeze(0)).mean().detach().cpu().item()),
                "ssim": float(ssim(cmp_img, raw_img).detach().cpu().item()),
                "l1": float(l1_loss(cmp_img, raw_img).detach().cpu().item()),
                "lpips": float(lpips_metric(cmp_img.unsqueeze(0), raw_img.unsqueeze(0))),
                "abs_error_mean": float(abs_err.mean().detach().cpu().item()),
                "abs_error_p50": float(torch.quantile(abs_err.reshape(-1), 0.50).detach().cpu().item()),
                "abs_error_p95": float(torch.quantile(abs_err.reshape(-1), 0.95).detach().cpu().item()),
                "abs_error_p99": float(torch.quantile(abs_err.reshape(-1), 0.99).detach().cpu().item()),
                "abs_error_max": float(abs_err.max().detach().cpu().item()),
            }
            per_view.append(row)
            if save_renders:
                save_image(out_dir / f"sample_{idx:03d}_before.png", raw_img)
                save_image(out_dir / f"sample_{idx:03d}_after.png", cmp_img)
                save_image(out_dir / f"sample_{idx:03d}_absdiff_x{diff_scale:g}.png", (abs_err * diff_scale).clamp(0, 1))
            if save_difference_plots:
                save_render_comparison_plot(out_dir / f"sample_{idx:03d}_comparison.png", raw_img, cmp_img, l1_value=row["l1"], psnr_value=row["psnr"], ssim_value=row["ssim"], diff_scale=diff_scale, title=f"Sample {idx}: {row['image_name']} @ t={row['timestamp']:.4f}")
    csv_path = out_dir / "per_view_quality.csv"
    write_per_view_csv(csv_path, per_view)
    plot_paths = save_quality_summary_plots(out_dir, per_view)
    summary: dict[str, float] = {}
    for key in ("psnr", "ssim", "l1", "lpips", "abs_error_mean", "abs_error_p95", "abs_error_p99", "abs_error_max"):
        vals = np.asarray([float(v[key]) for v in per_view], dtype=np.float64)
        summary[f"mean_{key}"] = float(vals.mean())
        summary[f"min_{key}"] = float(vals.min())
        summary[f"max_{key}"] = float(vals.max())
    return {"enabled": True, "render_mode": selected_render_mode(checkpoint, render_mode), "samples": len(per_view), "per_view_csv": str(csv_path), "plots": plot_paths, "summary": summary, "per_view": per_view}


def benchmark_pair(*, raw: GaussianModel, compressed: GaussianModel, checkpoint: Mapping[str, Any], cameras: Sequence[Any], background: torch.Tensor, render_mode: str, warmup: int, repeats: int, temporal_filter: bool, temporal_keyframes: int, temporal_window: int, temporal_threshold: float, disable_dropout: bool) -> dict[str, Any]:
    pipe_raw = make_pipe(checkpoint, render_mode=render_mode, disable_dropout=disable_dropout, temporal_filter=False, temporal_keyframes=0, temporal_window=temporal_window, temporal_threshold=temporal_threshold)
    pipe_cmp = make_pipe(checkpoint, render_mode=render_mode, disable_dropout=disable_dropout, temporal_filter=temporal_filter, temporal_keyframes=temporal_keyframes, temporal_window=temporal_window, temporal_threshold=temporal_threshold)
    log_progress(f"benchmark raw renderer: warmup={warmup}, repeats={repeats}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    raw_bench = benchmark_renderer(raw, cameras, pipe_raw, background, render, warmup=warmup, repeats=repeats)
    raw_vram = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    log_progress(f"benchmark compressed renderer: warmup={warmup}, repeats={repeats}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cmp_bench = benchmark_renderer(compressed, cameras, pipe_cmp, background, render, warmup=warmup, repeats=repeats)
    cmp_vram = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    raw_fps = float(raw_bench["fps"])
    cmp_fps = float(cmp_bench["fps"])
    fps_delta = cmp_fps - raw_fps
    vram_delta = None if raw_vram is None or cmp_vram is None else int(cmp_vram - raw_vram)
    return {
        "raw": raw_bench,
        "compressed": cmp_bench,
        "speedup": cmp_fps / max(raw_fps, 1e-9),
        "fps_delta": fps_delta,
        "fps_percent_change": 100.0 * fps_delta / max(raw_fps, 1e-9),
        "raw_peak_vram_bytes": raw_vram,
        "raw_peak_vram_human": human_bytes(raw_vram) if raw_vram is not None else None,
        "compressed_peak_vram_bytes": cmp_vram,
        "compressed_peak_vram_human": human_bytes(cmp_vram) if cmp_vram is not None else None,
        "peak_vram_delta_bytes": vram_delta,
        "peak_vram_delta_human": human_bytes(abs(vram_delta)) if vram_delta is not None else None,
        "peak_vram_percent_change": None if raw_vram is None or cmp_vram is None else 100.0 * float(cmp_vram - raw_vram) / max(float(raw_vram), 1.0),
        "peak_vram_reduction": reduction(raw_vram, cmp_vram) if raw_vram and cmp_vram else None,
    }


def temporal_filter_stats(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    filt = payload.get("temporal_visibility_filter")
    if filt is None:
        return None
    packed = np.asarray(filt["packed_masks"])
    shape = tuple(filt["shape"])
    masks = np.unpackbits(packed, axis=1)[:, : shape[1]].astype(bool)
    active = masks.sum(axis=1)
    return {"keyframes": int(shape[0]), "num_points": int(shape[1]), "active_min": int(active.min()), "active_mean": float(active.mean()), "active_max": int(active.max()), "active_ratio_mean": float(active.mean() / max(shape[1], 1)), "packed_mask_bytes": int(packed.nbytes)}


def parse_args():
    parser = argparse.ArgumentParser(description="Universal 4DGS checkpoint compression postprocess and diagnostics.")
    parser.add_argument("--ckpt-path", required=True, help="Input self-contained checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory for payload, metrics, renders, and plots")
    parser.add_argument("--output-name", default="checkpoint.universal_compressed.pt", help="Compressed payload filename")
    parser.add_argument("--device", default="cuda", help="Usually cuda. No automatic fallback is performed.")
    parser.add_argument("--codec", default="mobilegs", choices=("mobilegs", "nvq", "uniform", "float16", "float32"), help="Attribute compression codec. mobilegs uses GPCC xyz + NVQ codebooks for attributes + uniform opacity.")
    parser.add_argument("--uniform-bits", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--kmeans-iters", type=int, default=16)
    parser.add_argument("--sh-degree-cap", type=int, default=-1, help="Optional SH degree cap. -1 preserves the checkpoint model exactly.")
    parser.add_argument("--mobilegs-first-order-sh", action=argparse.BooleanOptionalAction, default=False, help="Mobile-GS-style first-order SH export; equivalent to --sh-degree-cap 1 when the checkpoint has higher SH.")
    parser.add_argument("--target-gaussians", type=int, default=0, help="Contribution-prune to this many Gaussians. 0 disables pruning by count.")
    parser.add_argument("--target-size-mb", type=float, default=0.0, help="Estimate a contribution-pruned point count for this checkpoint tensor budget. 0 disables size-target pruning.")
    parser.add_argument("--min-points", type=int, default=1024)
    parser.add_argument("--include-mlp", action=argparse.BooleanOptionalAction, default=True, help="Preserve optional sort-free/MobileGS MLP if checkpoint has one")
    parser.add_argument("--build-temporal-filter", action=argparse.BooleanOptionalAction, default=False, help="Build optional keyframe visibility masks for 4DGS-1K-style render acceleration")
    parser.add_argument("--temporal-keyframes", type=int, default=32)
    parser.add_argument("--temporal-mask-window", type=int, default=1)
    parser.add_argument("--temporal-mask-threshold", type=float, default=0.05)
    parser.add_argument("--views-per-keyframe", type=int, default=0)
    parser.add_argument("--camera-split", default="test", choices=("train", "test", "all"))
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--render-mode", default="checkpoint", choices=("checkpoint", "sorted", "sort_free"), help="checkpoint preserves the ablation/method renderer from run_config")
    parser.add_argument("--disable-dropout-render", action=argparse.BooleanOptionalAction, default=True, help="Disable random dropout during eval/benchmark even for DropoutGS checkpoints")
    parser.add_argument("--save-renders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-difference-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-gaussian-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-max-points", type=int, default=200000)
    parser.add_argument("--detail-quantile", type=float, default=0.95)
    parser.add_argument("--diff-scale", type=float, default=8.0)
    parser.add_argument("--size-report-depth", type=int, default=4, help="Reserved for future deeper reports; current report includes top-level checkpoint and payload breakdowns.")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / args.output_name
    plots_dir = out_dir / "plots"
    renders_dir = out_dir / "renders"

    log_progress("loading checkpoint and Gaussians")
    checkpoint, raw_gaussians, gaussian_kwargs = load_gaussians_from_checkpoint(args.ckpt_path, args.device)
    background = background_from_checkpoint(checkpoint, args.device)
    cameras, camera_split_used = available_cameras(checkpoint, args.camera_split, args.device)
    eval_cameras = cameras[: max(1, min(len(cameras), int(args.eval_samples)))]
    effective_render_mode = selected_render_mode(checkpoint, args.render_mode)

    raw_checkpoint_bytes = os.path.getsize(args.ckpt_path)
    raw_tensor_bytes = gaussian_tensor_size(raw_gaussians)
    raw_count = int(raw_gaussians.get_xyz.shape[0])
    log_progress(f"loaded {raw_count:,} Gaussians, checkpoint={human_bytes(raw_checkpoint_bytes)}, Gaussian tensors={human_bytes(raw_tensor_bytes)}")
    checkpoint_report = checkpoint_size_report(checkpoint, args.ckpt_path, out_dir)
    print_top_size_report(checkpoint_report, title="checkpoint size report")

    pre_snap = gaussian_snapshot(raw_gaussians)
    kept_idx, pruned_idx, _scores, requested_keep_count = predict_prune_indices(
        raw_gaussians,
        target_gaussians=args.target_gaussians,
        target_size_mb=args.target_size_mb,
        min_points=args.min_points,
        raw_tensor_bytes=raw_tensor_bytes,
    )

    log_progress("preparing compressed working copy")
    work_gaussians = load_gaussians_from_checkpoint(args.ckpt_path, args.device)[1]
    sh_cap = 1 if args.mobilegs_first_order_sh and args.sh_degree_cap < 0 else args.sh_degree_cap
    maybe_reduce_sh_degree(work_gaussians, sh_cap)
    log_progress(f"contribution pruning target keep count: {requested_keep_count:,} / {raw_count:,}")
    prune_report = prune_generic_contribution(
        work_gaussians,
        target_gaussians=args.target_gaussians,
        target_size_mb=args.target_size_mb,
        min_points=args.min_points,
        raw_tensor_bytes=raw_tensor_bytes,
    )
    log_progress(f"post-prune Gaussians: {int(work_gaussians.get_xyz.shape[0]):,}")

    temporal_filter = None
    if args.build_temporal_filter:
        log_progress("building temporal visibility filter")
        filter_pipe = make_pipe(checkpoint, render_mode="sorted", disable_dropout=True, temporal_filter=False, temporal_keyframes=0, temporal_window=args.temporal_mask_window, temporal_threshold=args.temporal_mask_threshold)
        train_cameras = cameras_from_checkpoint_scene(checkpoint.get("scene", {}), split="train", device=args.device)
        if not train_cameras:
            train_cameras = cameras
        temporal_filter = build_temporal_visibility_filter(work_gaussians, train_cameras, filter_pipe, background, render, keyframes=args.temporal_keyframes, views_per_keyframe=args.views_per_keyframe)
        log_progress("temporal visibility filter complete")

    log_progress(f"encoding payload with codec={args.codec}, GPCC xyz, codebooks={args.codebook_size}, block_size={args.block_size}")
    payload = capture_universal_payload(
        work_gaussians,
        codec=args.codec,
        uniform_bits=args.uniform_bits,
        codebook_size=args.codebook_size,
        block_size=args.block_size,
        kmeans_iters=args.kmeans_iters,
        include_mlp=args.include_mlp,
        temporal_visibility_filter=temporal_filter,
        seed=args.seed,
    )
    save_payload(payload, payload_path)
    payload_file_bytes = os.path.getsize(payload_path)
    payload_serialized_bytes = serialized_size(payload)
    payload_report = payload_size_report(payload, payload_path, out_dir)
    log_progress(f"saved compressed payload: {human_bytes(payload_file_bytes)}")
    log_progress(f"payload breakdown CSV: {payload_report['csv']}")

    log_progress("restoring compressed payload for verification/evaluation")
    restored = restore_universal_payload(load_payload(payload_path, map_location="cpu"), device=args.device, training_args=None)
    if restored.mobilegs_opacity_phi_nn is not None:
        restored.mobilegs_opacity_phi_nn.eval()

    gaussian_plots: dict[str, str] = {}
    if args.save_gaussian_plots:
        log_progress("writing 3D Gaussian diagnostic plots")
        gaussian_plots.update(save_gaussian_spatial_plots(plots_dir / "gaussians", pre_snap, kept_idx, pruned_idx, plot_max_points=args.plot_max_points, detail_quantile=args.detail_quantile, seed=args.seed))
        gaussian_plots.update(save_attribute_error_3d_plot(plots_dir / "gaussians", work_gaussians, restored, plot_max_points=args.plot_max_points, seed=args.seed + 27))

    log_progress("rendering before/after quality samples")
    quality = render_quality_and_plots(
        raw=raw_gaussians,
        compressed=restored,
        checkpoint=checkpoint,
        cameras=eval_cameras,
        background=background,
        out_dir=renders_dir,
        render_mode=args.render_mode,
        temporal_filter=args.build_temporal_filter,
        temporal_keyframes=args.temporal_keyframes,
        temporal_window=args.temporal_mask_window,
        temporal_threshold=args.temporal_mask_threshold,
        eval_samples=args.eval_samples,
        save_renders=args.save_renders,
        save_difference_plots=args.save_difference_plots,
        diff_scale=args.diff_scale,
        disable_dropout=args.disable_dropout_render,
    )

    log_progress("running render speed and VRAM benchmarks")
    benchmark = benchmark_pair(
        raw=raw_gaussians,
        compressed=restored,
        checkpoint=checkpoint,
        cameras=eval_cameras,
        background=background,
        render_mode=args.render_mode,
        warmup=args.warmup,
        repeats=args.repeats,
        temporal_filter=args.build_temporal_filter,
        temporal_keyframes=args.temporal_keyframes,
        temporal_window=args.temporal_mask_window,
        temporal_threshold=args.temporal_mask_threshold,
        disable_dropout=args.disable_dropout_render,
    )

    metrics = {
        "input_checkpoint": str(args.ckpt_path),
        "output_payload": str(payload_path),
        "format": UNIVERSAL_FORMAT,
        "camera_split_used": camera_split_used,
        "render_mode": {"requested": args.render_mode, "effective": effective_render_mode},
        "gaussian_kwargs": gaussian_kwargs,
        "counts": {
            "raw_gaussians": raw_count,
            "post_prune_gaussians": int(work_gaussians.get_xyz.shape[0]),
            "restored_gaussians": int(restored.get_xyz.shape[0]),
            "reduction": reduction(raw_count, int(restored.get_xyz.shape[0])),
        },
        "sizes": {
            "raw_checkpoint_bytes": raw_checkpoint_bytes,
            "raw_checkpoint_human": human_bytes(raw_checkpoint_bytes),
            "raw_gaussian_tensor_bytes": raw_tensor_bytes,
            "raw_gaussian_tensor_human": human_bytes(raw_tensor_bytes),
            "compressed_payload_serialized_bytes": payload_serialized_bytes,
            "compressed_file_bytes": payload_file_bytes,
            "compressed_file_human": human_bytes(payload_file_bytes),
            "size_delta_bytes": int(payload_file_bytes - raw_checkpoint_bytes),
            "size_delta_human": human_bytes(abs(int(payload_file_bytes - raw_checkpoint_bytes))),
            "reduction_vs_checkpoint": reduction(raw_checkpoint_bytes, payload_file_bytes),
            "reduction_vs_raw_gaussian_tensors": reduction(raw_tensor_bytes, payload_file_bytes),
        },
        "size_reports": {
            "checkpoint": checkpoint_report,
            "compressed_payload": payload_report,
        },
        "compression": {
            "profile": "mobilegs_postprocess" if args.codec == "mobilegs" else "universal",
            "codec": args.codec,
            "mobilegs_components": {
                "gpcc_xyz": True,
                "subvector_codebooks": args.codec in ("mobilegs", "nvq"),
                "uniform_opacity": args.codec == "mobilegs",
                "first_order_sh": bool(args.mobilegs_first_order_sh or sh_cap == 1),
                "contribution_pruning_by_target": bool(args.target_gaussians > 0 or args.target_size_mb > 0.0),
                "temporal_visibility_filter": bool(args.build_temporal_filter),
            },
            "uniform_bits": int(args.uniform_bits),
            "codebook_size": int(args.codebook_size),
            "block_size": int(args.block_size),
            "kmeans_iters": int(args.kmeans_iters),
            "sh_degree_cap": int(sh_cap),
            "mobilegs_first_order_sh": bool(args.mobilegs_first_order_sh),
            "include_mlp": bool(args.include_mlp),
            "pruning": prune_report,
            "temporal_visibility_filter": temporal_filter_stats(payload),
        },
        "attribute_errors": attribute_error_report(work_gaussians, restored),
        "quality": quality,
        "performance": benchmark,
        "visualizations": {
            "gaussian_plots": gaussian_plots,
            "render_dir": str(renders_dir),
            "plot_dir": str(plots_dir),
        },
    }
    metrics_path = out_dir / "compression_postprocess_metrics.json"
    write_json(metrics_path, json_safe(metrics))

    mean_psnr = quality.get("summary", {}).get("mean_psnr") if isinstance(quality, Mapping) else None
    log_progress("Universal compression postprocess complete")
    print("  payload:", payload_path)
    print("  metrics:", metrics_path)
    print("  checkpoint size report:", checkpoint_report["csv"])
    print("  compressed payload size report:", payload_report["csv"])
    print("  render mode:", effective_render_mode)
    print("  raw checkpoint:", human_bytes(raw_checkpoint_bytes))
    print("  raw Gaussian tensors:", human_bytes(raw_tensor_bytes))
    print("  compressed payload:", human_bytes(payload_file_bytes))
    print("  size delta:", f"{payload_file_bytes - raw_checkpoint_bytes:+d} bytes")
    print("  size reduction vs checkpoint:", f"{metrics['sizes']['reduction_vs_checkpoint']['ratio']:.3f}x" if metrics['sizes']['reduction_vs_checkpoint']['ratio'] else "n/a")
    print("  PSNR after vs before:", f"{mean_psnr:.3f} dB" if mean_psnr is not None else "n/a")
    print("  raw FPS:", f"{benchmark['raw']['fps']:.2f}")
    print("  compressed FPS:", f"{benchmark['compressed']['fps']:.2f}")
    print("  FPS delta:", f"{benchmark['fps_delta']:+.2f} ({benchmark['fps_percent_change']:+.2f}%)")
    print("  speedup:", f"{benchmark['speedup']:.2f}x")
    print("  raw peak VRAM:", benchmark.get("raw_peak_vram_human"))
    print("  compressed peak VRAM:", benchmark.get("compressed_peak_vram_human"))
    print("  peak VRAM delta:", (f"{benchmark['peak_vram_delta_bytes']:+d} bytes ({benchmark['peak_vram_percent_change']:+.2f}%)" if benchmark.get("peak_vram_delta_bytes") is not None else "n/a"))


if __name__ == "__main__":
    main()
