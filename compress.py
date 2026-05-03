import io
import os
import numpy as np
import torch
import torch.nn as nn

from scene.gaussian_model import GaussianModel
from utils.checkpoint_utils import load_checkpoint
from utils.gpcc_utils import calculate_morton_order

import argparse
import io
import json
import os

import numpy as np
import torch
import torch.nn as nn

def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def tensor_bytes(obj, seen=None):
    if seen is None:
        seen = set()

    if isinstance(obj, torch.nn.Parameter):
        obj = obj.data

    if torch.is_tensor(obj):
        storage = obj.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key in seen:
            return 0
        seen.add(key)
        return storage.nbytes()

    if isinstance(obj, np.ndarray):
        key = (obj.__array_interface__["data"][0], obj.nbytes)
        if key in seen:
            return 0
        seen.add(key)
        return obj.nbytes

    if isinstance(obj, dict):
        return sum(tensor_bytes(v, seen) for v in obj.values())

    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v, seen) for v in obj)

    if isinstance(obj, (bytes, bytearray)):
        return len(obj)

    return 0


def serialized_bytes(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.tell()


def reduction(before, after):
    if before <= 0:
        return float("nan"), float("nan")
    ratio = before / max(after, 1)
    pct = 100.0 * (1.0 - (after / before))
    return ratio, pct


def load_gaussians_from_checkpoint(
    ckpt_path,
    device="auto",
    model_overrides=None,
):
    device = resolve_device(device)
    model_overrides = model_overrides or {}
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    model_kwargs = dict(checkpoint["run_config"].get("gaussian_kwargs", {}))
    model_kwargs.update(model_overrides)
    if int(model_kwargs.get("gaussian_dim", 4)) != 4:
        raise ValueError("Only 4D Gaussian checkpoints are supported.")

    gaussians = GaussianModel(**model_kwargs)
    gaussians.restore(checkpoint["gaussians"], training_args=None)
    return gaussians, int(checkpoint["iteration"]), checkpoint, model_kwargs


def build_raw_gaussian_payload(gaussians):
    if gaussians.gaussian_dim != 4:
        raise ValueError("Only 4D Gaussian payloads are supported.")
    return {
        "format": "gaussian-raw-v1",
        "meta": {
            "gaussian_dim": gaussians.gaussian_dim,
            "rot_4d": gaussians.rot_4d,
            "isotropic_gaussians": gaussians.isotropic_gaussians,
            "force_sh_3d": gaussians.force_sh_3d,
            "max_sh_degree": gaussians.max_sh_degree,
            "active_sh_degree": gaussians.active_sh_degree,
            "max_sh_degree_t": gaussians.max_sh_degree_t,
            "active_sh_degree_t": gaussians.active_sh_degree_t,
            "time_duration": list(gaussians.time_duration),
            "prefilter_var": gaussians.prefilter_var,
        },
        "xyz": gaussians.get_xyz.detach().cpu(),
        "attr": {k: v.detach().cpu() for k, v in gaussians._compressible_tensors().items()},
    }


def get_compression_sort_idx(gaussians):
    xyz_q, _ = gaussians._quantize_xyz_u16(gaussians.get_xyz.detach())
    sort_idx = calculate_morton_order(xyz_q.int()).detach().cpu().long()
    return sort_idx


def collect_model_tensors(gaussians, sort_idx=None):
    def take(x):
        x = x.detach().cpu()
        return x if sort_idx is None else x[sort_idx]

    out = {
        "xyz": take(gaussians.get_xyz),
        "features_dc": take(gaussians._features_dc),
        "features_rest": take(gaussians._features_rest),
        "scaling": take(gaussians._scaling),
        "opacity": take(gaussians._opacity),
    }
    if not gaussians.isotropic_gaussians:
        out["rotation"] = take(gaussians._rotation)
    out["t"] = take(gaussians._t)
    out["scaling_t"] = take(gaussians._scaling_t)
    if gaussians.rot_4d and not gaussians.isotropic_gaussians:
        out["rotation_r"] = take(gaussians._rotation_r)
    return out


def compare_tensors(a, b):
    diff = (a.float() - b.float()).reshape(-1)
    ref = a.float().reshape(-1)

    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff * diff).mean()).item()
    max_abs = diff.abs().max().item()

    ref_abs_mean = ref.abs().mean().item()
    ref_rms = torch.sqrt((ref * ref).mean()).item()

    rel_mae = mae / (ref_abs_mean + 1e-12)
    rel_rmse = rmse / (ref_rms + 1e-12)

    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "rel_mae": rel_mae,
        "rel_rmse": rel_rmse,
    }


def validate_roundtrip(original_model, reconstructed_model, rel_rmse_threshold, max_abs_threshold):
    sort_idx = get_compression_sort_idx(original_model)

    orig = collect_model_tensors(original_model, sort_idx=sort_idx)
    recon = collect_model_tensors(reconstructed_model, sort_idx=None)

    per_tensor = {}
    worst_rel_rmse_name = None
    worst_rel_rmse = -1.0
    worst_max_abs_name = None
    worst_max_abs = -1.0

    for name in orig.keys():
        if name not in recon:
            raise KeyError(f"Missing tensor in reconstructed model: {name}")
        if orig[name].shape != recon[name].shape:
            raise ValueError(f"Shape mismatch for {name}: {orig[name].shape} vs {recon[name].shape}")

        stats = compare_tensors(orig[name], recon[name])
        per_tensor[name] = stats

        if stats["rel_rmse"] > worst_rel_rmse:
            worst_rel_rmse = stats["rel_rmse"]
            worst_rel_rmse_name = name

        if stats["max_abs"] > worst_max_abs:
            worst_max_abs = stats["max_abs"]
            worst_max_abs_name = name

    passed = (worst_rel_rmse <= rel_rmse_threshold) and (worst_max_abs <= max_abs_threshold)

    return {
        "passed": passed,
        "per_tensor": per_tensor,
        "worst_rel_rmse_name": worst_rel_rmse_name,
        "worst_rel_rmse": worst_rel_rmse,
        "worst_max_abs_name": worst_max_abs_name,
        "worst_max_abs": worst_max_abs,
    }


def print_validation_report(summary, rel_rmse_threshold, max_abs_threshold):
    print("\nvalidation:")
    for name, stats in summary["per_tensor"].items():
        print(
            f"  {name:12s} "
            f"MAE={stats['mae']:.6e} "
            f"RMSE={stats['rmse']:.6e} "
            f"MaxAbs={stats['max_abs']:.6e} "
            f"RelMAE={stats['rel_mae']:.6e} "
            f"RelRMSE={stats['rel_rmse']:.6e}"
        )

    print(f"\nworst rel RMSE: {summary['worst_rel_rmse_name']} = {summary['worst_rel_rmse']:.6e}")
    print(f"worst max abs : {summary['worst_max_abs_name']} = {summary['worst_max_abs']:.6e}")
    print(
        f"pass          : {summary['passed']} "
        f"(rel_rmse <= {rel_rmse_threshold:.1e}, max_abs <= {max_abs_threshold:.1e})"
    )

def resolve_device(requested_device=None):
    if requested_device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        return "cpu"

    return requested_device

def load_gaussians_from_memory(
    ckpt_obj,
    ckpt_path=None,
    device="auto",
    model_overrides=None,
):
    device = resolve_device(device)
    model_overrides = model_overrides or {}
    if not isinstance(ckpt_obj, dict) or "gaussians" not in ckpt_obj:
        raise ValueError("Expected a self-contained 4DGS checkpoint dictionary.")

    model_kwargs = dict(ckpt_obj["run_config"].get("gaussian_kwargs", {}))
    model_kwargs.update(model_overrides)
    gaussians = GaussianModel(**model_kwargs)
    gaussians.restore(ckpt_obj["gaussians"], training_args=None)
    if device != "cpu":
        # Checkpoints loaded with map_location=device already hold device tensors.
        # This function keeps CPU-loaded dicts on CPU to avoid silently migrating
        # optimizer-sized payloads outside the explicit load path.
        pass
    aux_tail = ckpt_obj["gaussians"][-1]
    return gaussians, int(ckpt_obj["iteration"]), aux_tail, model_kwargs


def load_original_and_reconstructed(
    ckpt_path,
    attr_bits=8,
    device="auto",
    compressed_path=None,
    model_overrides=None,
    rel_rmse_threshold=1.5e-2,
    max_abs_threshold=3e-2,
):
    device = resolve_device(device)
    model_overrides = model_overrides or {}

    original, iteration, aux_tail, model_kwargs = load_gaussians_from_checkpoint(
        ckpt_path, device=device, model_overrides=model_overrides
    )
    ckpt = load_checkpoint(ckpt_path, map_location=device)

    compressed = original.capture_compressed(attr_bits=attr_bits)

    if compressed_path is not None:
        parent = os.path.dirname(compressed_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        original.save_compressed(compressed_path, attr_bits=attr_bits)
        compressed_disk_size = os.path.getsize(compressed_path)
    else:
        compressed_disk_size = serialized_bytes(compressed)

    reconstructed = GaussianModel(**model_kwargs)
    reconstructed.restore_compressed(compressed, training_args=None, device=device)

    sort_idx = get_compression_sort_idx(original)
    orig = collect_model_tensors(original, sort_idx=sort_idx)
    recon = collect_model_tensors(reconstructed, sort_idx=None)

    raw_payload = build_raw_gaussian_payload(original)

    result = {
        "iteration": iteration,
        "device": device,
        "model_kwargs": model_kwargs,
        "aux_tail": aux_tail,
        "original_model": original,
        "reconstructed_model": reconstructed,
        "compressed": compressed,
        "orig_tensors": orig,
        "recon_tensors": recon,
        "raw_ckpt_file_size": os.path.getsize(ckpt_path),
        "raw_ckpt_ram_size": tensor_bytes(ckpt),
        "raw_payload_disk_size": serialized_bytes(raw_payload),
        "raw_payload_ram_size": tensor_bytes(raw_payload),
        "compressed_disk_size": compressed_disk_size,
        "compressed_ram_size": tensor_bytes(compressed),
        "validation": validate_roundtrip(
            original,
            reconstructed,
            rel_rmse_threshold=rel_rmse_threshold,
            max_abs_threshold=max_abs_threshold,
        ),
        "rel_rmse_threshold": rel_rmse_threshold,
        "max_abs_threshold": max_abs_threshold,
    }
    return result

def main(
    ckpt_path,
    compressed_path=None,
    attr_bits=8,
    device=None,
    model_overrides=None,
    rel_rmse_threshold=1.5e-2,
    max_abs_threshold=3e-2,
):
    device = resolve_device(device)
    model_overrides = model_overrides or {}

    file_size = os.path.getsize(ckpt_path)

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    gaussians, iteration, aux_tail, model_kwargs = load_gaussians_from_memory(
        ckpt, ckpt_path=ckpt_path, device=device, model_overrides=model_overrides
    )
    model_params = ckpt["gaussians"]
    ram_size = tensor_bytes(ckpt)

    raw_payload = build_raw_gaussian_payload(gaussians)
    raw_payload_disk_size = serialized_bytes(raw_payload)
    raw_payload_ram_size = tensor_bytes(raw_payload)

    compressed = gaussians.capture_compressed(attr_bits=attr_bits)
    compressed_disk_size = serialized_bytes(compressed)
    compressed_ram_size = tensor_bytes(compressed)

    if compressed_path:
        parent = os.path.dirname(compressed_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        gaussians.save_compressed(compressed_path, attr_bits=attr_bits)
        compressed_disk_size = os.path.getsize(compressed_path)

    reconstructed = GaussianModel(**model_kwargs)
    reconstructed.restore_compressed(compressed, training_args=None, device=device)

    validation = validate_roundtrip(
        gaussians,
        reconstructed,
        rel_rmse_threshold=rel_rmse_threshold,
        max_abs_threshold=max_abs_threshold,
    )

    ratio_ckpt_disk, pct_ckpt_disk = reduction(file_size, compressed_disk_size)
    ratio_raw_disk, pct_raw_disk = reduction(raw_payload_disk_size, compressed_disk_size)
    ratio_raw_ram, pct_raw_ram = reduction(raw_payload_ram_size, compressed_ram_size)

    print("path:", ckpt_path)
    print("iteration:", iteration)
    print("device:", device)
    print("attr_bits:", attr_bits)
    print("gaussian tuple length:", len(model_params))
    print("has MobileGS state:", aux_tail is not None)
    print("model kwargs:", model_kwargs)

    print("\nraw checkpoint:")
    print("  on disk:", human(file_size), f"({file_size} bytes)")
    print("  in RAM (tensor storage only):", human(ram_size), f"({ram_size} bytes)")

    if aux_tail is not None:
        aux_disk = serialized_bytes(aux_tail)
        aux_ram = tensor_bytes(aux_tail)
        print("\naux tail (not compressed by current Gaussian codec):")
        print("  on disk:", human(aux_disk), f"({aux_disk} bytes)")
        print("  in RAM:", human(aux_ram), f"({aux_ram} bytes)")

    print("\nraw gaussian payload:")
    print("  on disk:", human(raw_payload_disk_size), f"({raw_payload_disk_size} bytes)")
    print("  in RAM:", human(raw_payload_ram_size), f"({raw_payload_ram_size} bytes)")

    print("\ncompressed gaussian payload:")
    print("  on disk:", human(compressed_disk_size), f"({compressed_disk_size} bytes)")
    print("  in RAM:", human(compressed_ram_size), f"({compressed_ram_size} bytes)")

    print("\nreduction:")
    print(
        "  vs raw checkpoint on disk:",
        f"{ratio_ckpt_disk:.2f}x smaller ({pct_ckpt_disk:.2f}% reduction)",
    )
    print(
        "  vs raw gaussian payload on disk:",
        f"{ratio_raw_disk:.2f}x smaller ({pct_raw_disk:.2f}% reduction)",
    )
    print(
        "  vs raw gaussian payload in RAM:",
        f"{ratio_raw_ram:.2f}x smaller ({pct_raw_ram:.2f}% reduction)",
    )

    print_validation_report(
        validation,
        rel_rmse_threshold=rel_rmse_threshold,
        max_abs_threshold=max_abs_threshold,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-path",
        default="./4dgs/hpc_outs/output/dnerf/bouncingballs_simple/chkpnt_best.pth",
    )
    parser.add_argument(
        "--compressed-path",
        default="./4dgs/hpc_outs/output/dnerf/bouncingballs_simple/chkpnt_best_compressed.pth",
    )
    parser.add_argument("--attr-bits", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help='Use "auto" to prefer CUDA and fall back to CPU.',
    )
    parser.add_argument("--rel-rmse-threshold", type=float, default=1.5e-2)
    parser.add_argument("--max-abs-threshold", type=float, default=3e-2)
    parser.add_argument(
        "--model-overrides-json",
        default=None,
        help='JSON dict, e.g. \'{"time_duration":[-0.5,0.5],"prefilter_var":-1.0}\'',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_overrides = json.loads(args.model_overrides_json) if args.model_overrides_json else {}

    main(
        ckpt_path=args.ckpt_path,
        compressed_path=args.compressed_path,
        attr_bits=args.attr_bits,
        device=args.device,
        model_overrides=model_overrides,
        rel_rmse_threshold=args.rel_rmse_threshold,
        max_abs_threshold=args.max_abs_threshold,
    )