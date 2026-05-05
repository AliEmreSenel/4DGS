import io
import math
import os
import time
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from scene.cameras import Camera
from scene.gaussian_model import GaussianModel, coerce_time_duration
from utils.gpcc_utils import calculate_morton_order, compress_gpcc, decompress_gpcc
from utils.system_utils import mkdir_p

MOBILE_FORMAT = "mobile-gs-compressed-v1"


def human_bytes(n: int) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def serialized_size(obj) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return int(buf.tell())


def tensor_storage_bytes(obj, seen=None) -> int:
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
        return int(storage.nbytes())
    if isinstance(obj, np.ndarray):
        key = (int(obj.__array_interface__["data"][0]), int(obj.nbytes))
        if key in seen:
            return 0
        seen.add(key)
        return int(obj.nbytes)
    if isinstance(obj, Mapping):
        return sum(tensor_storage_bytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_storage_bytes(v, seen) for v in obj)
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    return 0


def _ensure_cpu_float(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().cpu().contiguous()


def _first_order_features(gaussians: GaussianModel):
    """Return SH tensors reduced to DC + first-order spatial SH.

    The Mobile-GS student/export path intentionally caps spatial SH at degree 1
    (four SH coefficients per RGB channel).  This function slices existing
    tensors and pads if a lower-degree model is exported.
    """
    f_dc = gaussians._features_dc.detach()
    f_rest = gaussians._features_rest.detach()
    need_rest = 3
    if f_rest.shape[1] >= need_rest:
        f_rest_1 = f_rest[:, :need_rest, :]
    else:
        pad = torch.zeros(
            (f_rest.shape[0], need_rest - f_rest.shape[1], f_rest.shape[2]),
            device=f_rest.device,
            dtype=f_rest.dtype,
        )
        f_rest_1 = torch.cat([f_rest, pad], dim=1)
    return f_dc.contiguous(), f_rest_1.contiguous()


def _quantize_u16(x: torch.Tensor):
    x = _ensure_cpu_float(x)
    x_min = x.amin(dim=0)
    x_max = x.amax(dim=0)
    extent = (x_max - x_min).clamp_min(1e-8)
    q = torch.round((x - x_min) * (65535.0 / extent)).to(torch.int32)
    return q, {"min": x_min, "max": x_max}


def _dequantize_u16(q: torch.Tensor, meta: Mapping, device="cuda"):
    q = q.to(device=device, dtype=torch.float32)
    x_min = meta["min"].to(device=device, dtype=torch.float32)
    x_max = meta["max"].to(device=device, dtype=torch.float32)
    extent = (x_max - x_min).clamp_min(1e-8)
    return q * (extent / 65535.0) + x_min


def _uniform_quantize(t: torch.Tensor, bits: int = 8):
    x = _ensure_cpu_float(t)
    shape = tuple(x.shape)
    flat = x.reshape(shape[0], -1)
    x_min = flat.amin(dim=0, keepdim=True)
    x_max = flat.amax(dim=0, keepdim=True)
    qmax = float((1 << int(bits)) - 1)
    step = ((x_max - x_min) / qmax).clamp_min(1e-8)
    q = torch.round((flat - x_min) / step).clamp(0, qmax).to(torch.uint8 if bits <= 8 else torch.int16)
    return {
        "codec": "uniform",
        "shape": list(shape),
        "bits": int(bits),
        "min": x_min.half(),
        "step": step.half(),
        "q": q.contiguous(),
    }


def _uniform_dequantize(pack: Mapping, device="cuda"):
    shape = tuple(pack["shape"])
    q = pack["q"].to(device=device, dtype=torch.float32)
    x_min = pack["min"].to(device=device, dtype=torch.float32)
    step = pack["step"].to(device=device, dtype=torch.float32)
    return (q * step + x_min).reshape(shape)


def _run_kmeans(x: torch.Tensor, k: int, iters: int, seed: int = 0):
    n = int(x.shape[0])
    k = int(max(1, min(k, n)))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    if n == k:
        centers = x.clone()
        labels = torch.arange(n, dtype=torch.long)
        return centers, labels
    perm = torch.randperm(n, generator=gen)[:k]
    centers = x[perm].clone()
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(max(1, int(iters))):
        # CPU cdist is acceptable at export time and keeps this path portable.
        dist = torch.cdist(x, centers)
        labels = torch.argmin(dist, dim=1)
        new_centers = centers.clone()
        for ci in range(k):
            mask = labels == ci
            if bool(mask.any()):
                new_centers[ci] = x[mask].mean(dim=0)
            else:
                ridx = torch.randint(0, n, (1,), generator=gen).item()
                new_centers[ci] = x[ridx]
        if torch.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return centers, labels


def nvq_encode_tensor(
    tensor: torch.Tensor,
    codebook_size: int = 256,
    block_size: int = 8,
    iters: int = 16,
    seed: int = 0,
):
    """Neural/vector-quantize a tensor by sub-vector codebooks.

    This is a deterministic export-time approximation of the Mobile-GS NVQ path:
    attributes are split into fixed-size subvectors, each subvector is assigned to
    a learned codeword, and only codebooks + indices are stored.
    """
    x = _ensure_cpu_float(tensor)
    shape = tuple(x.shape)
    flat = x.reshape(shape[0], -1)
    dim = int(flat.shape[1])
    block_size = max(1, int(block_size))
    num_blocks = int(math.ceil(dim / block_size))
    padded_dim = num_blocks * block_size
    if padded_dim != dim:
        flat = torch.nn.functional.pad(flat, (0, padded_dim - dim))
    blocks = flat.reshape(shape[0], num_blocks, block_size)
    codebooks = []
    indices = []
    for bi in range(num_blocks):
        centers, labels = _run_kmeans(
            blocks[:, bi, :].contiguous(),
            int(codebook_size),
            int(iters),
            seed=int(seed) + bi,
        )
        codebooks.append(centers.half())
        if centers.shape[0] <= 256:
            indices.append(labels.to(torch.uint8))
        elif centers.shape[0] <= 65536:
            indices.append(labels.to(torch.int16))
        else:
            indices.append(labels.to(torch.int32))
    return {
        "codec": "nvq",
        "shape": list(shape),
        "dim": dim,
        "block_size": block_size,
        "codebook_size": int(codebook_size),
        "codebooks": codebooks,
        "indices": torch.stack(indices, dim=1).contiguous(),
    }


def nvq_decode_tensor(pack: Mapping, device="cuda"):
    shape = tuple(pack["shape"])
    dim = int(pack["dim"])
    block_size = int(pack["block_size"])
    num_blocks = int(math.ceil(dim / block_size))
    idx = pack["indices"].long()
    blocks = []
    for bi in range(num_blocks):
        codebook = pack["codebooks"][bi].to(dtype=torch.float32)
        blocks.append(codebook[idx[:, bi]].to(device=device))
    flat = torch.cat(blocks, dim=1)[:, :dim]
    return flat.reshape(shape).contiguous()


def _encode_attr(name: str, tensor: torch.Tensor, *, nvq_names, codebook_size, block_size, iters, seed, uniform_bits):
    if name in nvq_names:
        return nvq_encode_tensor(tensor, codebook_size, block_size, iters, seed)
    return _uniform_quantize(tensor, bits=uniform_bits)


def _decode_attr(pack: Mapping, device="cuda"):
    codec = pack.get("codec")
    if codec == "nvq":
        return nvq_decode_tensor(pack, device=device)
    if codec == "uniform":
        return _uniform_dequantize(pack, device=device)
    raise ValueError(f"Unsupported mobile attr codec: {codec}")


def capture_mobile_payload(
    gaussians: GaussianModel,
    *,
    first_order_sh: bool = True,
    codebook_size: int = 256,
    block_size: int = 8,
    kmeans_iters: int = 16,
    uniform_bits: int = 8,
    include_mlp: bool = True,
    temporal_visibility_filter: Mapping | None = None,
):
    xyz_q, xyz_meta = _quantize_u16(gaussians.get_xyz)
    sort_idx = calculate_morton_order(xyz_q.int()).detach().cpu().long()
    xyz_q_sorted = xyz_q[sort_idx]

    if first_order_sh:
        features_dc, features_rest = _first_order_features(gaussians)
        max_sh_degree = 1
        active_sh_degree = min(1, int(gaussians.active_sh_degree))
        force_sh_3d = True
        max_sh_degree_t = 0
        active_sh_degree_t = 0
    else:
        features_dc = gaussians._features_dc.detach()
        features_rest = gaussians._features_rest.detach()
        max_sh_degree = int(gaussians.max_sh_degree)
        active_sh_degree = int(gaussians.active_sh_degree)
        force_sh_3d = bool(gaussians.force_sh_3d)
        max_sh_degree_t = int(gaussians.max_sh_degree_t)
        active_sh_degree_t = int(gaussians.active_sh_degree_t)

    attrs = {
        "features_dc": features_dc,
        "features_rest": features_rest,
        "scaling": gaussians._scaling.detach(),
        "opacity": gaussians._opacity.detach(),
        "t": gaussians._t.detach(),
        "scaling_t": gaussians._scaling_t.detach(),
    }
    if not gaussians.isotropic_gaussians:
        attrs["rotation"] = gaussians._rotation.detach()
    if gaussians.rot_4d and not gaussians.isotropic_gaussians:
        attrs["rotation_r"] = gaussians._rotation_r.detach()

    nvq_names = {"features_dc", "features_rest", "scaling", "t", "scaling_t"}
    if not gaussians.isotropic_gaussians:
        nvq_names.add("rotation")
    if gaussians.rot_4d and not gaussians.isotropic_gaussians:
        nvq_names.add("rotation_r")

    encoded_attrs = {}
    for name, tensor in attrs.items():
        encoded_attrs[name] = _encode_attr(
            name,
            tensor[sort_idx],
            nvq_names=nvq_names,
            codebook_size=codebook_size,
            block_size=block_size,
            iters=kmeans_iters,
            seed=17,
            uniform_bits=uniform_bits,
        )

    mlp_state = None
    if include_mlp and gaussians.mobilegs_opacity_phi_nn is not None:
        expected_input_dim = 3 * (4 if first_order_sh else gaussians.get_max_sh_channels) + 3 + 3 + 4 + 3
        first_weight = gaussians.mobilegs_opacity_phi_nn.backbone[0].weight
        trained_input_dim = int(first_weight.shape[1])
        if trained_input_dim != int(expected_input_dim):
            raise ValueError(
                "The Mobile-GS opacity/phi MLP input dimension does not match the exported SH layout: "
                f"trained_input_dim={trained_input_dim}, exported_input_dim={expected_input_dim}. "
                "Re-train with --mobilegs_force_first_order_sh for first-order export, "
                "or pass --keep-full-sh to mobile_export.py."
            )
        mlp_state = {
            k: v.detach().cpu() for k, v in gaussians.mobilegs_opacity_phi_nn.state_dict().items()
        }

    payload = {
        "format": MOBILE_FORMAT,
        "meta": {
            "gaussian_dim": int(gaussians.gaussian_dim),
            "rot_4d": bool(gaussians.rot_4d),
            "isotropic_gaussians": bool(gaussians.isotropic_gaussians),
            "force_sh_3d": bool(force_sh_3d),
            "max_sh_degree": int(max_sh_degree),
            "active_sh_degree": int(active_sh_degree),
            "max_sh_degree_t": int(max_sh_degree_t),
            "active_sh_degree_t": int(active_sh_degree_t),
            "time_duration": coerce_time_duration(gaussians.time_duration),
            "prefilter_var": float(gaussians.prefilter_var),
            "spatial_lr_scale": float(gaussians.spatial_lr_scale),
            "num_points": int(gaussians.get_xyz.shape[0]),
            "xyz_quant": xyz_meta,
            "first_order_sh": bool(first_order_sh),
            "codebook_size": int(codebook_size),
            "block_size": int(block_size),
            "uniform_bits": int(uniform_bits),
        },
        "xyz": compress_gpcc(xyz_q_sorted.cpu()),
        "attr": encoded_attrs,
        "mobilegs_opacity_phi_nn": mlp_state,
        "temporal_visibility_filter": temporal_visibility_filter,
    }
    if gaussians.env_map is not None and gaussians.env_map.numel() > 0:
        payload["env_map"] = gaussians.env_map.detach().cpu()
    return payload


def restore_mobile_payload(payload: Mapping, training_args=None, device="cuda") -> GaussianModel:
    if payload.get("format") != MOBILE_FORMAT:
        raise ValueError(f"Expected {MOBILE_FORMAT}, got {payload.get('format')}")
    meta = payload["meta"]
    model_kwargs = {
        "sh_degree": int(meta["max_sh_degree"]),
        "gaussian_dim": int(meta["gaussian_dim"]),
        "time_duration": coerce_time_duration(meta["time_duration"]),
        "rot_4d": bool(meta["rot_4d"]),
        "force_sh_3d": bool(meta["force_sh_3d"]),
        "sh_degree_t": int(meta.get("max_sh_degree_t", 0)),
        "prefilter_var": float(meta.get("prefilter_var", -1.0)),
        "isotropic_gaussians": bool(meta.get("isotropic_gaussians", False)),
    }
    gm = GaussianModel(**model_kwargs)
    gm.active_sh_degree = int(meta.get("active_sh_degree", gm.max_sh_degree))
    gm.active_sh_degree_t = int(meta.get("active_sh_degree_t", 0))
    gm.spatial_lr_scale = float(meta.get("spatial_lr_scale", 1.0))

    # The export path already Morton-sorts both xyz and every encoded attribute
    # before GPCC/NVQ encoding. Do not sort again on restore, otherwise xyz and
    # decoded attributes become desynchronized.
    xyz_q = decompress_gpcc(payload["xyz"]).to(device=device, dtype=torch.int32)
    xyz = _dequantize_u16(xyz_q, meta["xyz_quant"], device=device)

    attrs = {name: _decode_attr(pack, device=device) for name, pack in payload["attr"].items()}
    n = int(xyz.shape[0])
    if n != int(meta["num_points"]):
        raise ValueError(f"Decoded point count mismatch: {n} vs {meta['num_points']}")

    gm._xyz = torch.nn.Parameter(xyz.contiguous().requires_grad_(training_args is not None))
    gm._features_dc = torch.nn.Parameter(attrs["features_dc"].contiguous().requires_grad_(training_args is not None))
    gm._features_rest = torch.nn.Parameter(attrs["features_rest"].contiguous().requires_grad_(training_args is not None))
    gm._scaling = torch.nn.Parameter(attrs["scaling"].contiguous().requires_grad_(training_args is not None))
    gm._opacity = torch.nn.Parameter(attrs["opacity"].contiguous().requires_grad_(training_args is not None))
    gm._t = torch.nn.Parameter(attrs["t"].contiguous().requires_grad_(training_args is not None))
    gm._scaling_t = torch.nn.Parameter(attrs["scaling_t"].contiguous().requires_grad_(training_args is not None))
    if gm.isotropic_gaussians:
        gm._rotation = torch.empty((0, 4), device=device, dtype=xyz.dtype)
        gm._rotation_r = torch.empty((0, 4), device=device, dtype=xyz.dtype)
    else:
        gm._rotation = torch.nn.Parameter(attrs["rotation"].contiguous().requires_grad_(training_args is not None))
        if gm.rot_4d:
            gm._rotation_r = torch.nn.Parameter(attrs["rotation_r"].contiguous().requires_grad_(training_args is not None))

    mlp_state = payload.get("mobilegs_opacity_phi_nn")
    if mlp_state is not None:
        gm.mobilegs_opacity_phi_nn = gm._build_mobilegs_opacity_phi_nn()
        gm.mobilegs_opacity_phi_nn.load_state_dict({k: v.to(device) for k, v in mlp_state.items()})
        gm.mobilegs_opacity_phi_nn.eval()

    env_map = payload.get("env_map")
    gm.env_map = torch.empty(0, device=device) if env_map is None else env_map.to(device)
    gm.max_radii2D = torch.zeros((n,), device=device)
    gm.xyz_gradient_accum = torch.zeros((n, 1), device=device)
    gm.t_gradient_accum = torch.zeros((n, 1), device=device)
    gm.denom = torch.zeros((n, 1), device=device)
    gm.optimizer = None
    if training_args is not None:
        gm.training_setup(training_args)
    attach_temporal_visibility_filter(gm, payload.get("temporal_visibility_filter"), device=device)
    return gm


def camera_from_metadata(meta: Mapping, device="cuda", meta_only=True) -> Camera:
    return Camera(
        colmap_id=int(meta.get("colmap_id", meta.get("uid", 0))),
        R=np.asarray(meta["R"], dtype=np.float32),
        T=np.asarray(meta["T"], dtype=np.float32),
        FoVx=float(meta["FoVx"]),
        FoVy=float(meta["FoVy"]),
        image=None,
        gt_alpha_mask=None,
        image_name=str(meta.get("image_name", f"cam_{meta.get('uid', 0)}")),
        uid=int(meta.get("uid", 0)),
        trans=np.asarray(meta.get("trans", [0.0, 0.0, 0.0]), dtype=np.float32),
        scale=float(meta.get("scale", 1.0)),
        data_device=device,
        timestamp=float(meta.get("timestamp", 0.0)),
        cx=float(meta.get("cx", -1.0)),
        cy=float(meta.get("cy", -1.0)),
        fl_x=float(meta.get("fl_x", -1.0)),
        fl_y=float(meta.get("fl_y", -1.0)),
        resolution=list(meta["resolution"]),
        meta_only=meta_only,
    ).to(device, copy=False)


def cameras_from_checkpoint_scene(scene_meta: Mapping, split="train", device="cuda"):
    key = "train_cameras" if split == "train" else "test_cameras"
    return [camera_from_metadata(m, device=device, meta_only=True) for m in scene_meta.get(key, [])]


def _copy_camera_with_timestamp(cam: Camera, timestamp: float) -> Camera:
    new_cam = cam.to(copy=True)
    new_cam.timestamp = float(timestamp)
    return new_cam


def build_temporal_visibility_filter(
    gaussians: GaussianModel,
    cameras: Sequence[Camera],
    pipe,
    background: torch.Tensor,
    render_fn,
    *,
    keyframes: int = 32,
    views_per_keyframe: int = 0,
):
    """Build the 4DGS-1K-style keyframe visibility masks.

    For each keyframe timestamp, render training views at that timestamp and
    union the visible Gaussians.  These masks are exported and reused at
    inference, instead of thresholding the temporal marginal only.
    """
    if len(cameras) == 0 or gaussians.get_xyz.shape[0] == 0:
        return None
    keyframes = max(2, int(keyframes))
    t0, t1 = float(gaussians.time_duration[0]), float(gaussians.time_duration[1])
    times = torch.linspace(t0, t1, keyframes, device=gaussians.get_xyz.device)
    n = int(gaussians.get_xyz.shape[0])
    masks = []

    old_keyframes = getattr(pipe, "temporal_mask_keyframes", 0)
    old_mode = getattr(pipe, "temporal_mask_mode", "marginal")
    try:
        setattr(pipe, "temporal_mask_keyframes", 0)
        setattr(pipe, "temporal_mask_mode", "marginal")
        with torch.no_grad():
            for kt in times.tolist():
                # Prefer views naturally captured near this timestamp.  If a
                # dataset has only one camera per time, this behaves like the
                # original per-frame visibility list.  For sparse timestamps,
                # fall back to all views with timestamp overwritten to kt.
                cams_sorted = sorted(cameras, key=lambda c: abs(float(c.timestamp) - float(kt)))
                if views_per_keyframe and views_per_keyframe > 0:
                    cams_use = cams_sorted[: int(views_per_keyframe)]
                else:
                    nearest = abs(float(cams_sorted[0].timestamp) - float(kt))
                    cams_use = [c for c in cams_sorted if abs(abs(float(c.timestamp) - float(kt)) - nearest) < 1e-7]
                    if len(cams_use) == 0:
                        cams_use = cams_sorted
                mask = torch.zeros(n, dtype=torch.bool, device=gaussians.get_xyz.device)
                for cam in cams_use:
                    cam_kt = _copy_camera_with_timestamp(cam, kt)
                    pkg = render_fn(cam_kt, gaussians, pipe, background)
                    radii = pkg.get("radii")
                    if radii is not None and radii.numel() == n:
                        mask |= radii.reshape(-1) > 0
                if not bool(mask.any()):
                    # Conservative fallback avoids all-empty exported masks.
                    marginal = gaussians.get_marginal_t(float(kt))[:, 0]
                    mask = marginal > max(0.0, float(getattr(pipe, "temporal_mask_threshold", 0.05)))
                masks.append(mask.detach().cpu())
    finally:
        setattr(pipe, "temporal_mask_keyframes", old_keyframes)
        setattr(pipe, "temporal_mask_mode", old_mode)

    masks = torch.stack(masks, dim=0).contiguous()
    packed = np.packbits(masks.numpy().astype(np.uint8), axis=1)
    return {
        "format": "temporal-visibility-filter-v1",
        "times": times.detach().cpu(),
        "num_points": int(n),
        "packed_masks": packed,
        "shape": list(masks.shape),
    }


def attach_temporal_visibility_filter(gaussians: GaussianModel, filt, device="cuda"):
    if filt is None:
        return
    if filt.get("format") != "temporal-visibility-filter-v1":
        raise ValueError(f"Unsupported temporal filter format: {filt.get('format')}")
    shape = tuple(filt["shape"])
    packed = np.asarray(filt["packed_masks"])
    masks_np = np.unpackbits(packed, axis=1)[:, : shape[1]].astype(bool)
    masks = torch.from_numpy(masks_np).to(device=device, dtype=torch.bool)
    cache = {
        "mode": "visibility",
        "times": filt["times"].to(device=device, dtype=torch.float32),
        "masks": masks,
        "num_points": int(filt["num_points"]),
    }
    setattr(gaussians, "_temporal_visibility_mask_cache", cache)


def save_mobile_payload(payload: Mapping, path: str):
    parent = os.path.dirname(path)
    if parent:
        mkdir_p(parent)
    torch.save(dict(payload), path)


def load_mobile_payload(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def benchmark_renderer(gaussians, cameras, pipe, background, render_fn, warmup=10, repeats=100):
    if len(cameras) == 0:
        raise ValueError("No cameras available for benchmarking")
    warmup = max(0, int(warmup))
    repeats = max(1, int(repeats))
    for i in range(warmup):
        render_fn(cameras[i % len(cameras)], gaussians, pipe, background)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(repeats):
        render_fn(cameras[i % len(cameras)], gaussians, pipe, background)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {"frames": repeats, "seconds": elapsed, "fps": repeats / max(elapsed, 1e-9)}
