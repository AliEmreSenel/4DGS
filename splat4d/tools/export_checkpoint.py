#!/usr/bin/env python3
"""Export raw 3D/4D Gaussian checkpoints into `.splat4dpack` v3.

The exporter is intentionally native/Python-side because browser code must not
unpickle arbitrary PyTorch checkpoints. It preserves the checkpoint rendering
contract and writes all render-affecting data, including sort-free opacity/phi
MLP, temporal masks, compression payloads, env maps, camera metadata and unknown
extras.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, pickle, struct, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

MAGIC = b"S4DPK3\0\0"
VERSION = 3

CAPTURE_4D_NAMES = [
    "active_sh_degree", "xyz", "features_dc", "features_rest", "scaling", "rotation", "opacity",
    "max_radii2D", "xyz_gradient_accum", "t_gradient_accum", "denom", "optimizer",
    "spatial_lr_scale", "t", "scaling_t", "rotation_r", "rot_4d", "env_map", "active_sh_degree_t", "mobilegs_state"
]

@dataclass
class MlpExport:
    meta: Dict[str, Any]
    weights: np.ndarray

@dataclass
class ExportScene:
    mean4: np.ndarray
    scale4: np.ndarray
    ql: np.ndarray
    qr: np.ndarray
    opacity: np.ndarray
    appearance: np.ndarray
    flags: np.ndarray
    appearance_model: Dict[str, Any]
    render_policy: Dict[str, Any]
    schema: str
    source_keys: List[str]
    masks: Optional[Dict[str, Any]] = None
    mlp: Optional[MlpExport] = None
    env_map: Optional[Any] = None
    compression_meta: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


def torch_load(path: Path):
    try:
        import torch
    except Exception as e:
        raise SystemExit("PyTorch is required to export checkpoints. Install torch in the trusted export environment.") from e
    return torch.load(path, map_location="cpu", weights_only=False)


def to_numpy(x: Any, dtype=np.float32) -> Optional[np.ndarray]:
    if x is None: return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if dtype == np.uint32:
                return x.detach().cpu().numpy().astype(np.uint32, copy=False)
            return x.detach().cpu().float().numpy().astype(dtype, copy=False)
    except Exception:
        pass
    if isinstance(x, np.ndarray): return x.astype(dtype, copy=False)
    if isinstance(x, (list, tuple)): return np.asarray(x, dtype=dtype)
    if isinstance(x, (int, float)): return np.asarray([x], dtype=dtype)
    return None


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def identity_quats(n: int) -> np.ndarray:
    q = np.zeros((int(n), 4), dtype=np.float32)
    q[:, 0] = 1.0
    return q


def valid_quat_tensor(q: Any, n: int) -> Optional[np.ndarray]:
    """Return a normalized [N,4] quaternion tensor or None.

    Isotropic checkpoints in this repo intentionally store empty rotation tensors;
    GaussianModel.get_rotation()/get_rotation_r() synthesizes identity quaternions
    at render time.  Treating those empty tensors as real data produced all-zero
    quaternions in earlier web exports, which collapses the covariance and makes
    the render look completely unlike CUDA.
    """
    arr = to_numpy(q)
    if arr is None or arr.size == 0:
        return None
    arr = arr.reshape((arr.shape[0], -1)).astype(np.float32)
    if arr.shape[1] < 4:
        return None
    if arr.shape[0] != n:
        # np.resize repeats data, which is dangerous for rotations.  Use identity
        # unless the tensor is a singleton quaternion.
        if arr.shape[0] == 1:
            arr = np.repeat(arr[:, :4], n, axis=0)
        else:
            return None
    arr = arr[:, :4]
    nrm = np.linalg.norm(arr, axis=1, keepdims=True)
    bad = (~np.isfinite(nrm)) | (nrm < 1e-20)
    nrm[bad] = 1.0
    arr = arr / nrm
    if bad.any():
        arr[bad.reshape(-1)] = np.array([1, 0, 0, 0], dtype=np.float32)
    return arr.astype(np.float32)


def norm_quat(q):
    q = q.reshape((q.shape[0], -1))[:, :4].astype(np.float32)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n[n < 1e-20] = 1.0
    return q / n

def find_tensor(d: Dict[str, Any], names: Iterable[str], dtype=np.float32) -> Optional[np.ndarray]:
    normalized = {k.lower().replace('.', '_'): k for k in d.keys()}
    for name in names:
        for key in [name, name.lower(), name.replace('.', '_'), '_' + name]:
            if key in d:
                arr = to_numpy(d[key], dtype=dtype)
                if arr is not None: return arr
        nk = name.lower().replace('.', '_')
        if nk in normalized:
            arr = to_numpy(d[normalized[nk]], dtype=dtype)
            if arr is not None: return arr
    for k, v in d.items():
        lk = k.lower().replace('.', '_')
        for name in names:
            if lk.endswith(name.lower().replace('.', '_')):
                arr = to_numpy(v, dtype=dtype)
                if arr is not None: return arr
    return None


def unpack_capture_tuple(values: Tuple[Any, ...], *, root_meta: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
    """Unpack this repo's GaussianModel.capture() tuple.

    Current checkpoints from 4D-Gaussian-Splattering-Research-main save a
    self-contained dict with key ``gaussians`` containing this 20-field tuple.
    Older local experiments may have saved the tuple directly or as
    ``(gaussians_tuple, iteration)``.
    """
    if len(values) == 20:
        out = dict(zip(CAPTURE_4D_NAMES, values))
        schema = "4dgs-self-contained-capture-v20"
    elif len(values) == 19:
        # Same layout as v20 but without the final mobilegs_state.
        out = dict(zip(CAPTURE_4D_NAMES[:-1], values))
        out["mobilegs_state"] = None
        schema = "4dgs-capture-v19-no-mobilegs"
    elif len(values) == 18:
        # Older 4DGS snapshots before t_gradient_accum/mobilegs_state.
        names = [
            "active_sh_degree", "xyz", "features_dc", "features_rest", "scaling", "rotation", "opacity",
            "max_radii2D", "xyz_gradient_accum", "denom", "optimizer", "spatial_lr_scale",
            "t", "scaling_t", "rotation_r", "rot_4d", "env_map", "active_sh_degree_t",
        ]
        out = dict(zip(names, values))
        out["t_gradient_accum"] = None
        out["mobilegs_state"] = None
        schema = "4dgs-capture-v18-legacy"
    else:
        raise SystemExit(f"Unsupported 4DGS capture tuple length: {len(values)}")
    out["__capture_tensor_space"] = "4dgs-preactivation"
    if root_meta:
        out.update(root_meta)
    return out, schema


def _root_meta(obj: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for k in ["format", "iteration", "run_config", "scene", "requires_mobilegs"]:
        if k in obj:
            meta[f"root_{k}"] = obj[k]
    return meta


def flatten_state(obj: Any) -> Tuple[Dict[str, Any], str]:
    # Native format from this repo: utils.checkpoint_utils.CHECKPOINT_FORMAT ==
    # "4dgs-self-contained-v1".  The Gaussian tensors are not top-level; they are
    # stored under checkpoint["gaussians"] as GaussianModel.capture().
    if isinstance(obj, dict) and obj.get("format") == "4dgs-self-contained-v1" and "gaussians" in obj:
        g = obj["gaussians"]
        if isinstance(g, (tuple, list)):
            raw, schema = unpack_capture_tuple(tuple(g), root_meta=_root_meta(obj))
            return raw, schema
        if isinstance(g, dict):
            out = dict(g)
            out.update(_root_meta(obj))
            out["__capture_tensor_space"] = "4dgs-preactivation"
            return out, "4dgs-self-contained-dict-gaussians"
        raise SystemExit(f"Unsupported checkpoint['gaussians'] type: {type(g)}")

    # Legacy raw tuple, or (capture_tuple, iteration).
    if isinstance(obj, tuple):
        if len(obj) in (18, 19, 20):
            return unpack_capture_tuple(obj)
        if len(obj) == 2 and isinstance(obj[0], (tuple, list)):
            raw, schema = unpack_capture_tuple(tuple(obj[0]), root_meta={"root_iteration": obj[1]})
            return raw, f"legacy-pair:{schema}"

    # Compressed payloads produced by this repo.
    if isinstance(obj, dict):
        if obj.get("format") in {"gaussian-compressed-v1", "mobile-gs-compressed-v1", "universal-gaussian-compressed-v1", "gaussian-raw-v1"}:
            return dict(obj), str(obj.get("format"))
        for key in ["state_dict", "model", "point_cloud", "params", "splats"]:
            if key in obj and isinstance(obj[key], dict):
                out = dict(obj[key])
                for k, v in obj.items():
                    if k not in out: out[f"root_{k}"] = v
                return out, f"nested:{key}"
        if "gaussians" in obj and isinstance(obj["gaussians"], (tuple, list)):
            raw, schema = unpack_capture_tuple(tuple(obj["gaussians"]), root_meta=_root_meta(obj))
            return raw, f"dict-gaussians:{schema}"
        return dict(obj), "dict"
    if hasattr(obj, "state_dict"):
        return dict(obj.state_dict()), "module-state-dict"
    raise SystemExit(f"Unsupported checkpoint type: {type(obj)}")

def decode_huffman_pack(pack: Dict[str, Any]) -> np.ndarray:
    try:
        import dahuffman
    except Exception as e:
        raise SystemExit("Compressed checkpoint uses Huffman. Install dahuffman in the exporter environment.") from e
    codec = dahuffman.HuffmanCodec(code_table=pack["htable"])
    flat = np.asarray(codec.decode(pack["index"]), dtype=np.int32)
    shape = tuple(pack["shape"])
    flat = flat.reshape(shape[0], -1)
    xmin = to_numpy(pack["min"])
    step = to_numpy(pack["step"])
    x = flat.astype(np.float32) * step + xmin
    return x.reshape(shape).astype(np.float32)


def maybe_decode_compressed(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if raw.get("format") != "gaussian-compressed-v1": return None
    meta = raw.get("meta", {})
    decoded: Dict[str, Any] = {}
    # GPCC position decode requires tmc3. If unavailable, fail explicitly rather than silently dropping support.
    if "xyz" in raw:
        gpcc = Path("./mpeg-pcc-tmc13/build/tmc3/tmc3")
        if not gpcc.exists():
            raise SystemExit("Compressed checkpoint contains GPCC xyz. Install/build MPEG TMC13 or re-export from a restored native model.")
        # Use the source repo helper if available.
        sys.path.insert(0, str(Path.cwd()))
        try:
            from utils.gpcc_utils import decompress_gpcc, calculate_morton_order
            import torch
            xyz_q = decompress_gpcc(raw["xyz"]).to(dtype=torch.int32)
            sort_idx = calculate_morton_order(xyz_q.int())
            xyz_q = xyz_q[sort_idx].cpu().numpy().astype(np.float32)
            mn = to_numpy(meta["xyz_quant"]["min"])
            mx = to_numpy(meta["xyz_quant"]["max"])
            decoded["xyz"] = xyz_q * ((mx - mn) / 65535.0) + mn
        except Exception as e:
            raise SystemExit(f"Failed to decode GPCC compressed xyz: {e}") from e
    for name, p in raw.get("attr", {}).items():
        decoded[name] = decode_huffman_pack(p)
    decoded["_compressed_original"] = raw
    decoded["_compressed_meta"] = meta
    return decoded


def extract_mlp(raw: Dict[str, Any], n: int, app_stride: int) -> Optional[MlpExport]:
    # Supports the MobileOpacityPhiNN state dict created by capture(include_mobilegs=True).
    state = None
    mobile = raw.get("mobilegs_state") or raw.get("root_mobilegs_state")
    if isinstance(mobile, dict) and isinstance(mobile.get("model"), dict): state = mobile["model"]
    if state is None:
        # fallback: find actual layer tensors, not marker booleans/None such as mobilegs_state=None.
        candidates = {}
        for k, v in raw.items():
            lk = k.lower()
            if ("opacity_phi" in lk or "phi_head" in lk or "opacity_head" in lk or "backbone" in lk):
                if to_numpy(v) is not None:
                    candidates[k] = v
        if candidates: state = candidates
    if not state: return None

    layers = []
    weights = []
    def add_layer(name: str, wkey: str, bkey: str, activation: str):
        w = to_numpy(state.get(wkey))
        b = to_numpy(state.get(bkey))
        if w is None or b is None:
            # tolerate prefixes
            for k, v in state.items():
                if k.endswith(wkey): w = to_numpy(v)
                if k.endswith(bkey): b = to_numpy(v)
        if w is None or b is None: return False
        w = w.astype(np.float32); b = b.astype(np.float32).reshape(-1)
        wo = sum(x.size for x in weights); weights.append(w.reshape(-1))
        bo = sum(x.size for x in weights); weights.append(b)
        layers.append({"name": name, "in_dim": int(w.shape[1]), "out_dim": int(w.shape[0]), "weight_offset": int(wo), "weight_len": int(w.size), "bias_offset": int(bo), "bias_len": int(b.size), "activation": activation})
        return True

    # Sequential: backbone.0, backbone.2, backbone.4, then heads.
    ok = True
    ok &= add_layer("backbone.0", "backbone.0.weight", "backbone.0.bias", "relu")
    ok &= add_layer("backbone.2", "backbone.2.weight", "backbone.2.bias", "relu")
    ok &= add_layer("backbone.4", "backbone.4.weight", "backbone.4.bias", "relu")
    ok &= add_layer("phi_head", "phi_head.weight", "phi_head.bias", "relu")
    ok &= add_layer("opacity_head", "opacity_head.weight", "opacity_head.bias", "sigmoid")
    if not ok: raise SystemExit("Found sort-free MLP state but could not resolve all MobileOpacityPhiNN layers.")
    meta = {
        "purpose": "mobilegs-opacity-phi",
        "input_layout": ["normalized_sh", "viewdir3", "log_scale3", "rotation4", "time_features3"],
        "output_layout": ["phi_relu", "opacity_sigmoid"],
        "dtype": "f32",
        "layers": layers,
        "custom_formula": "weight = phi^2 + phi/depth^2 + exp(max_scale/depth)",
    }
    return MlpExport(meta=meta, weights=np.concatenate(weights).astype(np.float32))




def _find_repo_root_for_native_export(checkpoint_path: Path) -> Optional[Path]:
    """Find this 4DGS repo so the exporter can use GaussianModel getters.

    Manual tuple parsing is not reliable enough for CUDA parity: the repo has
    render-time activations and special cases (isotropic rotations, active SH
    promotion, force_sh_3d, temporal degree handling).  For self-contained
    checkpoints we should restore the real GaussianModel and read the exact
    tensors that gaussian_renderer.render passes to the CUDA extension.
    """
    candidates = []
    for base in [Path.cwd(), Path.cwd().parent, checkpoint_path.resolve().parent, checkpoint_path.resolve().parent.parent]:
        cur = base.resolve()
        for _ in range(5):
            candidates.append(cur)
            if cur.parent == cur:
                break
            cur = cur.parent
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if (c / 'scene' / 'gaussian_model.py').exists() and (c / 'gaussian_renderer' / '__init__.py').exists():
            return c
    return None


def _tensor_np(x: Any) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().contiguous().numpy().astype(np.float32, copy=False)
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)


def build_scene_via_repo_model(checkpoint: Dict[str, Any], checkpoint_path: Path, schema: str, source_keys: List[str]) -> Optional[ExportScene]:
    """Authoritative export path for 4dgs-self-contained-v1 checkpoints.

    This is intentionally preferred over build_scene(raw).  It imports the repo's
    GaussianModel, calls restore(), and reads get_xyz/get_scaling/get_rotation/
    get_opacity/get_features exactly as the Python interactive viewer does.
    """
    if not isinstance(checkpoint, dict) or checkpoint.get('format') != '4dgs-self-contained-v1' or 'gaussians' not in checkpoint:
        return None
    repo_root = _find_repo_root_for_native_export(checkpoint_path)
    if repo_root is None:
        print('warning: could not find 4DGS repo root; falling back to manual tuple parser', file=sys.stderr)
        return None
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from scene.gaussian_model import GaussianModel
    except Exception as e:
        print(f'warning: failed to import repo GaussianModel ({e}); falling back to manual parser', file=sys.stderr)
        return None

    run_config = checkpoint.get('run_config', {}) if isinstance(checkpoint.get('run_config', {}), dict) else {}
    gkwargs = dict(run_config.get('gaussian_kwargs', {}) if isinstance(run_config.get('gaussian_kwargs', {}), dict) else {})
    if int(gkwargs.get('gaussian_dim', 4) or 4) != 4:
        raise SystemExit('Only 4D Gaussian checkpoints are supported.')
    gm = GaussianModel(**gkwargs)
    gm.restore(checkpoint['gaussians'], None)
    # Match interactive_viewer.py: it promotes to the maximum SH degree for final rendering.
    gm.active_sh_degree = gm.max_sh_degree
    if hasattr(gm, 'active_sh_degree_t'):
        gm.active_sh_degree_t = max(int(getattr(gm, 'active_sh_degree_t', 0)), int(gkwargs.get('sh_degree_t', 0) or 0))

    xyz = _tensor_np(gm.get_xyz).reshape(-1, 3)
    n = int(xyz.shape[0])
    t = _tensor_np(gm.get_t).reshape(n, 1)
    mean4 = np.concatenate([xyz, t], axis=1).astype(np.float32)
    scaling = _tensor_np(gm.get_scaling).reshape(n, -1)
    if scaling.shape[1] == 1:
        scaling = np.repeat(scaling, 3, axis=1)
    scaling = scaling[:, :3]
    st = _tensor_np(gm.get_scaling_t).reshape(n, 1)
    scale4 = np.concatenate([scaling, st], axis=1).astype(np.float32)
    ql = _tensor_np(gm.get_rotation).reshape(n, 4)
    qr = _tensor_np(gm.get_rotation_r).reshape(n, 4)
    opacity = _tensor_np(gm.get_opacity).reshape(n).astype(np.float32)
    appearance = _tensor_np(gm.get_features).reshape(n, -1).astype(np.float32)

    coeff_count = max(1, appearance.shape[1] // 3)
    flags = np.zeros((n,), np.uint32)
    if bool(getattr(gm, 'isotropic_gaussians', False)):
        flags |= 2
    if bool(getattr(gm, 'rot_4d', False)):
        flags |= 4

    active_degree = int(getattr(gm, 'active_sh_degree', getattr(gm, 'max_sh_degree', 0)))
    active_degree_t = int(getattr(gm, 'active_sh_degree_t', getattr(gm, 'max_sh_degree_t', 0)))
    max_degree = int(getattr(gm, 'max_sh_degree', active_degree))
    max_degree_t = int(getattr(gm, 'max_sh_degree_t', active_degree_t))
    td = coerce_time_duration_export(getattr(gm, 'time_duration', gkwargs.get('time_duration', None)))
    time_span = float(td[1] - td[0]) if abs(float(td[1] - td[0])) > 1e-12 else 1.0
    force_sh_3d = bool(getattr(gm, 'force_sh_3d', False))
    prefilter_var = float(getattr(gm, 'prefilter_var', -1.0))
    render_args = {
        'gaussian_dim': 4,
        'rot_4d': bool(getattr(gm, 'rot_4d', False)),
        'force_sh_3d': force_sh_3d,
        'prefilter_var': prefilter_var,
        'scale_modifier': 1.0,
        'time_duration': [float(td[0]), float(td[1])],
        'time_span': time_span,
        'active_sh_degree': active_degree,
        'active_sh_degree_t': active_degree_t,
        'max_sh_degree': max_degree,
        'max_sh_degree_t': max_degree_t,
        'coeff_count': int(coeff_count),
        'export_backend': 'repo-gaussian-model-getters',
    }
    app_model = {
        'kind': 'sh' if coeff_count > 1 else 'rgb',
        'degree': max_degree,
        'degree_t': max_degree_t,
        'active_degree': active_degree,
        'active_degree_t': active_degree_t,
        'coeff_count': int(coeff_count),
        'storage': 'f32',
        'source_layout': 'GaussianModel.get_features_contiguous_flat',
    }
    raw_for_meta, _ = flatten_state(checkpoint)
    # Ensure camera metadata and run_config remain available under the names used by active_scene_cameras().
    raw_for_meta['scene'] = checkpoint.get('scene', {})
    raw_for_meta['root_scene'] = checkpoint.get('scene', {})
    raw_for_meta['root_run_config'] = run_config
    raw_for_meta['root_format'] = checkpoint.get('format')
    raw_for_meta['root_iteration'] = checkpoint.get('iteration')
    masks = extract_temporal_filter(raw_for_meta, n)
    mlp = extract_mlp(raw_for_meta, n, appearance.shape[1])
    native = render_type_from_checkpoint(raw_for_meta, mlp is not None)
    env = None
    try:
        env_tensor = getattr(gm, 'env_map', None)
        if env_tensor is not None and getattr(env_tensor, 'numel', lambda: 0)() > 0:
            env = _tensor_np(env_tensor)
    except Exception:
        env = None
    extras = {
        'format': checkpoint.get('format'),
        'iteration': checkpoint.get('iteration'),
        'run_config': run_config,
        'scene': checkpoint.get('scene', {}),
        'requires_mobilegs': checkpoint.get('requires_mobilegs', False),
        'export_backend': 'repo-gaussian-model-getters',
    }
    compressed_meta = {'source_was_compressed': False, 'decoded_for_render': False, 'schemes': [], 'has_codebooks': False, 'has_huffman': False, 'has_rvq': False}
    return ExportScene(mean4, scale4, ql, qr, opacity, appearance, flags, app_model, render_policy(native, mlp is not None), schema + ':repo-model-getters', source_keys, masks, mlp, env, compressed_meta, extras)

def render_policy(native: str, has_mlp: bool) -> Dict[str, Any]:
    # Render contracts are not quality presets. A checkpoint trained/exported for sorted alpha
    # must be rendered sorted; a checkpoint trained/exported for MobileGS sort-free must be
    # rendered through the sort-free opacity/phi MLP path. Cross-family overrides are forbidden.
    if native == "sort-free-mobilegs" and not has_mlp:
        raise SystemExit("sort-free-mobilegs checkpoint requires MobileGS opacity/phi MLP weights")
    if native == "sorted-alpha":
        forbidden = ["sort-free-mobilegs", "sort-free-weighted-oit", "webgl-preview"]
    elif native == "sort-free-mobilegs":
        forbidden = ["sorted-alpha", "sort-free-weighted-oit", "webgl-preview"]
    elif native == "sort-free-weighted-oit":
        forbidden = ["sorted-alpha", "sort-free-mobilegs", "webgl-preview"]
    else:
        raise SystemExit(f"unsupported native render type: {native}")
    return {
        "native_render_type": native,
        "default_render_type": native,
        "reference_render_type": native,
        "required_render_type": native,
        "allowed_render_types": [native],
        "forbidden_render_types": forbidden,
        "render_type_locked": True,
        "allow_url_override": False,
        "quality_presets": {}
    }




def nested_get(d: Any, path: List[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def coerce_time_duration_export(x: Any) -> Tuple[float, float]:
    if x is None:
        return (0.0, 1.0)
    try:
        arr = to_numpy(x)
        if arr is not None and arr.size >= 2:
            flat = arr.reshape(-1).astype(float)
            return (float(flat[0]), float(flat[1]))
    except Exception:
        pass
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return (float(x[0]), float(x[1]))
    return (0.0, 1.0)

def infer_sh_degrees(coeff_count: int, gkwargs: Dict[str, Any], raw: Dict[str, Any]) -> Tuple[int, int]:
    # Match GaussianModel.get_max_sh_channels.  Prefer run_config, then infer.
    max_sh_degree = int(gkwargs.get("sh_degree", gkwargs.get("max_sh_degree", 0)) or 0)
    max_sh_degree_t = int(gkwargs.get("sh_degree_t", gkwargs.get("max_sh_degree_t", 0)) or 0)
    force_sh_3d = truthy(gkwargs.get("force_sh_3d", raw.get("force_sh_3d", False)))
    if max_sh_degree <= 0:
        # Supported repo models are normally SH3/SH0; infer the largest spatial degree whose base divides coeff_count.
        for d in (3, 2, 1, 0):
            base = (d + 1) ** 2
            if coeff_count >= base and coeff_count % base == 0:
                max_sh_degree = d
                break
    if max_sh_degree_t <= 0 and not force_sh_3d:
        base = (max_sh_degree + 1) ** 2
        if base > 0 and coeff_count % base == 0:
            max_sh_degree_t = max(0, coeff_count // base - 1)
    return max_sh_degree, max_sh_degree_t

def truthy(x: Any) -> bool:
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "on"}
    return bool(x)


def is_capture_preactivation(raw: Dict[str, Any]) -> bool:
    return raw.get("__capture_tensor_space") == "4dgs-preactivation"


def _identity4() -> List[List[float]]:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _projection_from_camera(cam: Dict[str, Any]) -> List[List[float]]:
    # Mirrors utils.graphics_utils.getProjectionMatrix* and Camera.projection_matrix.transpose(0, 1)
    znear, zfar = 0.01, 100.0
    width, height = _camera_resolution(cam)
    cx = float(cam.get("cx", -1) if cam.get("cx", -1) is not None else -1)
    cy = float(cam.get("cy", -1) if cam.get("cy", -1) is not None else -1)
    fl_x = float(cam.get("fl_x", -1) if cam.get("fl_x", -1) is not None else -1)
    fl_y = float(cam.get("fl_y", -1) if cam.get("fl_y", -1) is not None else -1)
    if cx > 0 and cy > 0 and fl_x > 0 and fl_y > 0:
        top = cy / fl_y * znear
        bottom = -float(height - cy) / fl_y * znear
        left = -float(width - cx) / fl_x * znear
        right = cx / fl_x * znear
    else:
        fovx = float(cam.get("FoVx", cam.get("fovx", 1.0)))
        fovy = float(cam.get("FoVy", cam.get("fovy", 1.0)))
        import math
        top = math.tan(fovy * 0.5) * znear
        bottom = -top
        right = math.tan(fovx * 0.5) * znear
        left = -right
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] = 2.0 * znear / (right - left)
    p[1, 1] = 2.0 * znear / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = 1.0
    p[2, 2] = zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p.T.astype(np.float32).tolist()


def _view_from_camera(cam: Dict[str, Any]) -> Tuple[List[List[float]], List[float]]:
    # Mirrors utils.graphics_utils.getWorld2View2 and Camera.world_view_transform.transpose(0, 1).
    R = np.asarray(cam.get("R", np.eye(3)), dtype=np.float32).reshape(3, 3)
    T = np.asarray(cam.get("T", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
    trans = np.asarray(cam.get("trans", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
    scale = float(cam.get("scale", 1.0))
    rt = np.zeros((4, 4), dtype=np.float32)
    rt[:3, :3] = R.T
    rt[:3, 3] = T
    rt[3, 3] = 1.0
    try:
        c2w = np.linalg.inv(rt)
        center = c2w[:3, 3]
        center = (center + trans) * scale
        c2w[:3, 3] = center
        rt = np.linalg.inv(c2w).astype(np.float32)
    except Exception:
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    view = rt.T.astype(np.float32)
    try:
        center = np.linalg.inv(view)[3, :3].astype(np.float32)
    except Exception:
        pass
    return view.tolist(), [float(center[0]), float(center[1]), float(center[2])]


def _camera_resolution(cam: Dict[str, Any]) -> Tuple[int, int]:
    res = cam.get("resolution") or cam.get("image_size")
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        return int(res[0]), int(res[1])
    return int(cam.get("image_width", cam.get("width", 1))), int(cam.get("image_height", cam.get("height", 1)))


def _camera_frame_from_repo_metadata(cam: Any, split: str, index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(cam, dict):
        return None
    width, height = _camera_resolution(cam)
    view, eye = _view_from_camera(cam)
    projection = np.asarray(_projection_from_camera(cam), dtype=np.float32)
    view_np = np.asarray(view, dtype=np.float32)
    full_proj = (view_np @ projection).astype(np.float32)
    try:
        inv_view = np.linalg.inv(view_np).astype(np.float32).tolist()
    except Exception:
        inv_view = np.eye(4, dtype=np.float32).tolist()
    return {
        "name": str(cam.get("image_name", cam.get("name", f"{split}_{index:04d}"))),
        "view": view,
        "proj": full_proj.tolist(),
        "projection": projection.tolist(),
        "inv_view": inv_view,
        "camera_position": eye,
        "timestamp": float(cam.get("timestamp", 0.0) or 0.0),
        "width": width,
        "height": height,
        "split": split,
        "fovx": float(cam.get("FoVx", cam.get("fovx", 0.0)) or 0.0),
        "fovy": float(cam.get("FoVy", cam.get("fovy", 0.0)) or 0.0),
        "cx": float(cam.get("cx", -1.0) if cam.get("cx", -1.0) is not None else -1.0),
        "cy": float(cam.get("cy", -1.0) if cam.get("cy", -1.0) is not None else -1.0),
        "fl_x": float(cam.get("fl_x", -1.0) if cam.get("fl_x", -1.0) is not None else -1.0),
        "fl_y": float(cam.get("fl_y", -1.0) if cam.get("fl_y", -1.0) is not None else -1.0),
        "znear": float(cam.get("znear", 0.01) or 0.01),
        "zfar": float(cam.get("zfar", 100.0) or 100.0),
    }


def active_scene_cameras(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return canonical CameraFrame records for META.cameras.

    The repo self-contained checkpoint stores camera metadata at checkpoint["scene"].
    During export, this data may be present as raw["root_scene"], raw["scene"],
    or raw["source_scene"], depending on normalization stage and older patches.
    """
    scene = raw.get("root_scene") or raw.get("scene") or raw.get("source_scene")
    if not isinstance(scene, dict):
        return []
    out: List[Dict[str, Any]] = []
    for split, key in (("train", "train_cameras"), ("test", "test_cameras")):
        cams = scene.get(key, [])
        if not isinstance(cams, list):
            continue
        for i, cam in enumerate(cams):
            frame = _camera_frame_from_repo_metadata(cam, split, i)
            if frame is not None:
                out.append(frame)
    return out


def render_type_from_checkpoint(raw: Dict[str, Any], mlp_present: bool) -> str:
    # The repository stores the authoritative render family in run_config.args
    # and also writes requires_mobilegs=True for sort_free_render checkpoints.
    args = nested_get(raw, ["root_run_config", "args"], {})
    requires_mobilegs = truthy(raw.get("root_requires_mobilegs", False))
    sort_free = requires_mobilegs or truthy(args.get("sort_free_render", False))
    if sort_free:
        if not mlp_present:
            raise SystemExit(
                "Checkpoint is marked sort_free_render/requires_mobilegs, but the MobileGS opacity/phi MLP "
                "was not found in the checkpoint. Re-save with include_mobilegs=True or use the sorted checkpoint."
            )
        return "sort-free-mobilegs"
    # A bare MobileGS payload also implies sort-free.
    if raw.get("format") == "mobile-gs-compressed-v1" or raw.get("root_format") == "mobile-gs-compressed-v1":
        if not mlp_present:
            raise SystemExit("mobile-gs-compressed-v1 payload requires mobilegs_opacity_phi_nn weights")
        return "sort-free-mobilegs"
    return "sorted-alpha"


def u32_bitset_from_bool_masks(masks_bool: np.ndarray) -> np.ndarray:
    masks_bool = np.asarray(masks_bool, dtype=np.bool_)
    if masks_bool.ndim != 2:
        raise ValueError("temporal masks must be [K,N]")
    k, n = masks_bool.shape
    words_per = int(math.ceil(n / 32.0))
    words = np.zeros((k, words_per), dtype=np.uint32)
    for ki in range(k):
        idx = np.nonzero(masks_bool[ki])[0]
        for j in idx:
            words[ki, j // 32] |= np.uint32(1 << (j % 32))
    return words.reshape(-1)


def extract_temporal_filter(raw: Dict[str, Any], n: int) -> Optional[Dict[str, Any]]:
    # Exact exported visibility-filter format from utils.mobile_compression.
    filt = raw.get("temporal_visibility_filter") or raw.get("root_temporal_visibility_filter")
    if isinstance(filt, dict) and filt.get("format") == "temporal-visibility-filter-v1":
        shape = tuple(filt.get("shape", []))
        if len(shape) == 2 and int(shape[1]) == int(n):
            packed = np.asarray(filt["packed_masks"], dtype=np.uint8)
            masks_np = np.unpackbits(packed, axis=1)[:, :n].astype(bool)
            keyframes = to_numpy(filt.get("times"))
            if keyframes is None:
                keyframes = np.arange(masks_np.shape[0], dtype=np.float32)
            words = u32_bitset_from_bool_masks(masks_np)
            words_per = int(math.ceil(n / 32.0))
            return {
                "keyframes": keyframes.astype(np.float32).reshape(-1),
                "words": words.astype(np.uint32),
                "meta": {
                    "source_format": "temporal-visibility-filter-v1",
                    "keyframe_count": int(masks_np.shape[0]),
                    "words_per_mask": words_per,
                    "interpolation": "or-nearest-two",
                    "mask_order": "keyframe-major-u32-bitset",
                },
            }
    return None

def build_scene(raw: Dict[str, Any], schema: str, source_keys: List[str]) -> ExportScene:
    compressed_meta = {"source_was_compressed": False, "decoded_for_render": False, "schemes": [], "has_codebooks": False, "has_huffman": False, "has_rvq": False}
    decoded = maybe_decode_compressed(raw)
    if decoded is not None:
        # Keep root metadata from self-contained wrappers while replacing render tensors.
        root_bits = {k: v for k, v in raw.items() if k.startswith("root_")}
        raw = {**decoded, **root_bits}
        compressed_meta = {"source_was_compressed": True, "decoded_for_render": True, "schemes": ["gpcc-u16-xyz", "huffman-affine-attributes"], "has_codebooks": False, "has_huffman": True, "has_rvq": False}

    xyz = find_tensor(raw, ["_xyz", "xyz", "means3D", "means", "positions", "pos"])
    if xyz is None:
        ckpt_format = raw.get("root_format", raw.get("format", "unknown"))
        raise SystemExit(
            "Cannot find Gaussian positions after checkpoint normalization. "
            f"schema={schema}, checkpoint_format={ckpt_format}, top_keys={list(raw.keys())[:32]}. "
            "For this repo, positions must come from checkpoint['gaussians'][1] (GaussianModel._xyz)."
        )
    xyz = xyz.reshape((-1, xyz.shape[-1]))[:, :3].astype(np.float32)
    n = xyz.shape[0]

    t = find_tensor(raw, ["_t", "t", "timestamp", "timestamps", "mu_t", "mean_t", "time"])
    if t is None: t = np.zeros((n, 1), np.float32)
    t = np.resize(t.reshape((-1, 1)), (n, 1)).astype(np.float32)
    mean4 = np.concatenate([xyz, t], axis=1)

    preactivation = is_capture_preactivation(raw)
    scaling = find_tensor(raw, ["_scaling", "scaling", "scale", "scales", "log_scaling"])
    if scaling is None: scaling = np.full((n, 3), 0.01, np.float32)
    scaling = np.resize(scaling.reshape((scaling.shape[0], -1)), (n, max(1, scaling.reshape((scaling.shape[0], -1)).shape[1]))).astype(np.float32)
    # GaussianModel.capture() stores raw trainable _scaling, i.e. log-scale.
    if preactivation or any(k in raw for k in ["_scaling", "log_scaling"]): scaling = np.exp(scaling)
    if scaling.shape[1] == 1: scaling = np.repeat(scaling, 3, axis=1)
    scaling = np.maximum(scaling[:, :3], 1e-8)

    st = find_tensor(raw, ["_scaling_t", "scaling_t", "scale_t", "st", "time_scale", "sigma_t"])
    if st is None: st = np.full((n, 1), max(float(np.std(t)), 1/30), np.float32)
    st = np.resize(st.reshape((-1, 1)), (n,1)).astype(np.float32)
    # GaussianModel.capture() stores _scaling_t as log temporal scale too.
    if preactivation or any(k in raw for k in ["_scaling_t", "log_scaling_t"]): st = np.exp(st)
    st = np.maximum(st, 1e-8)
    scale4 = np.concatenate([scaling, st], axis=1)

    rot_raw = raw.get("_rotation", raw.get("rotation", raw.get("rot", raw.get("quaternion", raw.get("quat")))))
    ql_raw = raw.get("_rotation_l", raw.get("rotation_l", raw.get("q_left", raw.get("ql", raw.get("left_quat")))))
    qr_raw = raw.get("_rotation_r", raw.get("rotation_r", raw.get("q_right", raw.get("qr", raw.get("right_quat")))))
    ql = valid_quat_tensor(ql_raw, n)
    if ql is None:
        ql = valid_quat_tensor(rot_raw, n)
    if ql is None:
        ql = identity_quats(n)
    qr = valid_quat_tensor(qr_raw, n)
    if qr is None:
        qr = identity_quats(n)

    opacity = find_tensor(raw, ["_opacity", "opacity", "alpha", "density"])
    if opacity is None: opacity = np.full((n,1), 0.1, np.float32)
    opacity = np.resize(opacity.reshape((-1,1)), (n,1)).astype(np.float32)[:,0]
    # GaussianModel.capture() stores inverse-sigmoid opacity.  Generic state dicts
    # are treated as logits when out of [0,1].
    if preactivation or np.nanmin(opacity) < 0 or np.nanmax(opacity) > 1: opacity = sigmoid(opacity)
    opacity = np.clip(opacity, 0.0, 0.999).astype(np.float32)

    dc = find_tensor(raw, ["_features_dc", "features_dc", "f_dc", "rgb", "colors", "color"])
    rest = find_tensor(raw, ["_features_rest", "features_rest", "f_rest", "sh", "features"])
    if dc is None: dc = np.full((n, 3), 0.5, np.float32)
    dc = np.resize(dc.reshape((dc.shape[0], -1)), (n, max(3, dc.reshape((dc.shape[0], -1)).shape[1]))).astype(np.float32)
    base = dc[:, :3]
    if rest is not None:
        rest = np.resize(rest.reshape((rest.shape[0], -1)), (n, rest.reshape((rest.shape[0], -1)).shape[1])).astype(np.float32)
        appearance = np.concatenate([base, rest], axis=1).astype(np.float32)
    else:
        appearance = base.astype(np.float32)

    # Prefer checkpoint metadata over guessing.  4DGS with temporal SH can have
    # coeff_count != (degree+1)^2, so store both spatial and temporal degrees.
    gkwargs = nested_get(raw, ["root_run_config", "gaussian_kwargs"], {})
    active_sh = raw.get("active_sh_degree")
    active_sh_t = raw.get("active_sh_degree_t")
    coeff_count = max(1, appearance.shape[1] // 3)
    max_sh_degree, max_sh_degree_t = infer_sh_degrees(coeff_count, gkwargs, raw)
    # Match interactive_viewer.py: final rendering promotes spatial SH to max degree.
    active_degree = max_sh_degree
    active_degree_t = max(int(active_sh_t) if isinstance(active_sh_t, (int, np.integer)) else 0, max_sh_degree_t)
    force_sh_3d = truthy(gkwargs.get("force_sh_3d", raw.get("force_sh_3d", False)))
    prefilter_var = float(gkwargs.get("prefilter_var", raw.get("prefilter_var", -1.0)) or -1.0)
    td0, td1 = coerce_time_duration_export(gkwargs.get("time_duration", raw.get("time_duration", None)))
    render_args = {
        "gaussian_dim": int(gkwargs.get("gaussian_dim", 4) or 4),
        "rot_4d": bool(truthy(gkwargs.get("rot_4d", raw.get("rot_4d", False)))),
        "force_sh_3d": bool(force_sh_3d),
        "prefilter_var": float(prefilter_var),
        "scale_modifier": 1.0,
        "time_duration": [float(td0), float(td1)],
        "time_span": float(td1 - td0) if abs(td1 - td0) > 1e-12 else 1.0,
        "active_sh_degree": int(active_degree),
        "active_sh_degree_t": int(active_degree_t),
        "max_sh_degree": int(max_sh_degree),
        "max_sh_degree_t": int(max_sh_degree_t),
        "coeff_count": int(coeff_count),
    }
    app_model = {
        "kind": "sh" if coeff_count > 1 else "rgb",
        "degree": int(max_sh_degree),
        "degree_t": int(max_sh_degree_t),
        "active_degree": int(active_degree),
        "active_degree_t": int(active_degree_t),
        "coeff_count": int(coeff_count),
        "storage": "f32",
        "source_layout": "features_dc_plus_features_rest_flattened",
    } if coeff_count > 1 else {"kind": "rgb", "storage": "f32", "coeff_count": int(coeff_count)}

    flags = np.zeros((n,), np.uint32)
    if truthy(gkwargs.get("isotropic_gaussians", False)) or truthy(raw.get("isotropic_gaussians", False)):
        flags |= 2
    if render_args["rot_4d"]:
        flags |= 4

    # Temporal masks: first exact visibility filters; then direct u32 bitsets if present.
    masks = extract_temporal_filter(raw, n)
    if masks is None:
        words = find_tensor(raw, ["temporal_mask_words", "active_mask_words", "mask_words", "masks"], dtype=np.uint32)
        keyframes = find_tensor(raw, ["temporal_keyframes", "mask_keyframes", "keyframes"])
        if words is not None and keyframes is not None:
            words = words.astype(np.uint32).reshape(-1)
            keyframes = keyframes.astype(np.float32).reshape(-1)
            words_per = int(math.ceil(n / 32))
            if len(words) < words_per * len(keyframes):
                print("warning: temporal mask words shorter than expected; masks will be ignored", file=sys.stderr)
            else:
                masks = {"keyframes": keyframes, "words": words, "meta": {"source_format": "u32-bitset", "keyframe_count": int(len(keyframes)), "words_per_mask": words_per, "interpolation": "or-nearest-two", "mask_order": "keyframe-major-u32-bitset", "notes": "Renderer ORs nearest left/right keyframe masks; no render-family override."}}

    mlp = extract_mlp(raw, n, appearance.shape[1])
    native = render_type_from_checkpoint(raw, mlp is not None)

    env_map = raw.get("env_map", None)
    if env_map is None:
        env_map = raw.get("_env_map", None)
    extras = {
        "root_format": raw.get("root_format", raw.get("format")),
        "root_iteration": raw.get("root_iteration"),
        "run_config": raw.get("root_run_config"),
        "scene": raw.get("root_scene"),
        "requires_mobilegs": raw.get("root_requires_mobilegs"),
        "temporal_mask_config": {
            "threshold": nested_get(raw, ["root_run_config", "args", "temporal_mask_threshold"], None),
            "mode": nested_get(raw, ["root_run_config", "args", "temporal_mask_mode"], None),
            "keyframes": nested_get(raw, ["root_run_config", "args", "temporal_mask_keyframes"], None),
            "window": nested_get(raw, ["root_run_config", "args", "temporal_mask_window"], None),
        },
        "render_args": render_args,
    }
    for k in ["_compressed_original", "_compressed_meta"]:
        if k in raw: extras[k] = raw[k]
    return ExportScene(mean4, scale4, ql, qr, opacity, appearance, flags, app_model, render_policy(native, mlp is not None), schema, source_keys, masks, mlp, env_map, compressed_meta, extras)


def json_safe(x: Any) -> Any:
    if isinstance(x, np.ndarray): return {"shape": list(x.shape), "dtype": str(x.dtype)}
    try:
        import torch
        if isinstance(x, torch.Tensor): return {"shape": list(x.shape), "dtype": str(x.dtype)}
    except Exception: pass
    if isinstance(x, (str, int, float, bool)) or x is None: return x
    if isinstance(x, dict): return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [json_safe(v) for v in x[:32]]
    return repr(type(x))


def write_pack(scene: ExportScene, out: Path, source_path: Path, preserve_original: bool, name: Optional[str], source_bytes: bytes, gzip_pack: bool = False, archive_extras: bool = False):
    if out.suffix != ".splat4dpack":
        raise SystemExit(f"Output must end with .splat4dpack, got {out}. The WebGPU viewer supports v3 .splat4dpack only.")
    n = scene.mean4.shape[0]
    app_stride = scene.appearance.shape[1]
    provenance = {
        "source_path": str(source_path),
        "source_format": source_path.suffix,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "schema": scene.schema,
        "source_keys": scene.source_keys,
        "preserved_original": preserve_original,
    }
    meta = {
        "name": name or source_path.stem,
        "gaussian_count": int(n),
        "time_min": float(np.nanmin(scene.mean4[:,3])) if n else 0.0,
        "time_max": float(np.nanmax(scene.mean4[:,3])) if n else 0.0,
        "background": ([1.0, 1.0, 1.0, 1.0] if bool((scene.extras.get("scene") or {}).get("white_background", False)) else [0.0, 0.0, 0.0, 1.0]),
        "appearance_model": scene.appearance_model,
        "render_policy": scene.render_policy,
        "has_temporal_masks": scene.masks is not None,
        "has_sortfree_mlp": scene.mlp is not None,
        "has_env_map": scene.env_map is not None,
        "has_compression_payloads": bool(scene.compression_meta.get("source_was_compressed")),
        "provenance": provenance,
        "compression": scene.compression_meta,
        "cameras": active_scene_cameras(scene.extras),
        "custom": {
            "mlp_sh_dim": int(app_stride),
            "mlp_aux_dim": 3,
            "temporal_mask_config": scene.extras.get("temporal_mask_config"),
            "source_scene": json_safe(scene.extras.get("scene")),
            "run_config": json_safe(scene.extras.get("run_config")),
            "render_args": json_safe(scene.extras.get("render_args")),
        },
    }
    raw = bytearray(MAGIC + struct.pack('<I', VERSION))
    def chunk(tag: bytes, payload: bytes):
        raw.extend(tag); raw.extend(struct.pack('<Q', len(payload))); raw.extend(payload)
    chunk(b'META', json.dumps(meta, separators=(',', ':'), allow_nan=False).encode())
    rec = bytearray()
    for i in range(n):
        rec.extend(struct.pack('<4f4f4f4ffIIIIIII', *scene.mean4[i], *scene.scale4[i], *scene.ql[i], *scene.qr[i], float(scene.opacity[i]), i*app_stride, app_stride, 0, 0, int(scene.flags[i]), 0, 0))
    
    expected_rec_bytes = n * 96
    if len(rec) != expected_rec_bytes:
        raise SystemExit(f'Internal exporter error: GAUS chunk record size mismatch: {len(rec)} != {expected_rec_bytes}. Expected 96 bytes per GaussianRecordV3.')
    chunk(b'GAUS', bytes(rec))
    chunk(b'APPR', scene.appearance.astype('<f4', copy=False).tobytes())
    if scene.masks:
        chunk(b'TMSK', json.dumps(scene.masks['meta'], separators=(',', ':')).encode())
        chunk(b'KEYF', scene.masks['keyframes'].astype('<f4').tobytes())
        chunk(b'MASK', scene.masks['words'].astype('<u4').tobytes())
    if scene.mlp:
        chunk(b'MLPM', json.dumps(scene.mlp.meta, separators=(',', ':')).encode())
        chunk(b'MLPW', scene.mlp.weights.astype('<f4').tobytes())
    if scene.env_map is not None:
        arr = to_numpy(scene.env_map)
        if arr is not None: chunk(b'ENVM', arr.astype('<f4').tobytes())
    if preserve_original:
        chunk(b'ORIG', source_bytes)
    if archive_extras and scene.extras:
        try: chunk(b'EXTR', json.dumps(json_safe(scene.extras), separators=(',', ':')).encode())
        except Exception: pass
    data = bytes(raw)
    if gzip_pack:
        data = gzip.compress(data, compresslevel=6)
    out.write_bytes(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('-o', '--output', type=Path, required=True)
    ap.add_argument('--name', default=None)
    ap.add_argument('--preserve-original', action='store_true')
    ap.add_argument('--gzip', action='store_true', help='gzip-compress output; not recommended for browser/large models')
    ap.add_argument('--archive-extras', action='store_true', help='include large EXTR metadata chunk; off by default for web')
    args = ap.parse_args()
    source_bytes = args.checkpoint.read_bytes()
    obj = torch_load(args.checkpoint)
    raw, schema = flatten_state(obj)
    scene = build_scene_via_repo_model(obj, args.checkpoint, schema, sorted(map(str, raw.keys())))
    if scene is None:
        scene = build_scene(raw, schema, sorted(map(str, raw.keys())))
    write_pack(scene, args.output, args.checkpoint, args.preserve_original, args.name, source_bytes, gzip_pack=args.gzip, archive_extras=args.archive_extras)
    native = scene.render_policy["required_render_type"]
    print(f"wrote {args.output} ({scene.mean4.shape[0]} gaussians, required_render={native}, mlp={scene.mlp is not None}, masks={scene.masks is not None})")

if __name__ == '__main__': main()
