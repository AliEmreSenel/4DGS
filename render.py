#!/usr/bin/env python3
import argparse
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


def load_saved_namespace(cfg_path: Path):
    text = cfg_path.read_text().strip()
    if not text:
        raise ValueError(f"Empty cfg_args file: {cfg_path}")
    safe_globals = {"__builtins__": {}}
    safe_locals = {
        "Namespace": argparse.Namespace,
        "True": True,
        "False": False,
        "None": None,
    }
    obj = eval(text, safe_globals, safe_locals)
    if isinstance(obj, argparse.Namespace):
        return obj
    if isinstance(obj, dict):
        return argparse.Namespace(**obj)
    raise ValueError(f"Unexpected cfg_args object type from {cfg_path}: {type(obj)!r}")


def recursive_omegaconf_merge_into_namespace(ns, cfg):
    from omegaconf.dictconfig import DictConfig

    def rec(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                rec(key1, host[key])
        else:
            setattr(ns, key, host[key])

    for k in cfg.keys():
        rec(k, cfg)


def find_checkpoint(model_path: Path, which: str | None):
    if which:
        p = Path(which)
        if not p.is_absolute():
            p = model_path / which
        if p.exists():
            return p
        raise FileNotFoundError(p)

    best = model_path / "chkpnt_best.pth"
    if best.exists():
        return best
    ckpts = sorted(model_path.glob("chkpnt*.pth"))
    numeric = []
    for p in ckpts:
        stem = p.stem.replace("chkpnt", "")
        if stem.isdigit():
            numeric.append((int(stem), p))
    if numeric:
        return max(numeric)[1]
    raise FileNotFoundError(f"No checkpoint found in {model_path}")


def infer_checkpoint_layout(model_params):
    info = {
        "gaussian_dim": None,
        "rot_4d": None,
        "active_sh_degree_t": 0,
        "time_duration": None,
        "isotropic_gaussians": None,
    }
    if not isinstance(model_params, (tuple, list)):
        return info

    if len(model_params) in (19, 20):
        info["gaussian_dim"] = 4
        info["rot_4d"] = bool(model_params[16])
        try:
            scales = model_params[4]
            rot_l = model_params[5]
            rot_r = model_params[15]
            info["isotropic_gaussians"] = (
                hasattr(scales, "shape")
                and scales.ndim == 2
                and scales.shape[1] == 1
                and hasattr(rot_l, "numel")
                and rot_l.numel() == 0
                and hasattr(rot_r, "numel")
                and rot_r.numel() == 0
            )
        except Exception:
            pass
        try:
            info["active_sh_degree_t"] = int(model_params[18])
        except Exception:
            info["active_sh_degree_t"] = 0
        try:
            tvals = model_params[13]
            if hasattr(tvals, "detach"):
                tvals = tvals.detach().cpu().numpy()
            tmin = float(np.min(tvals))
            tmax = float(np.max(tvals))
            if np.isfinite(tmin) and np.isfinite(tmax):
                pad = max(1e-4, 0.02 * (tmax - tmin + 1e-6))
                info["time_duration"] = [tmin - pad, tmax + pad]
        except Exception:
            pass
    elif len(model_params) in (12, 13):
        info["gaussian_dim"] = 3
        info["rot_4d"] = False
        try:
            scales = model_params[4]
            rot_l = model_params[5]
            info["isotropic_gaussians"] = (
                hasattr(scales, "shape")
                and scales.ndim == 2
                and scales.shape[1] == 1
                and hasattr(rot_l, "numel")
                and rot_l.numel() == 0
            )
        except Exception:
            pass
    return info


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v.copy()
    return v / n


def camera_center_from_RT(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    return -(R @ T)


def camera_forward_from_R(R: np.ndarray) -> np.ndarray:
    return normalize(np.asarray(R[:, 2], dtype=np.float32))


def camera_from_eye_forward(
    eye, forward, world_up=np.array([0.0, 0.0, 1.0], dtype=np.float32)
):
    eye = np.asarray(eye, dtype=np.float32)
    forward = normalize(np.asarray(forward, dtype=np.float32))
    world_up = normalize(np.asarray(world_up, dtype=np.float32))

    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, alt_up)
        if np.linalg.norm(right) < 1e-6:
            alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            right = np.cross(forward, alt_up)
    right = normalize(right)
    down = normalize(np.cross(forward, right))

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T
    T = w2c[:3, 3]
    return R, T


def closest_point_to_rays(origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    for o, d in zip(origins, dirs):
        d = normalize(d.astype(np.float64))
        M = I - np.outer(d, d)
        A += M
        b += M @ o.astype(np.float64)
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.median(origins, axis=0)
    return x.astype(np.float32)


def pca_lateral_axis(points: np.ndarray) -> np.ndarray:
    pts = points - np.mean(points, axis=0, keepdims=True)
    cov = pts.T @ pts / max(1, pts.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axis = vecs[:, order[0]].astype(np.float32)
    axis[2] = 0.0
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return normalize(axis)


def triangle_wave(x: float) -> float:
    y = x % 1.0
    return 1.0 - abs(2.0 * y - 1.0)


def smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def quantize_times(cams, decimals: int = 6):
    groups = {}
    for cam in cams:
        key = round(float(cam.timestamp), decimals)
        groups.setdefault(key, []).append(cam)
    keys = sorted(groups.keys())
    return keys, groups


def build_time_slots(cams_all):
    centers = []
    forwards = []
    for cam in cams_all:
        centers.append(camera_center_from_RT(np.asarray(cam.R), np.asarray(cam.T)))
        forwards.append(camera_forward_from_R(np.asarray(cam.R)))
    centers = np.stack(centers, axis=0)
    forwards = np.stack(forwards, axis=0)
    anchor = closest_point_to_rays(centers, forwards)
    lateral = pca_lateral_axis(centers)

    times, groups = quantize_times(cams_all)
    min_count = min(len(groups[t]) for t in times)
    if min_count < 2:
        raise RuntimeError(
            "Need at least 2 cameras per timestamp for bounded novel rendering"
        )

    num_slots = min_count
    pos_by_t = []
    fwd_by_t = []
    spacing_by_t = []
    for t in times:
        g = groups[t]
        indexed = []
        for cam in g:
            c = camera_center_from_RT(np.asarray(cam.R), np.asarray(cam.T))
            f = camera_forward_from_R(np.asarray(cam.R))
            lateral_coord = float(np.dot(c - anchor, lateral))
            indexed.append((lateral_coord, c, f, cam))
        indexed.sort(key=lambda x: x[0])
        indexed = indexed[:num_slots]
        pos = np.stack([x[1] for x in indexed], axis=0)
        fwd = np.stack([x[2] for x in indexed], axis=0)
        pos_by_t.append(pos)
        fwd_by_t.append(fwd)
        if pos.shape[0] >= 2:
            spacing = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
            spacing_by_t.append(float(np.median(spacing)))
        else:
            spacing_by_t.append(0.0)

    return {
        "times": np.asarray(times, dtype=np.float32),
        "pos": np.stack(pos_by_t, axis=0).astype(np.float32),
        "fwd": np.stack(fwd_by_t, axis=0).astype(np.float32),
        "anchor": anchor.astype(np.float32),
        "lateral": lateral.astype(np.float32),
        "num_slots": num_slots,
        "median_spacing": float(np.median(np.asarray(spacing_by_t, dtype=np.float32))),
    }


def interp_along_time(arr: np.ndarray, times: np.ndarray, t: float) -> np.ndarray:
    if t <= float(times[0]):
        return arr[0].copy()
    if t >= float(times[-1]):
        return arr[-1].copy()
    hi = int(np.searchsorted(times, t, side="right"))
    lo = hi - 1
    t0 = float(times[lo])
    t1 = float(times[hi])
    a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    return (1.0 - a) * arr[lo] + a * arr[hi]


def interp_segment_extrapolated(arr: np.ndarray, s: float):
    n = arr.shape[0]
    if n == 1:
        return arr[0].copy(), arr[0].copy(), 0.0, 0, 0
    if s <= 0.0:
        i0, i1 = 0, 1
        a = s
    elif s >= n - 1:
        i0, i1 = n - 2, n - 1
        a = s - (n - 2)
    else:
        i0 = int(math.floor(s))
        i1 = min(n - 1, i0 + 1)
        a = s - i0
    base = (1.0 - a) * arr[i0] + a * arr[i1]
    return base, arr[i1] - arr[i0], float(a), i0, i1


def sample_bounded_novel(
    path_data,
    t: float,
    u: float,
    novel_phase: float,
    novel_strength: float,
    outward_scale: float,
    endpoint_overshoot: float,
    look_mode: str,
):
    times = path_data["times"]
    pos_slots = interp_along_time(path_data["pos"], times, t)
    fwd_slots = interp_along_time(path_data["fwd"], times, t)
    anchor = path_data["anchor"]
    median_spacing = max(1e-6, float(path_data["median_spacing"]))
    num_slots = pos_slots.shape[0]

    sweep = max(0.0, min(1.0, float(u)))
    sweep_ex = -endpoint_overshoot + sweep * (1.0 + 2.0 * endpoint_overshoot)
    s = sweep_ex * (num_slots - 1)

    eye_base, tangent_raw, _, i0, i1 = interp_segment_extrapolated(pos_slots, s)
    fwd_base, _, _, _, _ = interp_segment_extrapolated(fwd_slots, s)
    tangent = normalize(np.asarray(tangent_raw, dtype=np.float32))
    if np.linalg.norm(tangent) < 1e-6:
        tangent = path_data["lateral"]

    fwd_base = normalize(np.asarray(fwd_base, dtype=np.float32))
    novel_dir = fwd_base - tangent * float(np.dot(fwd_base, tangent))
    if np.linalg.norm(novel_dir) < 1e-6:
        novel_dir = anchor - eye_base
    novel_dir = normalize(novel_dir.astype(np.float32))

    radial = eye_base - anchor
    radial = (
        normalize(radial.astype(np.float32))
        if np.linalg.norm(radial) > 1e-6
        else np.zeros(3, dtype=np.float32)
    )

    edge_weight = smoothstep01(1.0 - abs(2.0 * sweep - 1.0))
    v = math.sin(2.0 * math.pi * novel_phase)
    eye = eye_base.copy()
    if abs(outward_scale) > 1e-8:
        eye = eye + outward_scale * radial * median_spacing
    if abs(novel_strength) > 1e-8:
        eye = eye + (edge_weight * novel_strength * v) * novel_dir * median_spacing

    if look_mode == "anchor":
        target = anchor
    elif look_mode == "blend_forward":
        target = eye + fwd_base
    elif look_mode == "camera_pair":
        pair_target = 0.5 * (
            pos_slots[i0] + fwd_slots[i0] + pos_slots[i1] + fwd_slots[i1]
        )
        target = 0.5 * pair_target + 0.5 * anchor
    else:
        target = 0.45 * anchor + 0.55 * (eye + fwd_base)
    forward = normalize((target - eye).astype(np.float32))
    return eye.astype(np.float32), forward.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Render bounded novel views along the reconstructed camera manifold"
    )
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--source_path", type=str, default=None)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--time_start", type=float, default=None)
    parser.add_argument("--time_end", type=float, default=None)
    parser.add_argument("--freeze_time", type=float, default=None)
    parser.add_argument(
        "--time_mode",
        choices=[
            "sync_arc_time",
            "sync_arc_freeze",
            "bounded_novel_time",
            "bounded_novel_freeze",
        ],
        default="bounded_novel_time",
    )
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--sweep_loops", type=float, default=1.0)
    parser.add_argument("--novel_loops", type=float, default=2.0)
    parser.add_argument(
        "--outward_scale",
        type=float,
        default=0.05,
        help="radial bow in units of median inter-camera spacing",
    )
    parser.add_argument(
        "--novel_strength",
        type=float,
        default=0.18,
        help="depth/off-manifold motion in units of median inter-camera spacing",
    )
    parser.add_argument(
        "--endpoint_overshoot",
        type=float,
        default=0.08,
        help="small left/right extrapolation beyond end cameras",
    )
    parser.add_argument(
        "--look_mode",
        choices=["blend_forward", "anchor", "mixed", "camera_pair"],
        default="mixed",
    )
    parser.add_argument("--skip_png", action="store_true")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--isotropic_gaussians", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    model_path = Path(args.model_path).resolve()
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else model_path / "renders" / "bounded_novel"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo_root))

    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render
    from scene import Scene
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel

    dummy_parser = argparse.ArgumentParser()
    lp = ModelParams(dummy_parser)
    pp = PipelineParams(dummy_parser)
    defaults = dummy_parser.parse_args([])
    resolved = argparse.Namespace(**vars(defaults))

    cfg_args_path = model_path / "cfg_args"
    if cfg_args_path.exists():
        saved = load_saved_namespace(cfg_args_path)
        vars(resolved).update(vars(saved))
    if args.config is not None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.config)
        recursive_omegaconf_merge_into_namespace(resolved, cfg)

    resolved.model_path = str(model_path)
    if args.source_path is not None:
        resolved.source_path = os.path.abspath(args.source_path)

    ckpt_path = find_checkpoint(model_path, args.checkpoint)
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    model_params, loaded_iter = torch.load(
        ckpt_path, map_location=map_location, weights_only=False
    )
    inferred = infer_checkpoint_layout(model_params)
    if inferred["gaussian_dim"] is not None:
        resolved.gaussian_dim = inferred["gaussian_dim"]
    if inferred["rot_4d"] is not None:
        resolved.rot_4d = inferred["rot_4d"]
    if inferred["isotropic_gaussians"] is True:
        resolved.isotropic_gaussians = True
    if getattr(resolved, "time_duration", None) is None or list(
        getattr(resolved, "time_duration", [-0.5, 0.5])
    ) == [-0.5, 0.5]:
        if inferred["time_duration"] is not None:
            resolved.time_duration = inferred["time_duration"]
    if args.isotropic_gaussians:
        resolved.isotropic_gaussians = True

    dataset = lp.extract(resolved)
    pipe = pp.extract(resolved)
    gaussian_dim = getattr(resolved, "gaussian_dim", 3)
    time_duration = getattr(resolved, "time_duration", [-0.5, 0.5])
    rot_4d = getattr(resolved, "rot_4d", False)
    force_sh_3d = getattr(resolved, "force_sh_3d", False)
    isotropic_gaussians = getattr(resolved, "isotropic_gaussians", False)
    num_pts = getattr(resolved, "num_pts", 100000)
    num_pts_ratio = getattr(resolved, "num_pts_ratio", 1.0)
    sh_degree_t = max(
        inferred["active_sh_degree_t"], 2 if getattr(pipe, "eval_shfs_4d", False) else 0
    )

    gaussians = GaussianModel(
        dataset.sh_degree,
        gaussian_dim=gaussian_dim,
        time_duration=time_duration,
        rot_4d=rot_4d,
        force_sh_3d=force_sh_3d,
        sh_degree_t=sh_degree_t,
        prefilter_var=dataset.prefilter_var,
        isotropic_gaussians=isotropic_gaussians,
    )
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=None,
        shuffle=False,
        resolution_scales=[1.0],
        num_pts=num_pts,
        num_pts_ratio=num_pts_ratio,
        time_duration=time_duration,
    )
    gaussians.restore(model_params, None)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    if hasattr(gaussians, "active_sh_degree_t"):
        try:
            gaussians.active_sh_degree_t = max(
                gaussians.active_sh_degree_t, inferred["active_sh_degree_t"]
            )
        except Exception:
            pass

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cams = list(scene.train_cameras[1.0])
    test_cams = list(scene.test_cameras[1.0])
    if args.split == "train":
        cams = train_cams
    elif args.split == "test":
        cams = test_cams
    else:
        cams = train_cams + test_cams
    if len(cams) == 0:
        raise RuntimeError("No cameras found in selected split")

    base_cam = cams[0]
    width = args.width or int(base_cam.image_width)
    height = args.height or int(base_cam.image_height)
    if getattr(base_cam, "cx", -1) > 0:
        scale_x = width / base_cam.image_width
        scale_y = height / base_cam.image_height
        cx = base_cam.cx * scale_x
        cy = base_cam.cy * scale_y
        fl_x = base_cam.fl_x * scale_x
        fl_y = base_cam.fl_y * scale_y
        use_intrinsics = True
        FoVx = FoVy = -1.0
    else:
        cx = cy = fl_x = fl_y = -1.0
        use_intrinsics = False
        FoVx = float(base_cam.FoVx)
        FoVy = float(base_cam.FoVy)

    path_data = build_time_slots(cams)
    available_times = path_data["times"]
    default_time_start = float(available_times[0])
    default_time_end = float(available_times[-1])
    time_start = default_time_start if args.time_start is None else args.time_start
    time_end = default_time_end if args.time_end is None else args.time_end
    freeze_time = (
        float(args.freeze_time)
        if args.freeze_time is not None
        else 0.5 * (time_start + time_end)
    )

    video_path = out_dir / "bounded_novel.mp4"
    frames_dir = out_dir / "frames"
    save_png = not args.skip_png
    if save_png:
        frames_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"bounded_novel: slots={path_data['num_slots']} median_spacing={path_data['median_spacing']:.4f} "
        f"time=[{time_start:.4f}, {time_end:.4f}] outward={args.outward_scale} "
        f"novel_strength={args.novel_strength} overshoot={args.endpoint_overshoot}"
    )

    writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8)
    try:
        for idx in range(args.frames):
            frac = 0.0 if args.frames == 1 else idx / (args.frames - 1)
            if args.time_mode in ("sync_arc_freeze", "bounded_novel_freeze"):
                t_val = freeze_time
            else:
                t_val = time_start + frac * (time_end - time_start)
            u = triangle_wave(frac * args.sweep_loops)
            novel_phase = frac * args.novel_loops
            eye, forward = sample_bounded_novel(
                path_data=path_data,
                t=t_val,
                u=u,
                novel_phase=novel_phase,
                novel_strength=(
                    args.novel_strength
                    if args.time_mode.startswith("bounded_novel")
                    else 0.0
                ),
                outward_scale=args.outward_scale,
                endpoint_overshoot=args.endpoint_overshoot,
                look_mode=args.look_mode,
            )
            R, T = camera_from_eye_forward(eye, forward)
            cam = Camera(
                colmap_id=0,
                R=R,
                T=T,
                FoVx=FoVx if not use_intrinsics else -1.0,
                FoVy=FoVy if not use_intrinsics else -1.0,
                image=torch.empty(0),
                gt_alpha_mask=None,
                image_name=f"bounded_novel_{idx:04d}",
                uid=idx,
                data_device="cuda",
                timestamp=float(t_val),
                cx=cx,
                cy=cy,
                fl_x=fl_x,
                fl_y=fl_y,
                resolution=(width, height),
                meta_only=True,
            ).cuda()
            with torch.no_grad():
                render_pkg = render(cam, gaussians, pipe, background)
                image = (
                    torch.clamp(render_pkg["render"], 0.0, 1.0)
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                )
            image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            writer.append_data(image_u8)
            if save_png:
                imageio.imwrite(frames_dir / f"{idx:04d}.png", image_u8)
            if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == args.frames:
                print(
                    f"[{idx+1}/{args.frames}] t={t_val:.4f} u={u:.3f} novel_phase={novel_phase:.3f}"
                )
    finally:
        writer.close()

    info = out_dir / "render_info.txt"
    info.write_text(
        f"mode=bounded_novel\n"
        f"checkpoint={ckpt_path}\n"
        f"loaded_iter={loaded_iter}\n"
        f"split={args.split}\n"
        f"frames={args.frames}\n"
        f"fps={args.fps}\n"
        f"time_mode={args.time_mode}\n"
        f"time_start={time_start}\n"
        f"time_end={time_end}\n"
        f"freeze_time={freeze_time}\n"
        f"sweep_loops={args.sweep_loops}\n"
        f"novel_loops={args.novel_loops}\n"
        f"outward_scale={args.outward_scale}\n"
        f"novel_strength={args.novel_strength}\n"
        f"endpoint_overshoot={args.endpoint_overshoot}\n"
        f"num_slots={path_data['num_slots']}\n"
        f"median_spacing={path_data['median_spacing']}\n"
    )
    print(f"Wrote video to {video_path}")
    if save_png:
        print(f"Wrote frames to {frames_dir}")


if __name__ == "__main__":
    main()
