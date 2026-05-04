#!/usr/bin/env python3
import argparse
import ast
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch



def coerce_time_duration(value):
    """Accept [0,1], ("0","1"), "[0.0, 1.0]", tensors, numpy arrays."""
    if isinstance(value, str):
        text = value.strip()
        try:
            value = ast.literal_eval(text)
        except Exception:
            value = text.strip("[]()").replace(",", " ").split()

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)

    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"Invalid time_duration: {value!r}")

    return [float(value[0]), float(value[1])]


def render_tensor_to_uint8(render_tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW float render tensor to HWC uint8 with minimal host traffic."""
    return (
        torch.clamp(render_tensor, 0.0, 1.0)
        .mul(255.0)
        .add_(0.5)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )


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


def look_at_colmap(
    eye,
    target,
    world_up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, alt_up)
        if np.linalg.norm(right) < 1e-6:
            alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            right = np.cross(forward, alt_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    down = np.cross(forward, right)
    down = down / (np.linalg.norm(down) + 1e-8)

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


def select_bounded_cameras(train_cams, test_cams, split: str):
    if split == "train":
        return train_cams
    if split == "test":
        return test_cams
    return train_cams + test_cams



class CheckpointScene:
    def __init__(self, scene_metadata, Camera):
        self.cameras_extent = float(scene_metadata["cameras_extent"])
        self.white_background = bool(scene_metadata["white_background"])
        self.train_cameras = {
            1.0: [camera_from_metadata(Camera, cam) for cam in scene_metadata.get("train_cameras", [])]
        }
        self.test_cameras = {
            1.0: [camera_from_metadata(Camera, cam) for cam in scene_metadata.get("test_cameras", [])]
        }


def camera_from_metadata(Camera, meta):
    resolution = meta.get("resolution") or [int(meta["width"]), int(meta["height"])]
    return Camera(
        colmap_id=int(meta.get("colmap_id", meta.get("uid", 0))),
        R=np.array(meta["R"], dtype=np.float32),
        T=np.array(meta["T"], dtype=np.float32),
        FoVx=float(meta["FoVx"]),
        FoVy=float(meta["FoVy"]),
        image=torch.empty((3, int(resolution[1]), int(resolution[0]))),
        gt_alpha_mask=None,
        image_name=str(meta.get("image_name", f"cam_{meta.get('uid', 0)}")),
        uid=int(meta.get("uid", meta.get("colmap_id", 0))),
        trans=np.array(meta.get("trans", [0.0, 0.0, 0.0]), dtype=np.float32),
        scale=float(meta.get("scale", 1.0)),
        data_device="cuda",
        timestamp=float(meta.get("timestamp", 0.0)),
        cx=float(meta.get("cx", -1)),
        cy=float(meta.get("cy", -1)),
        fl_x=float(meta.get("fl_x", -1)),
        fl_y=float(meta.get("fl_y", -1)),
        resolution=(int(resolution[0]), int(resolution[1])),
        image_path=None,
        meta_only=True,
    )


def select_orbit_cameras(train_cams, test_cams, split: str):
    if split == "train":
        return train_cams
    if split == "test":
        return test_cams if len(test_cams) > 0 else train_cams
    return test_cams if len(test_cams) > 0 else train_cams


def infer_resolution_and_intrinsics(base_cam, width_arg: int, height_arg: int):
    width = width_arg or int(base_cam.image_width)
    height = height_arg or int(base_cam.image_height)
    has_valid_intrinsics = all(
        float(getattr(base_cam, name, -1.0)) > 0.0
        for name in ("fl_x", "fl_y", "cx", "cy")
    )
    if has_valid_intrinsics:
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
    return width, height, use_intrinsics, FoVx, FoVy, cx, cy, fl_x, fl_y


def make_meta_camera(
    Camera,
    idx: int,
    image_name: str,
    R,
    T,
    timestamp: float,
    width: int,
    height: int,
    use_intrinsics: bool,
    FoVx: float,
    FoVy: float,
    cx: float,
    cy: float,
    fl_x: float,
    fl_y: float,
):
    return Camera(
        colmap_id=0,
        R=R,
        T=T,
        FoVx=FoVx if not use_intrinsics else -1.0,
        FoVy=FoVy if not use_intrinsics else -1.0,
        image=torch.empty(0),
        gt_alpha_mask=None,
        image_name=image_name,
        uid=idx,
        data_device="cuda",
        timestamp=float(timestamp),
        cx=cx,
        cy=cy,
        fl_x=fl_x,
        fl_y=fl_y,
        resolution=(width, height),
        meta_only=True,
    ).cuda()


def render_orbit_mode(
    args,
    scene,
    render,
    background,
    gaussians,
    pipe,
    Camera,
    time_duration,
    ckpt_path,
    loaded_iter,
    out_dir: Path,
):
    train_cams = list(scene.train_cameras[1.0])
    test_cams = list(scene.test_cameras[1.0])
    cam_list = select_orbit_cameras(train_cams, test_cams, args.split)
    if len(cam_list) == 0:
        raise RuntimeError("No cameras found in the selected split")

    centers = np.stack(
        [cam.camera_center.detach().cpu().numpy() for cam in cam_list], axis=0
    )
    forwards = np.stack(
        [camera_forward_from_R(np.asarray(cam.R)) for cam in cam_list], axis=0
    )
    center = closest_point_to_rays(centers, forwards)
    if not np.all(np.isfinite(center)):
        center = np.median(centers, axis=0).astype(np.float32)
    radius = (
        np.quantile(np.linalg.norm(centers - center[None], axis=1), 0.9)
        * args.radius_scale
    )
    if not np.isfinite(radius) or radius <= 1e-6:
        radius = max(scene.cameras_extent, 1.0) * args.radius_scale

    base_cam = cam_list[0]
    width, height, use_intrinsics, FoVx, FoVy, cx, cy, fl_x, fl_y = (
        infer_resolution_and_intrinsics(base_cam, args.width, args.height)
    )

    time_values = np.array([float(cam.timestamp) for cam in cam_list], dtype=np.float32)
    default_time_start = (
        float(np.min(time_values)) if len(time_values) else float(time_duration[0])
    )
    default_time_end = (
        float(np.max(time_values)) if len(time_values) else float(time_duration[1])
    )
    time_start = default_time_start if args.time_start is None else args.time_start
    time_end = default_time_end if args.time_end is None else args.time_end

    frames_dir = out_dir / "frames"
    save_png = bool(args.save_png)
    if save_png:
        frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "orbit_time.mp4"

    elev = math.radians(args.elevation_deg)
    look_at = center.copy()
    look_at[1] += args.look_at_offset_y

    writer = None if args.no_video else imageio.get_writer(
        video_path,
        fps=args.fps,
        codec=args.video_codec,
        quality=args.video_quality,
        macro_block_size=args.macro_block_size,
    )
    try:
        for idx in range(args.frames):
            frac = 0.0 if args.frames == 1 else idx / (args.frames - 1)
            az = math.radians(
                args.orbit_start_deg
                + frac * (args.orbit_end_deg - args.orbit_start_deg)
            )
            horiz = radius * math.cos(elev)
            z = center[2] + radius * math.sin(elev)
            eye = np.array(
                [
                    center[0] + horiz * math.cos(az),
                    center[1] + horiz * math.sin(az),
                    z,
                ],
                dtype=np.float32,
            )

            if args.time_mode == "orbit-only":
                t_val = (
                    float(args.freeze_time)
                    if args.freeze_time is not None
                    else 0.5 * (time_start + time_end)
                )
            elif args.time_mode == "time-only":
                t_val = time_start + frac * (time_end - time_start)
            else:
                t_val = time_start + frac * (time_end - time_start)

            if args.time_mode == "time-only":
                ref_cam = base_cam
                cam = make_meta_camera(
                    Camera,
                    idx=idx,
                    image_name=f"orbit_{idx:04d}",
                    R=ref_cam.R,
                    T=ref_cam.T,
                    timestamp=t_val,
                    width=width,
                    height=height,
                    use_intrinsics=use_intrinsics,
                    FoVx=FoVx,
                    FoVy=FoVy,
                    cx=cx,
                    cy=cy,
                    fl_x=fl_x,
                    fl_y=fl_y,
                )
            else:
                R, T = look_at_colmap(eye, look_at)
                cam = make_meta_camera(
                    Camera,
                    idx=idx,
                    image_name=f"orbit_{idx:04d}",
                    R=R,
                    T=T,
                    timestamp=t_val,
                    width=width,
                    height=height,
                    use_intrinsics=use_intrinsics,
                    FoVx=FoVx,
                    FoVy=FoVy,
                    cx=cx,
                    cy=cy,
                    fl_x=fl_x,
                    fl_y=fl_y,
                )

            with torch.no_grad():
                render_pkg = render(cam, gaussians, pipe, background)
                image_u8 = render_tensor_to_uint8(render_pkg["render"])
            if writer is not None:
                writer.append_data(image_u8)
            if save_png:
                imageio.imwrite(frames_dir / f"{idx:04d}.png", image_u8)
            if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == args.frames:
                print(
                    f"[{idx+1}/{args.frames}] t={t_val:.4f} az_deg={math.degrees(az):.2f}"
                )
    finally:
        if writer is not None:
            writer.close()

    info = out_dir / "render_info.txt"
    info.write_text(
        f"mode=orbit_time\n"
        f"checkpoint={ckpt_path}\n"
        f"loaded_iter={loaded_iter}\n"
        f"split={args.split}\n"
        f"frames={args.frames}\n"
        f"fps={args.fps}\n"
        f"video_codec={args.video_codec}\n"
        f"video_quality={args.video_quality}\n"
        f"no_video={args.no_video}\n"
        f"temporal_mask_threshold={getattr(pipe, 'temporal_mask_threshold', None)}\n"
        f"temporal_mask_keyframes={getattr(pipe, 'temporal_mask_keyframes', None)}\n"
        f"temporal_mask_window={getattr(pipe, 'temporal_mask_window', None)}\n"
        f"time_mode={args.time_mode}\n"
        f"time_start={time_start}\n"
        f"time_end={time_end}\n"
        f"freeze_time={args.freeze_time}\n"
        f"center={center.tolist()}\n"
        f"radius={radius}\n"
        f"orbit_start_deg={args.orbit_start_deg}\n"
        f"orbit_end_deg={args.orbit_end_deg}\n"
        f"elevation_deg={args.elevation_deg}\n"
        f"look_at_offset_y={args.look_at_offset_y}\n"
    )
    if writer is not None:
        print(f"Wrote video to {video_path}")
    if save_png:
        print(f"Wrote frames to {frames_dir}")


def render_bounded_mode(
    args,
    scene,
    render,
    background,
    gaussians,
    pipe,
    Camera,
    ckpt_path,
    loaded_iter,
    out_dir: Path,
):
    train_cams = list(scene.train_cameras[1.0])
    test_cams = list(scene.test_cameras[1.0])
    cams = select_bounded_cameras(train_cams, test_cams, args.split)
    if len(cams) == 0:
        raise RuntimeError("No cameras found in selected split")

    base_cam = cams[0]
    width, height, use_intrinsics, FoVx, FoVy, cx, cy, fl_x, fl_y = (
        infer_resolution_and_intrinsics(base_cam, args.width, args.height)
    )

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
    save_png = bool(args.save_png)
    if save_png:
        frames_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"bounded_novel: slots={path_data['num_slots']} median_spacing={path_data['median_spacing']:.4f} "
        f"time=[{time_start:.4f}, {time_end:.4f}] outward={args.outward_scale} "
        f"novel_strength={args.novel_strength} overshoot={args.endpoint_overshoot}"
    )

    writer = None if args.no_video else imageio.get_writer(
        video_path,
        fps=args.fps,
        codec=args.video_codec,
        quality=args.video_quality,
        macro_block_size=args.macro_block_size,
    )
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
            cam = make_meta_camera(
                Camera,
                idx=idx,
                image_name=f"bounded_novel_{idx:04d}",
                R=R,
                T=T,
                timestamp=t_val,
                width=width,
                height=height,
                use_intrinsics=use_intrinsics,
                FoVx=FoVx,
                FoVy=FoVy,
                cx=cx,
                cy=cy,
                fl_x=fl_x,
                fl_y=fl_y,
            )
            with torch.no_grad():
                render_pkg = render(cam, gaussians, pipe, background)
                image_u8 = render_tensor_to_uint8(render_pkg["render"])
            if writer is not None:
                writer.append_data(image_u8)
            if save_png:
                imageio.imwrite(frames_dir / f"{idx:04d}.png", image_u8)
            if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == args.frames:
                print(
                    f"[{idx+1}/{args.frames}] t={t_val:.4f} u={u:.3f} novel_phase={novel_phase:.3f}"
                )
    finally:
        if writer is not None:
            writer.close()

    info = out_dir / "render_info.txt"
    info.write_text(
        f"mode=bounded_novel\n"
        f"checkpoint={ckpt_path}\n"
        f"loaded_iter={loaded_iter}\n"
        f"split={args.split}\n"
        f"frames={args.frames}\n"
        f"fps={args.fps}\n"
        f"video_codec={args.video_codec}\n"
        f"video_quality={args.video_quality}\n"
        f"no_video={args.no_video}\n"
        f"temporal_mask_threshold={getattr(pipe, 'temporal_mask_threshold', None)}\n"
        f"temporal_mask_keyframes={getattr(pipe, 'temporal_mask_keyframes', None)}\n"
        f"temporal_mask_window={getattr(pipe, 'temporal_mask_window', None)}\n"
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


def main():
    parser = argparse.ArgumentParser(
        description="Render videos from a self-contained 4D Gaussian checkpoint"
    )
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--model_file", type=str, required=True)
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
            "orbit-time",
            "orbit-only",
            "time-only",
            "sync_arc_time",
            "sync_arc_freeze",
            "bounded_novel_time",
            "bounded_novel_freeze",
        ],
        default="orbit-time",
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
    parser.add_argument("--radius_scale", type=float, default=1.0)
    parser.add_argument("--elevation_deg", type=float, default=15.0)
    parser.add_argument("--look_at_offset_y", type=float, default=0.0)
    parser.add_argument("--orbit_start_deg", type=float, default=0.0)
    parser.add_argument("--orbit_end_deg", type=float, default=360.0)
    parser.add_argument("--save_png", action="store_true", help="Also write individual PNG frames next to the video")
    parser.add_argument("--no_video", action="store_true", help="Render PNG frames only; skip MP4 encoding")
    parser.add_argument("--video_codec", type=str, default="libx264")
    parser.add_argument("--video_quality", type=int, default=8)
    parser.add_argument("--macro_block_size", type=int, default=16)
    parser.add_argument(
        "--temporal_mask_threshold",
        type=float,
        default=None,
        help="Override pipeline temporal active-mask threshold; lower is safer, higher is faster",
    )
    parser.add_argument(
        "--temporal_mask_keyframes",
        type=int,
        default=None,
        help="Enable cached key-frame temporal active masks for inference when >1",
    )
    parser.add_argument(
        "--temporal_mask_window",
        type=int,
        default=None,
        help="Number of neighboring temporal keyframes to union on each side",
    )
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("render.py requires CUDA because the Gaussian rasterizers are CUDA extensions.")
    if args.no_video and not args.save_png:
        # Avoid a costly render loop that produces no output.
        args.save_png = True
        print("--no_video was set without --save_png; enabling --save_png automatically.")

    repo_root = Path(args.repo_root).resolve()
    model_file = Path(args.model_file).resolve()
    orbit_modes = {"orbit-time", "orbit-only", "time-only"}
    default_subdir = "orbit_time" if args.time_mode in orbit_modes else "bounded_novel"
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else model_file.parent / "renders" / default_subdir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo_root))

    from arguments import ModelParams, PipelineParams
    from utils.checkpoint_utils import checkpoint_args, load_checkpoint
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel

    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_payload = load_checkpoint(model_file, map_location=map_location)
    loaded_iter = int(checkpoint_payload["iteration"])
    ckpt_path = model_file

    dummy_parser = argparse.ArgumentParser()
    ModelParams(dummy_parser)
    pp = PipelineParams(dummy_parser)
    defaults = dummy_parser.parse_args([])
    resolved = argparse.Namespace(**vars(defaults))
    vars(resolved).update(vars(checkpoint_args(checkpoint_payload)))

    run_config = checkpoint_payload.get("run_config", {})
    gaussian_kwargs = dict(run_config.get("gaussian_kwargs", {}))
    if int(gaussian_kwargs.get("gaussian_dim", 4)) != 4:
        raise ValueError("Only 4D Gaussian checkpoints are supported.")

    pipe = pp.extract(resolved)
    if args.temporal_mask_threshold is not None:
        pipe.temporal_mask_threshold = float(args.temporal_mask_threshold)
    if args.temporal_mask_keyframes is not None:
        pipe.temporal_mask_keyframes = int(args.temporal_mask_keyframes)
    if args.temporal_mask_window is not None:
        pipe.temporal_mask_window = int(args.temporal_mask_window)

    gaussians = GaussianModel(**gaussian_kwargs)
    gaussians.restore(checkpoint_payload["gaussians"], None)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    if hasattr(gaussians, "active_sh_degree_t"):
        gaussians.active_sh_degree_t = max(
            gaussians.active_sh_degree_t,
            int(gaussian_kwargs.get("sh_degree_t", 0)),
        )

    scene = CheckpointScene(checkpoint_payload["scene"], Camera)
    time_duration = coerce_time_duration(
        gaussian_kwargs.get("time_duration", gaussians.time_duration)
    )
    gaussians.time_duration = time_duration
    gaussian_kwargs["time_duration"] = time_duration

    bg_color = [1, 1, 1] if scene.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.time_mode in orbit_modes:
        render_orbit_mode(
            args=args,
            scene=scene,
            render=render,
            background=background,
            gaussians=gaussians,
            pipe=pipe,
            Camera=Camera,
            time_duration=time_duration,
            ckpt_path=ckpt_path,
            loaded_iter=loaded_iter,
            out_dir=out_dir,
        )
    else:
        render_bounded_mode(
            args=args,
            scene=scene,
            render=render,
            background=background,
            gaussians=gaussians,
            pipe=pipe,
            Camera=Camera,
            ckpt_path=ckpt_path,
            loaded_iter=loaded_iter,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
