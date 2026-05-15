#!/usr/bin/env python3
"""Interactive 4D Gaussian checkpoint viewer.

This app opens a live window, renders a self-contained checkpoint, and lets you
fly through it with WASD and mouse-look controls.  It intentionally uses the same
CUDA rasterizer path as render.py so checkpoints saved by train.py can be viewed
without reconstructing a dataset folder.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch


WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return v / n


def coerce_time_duration(value) -> list[float]:
    """Accept [0, 1], (0, 1), "[0, 1]", tensors, numpy arrays, etc."""
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


def camera_center_from_rt(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    return -(np.asarray(R, dtype=np.float32) @ np.asarray(T, dtype=np.float32))


def camera_forward_from_r(R: np.ndarray) -> np.ndarray:
    return normalize(np.asarray(R, dtype=np.float32)[:, 2])


def yaw_pitch_from_forward(forward: np.ndarray) -> tuple[float, float]:
    f = normalize(forward)
    yaw = math.atan2(float(f[1]), float(f[0]))
    pitch = math.asin(max(-1.0, min(1.0, float(f[2]))))
    return yaw, pitch


def forward_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    cp = math.cos(pitch)
    return normalize(np.array([cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)], dtype=np.float32))


def camera_basis_from_yaw_pitch(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility helper for old yaw/pitch-only paths."""
    orientation = orientation_from_forward(forward_from_yaw_pitch(yaw, pitch))
    return orientation[:, 2], orientation[:, 0], orientation[:, 1]


def orientation_from_forward(forward: np.ndarray) -> np.ndarray:
    """Build a right/up/forward camera basis without introducing roll."""
    forward = normalize(forward)
    right = np.cross(WORLD_UP, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = normalize(right)
    up = normalize(np.cross(forward, right))
    return np.stack([right, up, forward], axis=1).astype(np.float32)


def normalize_orientation(orientation: np.ndarray) -> np.ndarray:
    """Return an orthonormal right/up/forward basis while preserving roll.

    This intentionally avoids Euler angles so looking up/down is never clamped.
    The mouse can spin continuously through full 360 degree pitch/yaw; R/T are
    the only controls that add roll around the current view direction.
    """
    ori = np.asarray(orientation, dtype=np.float32).reshape(3, 3)
    forward = normalize(ori[:, 2])
    up_hint = normalize(ori[:, 1])
    right = np.cross(up_hint, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(WORLD_UP, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = normalize(right)
    up = normalize(np.cross(forward, right))
    return np.stack([right, up, forward], axis=1).astype(np.float32)


def orientation_from_rt(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float32)
    # scene.cameras.Camera stores R as camera-to-world columns: right, down, forward.
    right = R[:, 0]
    up = -R[:, 1]
    forward = R[:, 2]
    return normalize_orientation(np.stack([right, up, forward], axis=1))


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = [float(v) for v in axis]
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float32,
    )


def rotate_orientation(orientation: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    if abs(float(angle)) < 1e-12:
        return normalize_orientation(orientation)
    return normalize_orientation(rotation_matrix(axis, angle) @ np.asarray(orientation, dtype=np.float32))


def orientation_debug_text(orientation: np.ndarray) -> str:
    orientation = normalize_orientation(orientation)
    f = orientation[:, 2]
    u = orientation[:, 1]
    return (
        f"look [{f[0]:.3f}, {f[1]:.3f}, {f[2]:.3f}] "
        f"up [{u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f}]"
    )


def right_from_forward(forward: np.ndarray) -> np.ndarray:
    # Game-style right vector: with Z-up coordinates, looking along +X means
    # +Y is screen-right.  Keeping right tied to WORLD_UP prevents accidental
    # roll when the mouse moves left/right.
    right = np.cross(WORLD_UP, forward)
    if np.linalg.norm(right) < 1e-6:
        yaw_fallback = np.array([forward[0], forward[1], 0.0], dtype=np.float32)
        if np.linalg.norm(yaw_fallback) > 1e-6:
            right = np.cross(WORLD_UP, normalize(yaw_fallback))
        else:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return normalize(right)


def rt_from_eye_forward(eye: np.ndarray, forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a no-roll COLMAP-style R, T used by scene.cameras.Camera."""
    eye = np.asarray(eye, dtype=np.float32)
    forward = normalize(forward)
    right = right_from_forward(forward)

    # Camera image Y points down, so the camera's local down vector is the
    # negative of the world-up-consistent camera up vector.
    up = normalize(np.cross(forward, right))
    if np.linalg.norm(up) < 1e-6:
        up = WORLD_UP.copy()
    down = -up

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T.astype(np.float32)
    T = w2c[:3, 3].astype(np.float32)
    return R, T


def rt_from_eye_orientation(eye: np.ndarray, orientation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a COLMAP-style R, T from a right/up/forward camera basis."""
    eye = np.asarray(eye, dtype=np.float32)
    orientation = normalize_orientation(orientation)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = orientation[:, 0]
    c2w[:3, 1] = -orientation[:, 1]
    c2w[:3, 2] = orientation[:, 2]
    c2w[:3, 3] = eye
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T.astype(np.float32)
    T = w2c[:3, 3].astype(np.float32)
    return R, T


def rt_from_eye_yaw_pitch(eye: np.ndarray, yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility helper for older yaw/pitch-only calls."""
    return rt_from_eye_orientation(eye, orientation_from_forward(forward_from_yaw_pitch(yaw, pitch)))


def closest_point_to_rays(origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    for o, d in zip(origins, dirs):
        d = normalize(d.astype(np.float64)).astype(np.float64)
        M = I - np.outer(d, d)
        A += M
        b += M @ o.astype(np.float64)
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.median(origins, axis=0)
    return x.astype(np.float32)


def clamp_time(t: float, t0: float, t1: float) -> float:
    lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
    return max(lo, min(hi, float(t)))


def wrap_time(t: float, t0: float, t1: float) -> float:
    if not math.isfinite(t0) or not math.isfinite(t1) or abs(t1 - t0) < 1e-8:
        return float(t0)
    span = t1 - t0
    return t0 + ((t - t0) % span)


def render_tensor_to_uint8(render_tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW float render tensor to HWC uint8 for pygame."""
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


@dataclass
class Intrinsics:
    width: int
    height: int
    fovx: float
    fovy: float
    cx: float = -1.0
    cy: float = -1.0
    fl_x: float = -1.0
    fl_y: float = -1.0

    @property
    def has_center_shift(self) -> bool:
        return self.cx > 0 and self.cy > 0 and self.fl_x > 0 and self.fl_y > 0


def copy_intrinsics(intr: Intrinsics) -> Intrinsics:
    return Intrinsics(
        width=int(intr.width),
        height=int(intr.height),
        fovx=float(intr.fovx),
        fovy=float(intr.fovy),
        cx=float(intr.cx),
        cy=float(intr.cy),
        fl_x=float(intr.fl_x),
        fl_y=float(intr.fl_y),
    )


def resized_intrinsics(base_intr: Intrinsics, width: int, height: int) -> Intrinsics:
    """Return camera intrinsics for a viewport without distorting aspect ratio.

    This function must be called with the original/base intrinsics, not with the
    currently resized intrinsics.  Resizing focal-length cameras by independent
    x/y scale factors keeps both old FOVs, which makes the projection disagree
    with a new window aspect ratio.  Instead, keep the base vertical FOV and
    derive the horizontal FOV from the requested viewport.  For center-shift
    cameras, preserve the optical-axis angular offset and use a square-pixel
    focal length so circles stay circular after arbitrary window resizes.
    """
    width = max(16, int(width))
    height = max(16, int(height))
    fovy = float(base_intr.fovy)

    if base_intr.has_center_shift:
        base_fl_x = max(1e-6, float(base_intr.fl_x))
        base_fl_y = max(1e-6, float(base_intr.fl_y))
        pixel_aspect = base_fl_x / base_fl_y
        if not math.isfinite(pixel_aspect) or pixel_aspect <= 0.0:
            pixel_aspect = 1.0

        fl_y = height / max(1e-6, 2.0 * math.tan(fovy * 0.5))
        fl_x = fl_y * pixel_aspect

        # Preserve the principal point as an angular offset from image center.
        offset_x = (float(base_intr.cx) - float(base_intr.width) * 0.5) / base_fl_x
        offset_y = (float(base_intr.cy) - float(base_intr.height) * 0.5) / base_fl_y
        cx = width * 0.5 + offset_x * fl_x
        cy = height * 0.5 + offset_y * fl_y

        fovx = 2.0 * math.atan(width / max(1e-6, 2.0 * fl_x))
        fovy = 2.0 * math.atan(height / max(1e-6, 2.0 * fl_y))
        return Intrinsics(width=width, height=height, fovx=fovx, fovy=fovy, cx=cx, cy=cy, fl_x=fl_x, fl_y=fl_y)

    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * (width / max(1.0, height)))
    return Intrinsics(width=width, height=height, fovx=fovx, fovy=fovy)


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def camera_meta_from_dict(meta: dict) -> SimpleNamespace:
    resolution = meta.get("resolution") or [int(meta["width"]), int(meta["height"])]
    return SimpleNamespace(
        colmap_id=int(meta.get("colmap_id", meta.get("uid", 0))),
        uid=int(meta.get("uid", meta.get("colmap_id", 0))),
        R=np.asarray(meta["R"], dtype=np.float32),
        T=np.asarray(meta["T"], dtype=np.float32),
        FoVx=float(meta.get("FoVx", -1.0)),
        FoVy=float(meta.get("FoVy", -1.0)),
        image_name=str(meta.get("image_name", f"cam_{meta.get('uid', 0)}")),
        timestamp=float(meta.get("timestamp", 0.0)),
        image_width=int(resolution[0]),
        image_height=int(resolution[1]),
        cx=float(meta.get("cx", -1.0)),
        cy=float(meta.get("cy", -1.0)),
        fl_x=float(meta.get("fl_x", -1.0)),
        fl_y=float(meta.get("fl_y", -1.0)),
        trans=np.asarray(meta.get("trans", [0.0, 0.0, 0.0]), dtype=np.float32),
        scale=float(meta.get("scale", 1.0)),
    )


def select_camera_metas(scene_meta: dict, split: str) -> tuple[list[SimpleNamespace], list[SimpleNamespace], list[SimpleNamespace]]:
    train = [camera_meta_from_dict(c) for c in scene_meta.get("train_cameras", [])]
    test = [camera_meta_from_dict(c) for c in scene_meta.get("test_cameras", [])]
    if split == "train":
        selected = train
    elif split == "test":
        selected = test if test else train
    else:
        selected = (test if test else []) + train
    if not selected:
        selected = train + test
    if not selected:
        raise RuntimeError("Checkpoint has no camera metadata; cannot choose an initial viewer pose.")
    return train, test, selected


def intrinsics_from_base(base_cam: SimpleNamespace, args: argparse.Namespace) -> Intrinsics:
    width = int(args.width or base_cam.image_width)
    height = int(args.height or base_cam.image_height)

    valid_focal = base_cam.fl_x > 0.0 and base_cam.fl_y > 0.0 and base_cam.cx > 0.0 and base_cam.cy > 0.0
    if args.fov_y_deg is not None:
        fovy = math.radians(float(args.fov_y_deg))
        fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * (width / max(1.0, height)))
        return Intrinsics(width=width, height=height, fovx=fovx, fovy=fovy)

    if valid_focal:
        sx = width / max(1, int(base_cam.image_width))
        sy = height / max(1, int(base_cam.image_height))
        fl_x = float(base_cam.fl_x) * sx
        fl_y = float(base_cam.fl_y) * sy
        cx = float(base_cam.cx) * sx
        cy = float(base_cam.cy) * sy
        fovx = 2.0 * math.atan(width / max(1e-6, 2.0 * fl_x))
        fovy = 2.0 * math.atan(height / max(1e-6, 2.0 * fl_y))
        return Intrinsics(width=width, height=height, fovx=fovx, fovy=fovy, cx=cx, cy=cy, fl_x=fl_x, fl_y=fl_y)

    fovx = float(base_cam.FoVx) if base_cam.FoVx > 0 else math.radians(60.0)
    fovy = float(base_cam.FoVy) if base_cam.FoVy > 0 else 2.0 * math.atan(math.tan(fovx * 0.5) * (height / max(1.0, width)))
    return Intrinsics(width=width, height=height, fovx=fovx, fovy=fovy)


class LiveCamera:
    """Minimal mutable camera object accepted by gaussian_renderer.render."""

    def __init__(self, intr: Intrinsics, device: torch.device, dtype: torch.dtype = torch.float32):
        from utils.graphics_utils import getProjectionMatrix, getProjectionMatrixCenterShift

        self.uid = 0
        self.colmap_id = 0
        self.image_name = "interactive_view"
        self.image_width = int(intr.width)
        self.image_height = int(intr.height)
        self.resolution = (self.image_width, self.image_height)
        self.FoVx = float(intr.fovx)
        self.FoVy = float(intr.fovy)
        self.cx = float(intr.cx)
        self.cy = float(intr.cy)
        self.fl_x = float(intr.fl_x)
        self.fl_y = float(intr.fl_y)
        self.znear = 0.01
        self.zfar = 100.0
        self.timestamp = 0.0
        self.data_device = device
        self._dtype = dtype
        self.R = np.eye(3, dtype=np.float32)
        self.T = np.zeros(3, dtype=np.float32)

        if intr.has_center_shift:
            proj = getProjectionMatrixCenterShift(
                self.znear,
                self.zfar,
                self.cx,
                self.cy,
                self.fl_x,
                self.fl_y,
                self.image_width,
                self.image_height,
            )
        else:
            proj = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy)
        self.projection_matrix = proj.transpose(0, 1).to(device=device, dtype=dtype)
        self.world_view_transform = torch.eye(4, device=device, dtype=dtype)
        self.full_proj_transform = torch.eye(4, device=device, dtype=dtype)
        self.camera_center = torch.zeros(3, device=device, dtype=dtype)

    def set_intrinsics(self, intr: Intrinsics) -> None:
        from utils.graphics_utils import getProjectionMatrix, getProjectionMatrixCenterShift

        self.image_width = int(intr.width)
        self.image_height = int(intr.height)
        self.resolution = (self.image_width, self.image_height)
        self.FoVx = float(intr.fovx)
        self.FoVy = float(intr.fovy)
        self.cx = float(intr.cx)
        self.cy = float(intr.cy)
        self.fl_x = float(intr.fl_x)
        self.fl_y = float(intr.fl_y)
        if intr.has_center_shift:
            proj = getProjectionMatrixCenterShift(
                self.znear,
                self.zfar,
                self.cx,
                self.cy,
                self.fl_x,
                self.fl_y,
                self.image_width,
                self.image_height,
            )
        else:
            proj = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy)
        self.projection_matrix = proj.transpose(0, 1).to(device=self.data_device, dtype=self._dtype)

    def update(self, eye: np.ndarray, forward: np.ndarray, timestamp: float) -> None:
        from utils.graphics_utils import getWorld2View2

        R, T = rt_from_eye_forward(eye, forward)
        self.R = R
        self.T = T
        self.timestamp = float(timestamp)
        world_view = torch.as_tensor(
            getWorld2View2(R, T), device=self.data_device, dtype=self._dtype
        ).transpose(0, 1)
        self.world_view_transform = world_view
        self.full_proj_transform = (
            world_view.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0)
        )
        self.camera_center = world_view.inverse()[3, :3]

    def update_angles(self, eye: np.ndarray, yaw: float, pitch: float, timestamp: float) -> None:
        self.update_orientation(eye, orientation_from_forward(forward_from_yaw_pitch(yaw, pitch)), timestamp)

    def update_orientation(self, eye: np.ndarray, orientation: np.ndarray, timestamp: float) -> None:
        from utils.graphics_utils import getWorld2View2

        R, T = rt_from_eye_orientation(eye, orientation)
        self.R = R
        self.T = T
        self.timestamp = float(timestamp)
        world_view = torch.as_tensor(
            getWorld2View2(R, T), device=self.data_device, dtype=self._dtype
        ).transpose(0, 1)
        self.world_view_transform = world_view
        self.full_proj_transform = (
            world_view.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0)
        )
        self.camera_center = world_view.inverse()[3, :3]

    def get_rays(self):
        # Used only when rendering a learned environment map. Keep it lazy so the
        # normal raster path does not allocate a full HxW grid every frame.
        ys, xs = torch.meshgrid(
            torch.arange(self.image_height, device=self.data_device, dtype=self._dtype) + 0.5,
            torch.arange(self.image_width, device=self.data_device, dtype=self._dtype) + 0.5,
            indexing="ij",
        )
        fl_x = self.fl_x if self.fl_x > 0 else self.image_width / (2.0 * math.tan(self.FoVx * 0.5))
        fl_y = self.fl_y if self.fl_y > 0 else self.image_height / (2.0 * math.tan(self.FoVy * 0.5))
        cx = self.cx if self.cx > 0 else self.image_width * 0.5
        cy = self.cy if self.cy > 0 else self.image_height * 0.5
        pts_view = torch.stack(
            [(ys - cy) / fl_y, (xs - cx) / fl_x, torch.ones_like(xs), torch.ones_like(xs)],
            dim=-1,
        )
        c2w = torch.linalg.inv(self.world_view_transform.transpose(0, 1))
        pts_world = pts_view @ c2w.T
        directions = pts_world[..., :3] - self.camera_center[None, None, :]
        return self.camera_center[None, None, :], directions / torch.norm(directions, dim=-1, keepdim=True).clamp_min(1e-8)


@dataclass
class ViewerState:
    repo_root: Path
    model_file: Path
    checkpoint_iteration: int
    render_fn: object
    gaussians: object
    pipe: object
    background: torch.Tensor
    scene_meta: dict
    train_cameras: list[SimpleNamespace]
    test_cameras: list[SimpleNamespace]
    selected_cameras: list[SimpleNamespace]
    base_camera: SimpleNamespace
    intrinsics: Intrinsics
    time_start: float
    time_end: float
    time_value: float
    white_background: bool
    cameras_extent: float


def load_viewer_state(args: argparse.Namespace) -> ViewerState:
    if not torch.cuda.is_available():
        raise RuntimeError("interactive_viewer.py requires CUDA because the Gaussian rasterizers are CUDA extensions.")

    repo_root = Path(args.repo_root).resolve()
    model_file = Path(args.model_file).resolve()
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    sys.path.insert(0, str(repo_root))

    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render as render_fn
    from scene.gaussian_model import GaussianModel
    from utils.checkpoint_utils import checkpoint_args, load_checkpoint

    checkpoint_payload = load_checkpoint(model_file, map_location="cuda")
    loaded_iter = int(checkpoint_payload["iteration"])

    dummy_parser = argparse.ArgumentParser(add_help=False)
    ModelParams(dummy_parser)
    pp = PipelineParams(dummy_parser)
    defaults = dummy_parser.parse_args([])
    resolved = argparse.Namespace(**vars(defaults))
    vars(resolved).update(vars(checkpoint_args(checkpoint_payload)))

    pipe = pp.extract(resolved)
    if args.temporal_mask_threshold is not None:
        pipe.temporal_mask_threshold = float(args.temporal_mask_threshold)
    if args.temporal_mask_keyframes is not None:
        pipe.temporal_mask_keyframes = int(args.temporal_mask_keyframes)
    if args.temporal_mask_window is not None:
        pipe.temporal_mask_window = int(args.temporal_mask_window)
    if args.temporal_mask_mode is not None:
        pipe.temporal_mask_mode = str(args.temporal_mask_mode)
    if args.sort_free_render is not None:
        pipe.sort_free_render = bool(args.sort_free_render)

    run_config = checkpoint_payload.get("run_config", {})
    gaussian_kwargs = dict(run_config.get("gaussian_kwargs", {}))
    if int(gaussian_kwargs.get("gaussian_dim", 4)) != 4:
        raise ValueError("Only 4D Gaussian checkpoints are supported by this viewer.")

    gaussians = GaussianModel(**gaussian_kwargs)
    gaussians.restore(checkpoint_payload["gaussians"], None)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    if hasattr(gaussians, "active_sh_degree_t"):
        gaussians.active_sh_degree_t = max(
            int(getattr(gaussians, "active_sh_degree_t", 0)),
            int(gaussian_kwargs.get("sh_degree_t", 0)),
        )

    scene_meta = dict(checkpoint_payload.get("scene", {}))
    train, test, selected = select_camera_metas(scene_meta, args.split)
    base = selected[0]
    if args.start_camera >= 0:
        base = selected[min(int(args.start_camera), len(selected) - 1)]

    time_duration = coerce_time_duration(gaussian_kwargs.get("time_duration", gaussians.time_duration))
    gaussians.time_duration = time_duration
    timestamp_values = np.asarray([float(c.timestamp) for c in selected], dtype=np.float32)
    default_t0 = float(np.min(timestamp_values)) if timestamp_values.size else float(time_duration[0])
    default_t1 = float(np.max(timestamp_values)) if timestamp_values.size else float(time_duration[1])
    if abs(default_t1 - default_t0) < 1e-8:
        default_t0, default_t1 = float(time_duration[0]), float(time_duration[1])
    time_start = default_t0 if args.time_start is None else float(args.time_start)
    time_end = default_t1 if args.time_end is None else float(args.time_end)
    if args.freeze_time is not None:
        time_value = clamp_time(float(args.freeze_time), time_start, time_end)
    else:
        time_value = clamp_time(float(base.timestamp), time_start, time_end)

    intr = intrinsics_from_base(base, args)
    white_background = bool(scene_meta.get("white_background", False))
    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    return ViewerState(
        repo_root=repo_root,
        model_file=model_file,
        checkpoint_iteration=loaded_iter,
        render_fn=render_fn,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        scene_meta=scene_meta,
        train_cameras=train,
        test_cameras=test,
        selected_cameras=selected,
        base_camera=base,
        intrinsics=intr,
        time_start=time_start,
        time_end=time_end,
        time_value=time_value,
        white_background=white_background,
        cameras_extent=float(scene_meta.get("cameras_extent", 1.0)),
    )


def print_controls() -> None:
    print(
        "\nInteractive controls\n"
        "  Mouse        look around: left/right yaw, up/down pitch, no pitch clamp\n"
        "  R/T          roll left/right around the current view direction\n"
        "  W/S or ↑/↓   move forward/back along the current look direction\n"
        "  A/D or ←/→   strafe left/right relative to the current look direction\n"
        "  Space/Ctrl   move up/down using the current vertical movement basis\n"
        "  U            toggle vertical movement basis: camera-local vs snapped/test-camera up\n"
        "  Q/E          scrub timestamp backward/forward\n"
        "  Shift        speed boost\n"
        "  +/-          change movement speed\n"
        "  P            pause/resume time playback\n"
        "  C / Shift+C  snap next/previous camera in the selected split\n"
        "  F1/F2        snap previous/next training camera\n"
        "  F3/F4        snap previous/next test camera\n"
        "  I            toggle info overlay: FPS, VRAM, gaussian count, pose, time\n"
        "  F            toggle fullscreen and render at fullscreen size if VRAM allows\n"
        "  Resize       render at the current window size if VRAM allows\n"
        "  Home         reset pose to the checkpoint camera\n"
        "  Tab          toggle mouse capture\n"
        "  H            print controls\n"
        "  Esc          quit\n"
    )


def set_mouse_capture(pygame, enabled: bool) -> bool:
    pygame.event.set_grab(enabled)
    pygame.mouse.set_visible(not enabled)
    relative_enabled = False
    if hasattr(pygame.mouse, "set_relative_mode"):
        try:
            pygame.mouse.set_relative_mode(enabled)
            relative_enabled = bool(enabled)
        except Exception:
            relative_enabled = False
    pygame.mouse.get_rel()
    return relative_enabled


def display_flags(pygame, fullscreen: bool) -> int:
    flags = pygame.HWSURFACE | pygame.DOUBLEBUF
    if fullscreen:
        flags |= pygame.FULLSCREEN
    else:
        flags |= getattr(pygame, "RESIZABLE", 0)
    return flags


def make_display(pygame, size: tuple[int, int], fullscreen: bool, vsync: bool):
    # In fullscreen, (0, 0) requests the desktop resolution from SDL.  We then
    # read screen.get_size() and resize the Gaussian render target to match.
    mode_size = (0, 0) if fullscreen else (max(16, int(size[0])), max(16, int(size[1])))
    return pygame.display.set_mode(mode_size, display_flags(pygame, fullscreen), vsync=1 if vsync else 0)


def blit_surface_preserve_aspect(pygame, screen, surface, display_size: tuple[int, int]) -> None:
    """Blit a fallback render without stretching it to the wrong aspect ratio."""
    sw, sh = surface.get_size()
    dw, dh = max(1, int(display_size[0])), max(1, int(display_size[1]))
    if (sw, sh) == (dw, dh):
        screen.blit(surface, (0, 0))
        return

    scale = min(dw / max(1, sw), dh / max(1, sh))
    target_w = max(1, int(round(sw * scale)))
    target_h = max(1, int(round(sh * scale)))
    x = (dw - target_w) // 2
    y = (dh - target_h) // 2
    screen.fill((0, 0, 0))
    screen.blit(pygame.transform.smoothscale(surface, (target_w, target_h)), (x, y))


def recenter_mouse_if_needed(pygame, app) -> None:
    """Center only for the non-relative fallback path.

    Calling set_pos every frame injects synthetic mouse deltas on some SDL
    backends, which feels like jitter/rubber-banding.  The normal path uses SDL
    relative mouse mode and never warps the pointer.  This fallback recenters
    only immediately after consuming an absolute mouse offset.
    """
    if app.get("mouse_grab", False) and not app.get("relative_mouse", False):
        w, h = app.get("display_size", (0, 0))
        if w > 0 and h > 0:
            pygame.mouse.set_pos((int(w) // 2, int(h) // 2))
            pygame.mouse.get_rel()
            try:
                pygame.event.clear(pygame.MOUSEMOTION)
            except Exception:
                pass


def refresh_mouse_capture(pygame, app) -> None:
    app["relative_mouse"] = set_mouse_capture(pygame, bool(app.get("mouse_grab", False)))
    app["mouse_dx"] = 0.0
    app["mouse_dy"] = 0.0
    if not app["relative_mouse"]:
        recenter_mouse_if_needed(pygame, app)


def set_render_target_size(
    pygame,
    state: ViewerState,
    live_cam: LiveCamera,
    app,
    size: tuple[int, int],
    reason: str = "resize",
) -> None:
    width = max(16, int(size[0]))
    height = max(16, int(size[1]))
    new_size = (width, height)
    if tuple(app.get("render_size", (0, 0))) == new_size:
        return

    base_intrinsics = app.get("base_intrinsics", state.intrinsics)
    state.intrinsics = resized_intrinsics(base_intrinsics, width, height)
    live_cam.set_intrinsics(state.intrinsics)
    app["render_size"] = new_size
    app["surface"] = pygame.Surface(new_size)
    app["resize_status"] = f"rendering {width}x{height} after {reason}"
    print(f"Render target resized to {width}x{height} ({reason}).")


def resize_window_and_render_target(
    pygame,
    state: ViewerState,
    live_cam: LiveCamera,
    app,
    size: tuple[int, int],
    reason: str = "resize",
) -> None:
    width = max(16, int(size[0]))
    height = max(16, int(size[1]))
    if not bool(app.get("fullscreen", False)):
        app["screen"] = make_display(pygame, (width, height), False, app["vsync"])
        app["windowed_size"] = app["screen"].get_size()
    app["display_size"] = app["screen"].get_size()
    set_render_target_size(pygame, state, live_cam, app, app["display_size"], reason)
    refresh_mouse_capture(pygame, app)


def fallback_to_last_good_render_size(pygame, state: ViewerState, live_cam: LiveCamera, app, exc: BaseException) -> None:
    last_good_intr = app.get("last_good_intrinsics")
    if last_good_intr is None:
        raise exc
    state.intrinsics = copy_intrinsics(last_good_intr)
    live_cam.set_intrinsics(state.intrinsics)
    app["render_size"] = (state.intrinsics.width, state.intrinsics.height)
    app["surface"] = pygame.Surface(app["render_size"])
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    message = (
        "Could not render at the window/fullscreen size because CUDA ran out of VRAM. "
        f"Falling back to {state.intrinsics.width}x{state.intrinsics.height} and scaling it to the display."
    )
    app["resize_status"] = message
    print(message)


def toggle_fullscreen(pygame, state: ViewerState, live_cam: LiveCamera, app) -> None:
    app["fullscreen"] = not bool(app.get("fullscreen", False))
    if app["fullscreen"]:
        if not app.get("windowed_size"):
            app["windowed_size"] = app.get("display_size", app.get("render_size", (1280, 720)))
        app["screen"] = make_display(pygame, (0, 0), True, app["vsync"])
    else:
        app["screen"] = make_display(pygame, app.get("windowed_size", app.get("render_size", (1280, 720))), False, app["vsync"])

    app["display_size"] = app["screen"].get_size()
    set_render_target_size(pygame, state, live_cam, app, app["display_size"], "fullscreen" if app["fullscreen"] else "windowed")
    refresh_mouse_capture(pygame, app)


def apply_camera_pose(app, cam: SimpleNamespace, split_name: str, index: int) -> None:
    app["eye"] = camera_center_from_rt(cam.R, cam.T)
    orientation = orientation_from_rt(cam.R)
    app["orientation"] = orientation
    app["active_camera_up"] = orientation[:, 1].copy()
    app["time_value"] = clamp_time(float(cam.timestamp), app["time_start"], app["time_end"])
    app["snap_split"] = split_name
    app["snap_index"] = int(index)
    app["snap_name"] = str(cam.image_name)
    refresh_locked_vertical_basis(app)


def snap_camera(app, cameras: list[SimpleNamespace], split_name: str, direction: int) -> None:
    if not cameras:
        print(f"No {split_name} cameras are available in this checkpoint.")
        return
    current_split = app.get("snap_split")
    current_index = int(app.get("snap_index", -1))
    if current_split != split_name or current_index < 0 or current_index >= len(cameras):
        current_index = -1 if direction >= 0 else 0
    next_index = (current_index + int(direction)) % len(cameras)
    apply_camera_pose(app, cameras[next_index], split_name, next_index)
    print(f"Snapped to {split_name} camera {next_index + 1}/{len(cameras)}: {app['snap_name']} t={app['time_value']:.6f}")


def vertical_basis_label(app) -> str:
    return str(app.get("nav_up_label", "camera-local"))


def refresh_locked_vertical_basis(app) -> None:
    if app.get("nav_up_mode", "camera") != "locked":
        return
    up = normalize(app.get("active_camera_up", normalize_orientation(app["orientation"])[:, 1]))
    if np.linalg.norm(up) < 1e-6:
        up = normalize_orientation(app["orientation"])[:, 1]
    app["nav_up_axis"] = up.copy()
    split = str(app.get("snap_split", "snapped"))
    app["nav_up_label"] = f"{split} camera-up"


def toggle_vertical_basis(app) -> None:
    current_mode = str(app.get("nav_up_mode", "camera"))
    if current_mode == "camera":
        up = normalize(app.get("active_camera_up", normalize_orientation(app["orientation"])[:, 1]))
        if np.linalg.norm(up) < 1e-6:
            up = normalize_orientation(app["orientation"])[:, 1]
        app["nav_up_mode"] = "locked"
        app["nav_up_axis"] = up.copy()
        split = str(app.get("snap_split", "snapped"))
        app["nav_up_label"] = f"{split} camera-up"
        print(f"Vertical movement now uses {vertical_basis_label(app)}: Space moves +up, Ctrl moves -up.")
    else:
        app["nav_up_mode"] = "camera"
        app["nav_up_axis"] = None
        app["nav_up_label"] = "camera-local"
        print("Vertical movement now uses camera-local up/down.")


def vertical_direction(app, orientation: np.ndarray) -> np.ndarray:
    if app.get("nav_up_mode", "camera") == "locked":
        axis = app.get("nav_up_axis")
        if axis is not None:
            axis = normalize(axis)
            if np.linalg.norm(axis) > 1e-6:
                return axis
    return orientation[:, 1]


def handle_discrete_key(event_key: int, app, state: ViewerState, live_cam: LiveCamera, mods: int = 0) -> None:
    import pygame

    shift_down = bool(mods & (pygame.KMOD_SHIFT | pygame.KMOD_LSHIFT | pygame.KMOD_RSHIFT))

    if event_key == pygame.K_ESCAPE:
        app["running"] = False
    elif event_key == pygame.K_TAB:
        app["mouse_grab"] = not app["mouse_grab"]
        refresh_mouse_capture(pygame, app)
    elif event_key == pygame.K_p:
        app["paused"] = not app["paused"]
    elif event_key == pygame.K_i:
        app["show_info"] = not app["show_info"]
    elif event_key == pygame.K_f:
        toggle_fullscreen(pygame, state, live_cam, app)
    elif event_key == pygame.K_u:
        toggle_vertical_basis(app)
    elif event_key in (pygame.K_EQUALS, getattr(pygame, "K_PLUS", pygame.K_EQUALS), pygame.K_KP_PLUS):
        app["move_speed"] *= 1.25
    elif event_key in (pygame.K_MINUS, getattr(pygame, "K_UNDERSCORE", pygame.K_MINUS), pygame.K_KP_MINUS):
        app["move_speed"] /= 1.25
    elif event_key == pygame.K_HOME:
        app["eye"] = app["initial_eye"].copy()
        app["orientation"] = app["initial_orientation"].copy()
        app["active_camera_up"] = app["initial_orientation"][:, 1].copy()
        app["time_value"] = float(app["initial_time"])
        app["snap_split"] = "initial"
        app["snap_index"] = int(app.get("initial_snap_index", 0))
        app["snap_name"] = str(app.get("initial_snap_name", "initial"))
        refresh_locked_vertical_basis(app)
    elif event_key == pygame.K_c:
        snap_camera(app, state.selected_cameras, str(app.get("selected_split_label", "selected")), -1 if shift_down else 1)
    elif event_key == pygame.K_F1:
        snap_camera(app, state.train_cameras, "train", -1)
    elif event_key == pygame.K_F2:
        snap_camera(app, state.train_cameras, "train", 1)
    elif event_key == pygame.K_F3:
        snap_camera(app, state.test_cameras, "test", -1)
    elif event_key == pygame.K_F4:
        snap_camera(app, state.test_cameras, "test", 1)
    elif event_key == pygame.K_h:
        print_controls()


def update_motion(keys, app, dt: float) -> None:
    import pygame

    # True 6DOF game-style translation/roll from the current camera basis.
    # No Euler pitch/yaw state is used here, so nothing clamps at vertical look.
    orientation = normalize_orientation(app["orientation"])
    right = orientation[:, 0]
    up = vertical_direction(app, orientation)
    forward = orientation[:, 2]

    roll_step = float(app["roll_speed"]) * float(dt)
    if keys[pygame.K_r]:
        orientation = rotate_orientation(orientation, forward, -roll_step)
    if keys[pygame.K_t]:
        orientation = rotate_orientation(orientation, forward, roll_step)
    app["orientation"] = orientation
    right = orientation[:, 0]
    up = vertical_direction(app, orientation)
    forward = orientation[:, 2]

    move = np.zeros(3, dtype=np.float32)
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        move += forward
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        move -= forward
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        move += right
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        move -= right
    if keys[pygame.K_SPACE]:
        move += up
    if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
        move -= up

    if np.linalg.norm(move) > 1e-6:
        speed = app["move_speed"] * (app["boost"] if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.0)
        app["eye"] = app["eye"] + normalize(move) * float(speed * dt)

    time_step = app["time_scrub_step"] * dt
    if keys[pygame.K_q]:
        app["time_value"] = clamp_time(app["time_value"] - time_step, app["time_start"], app["time_end"])
    if keys[pygame.K_e]:
        app["time_value"] = clamp_time(app["time_value"] + time_step, app["time_start"], app["time_end"])


def cuda_memory_text() -> str:
    if not torch.cuda.is_available():
        return "VRAM n/a"
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 2)
    return f"VRAM alloc/res/peak/total {allocated:.0f}/{reserved:.0f}/{peak:.0f}/{total:.0f} MB"


def collect_info_lines(state: ViewerState, app, clock, render_ms_ema: float | None) -> list[str]:
    xyz = state.gaussians.get_xyz
    gaussian_count = int(xyz.shape[0]) if hasattr(xyz, "shape") else -1
    eye = app["eye"]
    render_ms_text = "n/a" if render_ms_ema is None else f"{render_ms_ema:.1f} ms"
    display_size = tuple(app.get("display_size", (state.intrinsics.width, state.intrinsics.height)))
    render_size = tuple(app.get("render_size", (state.intrinsics.width, state.intrinsics.height)))
    lines = [
        f"FPS {clock.get_fps():.1f} | render {render_ms_text} | target {app['target_fps']}",
        cuda_memory_text(),
        f"Gaussians {gaussian_count:,} | render {render_size[0]}x{render_size[1]} | display {display_size[0]}x{display_size[1]}",
        f"time {app['time_value']:.6f} / [{app['time_start']:.6f}, {app['time_end']:.6f}] {'paused' if app['paused'] else 'playing'}",
        f"pose {app.get('snap_split', 'free')} #{int(app.get('snap_index', 0)) + 1}: {app.get('snap_name', '')}",
        f"eye [{eye[0]:.3f}, {eye[1]:.3f}, {eye[2]:.3f}] {orientation_debug_text(app['orientation'])}",
        f"speed {app['move_speed']:.3f} boost x{app['boost']:.1f} roll {math.degrees(app['roll_speed']):.1f} deg/s | vertical {vertical_basis_label(app)}",
    ]
    if app.get("resize_status"):
        lines.append(str(app["resize_status"]))
    return lines


def draw_info_overlay(screen, font, lines: list[str]) -> None:
    import pygame

    padding = 8
    line_surfaces = [font.render(line, True, (255, 255, 255)) for line in lines]
    width = max(s.get_width() for s in line_surfaces) + padding * 2
    height = sum(s.get_height() for s in line_surfaces) + padding * 2 + max(0, len(line_surfaces) - 1) * 2
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 165))
    y = padding
    for surf in line_surfaces:
        overlay.blit(surf, (padding, y))
        y += surf.get_height() + 2
    screen.blit(overlay, (10, 10))


def run_viewer(args: argparse.Namespace) -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    state = load_viewer_state(args)

    centers = np.stack([camera_center_from_rt(c.R, c.T) for c in state.selected_cameras], axis=0)
    forwards = np.stack([camera_forward_from_r(c.R) for c in state.selected_cameras], axis=0)
    scene_center = closest_point_to_rays(centers, forwards)
    if not np.all(np.isfinite(scene_center)):
        scene_center = np.median(centers, axis=0).astype(np.float32)

    initial_eye = camera_center_from_rt(state.base_camera.R, state.base_camera.T)
    initial_orientation = orientation_from_rt(state.base_camera.R)
    move_speed = float(args.move_speed) if args.move_speed is not None else max(0.05, state.cameras_extent * 0.75)
    time_span = max(1e-6, abs(state.time_end - state.time_start))

    base_intrinsics = copy_intrinsics(state.intrinsics)
    live_cam = LiveCamera(state.intrinsics, device=torch.device("cuda"), dtype=torch.float32)
    pygame.init()
    display_size = (state.intrinsics.width, state.intrinsics.height)
    screen = make_display(pygame, display_size, fullscreen=False, vsync=bool(args.vsync))
    display_size = screen.get_size()
    if display_size != (state.intrinsics.width, state.intrinsics.height):
        state.intrinsics = resized_intrinsics(base_intrinsics, display_size[0], display_size[1])
        live_cam.set_intrinsics(state.intrinsics)
    render_size = display_size
    surface = pygame.Surface(render_size)
    pygame.display.set_caption("4DGS Interactive Viewer - loading")
    font = pygame.font.Font(None, 18)

    app = {
        "running": True,
        "mouse_grab": True,
        "show_info": bool(args.show_info),
        "paused": bool(args.pause_time or args.freeze_time is not None or args.time_speed == 0.0),
        "eye": initial_eye.copy(),
        "orientation": initial_orientation.copy(),
        "initial_eye": initial_eye.copy(),
        "initial_orientation": initial_orientation.copy(),
        "active_camera_up": initial_orientation[:, 1].copy(),
        "nav_up_mode": "camera",
        "nav_up_axis": None,
        "nav_up_label": "camera-local",
        "time_value": float(state.time_value),
        "initial_time": float(state.time_value),
        "time_start": float(state.time_start),
        "time_end": float(state.time_end),
        "move_speed": move_speed,
        "boost": float(args.boost),
        "roll_speed": float(args.roll_speed),
        "time_scrub_step": time_span * float(args.scrub_seconds_per_second),
        "target_fps": int(args.fps),
        "display_size": display_size,
        "windowed_size": display_size,
        "render_size": render_size,
        "screen": screen,
        "surface": surface,
        "last_good_intrinsics": copy_intrinsics(state.intrinsics),
        "base_intrinsics": base_intrinsics,
        "fullscreen": False,
        "vsync": bool(args.vsync),
        "relative_mouse": False,
        "mouse_dx": 0.0,
        "mouse_dy": 0.0,
        "selected_split_label": str(args.split if args.split != "all" else "selected"),
        "snap_split": str(args.split if args.split != "all" else "selected"),
        "snap_index": int(args.start_camera if args.start_camera >= 0 else 0),
        "snap_name": str(state.base_camera.image_name),
        "initial_snap_index": int(args.start_camera if args.start_camera >= 0 else 0),
        "initial_snap_name": str(state.base_camera.image_name),
    }
    refresh_mouse_capture(pygame, app)

    print(
        f"Loaded {state.model_file}\n"
        f"  iteration={state.checkpoint_iteration} gaussians={int(state.gaussians.get_xyz.shape[0])}\n"
        f"  resolution={state.intrinsics.width}x{state.intrinsics.height} target_fps={args.fps}\n"
        f"  time=[{state.time_start:.6f}, {state.time_end:.6f}] start={state.time_value:.6f}\n"
        f"  move_speed={move_speed:.4f} scene_center={scene_center.tolist()}"
    )
    print_controls()

    clock = pygame.time.Clock()
    last_title_update = 0.0
    render_ms_ema = None
    try:
        while app["running"]:
            dt = clock.tick(int(args.fps)) / 1000.0
            if dt <= 0.0:
                dt = 1.0 / max(1.0, float(args.fps))
            dt = min(dt, 0.1)

            app["mouse_dx"] = 0.0
            app["mouse_dy"] = 0.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    app["running"] = False
                elif event.type == pygame.VIDEORESIZE:
                    resize_window_and_render_target(
                        pygame,
                        state,
                        live_cam,
                        app,
                        (getattr(event, "w", app["display_size"][0]), getattr(event, "h", app["display_size"][1])),
                        "window resize",
                    )
                elif event.type == pygame.KEYDOWN:
                    handle_discrete_key(event.key, app, state, live_cam, getattr(event, "mod", 0))
                elif event.type == pygame.MOUSEMOTION and app["mouse_grab"]:
                    rel = getattr(event, "rel", (0, 0))
                    app["mouse_dx"] += float(rel[0])
                    app["mouse_dy"] += float(rel[1])

            if app["mouse_grab"]:
                dx = float(app.get("mouse_dx", 0.0))
                dy = float(app.get("mouse_dy", 0.0))

                if not app.get("relative_mouse", False):
                    # Absolute-position fallback for platforms without SDL
                    # relative mouse mode.  Read offset from center, then warp
                    # once and immediately drop the synthetic motion event.
                    w, h = app.get("display_size", (0, 0))
                    if w > 0 and h > 0:
                        cx, cy = int(w) // 2, int(h) // 2
                        mx, my = pygame.mouse.get_pos()
                        dx += float(mx - cx)
                        dy += float(my - cy)
                        if mx != cx or my != cy:
                            pygame.mouse.set_pos((cx, cy))
                            pygame.mouse.get_rel()
                            try:
                                pygame.event.clear(pygame.MOUSEMOTION)
                            except Exception:
                                pass

                if dx or dy:
                    orientation = normalize_orientation(app["orientation"])
                    # Mouse X yaws around the current camera-up axis.
                    # Mouse Y pitches around the current camera-right axis.
                    # No Euler angle is stored, so pitch can pass vertical and spin 360 degrees.
                    if dx:
                        orientation = rotate_orientation(orientation, orientation[:, 1], dx * float(args.mouse_sensitivity))
                    if dy:
                        orientation = rotate_orientation(orientation, orientation[:, 0], dy * float(args.mouse_sensitivity))
                    app["orientation"] = orientation
            else:
                pygame.mouse.get_rel()

            keys = pygame.key.get_pressed()
            update_motion(keys, app, dt)
            if not app["paused"]:
                app["time_value"] = wrap_time(
                    app["time_value"] + float(args.time_speed) * dt,
                    app["time_start"],
                    app["time_end"],
                )

            live_cam.update_orientation(app["eye"], app["orientation"], app["time_value"])

            t_render0 = time.perf_counter()
            try:
                with torch.inference_mode():
                    render_pkg = state.render_fn(live_cam, state.gaussians, state.pipe, state.background)
                    frame = render_tensor_to_uint8(render_pkg["render"])
            except RuntimeError as exc:
                if is_cuda_oom(exc):
                    fallback_to_last_good_render_size(pygame, state, live_cam, app, exc)
                    continue
                raise
            render_ms = (time.perf_counter() - t_render0) * 1000.0
            render_ms_ema = render_ms if render_ms_ema is None else (0.9 * render_ms_ema + 0.1 * render_ms)
            app["last_good_intrinsics"] = copy_intrinsics(state.intrinsics)

            # pygame expects W x H x C for surfarray.blit_array.
            surface = app["surface"]
            pygame.surfarray.blit_array(surface, np.transpose(frame, (1, 0, 2)))
            screen = app["screen"]
            display_size = screen.get_size()
            app["display_size"] = display_size
            blit_surface_preserve_aspect(pygame, screen, surface, display_size)
            if app["show_info"]:
                draw_info_overlay(screen, font, collect_info_lines(state, app, clock, render_ms_ema))
            pygame.display.flip()

            now = time.perf_counter()
            if now - last_title_update > 0.25:
                pygame.display.set_caption(
                    "4DGS Interactive Viewer | "
                    f"{clock.get_fps():5.1f} fps | render {render_ms_ema:5.1f} ms | "
                    f"t={app['time_value']:.4f} | speed={app['move_speed']:.3f} | vertical={vertical_basis_label(app)}"
                )
                last_title_update = now
    finally:
        set_mouse_capture(pygame, False)
        pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open a 60 FPS interactive WASD/mouse viewer for a self-contained 4DGS checkpoint."
    )
    parser.add_argument("--repo_root", type=str, default=".", help="Repository root containing gaussian_renderer, scene, utils, etc.")
    parser.add_argument("--model_file", type=str, required=True, help="Path to a self-contained checkpoint .pth")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test", help="Camera split used for initial pose and intrinsics")
    parser.add_argument("--start_camera", type=int, default=0, help="Index within the selected split to use as the starting pose")
    parser.add_argument("--width", type=int, default=0, help="Window/render width. Defaults to checkpoint camera width.")
    parser.add_argument("--height", type=int, default=0, help="Window/render height. Defaults to checkpoint camera height.")
    parser.add_argument("--fov_y_deg", type=float, default=None, help="Override vertical field of view in degrees.")
    parser.add_argument("--fps", type=int, default=60, help="Target window frame rate.")
    parser.add_argument("--vsync", action="store_true", help="Ask pygame/SDL for vsync when creating the window.")

    parser.add_argument("--move_speed", type=float, default=None, help="World units per second. Defaults to 0.75 * cameras_extent.")
    parser.add_argument("--boost", type=float, default=4.0, help="Shift-key movement multiplier.")
    parser.add_argument("--mouse_sensitivity", type=float, default=0.002, help="Radians per mouse pixel.")
    parser.add_argument("--roll_speed", type=float, default=1.5, help="Radians per second for R/T roll controls.")
    parser.add_argument("--show_info", action="store_true", help="Start with the FPS/VRAM/Gaussian-count overlay visible.")

    parser.add_argument("--time_start", type=float, default=None)
    parser.add_argument("--time_end", type=float, default=None)
    parser.add_argument("--freeze_time", type=float, default=None, help="Start at and pause a fixed timestamp.")
    parser.add_argument("--time_speed", type=float, default=0.2, help="Checkpoint time units per second; set 0 to keep time frozen.")
    parser.add_argument("--pause_time", action="store_true", help="Start with time playback paused.")
    parser.add_argument(
        "--scrub_seconds_per_second",
        type=float,
        default=0.35,
        help="Fraction of the time range scrubbed per held bracket key second.",
    )

    parser.add_argument("--temporal_mask_threshold", type=float, default=None)
    parser.add_argument("--temporal_mask_keyframes", type=int, default=None)
    parser.add_argument("--temporal_mask_window", type=int, default=None)
    parser.add_argument("--temporal_mask_mode", choices=["marginal", "visibility"], default=None)
    parser.add_argument("--sort_free_render", dest="sort_free_render", action="store_true", default=None)
    parser.add_argument("--no_sort_free_render", dest="sort_free_render", action="store_false")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_viewer(args)


if __name__ == "__main__":
    main()
