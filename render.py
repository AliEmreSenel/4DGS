#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
import queue
import sys
import threading
from dataclasses import dataclass
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch


@dataclass
class InFlightFrame:
    idx: int
    t: float
    ready_event: torch.cuda.Event
    cpu_image_u8: torch.Tensor
    # Keep GPU tensors/camera alive until the event signals completion.
    keepalive: tuple[Any, ...]


class AsyncFrameWriter:
    def __init__(self, video_path: Path, fps: int, frames_dir: Path, save_png: bool, max_queue: int, total_frames: int):
        self.frames_dir = frames_dir
        self.save_png = save_png
        self.total_frames = total_frames
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, max_queue))
        self._exception: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            args=(video_path, fps),
            daemon=True,
        )
        self._thread.start()

    def _run(self, video_path: Path, fps: int):
        writer = None
        try:
            writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8)
            while True:
                item = self._queue.get()
                if item is None:
                    break
                idx, t, image = item
                if self.save_png:
                    frame_path = self.frames_dir / f"{idx:04d}.png"
                    imageio.imwrite(frame_path, image)
                writer.append_data(image)
                if (idx + 1) % 25 == 0 or idx == 0 or (idx + 1) == self.total_frames:
                    print(f"[{idx+1}/{self.total_frames}] queued frame written t={t:.4f}")
        except Exception as e:
            self._exception = e
        finally:
            if writer is not None:
                writer.close()

    def submit(self, idx: int, t: float, image: np.ndarray):
        while True:
            if self._exception is not None:
                raise RuntimeError("Asynchronous frame writer failed") from self._exception
            try:
                self._queue.put((idx, t, image), timeout=0.2)
                return
            except queue.Full:
                continue

    def close(self):
        while True:
            if self._exception is not None:
                break
            try:
                self._queue.put(None, timeout=0.2)
                break
            except queue.Full:
                continue
        self._thread.join()
        if self._exception is not None:
            raise RuntimeError("Asynchronous frame writer failed") from self._exception


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
    try:
        obj = eval(text, safe_globals, safe_locals)
    except Exception as e:
        raise ValueError(f"Could not parse cfg_args from {cfg_path}: {e}") from e

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


def look_at_colmap(eye, target, world_up=np.array([0.0, 0.0, 1.0], dtype=np.float32)):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
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


def main():
    parser = argparse.ArgumentParser(
        description="Render a 360-over-time video from a 4DGS checkpoint"
    )
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--source_path", type=str, default=None)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--radius_scale", type=float, default=1.0)
    parser.add_argument("--elevation_deg", type=float, default=15.0)
    parser.add_argument("--look_at_offset_y", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--time_start", type=float, default=None)
    parser.add_argument("--time_end", type=float, default=None)
    parser.add_argument(
        "--time_mode",
        type=str,
        choices=["orbit-time", "orbit-only", "time-only"],
        default="orbit-time",
    )
    parser.add_argument("--freeze_time", type=float, default=None)
    parser.add_argument("--orbit_start_deg", type=float, default=0.0)
    parser.add_argument("--orbit_end_deg", type=float, default=360.0)
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--isotropic_gaussians", action="store_true")
    parser.add_argument(
        "--render_concurrency",
        type=int,
        default=0,
        help="Max in-flight render jobs (0 = auto).",
    )
    parser.add_argument(
        "--render_streams",
        type=int,
        default=1,
        help="CUDA streams for dispatching render jobs. Values >1 are experimental.",
    )
    parser.add_argument(
        "--writer_queue",
        type=int,
        default=16,
        help="Max queued frames waiting to be encoded/written.",
    )
    parser.add_argument(
        "--skip_png",
        action="store_true",
        help="Skip per-frame PNG files and only write MP4.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    model_path = Path(args.model_path).resolve()
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else model_path / "renders" / "orbit_time"
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
        ckpt_path,
        map_location=map_location,
        weights_only=False,
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

    if not os.path.exists(dataset.source_path):
        raise FileNotFoundError(
            f"Dataset source_path does not exist: {dataset.source_path}. "
            f"Pass --source_path or --config configs/...yaml."
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

    cam_list = (
        scene.test_cameras[1.0]
        if args.split == "test" and len(scene.test_cameras[1.0]) > 0
        else scene.train_cameras[1.0]
    )
    if len(cam_list) == 0:
        raise RuntimeError("No cameras found in the selected split")

    centers = np.stack([cam.camera_center.cpu().numpy() for cam in cam_list], axis=0)
    center = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    radius = (
        np.quantile(np.linalg.norm(centers - center[None], axis=1), 0.9)
        * args.radius_scale
    )
    if not np.isfinite(radius) or radius <= 1e-6:
        radius = max(scene.cameras_extent, 1.0) * args.radius_scale

    base_cam = cam_list[0]
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

    time_values = np.array([float(cam.timestamp) for cam in cam_list], dtype=np.float32)
    default_time_start = (
        float(np.min(time_values)) if len(time_values) else float(time_duration[0])
    )
    default_time_end = (
        float(np.max(time_values)) if len(time_values) else float(time_duration[1])
    )
    time_start = default_time_start if args.time_start is None else args.time_start
    time_end = default_time_end if args.time_end is None else args.time_end

    save_png = not args.skip_png
    frames_dir = out_dir / "frames"
    if save_png:
        frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "orbit_time.mp4"

    elev = math.radians(args.elevation_deg)
    look_at = center.copy()
    look_at[1] += args.look_at_offset_y

    auto_concurrency = max(2, min(16, (os.cpu_count() or 8) // 2))
    max_in_flight = args.render_concurrency if args.render_concurrency > 0 else auto_concurrency
    max_in_flight = max(1, max_in_flight)
    num_streams = max(1, min(args.render_streams, max_in_flight))
    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    print(
        "Render pipeline settings: "
        f"in_flight={max_in_flight}, streams={num_streams}, writer_queue={args.writer_queue}, "
        f"save_png={save_png}"
    )

    def build_camera(idx: int):
        frac = 0.0 if args.frames == 1 else idx / (args.frames - 1)
        az = math.radians(
            args.orbit_start_deg + frac * (args.orbit_end_deg - args.orbit_start_deg)
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
                args.freeze_time
                if args.freeze_time is not None
                else 0.5 * (time_start + time_end)
            )
        elif args.time_mode == "time-only":
            eye = centers[0]
            t_val = time_start + frac * (time_end - time_start)
        else:
            t_val = time_start + frac * (time_end - time_start)

        if args.time_mode == "time-only":
            ref_cam = base_cam
            cam = Camera(
                colmap_id=0,
                R=ref_cam.R,
                T=ref_cam.T,
                FoVx=FoVx if not use_intrinsics else -1.0,
                FoVy=FoVy if not use_intrinsics else -1.0,
                image=torch.empty(0),
                gt_alpha_mask=None,
                image_name=f"orbit_{idx:04d}",
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
        else:
            R, T = look_at_colmap(eye, look_at)
            cam = Camera(
                colmap_id=0,
                R=R,
                T=T,
                FoVx=FoVx if not use_intrinsics else -1.0,
                FoVy=FoVy if not use_intrinsics else -1.0,
                image=torch.empty(0),
                gt_alpha_mask=None,
                image_name=f"orbit_{idx:04d}",
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

        return cam, float(t_val)

    def enqueue_render(idx: int) -> InFlightFrame:
        stream = streams[idx % len(streams)]
        cam, t_val = build_camera(idx)
        with torch.cuda.stream(stream):
            render_pkg = render(cam, gaussians, pipe, background)
            render_tensor = render_pkg["render"]
            gpu_image = torch.clamp(render_tensor, 0.0, 1.0).permute(1, 2, 0).contiguous()
            gpu_image_u8 = (gpu_image * 255.0).to(torch.uint8)
            cpu_image_u8 = torch.empty(
                gpu_image_u8.shape,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            cpu_image_u8.copy_(gpu_image_u8, non_blocking=True)

            ready_event = torch.cuda.Event(blocking=False)
            ready_event.record(stream)

        return InFlightFrame(
            idx=idx,
            t=t_val,
            ready_event=ready_event,
            cpu_image_u8=cpu_image_u8,
            keepalive=(cam, gpu_image_u8, render_tensor),
        )

    writer = AsyncFrameWriter(
        video_path=video_path,
        fps=args.fps,
        frames_dir=frames_dir,
        save_png=save_png,
        max_queue=args.writer_queue,
        total_frames=args.frames,
    )

    pending: dict[int, InFlightFrame] = {}
    next_to_write = 0

    def flush_ready(blocking: bool):
        nonlocal next_to_write
        while next_to_write in pending:
            current = pending[next_to_write]
            if not current.ready_event.query():
                if not blocking:
                    break
                current.ready_event.synchronize()

            image = current.cpu_image_u8.numpy().copy()
            writer.submit(current.idx, current.t, image)
            del pending[next_to_write]
            next_to_write += 1
            blocking = False

    try:
        for idx in range(args.frames):
            pending[idx] = enqueue_render(idx)
            flush_ready(blocking=False)
            if len(pending) >= max_in_flight:
                flush_ready(blocking=True)

        while pending:
            flush_ready(blocking=True)
    finally:
        writer.close()

    meta_path = out_dir / "render_info.txt"
    meta_path.write_text(
        f"checkpoint={ckpt_path}\n"
        f"loaded_iter={loaded_iter}\n"
        f"split={args.split}\n"
        f"frames={args.frames}\n"
        f"fps={args.fps}\n"
        f"center={center.tolist()}\n"
        f"radius={radius}\n"
        f"time_start={time_start}\n"
        f"time_end={time_end}\n"
        f"time_mode={args.time_mode}\n"
        f"gaussian_dim={gaussian_dim}\n"
        f"rot_4d={rot_4d}\n"
        f"sh_degree_t={sh_degree_t}\n"
        f"render_concurrency={max_in_flight}\n"
        f"render_streams={num_streams}\n"
        f"writer_queue={args.writer_queue}\n"
        f"save_png={save_png}\n"
    )
    print(f"Wrote video to {video_path}")
    if save_png:
        print(f"Wrote frames to {frames_dir}")


if __name__ == "__main__":
    main()
