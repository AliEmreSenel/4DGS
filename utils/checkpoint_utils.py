import argparse
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

CHECKPOINT_FORMAT = "4dgs-self-contained-v1"


def _plain_value(value: Any) -> Any:
    if isinstance(value, argparse.Namespace):
        return namespace_to_plain_dict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(v) for v in value]
    return str(value)


def namespace_to_plain_dict(ns: argparse.Namespace | Any) -> Dict[str, Any]:
    return {str(k): _plain_value(v) for k, v in vars(ns).items()}


def namespace_from_plain_dict(data: Mapping[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(**dict(data))


def camera_to_metadata(cam: Any) -> Dict[str, Any]:
    resolution = getattr(cam, "resolution", None)
    if resolution is None:
        resolution = [int(cam.image_width), int(cam.image_height)]
    return {
        "colmap_id": int(getattr(cam, "colmap_id", getattr(cam, "uid", 0))),
        "uid": int(getattr(cam, "uid", 0)),
        "R": np.asarray(cam.R, dtype=np.float32).tolist(),
        "T": np.asarray(cam.T, dtype=np.float32).tolist(),
        "FoVx": float(cam.FoVx),
        "FoVy": float(cam.FoVy),
        "image_name": str(getattr(cam, "image_name", f"cam_{getattr(cam, 'uid', 0)}")),
        "timestamp": float(getattr(cam, "timestamp", 0.0)),
        "resolution": [int(resolution[0]), int(resolution[1])],
        "cx": float(getattr(cam, "cx", -1)),
        "cy": float(getattr(cam, "cy", -1)),
        "fl_x": float(getattr(cam, "fl_x", -1)),
        "fl_y": float(getattr(cam, "fl_y", -1)),
        "trans": np.asarray(getattr(cam, "trans", [0.0, 0.0, 0.0]), dtype=np.float32).tolist(),
        "scale": float(getattr(cam, "scale", 1.0)),
    }


def scene_to_metadata(scene: Any) -> Dict[str, Any]:
    train = list(scene.train_cameras.get(1.0, []))
    test = list(scene.test_cameras.get(1.0, []))
    return {
        "cameras_extent": float(scene.cameras_extent),
        "white_background": bool(scene.white_background),
        "train_cameras": [camera_to_metadata(cam) for cam in train],
        "test_cameras": [camera_to_metadata(cam) for cam in test],
    }


def build_run_config(args: argparse.Namespace, gaussian_kwargs: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "args": namespace_to_plain_dict(args),
        "gaussian_kwargs": _plain_value(dict(gaussian_kwargs or {})),
    }


def build_checkpoint(
    *,
    gaussians: Any,
    iteration: int,
    run_config: Mapping[str, Any],
    scene_metadata: Mapping[str, Any],
    include_mobilegs: bool,
) -> Dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "iteration": int(iteration),
        "gaussians": gaussians.capture(include_mobilegs=include_mobilegs),
        "run_config": _plain_value(run_config),
        "scene": _plain_value(scene_metadata),
        "requires_mobilegs": bool(include_mobilegs),
    }


def save_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> None:
    torch.save(dict(checkpoint), str(path))


def load_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> Dict[str, Any]:
    checkpoint = torch.load(str(path), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a {CHECKPOINT_FORMAT} checkpoint. Regenerate it with this codebase."
        )
    return checkpoint


def checkpoint_args(checkpoint: Mapping[str, Any]) -> argparse.Namespace:
    run_config = checkpoint.get("run_config", {})
    args = run_config.get("args", {}) if isinstance(run_config, Mapping) else {}
    return namespace_from_plain_dict(args)
