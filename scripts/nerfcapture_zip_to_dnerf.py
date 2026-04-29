#!/usr/bin/env python3
"""Convert a NeRF Capture export zip into this repo's D-NeRF-style dataset format.

Output layout:
  <output_dir>/
    train/*.png
    val/*.png
    transforms_train.json
    transforms_test.json
    transforms_val.json

The generated JSON matches what scene/dataset_readers.py expects:
- top-level camera intrinsics when available
- frames entries with file_path (without extension), time in [0, 1], transform_matrix
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

COMMON_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path, help="Path to NeRF Capture export zip")
    parser.add_argument("output_dir", type=Path, help="Directory to write converted D-NeRF-style dataset")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio in (0,1). Ignored when --val-every > 0 (default: 0.1)",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=0,
        help="If > 0, put every k-th frame (after time-sort) into val (e.g. 8)",
    )
    parser.add_argument(
        "--split-mode",
        choices=("interleaved", "random"),
        default="interleaved",
        help="How to pick val frames when --val-every is 0 (default: interleaved)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for random split mode")
    parser.add_argument(
        "--output-ext",
        choices=(".png",),
        default=".png",
        help="Output image extension. Keep .png for this project (default: .png)",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_transforms_json(extracted_root: Path) -> Path:
    preferred = [
        extracted_root / "transforms.json",
        extracted_root / "dataset.json",
        extracted_root / "metadata.json",
    ]
    for p in preferred:
        if p.exists():
            data = _read_json(p)
            if isinstance(data.get("frames"), list):
                return p

    candidates = sorted(extracted_root.rglob("*.json"))
    for p in candidates:
        try:
            data = _read_json(p)
        except Exception:
            continue
        frames = data.get("frames")
        if not isinstance(frames, list) or not frames:
            continue
        first = frames[0]
        if isinstance(first, dict) and "transform_matrix" in first and "file_path" in first:
            return p

    raise FileNotFoundError(
        "Could not find a transforms JSON with frames containing file_path and transform_matrix."
    )


def _build_basename_index(root: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in COMMON_IMAGE_EXTS:
            idx.setdefault(p.name, []).append(p)
    return idx


def _resolve_image_path(root: Path, file_path: str, basename_idx: dict[str, list[Path]]) -> Path:
    raw = Path(file_path)

    # Direct relative path.
    direct = root / raw
    if direct.exists() and direct.is_file():
        return direct

    # Try appending common image extensions if metadata omitted extension.
    for ext in COMMON_IMAGE_EXTS:
        cand = root / (str(raw) + ext)
        if cand.exists() and cand.is_file():
            return cand

    # Fallback by basename if directory layout differs.
    basename = raw.name
    if "." not in basename:
        for ext in COMMON_IMAGE_EXTS:
            hit = basename + ext
            if hit in basename_idx and len(basename_idx[hit]) == 1:
                return basename_idx[hit][0]
    else:
        if basename in basename_idx and len(basename_idx[basename]) == 1:
            return basename_idx[basename][0]

    raise FileNotFoundError(f"Could not resolve image for frame path '{file_path}'")


def _normalize_times(frames: list[dict[str, Any]]) -> list[float]:
    raw_times: list[float | None] = []
    for i, fr in enumerate(frames):
        t = fr.get("time", fr.get("timestamp", None))
        if t is None:
            raw_times.append(None)
        else:
            try:
                raw_times.append(float(t))
            except Exception:
                raise ValueError(f"Invalid time/timestamp value at frame {i}: {t}")

    if all(t is None for t in raw_times):
        n = len(frames)
        if n == 1:
            return [0.0]
        return [i / float(n - 1) for i in range(n)]

    # Replace missing values with index before normalization.
    filled = [float(i) if t is None else float(t) for i, t in enumerate(raw_times)]
    t_min, t_max = min(filled), max(filled)

    if math.isclose(t_min, t_max):
        return [0.0 for _ in filled]

    return [(t - t_min) / (t_max - t_min) for t in filled]


def _pick_val_indices(num_frames: int, val_ratio: float, val_every: int, split_mode: str, seed: int) -> set[int]:
    if num_frames < 2:
        raise ValueError("Need at least 2 frames to create train/val split.")

    if val_every > 0:
        val = {i for i in range(num_frames) if i % val_every == 0}
        if len(val) == num_frames:
            val.remove(0)
        return val

    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1) when --val-every is 0.")

    n_val = max(1, int(round(num_frames * val_ratio)))
    n_val = min(n_val, num_frames - 1)

    if split_mode == "random":
        rng = random.Random(seed)
        return set(rng.sample(list(range(num_frames)), n_val))

    # Interleaved across time: pick approximately evenly spaced indices.
    if n_val == 1:
        return {num_frames // 2}

    val_indices = {
        int(round(i * (num_frames - 1) / float(n_val - 1)))
        for i in range(n_val)
    }
    if len(val_indices) == num_frames:
        val_indices.discard(0)
    return val_indices


def _top_level_intrinsics(meta: dict[str, Any], frames: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # Preserve camera_angle_x if provided.
    if "camera_angle_x" in meta:
        out["camera_angle_x"] = float(meta["camera_angle_x"])

    # Prefer explicit fx/fy/cx/cy so loader uses intrinsics exactly.
    keys = ("fl_x", "fl_y", "cx", "cy")
    if all(k in meta for k in keys):
        out["fl_x"] = float(meta["fl_x"])
        out["fl_y"] = float(meta["fl_y"])
        out["cx"] = float(meta["cx"])
        out["cy"] = float(meta["cy"])
    else:
        # Fallback from first frame if frame-level intrinsics exist.
        f0 = frames[0]
        if all(k in f0 for k in keys):
            out["fl_x"] = float(f0["fl_x"])
            out["fl_y"] = float(f0["fl_y"])
            out["cx"] = float(f0["cx"])
            out["cy"] = float(f0["cy"])

    out["w"] = int(meta.get("w", width))
    out["h"] = int(meta.get("h", height))
    return out


def _validate_transform_matrix(m: Any, idx: int) -> None:
    if not isinstance(m, list) or len(m) != 4:
        raise ValueError(f"Frame {idx}: transform_matrix must be 4x4 list")
    for r in m:
        if not isinstance(r, list) or len(r) != 4:
            raise ValueError(f"Frame {idx}: transform_matrix must be 4x4 list")


def _copy_as_png(src: Path, dst: Path) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA")
        im.save(dst)
        return im.size


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _validate_output(output_dir: Path, transforms_name: str, extension: str) -> None:
    meta = _read_json(output_dir / transforms_name)
    if "frames" not in meta or not isinstance(meta["frames"], list):
        raise ValueError(f"{transforms_name} missing frames list")

    for i, fr in enumerate(meta["frames"]):
        if "file_path" not in fr or "transform_matrix" not in fr or "time" not in fr:
            raise ValueError(f"{transforms_name} frame {i} missing required keys")
        t = float(fr["time"])
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"{transforms_name} frame {i} has time outside [0,1]: {t}")
        _validate_transform_matrix(fr["transform_matrix"], i)
        image_path = output_dir / (fr["file_path"] + extension)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image referenced by {transforms_name} frame {i}: {image_path}")


def convert(zip_path: Path, output_dir: Path, val_ratio: float, val_every: int, split_mode: str, seed: int, output_ext: str) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nerfcap_extract_") as tmp_dir:
        extract_root = Path(tmp_dir) / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)

        transforms_path = _find_transforms_json(extract_root)
        meta = _read_json(transforms_path)
        frames = meta.get("frames", [])
        if not isinstance(frames, list) or not frames:
            raise ValueError("Input transforms JSON has no frames.")

        basename_idx = _build_basename_index(extract_root)
        resolved = []
        for i, fr in enumerate(frames):
            if "file_path" not in fr:
                raise ValueError(f"Frame {i} missing file_path")
            if "transform_matrix" not in fr:
                raise ValueError(f"Frame {i} missing transform_matrix")
            _validate_transform_matrix(fr["transform_matrix"], i)
            img = _resolve_image_path(extract_root, str(fr["file_path"]), basename_idx)
            resolved.append((i, fr, img))

        times = _normalize_times([fr for _, fr, _ in resolved])

        # Sort by normalized time to get stable temporal splitting.
        merged = sorted(
            [(orig_i, fr, img, times[k]) for k, (orig_i, fr, img) in enumerate(resolved)],
            key=lambda x: (x[3], x[0]),
        )

        val_indices = _pick_val_indices(
            num_frames=len(merged),
            val_ratio=val_ratio,
            val_every=val_every,
            split_mode=split_mode,
            seed=seed,
        )

        # Clear previous output images if present so stale files do not linger.
        for sub in ("train", "val"):
            subdir = output_dir / sub
            if subdir.exists():
                shutil.rmtree(subdir)
            subdir.mkdir(parents=True, exist_ok=True)

        train_frames: list[dict[str, Any]] = []
        val_frames: list[dict[str, Any]] = []

        width = height = None
        train_count = val_count = 0

        for sorted_i, (_, fr, img_path, t) in enumerate(merged):
            split = "val" if sorted_i in val_indices else "train"
            if split == "train":
                stem = f"r_{train_count:05d}"
                train_count += 1
            else:
                stem = f"r_{val_count:05d}"
                val_count += 1

            dst_img = output_dir / split / f"{stem}{output_ext}"
            w, h = _copy_as_png(img_path, dst_img)
            if width is None:
                width, height = w, h

            out_frame = {
                "file_path": f"./{split}/{stem}",
                "time": float(max(0.0, min(1.0, t))),
                "transform_matrix": fr["transform_matrix"],
            }

            # Preserve frame-level intrinsics if present.
            for key in ("fl_x", "fl_y", "cx", "cy"):
                if key in fr:
                    out_frame[key] = float(fr[key])

            if "rotation" in fr:
                out_frame["rotation"] = fr["rotation"]

            if split == "train":
                train_frames.append(out_frame)
            else:
                val_frames.append(out_frame)

        assert width is not None and height is not None

        top = _top_level_intrinsics(meta, [fr for _, fr, _, _ in merged], width, height)

        train_json = dict(top)
        train_json["frames"] = train_frames

        val_json = dict(top)
        val_json["frames"] = val_frames

        _write_json(output_dir / "transforms_train.json", train_json)
        _write_json(output_dir / "transforms_test.json", val_json)
        _write_json(output_dir / "transforms_val.json", val_json)

        _validate_output(output_dir, "transforms_train.json", output_ext)
        _validate_output(output_dir, "transforms_test.json", output_ext)

        print("Conversion complete.")
        print(f"Input zip: {zip_path}")
        print(f"Found metadata: {transforms_path}")
        print(f"Output dir: {output_dir}")
        print(f"Train frames: {len(train_frames)}")
        print(f"Val frames:   {len(val_frames)}")
        print("Wrote transforms_train.json, transforms_test.json, transforms_val.json")


def main() -> None:
    args = _parse_args()
    convert(
        zip_path=args.zip_path,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        val_every=args.val_every,
        split_mode=args.split_mode,
        seed=args.seed,
        output_ext=args.output_ext,
    )


if __name__ == "__main__":
    main()
