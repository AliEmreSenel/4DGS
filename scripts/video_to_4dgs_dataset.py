#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
import numpy as np
try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise SystemExit("opencv-python or opencv-python-headless is required") from exc
try:
    from PIL import Image, ImageDraw
except Exception as exc:  # pragma: no cover
    raise SystemExit("Pillow is required") from exc
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".MP4", ".MOV", ".M4V", ".AVI", ".MKV", ".WEBM"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
class CommandError(RuntimeError):
    pass
class Logger:
    def __init__(self, log_dir: Path, verbose: bool = True):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.pipeline_log = self.log_dir / "pipeline.log"
        self.command_counter = 0
        self._captures: dict[tuple[str, ...], str] = {}
    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        with self.pipeline_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.verbose:
            print(line, flush=True)
    def save_text(self, name: str, text: str) -> Path:
        path = self.log_dir / name
        path.write_text(text, encoding="utf-8")
        return path
    def run(self, cmd: Sequence[str], cwd: Path | None = None, label: str | None = None) -> None:
        self.command_counter += 1
        tag = f"{self.command_counter:02d}_{label or Path(cmd[0]).name}"
        self.log(f"Running command [{tag}]: {' '.join(cmd)}")
        start = time.time()
        proc = subprocess.run(
            list(cmd),
            cwd=None if cwd is None else str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        elapsed = time.time() - start
        stdout_path = self.log_dir / f"{tag}.stdout.log"
        stderr_path = self.log_dir / f"{tag}.stderr.log"
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        self.log(f"Command [{tag}] finished with code {proc.returncode} in {elapsed:.2f}s")
        if proc.stdout.strip():
            self.log(f"stdout log: {stdout_path}")
        if proc.stderr.strip():
            self.log(f"stderr log: {stderr_path}")
        if proc.returncode != 0:
            raise CommandError(
                f"Command failed: {' '.join(cmd)}\n\n"
                f"stdout log: {stdout_path}\n"
                f"stderr log: {stderr_path}\n\n"
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )
    def capture(self, cmd: Sequence[str], cwd: Path | None = None, label: str | None = None) -> str:
        key = tuple(cmd) + ((f"cwd={cwd}",) if cwd is not None else ())
        if key in self._captures:
            return self._captures[key]
        self.command_counter += 1
        tag = f"{self.command_counter:02d}_{label or Path(cmd[0]).name}"
        self.log(f"Capturing command [{tag}]: {' '.join(cmd)}")
        proc = subprocess.run(
            list(cmd),
            cwd=None if cwd is None else str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_path = self.log_dir / f"{tag}.stdout.log"
        stderr_path = self.log_dir / f"{tag}.stderr.log"
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            raise CommandError(
                f"Command failed: {' '.join(cmd)}\n\n"
                f"stdout log: {stdout_path}\n"
                f"stderr log: {stderr_path}\n\n"
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )
        text = proc.stdout.strip() if proc.stdout.strip() else proc.stderr.strip()
        self._captures[key] = text
        return text
# --- utility helpers carried over from v3 ---
def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"Required binary '{name}' not found in PATH")
def ffprobe_duration(video_path: Path, logger: Logger) -> float:
    out = logger.capture([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ], label=f"ffprobe_duration_{video_path.stem}")
    return float(out)
def ffprobe_video_size(video_path: Path, logger: Logger) -> tuple[int, int]:
    out = logger.capture([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(video_path)
    ], label=f"ffprobe_size_{video_path.stem}")
    payload = json.loads(out)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    return int(streams[0]["width"]), int(streams[0]["height"])
def extract_audio(video_path: Path, wav_path: Path, sr: int, logger: Logger) -> None:
    logger.run([
        "ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr),
        "-c:a", "pcm_s16le", str(wav_path)
    ], label=f"extract_audio_{video_path.stem}")
def load_wav_mono_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sampwidth}")
        raw = wf.readframes(nframes)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    data /= 32768.0
    return data, fr
def smooth_abs_envelope(y: np.ndarray, sr: int, window_ms: float = 8.0) -> np.ndarray:
    win = max(1, int(round(sr * window_ms / 1000.0)))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(np.abs(y), kernel, mode="same").astype(np.float32)
def onset_envelope(y: np.ndarray, sr: int, hop: int = 256, win: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    n = 1 + max(0, (len(y) - win) // hop)
    env = np.empty(n, dtype=np.float32)
    window = np.hanning(win).astype(np.float32)
    for i in range(n):
        start = i * hop
        frame = y[start:start + win]
        if len(frame) < win:
            frame = np.pad(frame, (0, win - len(frame)))
        frame = frame * window
        env[i] = np.sqrt(np.mean(frame * frame) + 1e-12)
    smooth = np.convolve(env, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    diff = np.diff(smooth, prepend=smooth[0])
    diff = np.maximum(diff, 0.0)
    times = (np.arange(len(diff), dtype=np.float32) * hop + win * 0.5) / float(sr)
    return diff.astype(np.float32), times
def refine_peak_time(y: np.ndarray, sr: int, coarse_time: float, search_radius: float = 0.2) -> float:
    env = smooth_abs_envelope(y, sr)
    lo = max(0, int(round((coarse_time - search_radius) * sr)))
    hi = min(len(env), int(round((coarse_time + search_radius) * sr)))
    if hi <= lo:
        return coarse_time
    rel = int(np.argmax(env[lo:hi]))
    return (lo + rel) / float(sr)
def detect_peak_in_window(y: np.ndarray, sr: int, duration: float, window_start: float, window_end: float, label: str) -> tuple[float, float]:
    onset, times = onset_envelope(y, sr)
    window_start = max(0.0, min(window_start, duration))
    window_end = max(window_start, min(window_end, duration))
    mask = (times >= window_start) & (times <= window_end)
    if not np.any(mask):
        raise RuntimeError(f"No samples available while searching for {label} clap in [{window_start:.3f}, {window_end:.3f}]")
    indices = np.flatnonzero(mask)
    peak_index = int(indices[np.argmax(onset[mask])])
    coarse = float(times[peak_index])
    fine = refine_peak_time(y, sr, coarse)
    return fine, float(onset[peak_index])
def detect_start_clap(y: np.ndarray, sr: int, duration: float, start_window: float, start_after: float = 0.0) -> tuple[float, float]:
    return detect_peak_in_window(y, sr, duration, max(0.0, start_after), min(start_window, duration), "start")
def detect_end_clap(y: np.ndarray, sr: int, duration: float, end_window: float) -> tuple[float, float]:
    return detect_peak_in_window(y, sr, duration, max(0.0, duration - min(end_window, duration)), duration, "end")
def detect_peak_candidates_in_window(
    y: np.ndarray,
    sr: int,
    duration: float,
    window_start: float,
    window_end: float,
    label: str,
    max_candidates: int = 24,
    min_separation: float = 0.25,
) -> list[tuple[float, float]]:
    onset, times = onset_envelope(y, sr)
    window_start = max(0.0, min(window_start, duration))
    window_end = max(window_start, min(window_end, duration))
    mask = (times >= window_start) & (times <= window_end)
    if not np.any(mask):
        raise RuntimeError(f"No samples available while searching for {label} clap in [{window_start:.3f}, {window_end:.3f}]")
    indices = np.flatnonzero(mask)
    sorted_indices = indices[np.argsort(onset[indices])[::-1]]
    candidates: list[tuple[float, float]] = []
    for idx in sorted_indices:
        coarse = float(times[int(idx)])
        if any(abs(coarse - existing_time) < min_separation for existing_time, _ in candidates):
            continue
        fine = refine_peak_time(y, sr, coarse)
        if any(abs(fine - existing_time) < min_separation for existing_time, _ in candidates):
            continue
        candidates.append((float(fine), float(onset[int(idx)])))
        if len(candidates) >= max_candidates:
            break
    if not candidates:
        raise RuntimeError(f"No {label} clap candidates found in [{window_start:.3f}, {window_end:.3f}]")
    return candidates
def detect_start_end_claps(
    y: np.ndarray,
    sr: int,
    duration: float,
    start_window: float,
    end_window: float,
    min_interval: float,
    start_after: float = 0.0,
) -> tuple[float, float, float, float]:
    start_candidates = detect_peak_candidates_in_window(
        y, sr, duration, max(0.0, start_after), min(start_window, duration), "start"
    )
    end_candidates = detect_peak_candidates_in_window(
        y, sr, duration, max(0.0, duration - min(end_window, duration)), duration, "end"
    )
    best: tuple[float, float, float, float, float] | None = None
    for start_time, start_strength in start_candidates:
        for end_time, end_strength in end_candidates:
            interval = float(end_time) - float(start_time)
            if interval < min_interval:
                continue
            score = float(start_strength) + float(end_strength)
            # Prefer the strongest pair. For ties, prefer an earlier start and later end.
            key = (score, -float(start_time), float(end_time), interval)
            if best is None or key > best[0:4]:
                best = (score, -float(start_time), float(end_time), interval, float(start_strength) + float(end_strength))
                best_pair = (float(start_time), float(start_strength), float(end_time), float(end_strength))
    if best is None:
        start_preview = [{"time": round(t, 6), "strength": round(v, 6)} for t, v in start_candidates[:5]]
        end_preview = [{"time": round(t, 6), "strength": round(v, 6)} for t, v in end_candidates[:5]]
        raise RuntimeError(
            "Could not find distinct start/end clap pair "
            f"at least {min_interval:.3f}s apart. "
            f"Top start candidates={start_preview}; top end candidates={end_preview}. "
            "Try reducing --clap-min-interval, narrowing the overlapping window, "
            "or increasing --clap-end-window if the end clap is earlier than expected."
        )
    return best_pair
def resolve_video_inputs(items: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    def add_file(p: Path) -> None:
        q = p.expanduser().resolve()
        if q in seen:
            return
        if q.is_file() and q.suffix in VIDEO_EXTS:
            seen.add(q)
            resolved.append(q)
    for item in items:
        raw = Path(item).expanduser()
        if raw.exists():
            if raw.is_dir():
                for child in sorted(raw.iterdir()):
                    add_file(child)
            else:
                add_file(raw)
            continue
        matches = sorted(Path().glob(item))
        if matches:
            for match in matches:
                if match.is_dir():
                    for child in sorted(match.iterdir()):
                        add_file(child)
                else:
                    add_file(match)
            continue
        raise FileNotFoundError(f"Input path not found: {item}")
    if len(resolved) < 2:
        raise ValueError("Need at least 2 videos after resolving inputs")
    return resolved
def output_size_from_width(source_width: int, source_height: int, width: int | None) -> tuple[int, int]:
    if width is None or width == source_width:
        return source_width, source_height
    scaled_h = int(round(source_height * (float(width) / float(source_width))))
    if scaled_h % 2 == 1:
        scaled_h += 1
    return width, scaled_h
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
def maybe_clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
def camera_name_from_index(idx: int) -> str:
    return f"cam{idx:02d}"
def parse_per_video_float_overrides(items: Sequence[str] | None, videos: list[Path], option_name: str) -> dict[int, float]:
    """Parse KEY=SECONDS overrides where KEY can be camNN, index, filename, stem, or path.

    The argparse option uses action=append, but each item may also contain a
    comma-separated list so both of these work:
      --clap-start-after cam02=3.0 --clap-start-after cam04=2.5
      --clap-start-after cam02=3.0,cam04=2.5
    """
    overrides: dict[int, float] = {}
    if not items:
        return overrides
    key_to_index: dict[str, int] = {}
    for i, video in enumerate(videos):
        aliases = {
            str(i),
            camera_name_from_index(i),
            video.name,
            video.stem,
            str(video),
            str(video.resolve()),
        }
        for alias in aliases:
            key_to_index[alias] = i
    parts: list[str] = []
    for item in items:
        parts.extend([piece.strip() for piece in str(item).split(",") if piece.strip()])
    for part in parts:
        if "=" not in part:
            raise ValueError(f"{option_name} expects KEY=SECONDS, got {part!r}")
        raw_key, raw_value = part.split("=", 1)
        key = raw_key.strip()
        if key not in key_to_index:
            available = []
            for i, video in enumerate(videos):
                available.append(f"{camera_name_from_index(i)} / {video.name} / {video.stem}")
            raise ValueError(
                f"Unknown video key {key!r} in {option_name}. "
                f"Use one of: {available}"
            )
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{option_name} value for {key!r} must be a number of seconds, got {raw_value!r}") from exc
        if value < 0.0:
            raise ValueError(f"{option_name} value for {key!r} must be non-negative, got {value}")
        overrides[key_to_index[key]] = value
    return overrides
def collect_image_inventory(root: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    total = 0
    if root.exists():
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                count = sum(1 for p in sub.iterdir() if p.suffix in IMAGE_EXTS)
                counts[sub.name] = count
                total += count
    return {"root": str(root), "total_images": total, "per_folder": counts}
def list_camera_dirs(root: Path) -> list[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir()] if root.exists() else []
def list_image_files(cam_dir: Path) -> list[Path]:
    return sorted([p for p in cam_dir.iterdir() if p.suffix in IMAGE_EXTS])
def harmonize_extracted_images(root: Path, logger: Logger, trim: bool = True) -> tuple[int, dict[str, int]]:
    cam_dirs = list_camera_dirs(root)
    if not cam_dirs:
        raise RuntimeError(f"No camera directories found under {root}")
    counts = {cam_dir.name: len(list_image_files(cam_dir)) for cam_dir in cam_dirs}
    if any(v <= 0 for v in counts.values()):
        raise RuntimeError(f"At least one camera directory under {root} is empty: {counts}")
    min_count = min(counts.values())
    max_count = max(counts.values())
    logger.log(f"Frame counts under {root}: {counts}; synchronized usable count={min_count}")
    if trim and max_count != min_count:
        for cam_dir in cam_dirs:
            files = list_image_files(cam_dir)
            extras = files[min_count:]
            if extras:
                for p in extras:
                    p.unlink()
                logger.log(f"Trimmed {len(extras)} trailing frame(s) from {cam_dir.name} to enforce equal-length synchronization")
        counts = {cam_dir.name: len(list_image_files(cam_dir)) for cam_dir in cam_dirs}
        logger.log(f"Counts after trimming under {root}: {counts}")
    return min_count, counts
def verify_sample_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size
def read_image_times_manifest(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows
def build_pose_indices(nframes: int, fps_final: float, fps_colmap: float) -> list[int]:
    if nframes <= 0:
        return []
    step = max(1, int(round(float(fps_final) / float(fps_colmap))))
    indices = list(range(0, nframes, step))
    if indices[-1] != nframes - 1:
        indices.append(nframes - 1)
    return sorted(set(indices))
def extract_synced_pngs(videos: list[Path], starts: list[float], overlap_duration: float, out_root: Path, fps: float, width: int | None, jpeg_quality: int, logger: Logger, force: bool, source_ends: list[float] | None = None, time_warp: bool = False) -> int:
    meta_path = out_root.parent / f"{out_root.name}_meta.json"
    inventory_path = out_root.parent / f"{out_root.name}_inventory.json"
    meta = {
        "videos": [str(v.resolve()) for v in videos],
        "starts": [round(x, 6) for x in starts],
        "source_ends": None if source_ends is None else [round(x, 6) for x in source_ends],
        "duration": round(overlap_duration, 6),
        "time_warp": bool(time_warp),
        "fps": round(float(fps), 6),
        "width": width,
        "jpeg_quality": int(jpeg_quality),
    }
    inv = collect_image_inventory(out_root)
    old_meta = load_json(meta_path)
    if not force and inv["total_images"] > 0:
        exact_match = old_meta == meta
        same_cam_count = len(inv["per_folder"]) == len(videos) and all(v > 0 for v in inv["per_folder"].values())
        if exact_match:
            logger.log(f"Reusing extracted images under {out_root} using exact cache metadata match ({inv['total_images']} files)")
            nframes, counts = harmonize_extracted_images(out_root, logger, trim=True)
            write_json(inventory_path, collect_image_inventory(out_root))
            return nframes
        if same_cam_count:
            logger.log(
                f"Existing extracted images under {out_root} have all camera folders but extraction metadata changed; "
                f"re-extracting to avoid stale synchronization. Old meta={old_meta} new meta={meta}"
            )
        else:
            logger.log(f"Extraction cache miss for {out_root}: old_meta={old_meta} current_meta={meta} inventory={inv}")
    maybe_clear_dir(out_root)
    for i, video in enumerate(videos):
        cam_dir = out_root / camera_name_from_index(i)
        cam_dir.mkdir(parents=True, exist_ok=True)
        src_w, src_h = ffprobe_video_size(video, logger)
        out_w, out_h = output_size_from_width(src_w, src_h, width)
        if source_ends is not None:
            source_duration = float(source_ends[i]) - float(starts[i])
            if source_duration <= 0.0:
                raise RuntimeError(
                    f"Invalid synchronized source interval for {video}: "
                    f"start={starts[i]:.6f}, end={source_ends[i]:.6f}"
                )
            if time_warp:
                # Both-clap mode uses an affine per-video time mapping so the
                # detected start clap lands at t=0 and the detected end clap
                # lands at the shared output duration for every camera.
                scale = float(overlap_duration) / source_duration
                vf = f"trim=start={starts[i]:.6f}:end={source_ends[i]:.6f},setpts=(PTS-STARTPTS)*{scale:.12f},fps={fps:.8f}"
            else:
                vf = f"trim=start={starts[i]:.6f}:duration={overlap_duration:.6f},setpts=PTS-STARTPTS,fps={fps:.8f}"
        else:
            vf = f"trim=start={starts[i]:.6f}:duration={overlap_duration:.6f},setpts=PTS-STARTPTS,fps={fps:.8f}"
        if out_w != src_w or out_h != src_h:
            vf += f",scale={out_w}:{out_h}:flags=lanczos"
        pattern = str(cam_dir / "%06d.png")
        logger.run([
            "ffmpeg", "-y", "-i", str(video), "-vf", vf, "-vsync", "vfr", pattern
        ], label=f"extract_{video.stem}")
    nframes, counts = harmonize_extracted_images(out_root, logger, trim=True)
    inv = collect_image_inventory(out_root)
    write_json(meta_path, meta)
    write_json(inventory_path, inv)
    logger.log(f"Extracted synced images inventory: {inv}")
    logger.log(f"Extraction metadata written to {meta_path}")
    return nframes
def build_pose_subset_from_full(full_raw_images: Path, pose_images: Path, pose_indices: list[int], nframes: int, mask_frac: float, meta_path: Path, force: bool, logger: Logger) -> list[list[str]]:
    source_nframes, source_counts = harmonize_extracted_images(full_raw_images, logger, trim=True)
    if source_nframes != nframes:
        logger.log(f"Adjusted dataset frame count from requested {nframes} to synchronized source count {source_nframes}")
        nframes = source_nframes
    safe_pose_indices = [int(x) for x in pose_indices if 0 <= int(x) < nframes]
    safe_pose_set = set(safe_pose_indices)
    dropped = [int(x) for x in pose_indices if int(x) not in safe_pose_set]
    if dropped:
        logger.log(f"Dropped out-of-range pose indices after synchronization trimming: {dropped}")
    meta = {
        "source": str(full_raw_images.resolve()),
        "pose_indices": safe_pose_indices,
        "nframes": int(nframes),
        "mask_frac": round(float(mask_frac), 6),
        "source_counts": source_counts,
        "pose_ext": ".jpg",
    }
    inv = collect_image_inventory(pose_images)
    reuse_pose_subset = (not force) and load_json(meta_path) == meta and inv["total_images"] > 0
    if reuse_pose_subset:
        bad_ext = []
        for cam_dir in sorted(p for p in pose_images.iterdir() if p.is_dir()):
            for file_path in list_image_files(cam_dir):
                if file_path.suffix.lower() != ".jpg":
                    bad_ext.append(str(file_path))
                    if len(bad_ext) >= 3:
                        break
            if bad_ext:
                break
        if bad_ext:
            logger.log(f"Discarding cached pose subset because non-JPG files were found: {bad_ext}")
            reuse_pose_subset = False
    if reuse_pose_subset:
        logger.log(f"Reusing masked COLMAP pose images under {pose_images} ({inv['total_images']} files)")
    else:
        maybe_clear_dir(pose_images)
        draw_frac = max(0.0, min(float(mask_frac), 0.95))
        for cam_dir in sorted(p for p in full_raw_images.iterdir() if p.is_dir()):
            dst_cam = pose_images / cam_dir.name
            dst_cam.mkdir(parents=True, exist_ok=True)
            files = sorted([p for p in cam_dir.iterdir() if p.suffix in IMAGE_EXTS])
            logger.log(f"Building COLMAP pose subset for {cam_dir.name}: {len(files)} source frames, {len(safe_pose_indices)} selected")
            for idx in safe_pose_indices:
                if idx >= len(files):
                    logger.log(f"Skipping pose subset frame {idx} for {cam_dir.name} because only {len(files)} frames are available")
                    continue
                src = files[idx]
                dst = dst_cam / f"{src.stem}.jpg"
                with Image.open(src) as im:
                    rgb = im.convert("RGB")
                    w, h = rgb.size
                    if draw_frac > 0.0:
                        mask_w = int(round(w * draw_frac))
                        mask_h = int(round(h * draw_frac))
                        x0 = max(0, (w - mask_w) // 2)
                        y0 = max(0, (h - mask_h) // 2)
                        x1 = min(w, x0 + mask_w)
                        y1 = min(h, y0 + mask_h)
                        draw = ImageDraw.Draw(rgb)
                        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
                    rgb.save(dst)
        write_json(meta_path, meta)
    manifest_rows = []
    for cam_dir in sorted(p for p in pose_images.iterdir() if p.is_dir()):
        files = sorted([p for p in cam_dir.iterdir() if p.suffix in IMAGE_EXTS])
        for src in files:
            frame_index = int(src.stem) - 1
            time_norm = 0.0 if nframes <= 1 else frame_index / float(nframes - 1)
            manifest_rows.append([src.stem, cam_dir.name, str(frame_index), f"{time_norm:.8f}", f"{time_norm:.8f}"])
    return manifest_rows
def collect_root_image_inventory(root: Path) -> dict[str, Any]:
    total = 0
    names: list[str] = []
    if root.exists():
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix in IMAGE_EXTS:
                names.append(p.name)
        total = len(names)
    return {"root": str(root), "total_images": total, "files": names[:20], "has_more": total > 20}
def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
def build_pose_flat_import_from_subset(pose_images: Path, flat_root: Path, nframes: int, csv_path: Path, meta_path: Path, force: bool, logger: Logger) -> list[dict[str, Any]]:
    cam_dirs = [p for p in sorted(pose_images.iterdir()) if p.is_dir()]
    source_listing: dict[str, list[str]] = {}
    for cam_dir in cam_dirs:
        source_listing[cam_dir.name] = [p.name for p in list_image_files(cam_dir)]
    expected_total = sum(len(v) for v in source_listing.values())
    meta = {
        "source": str(pose_images.resolve()),
        "nframes": int(nframes),
        "source_listing": source_listing,
    }
    inv = collect_root_image_inventory(flat_root)
    if (not force) and load_json(meta_path) == meta and inv["total_images"] == expected_total and csv_path.exists():
        logger.log(f"Reusing flat COLMAP import images under {flat_root} ({inv['total_images']} files)")
        rows = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "flat_name": row["flat_name"],
                    "camera_name": row["camera_name"],
                    "source_name": row["source_name"],
                    "frame_index": int(row["frame_index"]),
                    "time_norm": float(row["time_norm"]),
                    "local_time_seconds": float(row["local_time_seconds"]),
                })
        return rows
    maybe_clear_dir(flat_root)
    rows: list[dict[str, Any]] = []
    for cam_dir in cam_dirs:
        files = list_image_files(cam_dir)
        logger.log(f"Flattening cached pose subset for COLMAP import from {cam_dir.name}: {len(files)} files")
        for src in files:
            frame_index = int(src.stem) - 1
            time_norm = 0.0 if nframes <= 1 else frame_index / float(nframes - 1)
            flat_name = f"{cam_dir.name}__{src.name}"
            dst = flat_root / flat_name
            hardlink_or_copy(src, dst)
            rows.append({
                "flat_name": flat_name,
                "camera_name": cam_dir.name,
                "source_name": src.name,
                "frame_index": frame_index,
                "time_norm": round(float(time_norm), 8),
                "local_time_seconds": round(float(time_norm), 8),
            })
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["flat_name", "camera_name", "source_name", "frame_index", "time_norm", "local_time_seconds"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(meta_path, meta)
    return rows
def build_per_camera_import_roots_from_flat(flat_root: Path, rows: list[dict[str, Any]], bycam_root: Path, meta_path: Path, force: bool, logger: Logger) -> None:
    expected = {}
    for row in rows:
        expected.setdefault(str(row["camera_name"]), []).append(str(row["flat_name"]))
    meta = {"flat_root": str(flat_root.resolve()), "expected": {k: sorted(v) for k, v in expected.items()}}
    ok = True
    if (not force) and load_json(meta_path) == meta and bycam_root.exists():
        for cam_name, names in meta["expected"].items():
            cam_dir = bycam_root / cam_name
            actual = sorted(p.name for p in cam_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if cam_dir.exists() else []
            if actual != names:
                ok = False
                break
        if ok:
            logger.log(f"Reusing per-camera COLMAP import roots under {bycam_root}")
            return
    maybe_clear_dir(bycam_root)
    for cam_name, names in sorted(meta["expected"].items()):
        cam_dir = bycam_root / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        logger.log(f"Preparing per-camera COLMAP import root for {cam_name}: {len(names)} files")
        for name in names:
            src = flat_root / name
            dst = cam_dir / name
            hardlink_or_copy(src, dst)
    write_json(meta_path, meta)
def group_pose_rows_by_camera(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row["camera_name"]), []).append(row)
    for cam_name in out:
        out[cam_name] = sorted(out[cam_name], key=lambda r: (int(r["frame_index"]), str(r["flat_name"])))
    return out
def database_stats(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"images": 0, "frames": 0, "keypoints": 0, "descriptors": 0, "matches": 0, "two_view_geometries": 0, "cameras": 0, "tables": []}
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    stats: dict[str, Any] = {}
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        stats["tables"] = [str(row[0]) for row in cur.fetchall()]
    except sqlite3.Error:
        stats["tables"] = []
    for table in ["images", "frames", "keypoints", "descriptors", "matches", "two_view_geometries", "cameras", "pose_priors", "rigs", "sensors"]:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = int(cur.fetchone()[0])
        except sqlite3.Error:
            stats[table] = 0
    try:
        cur.execute("PRAGMA table_info(keypoints)")
        cols = [str(r[1]) for r in cur.fetchall()]
        if "rows" in cols:
            cur.execute("SELECT COALESCE(SUM(rows),0) FROM keypoints")
            stats["keypoints_sum_rows"] = int(cur.fetchone()[0])
        else:
            stats["keypoints_sum_rows"] = int(stats.get("keypoints", 0))
    except sqlite3.Error:
        stats["keypoints_sum_rows"] = 0
    try:
        cur.execute("PRAGMA table_info(descriptors)")
        cols = [str(r[1]) for r in cur.fetchall()]
        if "rows" in cols:
            cur.execute("SELECT COALESCE(SUM(rows),0) FROM descriptors")
            stats["descriptors_sum_rows"] = int(cur.fetchone()[0])
        else:
            stats["descriptors_sum_rows"] = int(stats.get("descriptors", 0))
    except sqlite3.Error:
        stats["descriptors_sum_rows"] = 0
    con.close()
    return stats

def has_usable_colmap_features(stats: dict[str, Any]) -> bool:
    return any(int(stats.get(k, 0)) > 0 for k in ["images", "frames", "keypoints", "keypoints_sum_rows", "descriptors", "descriptors_sum_rows"])
def read_log_tail(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])
def reset_colmap_workspace(colmap_dir: Path, sparse_dir: Path, sparse_txt: Path) -> Path:
    maybe_clear_dir(colmap_dir)
    maybe_clear_dir(sparse_dir)
    maybe_clear_dir(sparse_txt)
    db_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    sparse_txt.mkdir(parents=True, exist_ok=True)
    return db_path
def parse_colmap_cameras_txt(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(x) for x in parts[4:]]
        rows.append({"camera_id": cid, "model": model, "width": width, "height": height, "params": params})
    return rows
def parse_colmap_images_txt(path: Path) -> list[dict[str, Any]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    rows = []
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        rows.append({
            "image_id": int(parts[0]),
            "qvec": [float(x) for x in parts[1:5]],
            "tvec": [float(x) for x in parts[5:8]],
            "camera_id": int(parts[8]),
            "name": parts[9],
        })
    return rows
def parse_colmap_points_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz = []
    rgb = []
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.uint8)
def qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)
def colmap_w2c_to_blender_c2w(qvec: Sequence[float], tvec: Sequence[float]) -> np.ndarray:
    R = qvec_to_rotmat(qvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3:] = t
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1.0
    return c2w
def camera_intrinsics_from_colmap(cam: dict[str, Any]) -> tuple[float, float, float, float]:
    model = cam["model"]
    params = cam["params"]
    w = float(cam["width"])
    h = float(cam["height"])
    if model == "SIMPLE_PINHOLE":
        f = params[0]
        return f, f, w / 2.0, h / 2.0
    if model == "PINHOLE":
        return params[0], params[1], params[2], params[3]
    if model == "SIMPLE_RADIAL":
        f = params[0]
        return f, f, params[1], params[2]
    if model == "RADIAL":
        f = params[0]
        return f, f, params[1], params[2]
    return params[0], params[0], w / 2.0, h / 2.0
def write_points3d_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(xyz.shape[0])
    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ])
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
def infer_camera_names_from_dirs(root: Path) -> list[str]:
    return [p.name for p in sorted(root.iterdir()) if p.is_dir()]
def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)
def slerp_qvec(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    qa = np.asarray(q0, dtype=np.float64)
    qb = np.asarray(q1, dtype=np.float64)
    qa = qa / max(np.linalg.norm(qa), 1e-12)
    qb = qb / max(np.linalg.norm(qb), 1e-12)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        out = qa + t * (qb - qa)
        return out / max(np.linalg.norm(out), 1e-12)
    theta0 = math.acos(dot)
    sin_theta0 = math.sin(theta0)
    theta = theta0 * t
    s0 = math.sin(theta0 - theta) / sin_theta0
    s1 = math.sin(theta) / sin_theta0
    out = s0 * qa + s1 * qb
    return out / max(np.linalg.norm(out), 1e-12)
def interpolate_c2w(c2w0: np.ndarray, c2w1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = rotmat_to_qvec(c2w0[:3, :3])
    q1 = rotmat_to_qvec(c2w1[:3, :3])
    q = slerp_qvec(q0, q1, alpha)
    R = qvec_to_rotmat(q)
    t = (1.0 - alpha) * c2w0[:3, 3] + alpha * c2w1[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = t
    return out
def interpolate_track_sample(samples: list[dict[str, Any]], frame_index: int) -> dict[str, Any]:
    if not samples:
        raise RuntimeError("Cannot interpolate an empty camera track")
    if len(samples) == 1:
        return samples[0]
    frames = [int(s["frame_index"]) for s in samples]
    if frame_index <= frames[0]:
        return samples[0]
    if frame_index >= frames[-1]:
        return samples[-1]
    for left, right in zip(samples[:-1], samples[1:]):
        f0 = int(left["frame_index"])
        f1 = int(right["frame_index"])
        if f0 <= frame_index <= f1:
            if f1 == f0:
                return left
            alpha = (frame_index - f0) / float(f1 - f0)
            return {
                "frame_index": frame_index,
                "camera_id": left["camera_id"],
                "c2w": interpolate_c2w(left["c2w"], right["c2w"], alpha),
            }
    return samples[-1]
def build_frames_for_dataset(full_raw_images: Path, cameras: list[dict[str, Any]], images: list[dict[str, Any]], nframes: int, test_camera: int, image_meta_by_name: dict[str, dict[str, Any]], logger: Logger, split_mode: str = "random_every", test_every: int = 20, test_seed: int = 0) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, int]]:
    cam_by_id = {cam["camera_id"]: cam for cam in cameras}
    solved_samples_by_cam: dict[str, list[dict[str, Any]]] = {}
    missing_meta = []
    for img in images:
        key = Path(img["name"]).name
        meta = image_meta_by_name.get(key)
        if meta is None:
            missing_meta.append(img["name"])
            continue
        cam_name = str(meta["camera_name"])
        solved_samples_by_cam.setdefault(cam_name, []).append({
            "frame_index": int(meta["frame_index"]),
            "camera_id": int(img["camera_id"]),
            "c2w": colmap_w2c_to_blender_c2w(img["qvec"], img["tvec"]),
            "image_name": key,
        })
    if missing_meta:
        logger.log(f"Warning: {len(missing_meta)} COLMAP image(s) had no pose-manifest entry; examples={missing_meta[:5]}")
    all_frames = []
    registered_per_camera: dict[str, int] = {}
    cam_names = infer_camera_names_from_dirs(full_raw_images)
    if not cam_names:
        raise RuntimeError(f"No camera directories found under {full_raw_images}")
    root_common = None
    frames_by_camera: dict[str, list[dict[str, Any]]] = {}
    for cam_idx, cam_name in enumerate(cam_names):
        raw_samples = sorted(solved_samples_by_cam.get(cam_name, []), key=lambda s: int(s["frame_index"]))
        deduped: list[dict[str, Any]] = []
        seen_frames = set()
        for sample in raw_samples:
            fi = int(sample["frame_index"])
            if fi in seen_frames:
                continue
            deduped.append(sample)
            seen_frames.add(fi)
        samples = deduped
        registered_per_camera[cam_name] = len(samples)
        logger.log(f"Registered pose samples for {cam_name}: {len(samples)} frame(s)")
        if not samples:
            raise RuntimeError(f"COLMAP did not register any pose samples for camera {cam_name}")
        ref_cam = cam_by_id[int(samples[0]["camera_id"])]
        fl_x, fl_y, cx, cy = camera_intrinsics_from_colmap(ref_cam)
        if root_common is None:
            root_common = {
                "fl_x": float(fl_x),
                "fl_y": float(fl_y),
                "cx": float(cx),
                "cy": float(cy),
                "w": int(ref_cam["width"]),
                "h": int(ref_cam["height"]),
            }
        files = sorted([p for p in (full_raw_images / cam_name).iterdir() if p.suffix in IMAGE_EXTS])
        if len(files) < nframes:
            raise RuntimeError(f"Camera {cam_name} has {len(files)} frames but expected at least {nframes}")
        files = files[:nframes]
        cam_frames: list[dict[str, Any]] = []
        for frame_index, img_path in enumerate(files):
            solved = interpolate_track_sample(samples, frame_index)
            c2w = solved["c2w"]
            time_norm = 0.0 if nframes <= 1 else frame_index / float(nframes - 1)
            frame = {
                "file_path": f"images/{cam_name}/{img_path.stem}",
                "time": round(float(time_norm), 8),
                "transform_matrix": [[float(x) for x in row] for row in c2w],
                "fl_x": float(fl_x),
                "fl_y": float(fl_y),
                "cx": float(cx),
                "cy": float(cy),
                "w": int(ref_cam["width"]),
                "h": int(ref_cam["height"]),
            }
            cam_frames.append(frame)
            all_frames.append(frame)
        frames_by_camera[cam_name] = cam_frames
    assert root_common is not None

    train_frames: list[dict[str, Any]] = []
    test_frames: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}
    if split_mode == "camera_holdout":
        logger.log(f"Using camera holdout split with test camera index {test_camera}")
        for cam_idx, cam_name in enumerate(cam_names):
            cam_frames = frames_by_camera[cam_name]
            if cam_idx == test_camera:
                test_frames.extend(cam_frames)
                split_counts[cam_name] = {"train": 0, "test": len(cam_frames)}
            else:
                train_frames.extend(cam_frames)
                split_counts[cam_name] = {"train": len(cam_frames), "test": 0}
    else:
        if test_every < 2:
            raise RuntimeError(f"test_every must be >= 2 for split_mode={split_mode}, got {test_every}")
        logger.log(f"Using random per-camera split with approximately 1/{test_every} frames held out for test, seed={test_seed}")
        for cam_idx, cam_name in enumerate(cam_names):
            cam_frames = frames_by_camera[cam_name]
            n_cam = len(cam_frames)
            n_test = max(1, n_cam // test_every)
            local_rng = random.Random(test_seed + cam_idx * 1000003)
            test_indices = sorted(local_rng.sample(range(n_cam), n_test))
            test_index_set = set(test_indices)
            logger.log(f"Split {cam_name}: total={n_cam}, test={n_test}, first_test_indices={test_indices[:10]}")
            for idx, frame in enumerate(cam_frames):
                if idx in test_index_set:
                    test_frames.append(frame)
                else:
                    train_frames.append(frame)
            split_counts[cam_name] = {"train": n_cam - n_test, "test": n_test}
    logger.log(f"Split summary per camera: {split_counts}")
    logger.log(f"Final split sizes: train={len(train_frames)}, test={len(test_frames)}, all={len(all_frames)}")
    return {**root_common, "frames": train_frames}, {**root_common, "frames": test_frames}, {**root_common, "frames": all_frames}, registered_per_camera
def write_repo_yaml(path: Path, source_path: Path, model_path: Path) -> None:
    text = f"""gaussian_dim: 4
time_duration: [0.0, 1.0]
num_pts: 100_000
num_pts_ratio: 1.0
rot_4d: True
force_sh_3d: False
batch_size: 2
exhaust_test: True
ModelParams:
  sh_degree: 3
  source_path: \"{source_path.as_posix()}\"
  model_path: \"{model_path.as_posix()}\"
  images: \"images\"
  resolution: 3
  white_background: False
  data_device: \"cuda\"
  eval: True
  extension: \".png\"
  num_extra_pts: 0
  loaded_pth: \"\"
  frame_ratio: 1
  dataloader: False
PipelineParams:
  convert_SHs_python: False
  compute_cov3D_python: False
  debug: False
  env_map_res: 0
  env_optimize_until: 1000000000
  env_optimize_from: 0
  eval_shfs_4d: True
OptimizationParams:
  iterations: 10_000
  position_lr_init: 0.00016
  position_t_lr_init: -1.0
  position_lr_final: 0.0000016
  position_lr_delay_mult: 0.01
  position_lr_max_steps: 10_000
  feature_lr: 0.0025
  opacity_lr: 0.05
  scaling_lr: 0.005
  rotation_lr: 0.001
  percent_dense: 0.01
  lambda_dssim: 0.2
  thresh_opa_prune: 0.005
  densification_interval: 100
  opacity_reset_interval: 3000
  densify_from_iter: 500
  densify_until_iter: 7_500
  densify_grad_threshold: 0.0002
  densify_grad_t_threshold: 0.000005
  densify_until_num_points: -1
  final_prune_from_iter: -1
  sh_increase_interval: 1000
  lambda_opa_mask: 0.0
  lambda_rigid: 0.0
  lambda_motion: 0.0
  lambda_depth: 0.0
  enable_spatio_temporal_pruning: True
  spatio_temporal_pruning_ratio: 0.1
  spatio_temporal_pruning_from_iter: 500
  spatio_temporal_pruning_until_iter: 7_500
  spatio_temporal_pruning_interval: 100
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
def _common_overlap_from_sync_points(durations: list[float], sync_points: list[float]) -> tuple[list[float], float]:
    if len(durations) != len(sync_points):
        raise ValueError("durations and sync_points must have the same length")
    # Treat each sync point as the same real-world event.  Local video time t
    # maps to event-relative global time g = t - sync_point.  The common
    # interval is the intersection of all event-relative video intervals.
    global_start = max(-float(t) for t in sync_points)
    global_end = min(float(durations[i]) - float(sync_points[i]) for i in range(len(durations)))
    overlap = global_end - global_start
    starts = [float(sync_points[i]) + global_start for i in range(len(sync_points))]
    return starts, float(overlap)
def _extract_audio_for_videos(videos: list[Path], audio_dir: Path, audio_sr: int, logger: Logger) -> tuple[list[float], list[Path]]:
    durations = [ffprobe_duration(v, logger) for v in videos]
    wavs: list[Path] = []
    for v in videos:
        wav_path = audio_dir / f"{v.stem}.wav"
        extract_audio(v, wav_path, audio_sr, logger)
        wavs.append(wav_path)
    return durations, wavs
def determine_sync_from_claps(
    videos: list[Path],
    audio_dir: Path,
    audio_sr: int,
    sync_mode: str,
    clap_start_window: float,
    clap_end_window: float,
    clap_min_interval: float,
    clap_start_after: dict[int, float],
    logger: Logger,
) -> dict[str, Any]:
    normalized_mode = "both" if sync_mode == "both_claps" else sync_mode
    durations, wavs = _extract_audio_for_videos(videos, audio_dir, audio_sr, logger)
    start_claps: list[float] | None = None
    end_claps: list[float] | None = None
    start_strengths: list[float] | None = None
    end_strengths: list[float] | None = None
    if normalized_mode in {"start_clap", "both"}:
        start_claps = []
        start_strengths = []
    if normalized_mode in {"end_clap", "both"}:
        end_claps = []
        end_strengths = []
    for idx, (wav_path, duration) in enumerate(zip(wavs, durations)):
        y, sr = load_wav_mono_pcm16(wav_path)
        start_after = float(clap_start_after.get(idx, 0.0))
        if normalized_mode == "both":
            assert start_claps is not None and start_strengths is not None
            assert end_claps is not None and end_strengths is not None
            start_time, start_strength, end_time, end_strength = detect_start_end_claps(
                y, sr, duration, clap_start_window, clap_end_window, clap_min_interval, start_after=start_after
            )
            start_claps.append(float(start_time))
            start_strengths.append(float(start_strength))
            end_claps.append(float(end_time))
            end_strengths.append(float(end_strength))
            continue
        if start_claps is not None and start_strengths is not None:
            clap_time, strength = detect_start_clap(y, sr, duration, clap_start_window, start_after=start_after)
            start_claps.append(float(clap_time))
            start_strengths.append(float(strength))
        if end_claps is not None and end_strengths is not None:
            clap_time, strength = detect_end_clap(y, sr, duration, clap_end_window)
            end_claps.append(float(clap_time))
            end_strengths.append(float(strength))
    source_ends: list[float] | None = None
    time_warp = False
    if normalized_mode == "start_clap":
        assert start_claps is not None
        starts, overlap = _common_overlap_from_sync_points(durations, start_claps)
        offsets = [float(t) - float(start_claps[0]) for t in start_claps]
    elif normalized_mode == "end_clap":
        assert end_claps is not None
        starts, overlap = _common_overlap_from_sync_points(durations, end_claps)
        offsets = [float(t) - float(end_claps[0]) for t in end_claps]
    elif normalized_mode == "both":
        assert start_claps is not None and end_claps is not None
        source_ends = [float(t) for t in end_claps]
        intervals = [source_ends[i] - float(start_claps[i]) for i in range(len(videos))]
        bad = [i for i, interval in enumerate(intervals) if interval < clap_min_interval]
        if bad:
            details = {camera_name_from_index(i): {"start": start_claps[i], "end": source_ends[i], "interval": intervals[i], "min_interval": clap_min_interval} for i in bad}
            raise RuntimeError(f"Invalid start/end clap ordering or too-small clap interval: {details}")
        starts = [float(t) for t in start_claps]
        overlap = min(intervals)
        offsets = [float(t) - float(start_claps[0]) for t in start_claps]
        time_warp = True
    else:
        raise ValueError(f"Unsupported sync mode: {sync_mode}")
    if overlap <= 0.0:
        raise RuntimeError(f"No shared synchronized interval found for sync mode {sync_mode}: overlap={overlap:.6f}s")
    sync_info: dict[str, Any] = {
        "mode": normalized_mode,
        "videos": [str(v) for v in videos],
        "durations": durations,
        "start_claps": start_claps,
        "end_claps": end_claps,
        "start_strengths": start_strengths,
        "end_strengths": end_strengths,
        "clap_start_after": {camera_name_from_index(i): float(v) for i, v in sorted(clap_start_after.items())},
        "offsets_vs_video0": offsets,
        "starts": starts,
        "source_ends": source_ends,
        "time_warp": time_warp,
        "overlap_duration": float(overlap),
    }
    if normalized_mode == "both" and start_claps is not None and end_claps is not None:
        intervals = [float(end_claps[i]) - float(start_claps[i]) for i in range(len(videos))]
        sync_info["clap_intervals"] = intervals
        sync_info["time_scales"] = [float(overlap) / interval for interval in intervals]
    return sync_info
def colmap_help(logger: Logger, subcommand: str) -> str:
    return logger.capture(["colmap", subcommand, "-h"], label=f"colmap_{subcommand}_help")
def colmap_option_supported(help_text: str, option: str) -> bool:
    return option in help_text
def build_colmap_cmd(logger: Logger, subcommand: str, base_args: list[str], optional_pairs: list[tuple[str, str]]) -> list[str]:
    help_text = colmap_help(logger, subcommand)
    cmd = ["colmap", subcommand] + base_args
    included: list[str] = []
    skipped: list[str] = []
    for key, value in optional_pairs:
        if colmap_option_supported(help_text, key):
            cmd.extend([key, value])
            included.append(f"{key}={value}")
        else:
            skipped.append(key)
    logger.log(f"COLMAP {subcommand}: included optional args: {included if included else 'none'}")
    if skipped:
        logger.log(f"COLMAP {subcommand}: skipped unsupported args: {skipped}")
    return cmd
def main() -> int:
    ap = argparse.ArgumentParser(description="Build a trainable 4DGS dataset directly from synced phone videos")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene-name", default=None)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--colmap-fps", type=float, default=6.0)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--jpeg-quality", type=int, default=2)
    ap.add_argument("--audio-sr", type=int, default=16000)
    ap.add_argument("--sync-mode", choices=["start_clap", "end_clap", "both", "both_claps"], default="end_clap")
    ap.add_argument("--clap-start-window", type=float, default=8.0)
    ap.add_argument("--clap-end-window", type=float, default=8.0)
    ap.add_argument("--clap-min-interval", type=float, default=1.0, help="Minimum seconds between detected start and end claps in --sync-mode both")
    ap.add_argument("--clap-start-after", action="append", default=[], metavar="VIDEO=SECONDS", help="Ignore start-clap candidates before SECONDS for one video. VIDEO can be camNN, zero-based index, filename, stem, or path. May be repeated or comma-separated.")
    ap.add_argument("--mask-frac", type=float, default=0.35)
    ap.add_argument("--camera-model", default="SIMPLE_PINHOLE")
    ap.add_argument("--test-camera", type=int, default=0)
    ap.add_argument("--split-mode", choices=["random_every", "camera_holdout"], default="random_every")
    ap.add_argument("--test-every", type=int, default=20)
    ap.add_argument("--test-seed", type=int, default=0)
    ap.add_argument("--force-reextract", action="store_true")
    ap.add_argument("--force-colmap", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.out).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    scene_name = args.scene_name or out_root.name
    prep_dir = out_root / "_prep"
    logs_dir = prep_dir / "logs"
    logger = Logger(logs_dir, verbose=True if args.verbose else True)
    require_binary("ffmpeg")
    require_binary("ffprobe")
    require_binary("colmap")
    logger.log(f"Starting pipeline with args: {vars(args)}")
    logger.log(f"COLMAP binary: {shutil.which('colmap')}")
    try:
        logger.log(f"COLMAP help first line: {logger.capture(['colmap', '-h'], label='colmap_root_help').splitlines()[0]}")
    except Exception as exc:
        logger.log(f"Could not capture COLMAP root help: {exc}")
    videos = resolve_video_inputs(args.videos)
    logger.log(f"Resolved videos: {[str(v) for v in videos]}")
    audio_dir = prep_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    clap_start_after = parse_per_video_float_overrides(args.clap_start_after, videos, "--clap-start-after")
    if clap_start_after:
        start_after_summary = {camera_name_from_index(i): v for i, v in sorted(clap_start_after.items())}
        logger.log(f"Per-video start-clap lower bounds: {start_after_summary}")
    sync_info = determine_sync_from_claps(
        videos=videos,
        audio_dir=audio_dir,
        audio_sr=args.audio_sr,
        sync_mode=args.sync_mode,
        clap_start_window=args.clap_start_window,
        clap_end_window=args.clap_end_window,
        clap_min_interval=args.clap_min_interval,
        clap_start_after=clap_start_after,
        logger=logger,
    )
    starts = [float(x) for x in sync_info["starts"]]
    overlap_duration = float(sync_info["overlap_duration"])
    source_ends = sync_info.get("source_ends")
    if source_ends is not None:
        source_ends = [float(x) for x in source_ends]
    time_warp = bool(sync_info.get("time_warp", False))
    write_json(prep_dir / "sync.json", sync_info)
    logger.log(f"Sync summary: {sync_info}")
    if overlap_duration <= 0.1:
        raise RuntimeError(f"Shared overlap duration is too small: {overlap_duration:.3f}s")
    full_raw_images = prep_dir / "full_raw_images"
    full_raw_images.parent.mkdir(parents=True, exist_ok=True)
    nframes = extract_synced_pngs(
        videos=videos,
        starts=starts,
        overlap_duration=overlap_duration,
        out_root=full_raw_images,
        fps=args.fps,
        width=args.width,
        jpeg_quality=args.jpeg_quality,
        logger=logger,
        force=args.force_reextract,
        source_ends=source_ends,
        time_warp=time_warp,
    )
    logger.log(f"Final dataset extraction frame count: {nframes}")
    if nframes < 2:
        raise RuntimeError(f"Too few synced frames extracted: {nframes}")
    pose_indices = build_pose_indices(nframes, args.fps, args.colmap_fps)
    logger.log(f"Pose subset uses {len(pose_indices)} frame indices out of {nframes}: first={pose_indices[:10]} last={pose_indices[-10:]}")
    pose_images = prep_dir / "pose_images"
    pose_images_nomask = prep_dir / "pose_images_nomask"
    pose_manifest_rows = build_pose_subset_from_full(
        full_raw_images=full_raw_images,
        pose_images=pose_images,
        pose_indices=pose_indices,
        nframes=nframes,
        mask_frac=args.mask_frac,
        meta_path=prep_dir / "pose_subset_meta.json",
        force=args.force_reextract,
        logger=logger,
    )
    pose_manifest_rows_nomask = build_pose_subset_from_full(
        full_raw_images=full_raw_images,
        pose_images=pose_images_nomask,
        pose_indices=pose_indices,
        nframes=nframes,
        mask_frac=0.0,
        meta_path=prep_dir / "pose_subset_nomask_meta.json",
        force=args.force_reextract,
        logger=logger,
    )
    with (prep_dir / "manifest_pose.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "camera_name", "frame_index", "time_norm", "local_time_seconds"])
        writer.writerows(pose_manifest_rows)
    with (prep_dir / "manifest_pose_nomask.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "camera_name", "frame_index", "time_norm", "local_time_seconds"])
        writer.writerows(pose_manifest_rows_nomask)
    pose_images_flat = prep_dir / "pose_images_flat"
    pose_images_flat_nomask = prep_dir / "pose_images_flat_nomask"
    pose_flat_manifest_csv = prep_dir / "manifest_pose_flat.csv"
    pose_flat_manifest_csv_nomask = prep_dir / "manifest_pose_flat_nomask.csv"
    pose_flat_rows = build_pose_flat_import_from_subset(
        pose_images=pose_images,
        flat_root=pose_images_flat,
        nframes=nframes,
        csv_path=pose_flat_manifest_csv,
        meta_path=prep_dir / "pose_flat_meta.json",
        force=args.force_reextract,
        logger=logger,
    )
    pose_flat_rows_nomask = build_pose_flat_import_from_subset(
        pose_images=pose_images_nomask,
        flat_root=pose_images_flat_nomask,
        nframes=nframes,
        csv_path=pose_flat_manifest_csv_nomask,
        meta_path=prep_dir / "pose_flat_nomask_meta.json",
        force=args.force_reextract,
        logger=logger,
    )
    pose_rows_by_camera = group_pose_rows_by_camera(pose_flat_rows)
    pose_rows_by_camera_nomask = group_pose_rows_by_camera(pose_flat_rows_nomask)
    pose_import_by_camera = prep_dir / "pose_import_by_camera"
    pose_import_by_camera_nomask = prep_dir / "pose_import_by_camera_nomask"
    build_per_camera_import_roots_from_flat(
        flat_root=pose_images_flat,
        rows=pose_flat_rows,
        bycam_root=pose_import_by_camera,
        meta_path=prep_dir / "pose_import_by_camera_meta.json",
        force=args.force_reextract,
        logger=logger,
    )
    build_per_camera_import_roots_from_flat(
        flat_root=pose_images_flat_nomask,
        rows=pose_flat_rows_nomask,
        bycam_root=pose_import_by_camera_nomask,
        meta_path=prep_dir / "pose_import_by_camera_nomask_meta.json",
        force=args.force_reextract,
        logger=logger,
    )
    full_inventory = collect_image_inventory(full_raw_images)
    pose_inventory = collect_image_inventory(pose_images)
    pose_inventory_nomask = collect_image_inventory(pose_images_nomask)
    pose_flat_inventory = collect_root_image_inventory(pose_images_flat)
    pose_flat_inventory_nomask = collect_root_image_inventory(pose_images_flat_nomask)
    write_json(prep_dir / "full_raw_inventory.json", full_inventory)
    write_json(prep_dir / "pose_inventory.json", pose_inventory)
    write_json(prep_dir / "pose_inventory_nomask.json", pose_inventory_nomask)
    write_json(prep_dir / "pose_flat_inventory.json", pose_flat_inventory)
    write_json(prep_dir / "pose_flat_inventory_nomask.json", pose_flat_inventory_nomask)
    logger.log(f"Full extraction inventory: {full_inventory['total_images']} images")
    logger.log(f"Pose subset inventory: {pose_inventory['total_images']} images")
    logger.log(f"Flat COLMAP pose inventory: {pose_flat_inventory['total_images']} images")
    logger.log(f"Unmasked pose subset inventory: {pose_inventory_nomask['total_images']} images")
    logger.log(f"Unmasked flat COLMAP pose inventory: {pose_flat_inventory_nomask['total_images']} images")
    per_cam_import_inventory = collect_image_inventory(pose_import_by_camera)
    per_cam_import_inventory_nomask = collect_image_inventory(pose_import_by_camera_nomask)
    write_json(prep_dir / "pose_import_by_camera_inventory.json", per_cam_import_inventory)
    write_json(prep_dir / "pose_import_by_camera_nomask_inventory.json", per_cam_import_inventory_nomask)
    logger.log(f"Per-camera COLMAP feature roots inventory: {per_cam_import_inventory['total_images']} images")
    logger.log(f"Unmasked per-camera COLMAP feature roots inventory: {per_cam_import_inventory_nomask['total_images']} images")
    if pose_inventory["total_images"] == 0:
        raise RuntimeError(f"Pose image directory is empty: {pose_images}")
    if pose_flat_inventory["total_images"] == 0:
        raise RuntimeError(f"Flat COLMAP pose image directory is empty: {pose_images_flat}")
    first_pose_image = next(pose_images_flat_nomask.glob("*.jpg"), None)
    if first_pose_image is None:
        first_pose_image = next(pose_images_flat_nomask.glob("*.png"), None)
    if first_pose_image is None:
        first_pose_image = next(pose_images_flat.glob("*.jpg"), None)
    if first_pose_image is None:
        first_pose_image = next(pose_images_flat.glob("*.png"), None)
    if first_pose_image is None:
        raise RuntimeError(f"No flat pose images found under {pose_images_flat}")
    w0, h0 = verify_sample_image(first_pose_image)
    logger.log(f"Verified sample pose image {first_pose_image} with size {w0}x{h0}")
    db_path = prep_dir / "colmap" / "database.db"
    colmap_dir = prep_dir / "colmap"
    sparse_dir = colmap_dir / "sparse"
    sparse_txt = prep_dir / "sparse_txt"
    colmap_meta_path = prep_dir / "colmap_meta.json"
    colmap_meta = {
        "camera_model": args.camera_model,
        "mask_frac": round(float(args.mask_frac), 6),
        "pose_indices": [int(x) for x in pose_indices],
        "videos": [str(v.resolve()) for v in videos],
        "pose_image_total": int(pose_inventory["total_images"]),
        "pose_flat_image_total": int(pose_flat_inventory["total_images"]),
        "pose_flat_nomask_image_total": int(pose_flat_inventory_nomask["total_images"]),
        "colmap_fps": float(args.colmap_fps),
    }
    can_reuse_colmap = (
        (not args.force_colmap)
        and load_json(colmap_meta_path) == colmap_meta
        and db_path.exists()
        and (sparse_txt / "cameras.txt").exists()
        and (sparse_txt / "images.txt").exists()
        and (sparse_txt / "points3D.txt").exists()
    )
    if can_reuse_colmap:
        logger.log("Reusing existing COLMAP outputs")
        stats_after_features = database_stats(db_path)
        stats_after_matches = stats_after_features.copy()
    else:
        feature_help = colmap_help(logger, "feature_extractor")
        supports_single_camera = colmap_option_supported(feature_help, "--ImageReader.single_camera")
        feature_attempts: list[dict[str, Any]] = []
        def run_feature_attempt(name: str, cmd: list[str], log_label: str) -> tuple[Path, dict[str, int]]:
            nonlocal db_path, sparse_dir, sparse_txt
            db_path = reset_colmap_workspace(colmap_dir, sparse_dir, sparse_txt)
            logger.log(f"Trying COLMAP feature import strategy '{name}'")
            logger.run(cmd, label=log_label)
            stats = database_stats(db_path)
            write_json(prep_dir / f"database_stats_after_features_{name}.json", stats)
            stderr_tail = read_log_tail(logger.log_dir / f"{logger.command_counter:02d}_{log_label}.stderr.log")
            feature_attempts.append({"strategy": name, "cmd": cmd, "stats": stats, "stderr_tail": stderr_tail})
            logger.log(f"Feature import strategy '{name}' stats: {stats}")
            return db_path, stats
        base_common = ["--SiftExtraction.max_num_features", "8192"]
        if "--SiftExtraction.use_gpu" in feature_help:
            base_common.extend(["--SiftExtraction.use_gpu", "0"])
        if "--SiftExtraction.num_threads" in feature_help:
            base_common.extend(["--SiftExtraction.num_threads", "-1"])
        if "--ImageReader.default_focal_length_factor" in feature_help:
            base_common.extend(["--ImageReader.default_focal_length_factor", "1.2"])
        def try_import_variant(prefix: str, flat_root: Path, flat_inventory: dict[str, Any], bycam_root: Path, rows_by_camera_local: dict[str, list[dict[str, Any]]], use_camera_model: bool) -> dict[str, Any]:
            flat_cmd = [
                "colmap", "feature_extractor",
                "--database_path", str(colmap_dir / "database.db"),
                "--image_path", str(flat_root),
            ]
            if use_camera_model:
                flat_cmd.extend(["--ImageReader.camera_model", args.camera_model])
            flat_cmd += base_common
            logger.log(f"COLMAP import preflight {prefix} flat root: {flat_inventory['total_images']} images under {flat_root}")
            _, stats = run_feature_attempt(f"{prefix}_flat_{'with_model' if use_camera_model else 'default'}", flat_cmd, f"colmap_feature_extractor_{prefix}_{'with_model' if use_camera_model else 'default'}")
            if has_usable_colmap_features(stats):
                return stats
            db_local = reset_colmap_workspace(colmap_dir, sparse_dir, sparse_txt)
            logger.log(f"Falling back to per-camera COLMAP feature extraction roots under {bycam_root} for {prefix}")
            for cam_name, rows in sorted(rows_by_camera_local.items()):
                cam_root = bycam_root / cam_name
                cmd = [
                    "colmap", "feature_extractor",
                    "--database_path", str(db_local),
                    "--image_path", str(cam_root),
                ] + base_common
                if use_camera_model:
                    cmd.extend(["--ImageReader.camera_model", args.camera_model])
                if supports_single_camera:
                    cmd.extend(["--ImageReader.single_camera", "1"])
                logger.log(f"COLMAP import preflight for {prefix}/{cam_name}: {len(rows)} images under {cam_root}")
                logger.run(cmd, label=f"colmap_feature_extractor_{prefix}_{cam_name}")
            stats = database_stats(db_local)
            feature_attempts.append({"strategy": f"{prefix}_per_camera", "stats": stats})
            return stats
        stats_after_features = try_import_variant("nomask", pose_images_flat_nomask, pose_flat_inventory_nomask, pose_import_by_camera_nomask, pose_rows_by_camera_nomask, True)
        if not has_usable_colmap_features(stats_after_features):
            stats_after_features = try_import_variant("nomask", pose_images_flat_nomask, pose_flat_inventory_nomask, pose_import_by_camera_nomask, pose_rows_by_camera_nomask, False)
        if not has_usable_colmap_features(stats_after_features):
            stats_after_features = try_import_variant("masked", pose_images_flat, pose_flat_inventory, pose_import_by_camera, pose_rows_by_camera, True)
        if not has_usable_colmap_features(stats_after_features):
            stats_after_features = try_import_variant("masked", pose_images_flat, pose_flat_inventory, pose_import_by_camera, pose_rows_by_camera, False)
        stats_after_matches = stats_after_features.copy()
        write_json(prep_dir / "feature_import_attempts.json", feature_attempts)
        write_json(prep_dir / "database_stats_after_features.json", stats_after_features)
        logger.log(f"Database stats after features: {stats_after_features}")
        if not has_usable_colmap_features(stats_after_features):
            raise RuntimeError(
                "COLMAP feature_extractor did not produce usable image/frame/feature records after multiple import strategies. "
                "See _prep/feature_import_attempts.json and the feature_extractor stderr logs. "
                f"Final database stats: {stats_after_features}"
            )
        matcher_cmd = build_colmap_cmd(
            logger,
            "exhaustive_matcher",
            ["--database_path", str(db_path)],
            [
                ("--SiftMatching.use_gpu", "0"),
                ("--SiftMatching.num_threads", "-1"),
            ],
        )
        logger.run(matcher_cmd, label="colmap_exhaustive_matcher")
        stats_after_matches = database_stats(db_path)
        write_json(prep_dir / "database_stats_after_matches.json", stats_after_matches)
        logger.log(f"Database stats after matches: {stats_after_matches}")
        if int(stats_after_matches.get("two_view_geometries", 0)) == 0 and int(stats_after_matches.get("matches", 0)) == 0:
            raise RuntimeError(
                "COLMAP produced no verified image pairs. Try smaller --mask-frac, smaller --colmap-fps, or stronger background texture. "
                f"Database stats: {stats_after_matches}"
            )
        mapper_cmd = build_colmap_cmd(
            logger,
            "mapper",
            [
                "--database_path", str(db_path),
                "--image_path", str(pose_images_flat),
                "--output_path", str(sparse_dir),
            ],
            [
                ("--Mapper.multiple_models", "0"),
                ("--Mapper.init_min_num_inliers", "20"),
                ("--Mapper.abs_pose_min_num_inliers", "20"),
            ],
        )
        logger.run(mapper_cmd, label="colmap_mapper")
        model0 = sparse_dir / "0"
        if not model0.exists():
            raise RuntimeError("COLMAP mapping did not produce sparse/0. Check _prep/logs and try --colmap-fps 3 --mask-frac 0.2 --width 1280")
        logger.run([
            "colmap", "model_converter",
            "--input_path", str(model0),
            "--output_path", str(sparse_txt),
            "--output_type", "TXT",
        ], label="colmap_model_converter")
        write_json(colmap_meta_path, colmap_meta)
    cameras = parse_colmap_cameras_txt(sparse_txt / "cameras.txt")
    images = parse_colmap_images_txt(sparse_txt / "images.txt")
    xyz, rgb = parse_colmap_points_txt(sparse_txt / "points3D.txt")
    logger.log(f"Parsed COLMAP text model: {len(cameras)} cameras, {len(images)} registered images, {len(xyz)} points")
    if not images:
        raise RuntimeError("COLMAP registered zero images in the sparse model")
    images_dir = out_root / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    shutil.copytree(full_raw_images, images_dir)
    points3d_path = out_root / "points3d.ply"
    if len(xyz) == 0:
        xyz = np.random.uniform(-1.0, 1.0, size=(100000, 3)).astype(np.float32)
        rgb = np.full((100000, 3), 127, dtype=np.uint8)
    write_points3d_ply(points3d_path, xyz, rgb)
    pose_meta_by_name = {row["flat_name"]: row for row in pose_flat_rows}
    transforms_train, transforms_test, transforms_all, registered_per_camera = build_frames_for_dataset(
        full_raw_images=full_raw_images,
        cameras=cameras,
        images=images,
        nframes=nframes,
        test_camera=args.test_camera,
        image_meta_by_name=pose_meta_by_name,
        logger=logger,
    )
    write_json(out_root / "transforms_train.json", transforms_train)
    write_json(out_root / "transforms_test.json", transforms_test)
    write_json(out_root / "transforms_val.json", transforms_test)
    write_json(out_root / "transforms_all.json", transforms_all)
    cfg_path = repo_root / "configs" / "dnerf" / f"{scene_name}.yaml"
    model_path = repo_root / "output" / "dnerf" / scene_name
    write_repo_yaml(cfg_path, out_root, model_path)
    run_train = out_root / "run_train.sh"
    run_train.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\nuv run train.py --config configs/dnerf/{scene_name}.yaml\n", encoding="utf-8")
    os.chmod(run_train, 0o755)
    summary = {
        "scene_name": scene_name,
        "videos": [str(v) for v in videos],
        "nframes": nframes,
        "pose_frames": len(pose_indices),
        "sync": sync_info,
        "full_inventory": full_inventory,
        "pose_inventory": pose_inventory,
        "pose_flat_inventory": pose_flat_inventory,
        "database_stats_after_features": stats_after_features,
        "database_stats_after_matches": stats_after_matches,
        "colmap_registered_images": len(images),
        "registered_pose_samples_per_camera": registered_per_camera,
        "colmap_points": int(len(xyz)),
        "dataset_root": str(out_root),
        "config": str(cfg_path),
        "train_command": f"uv run train.py --config configs/dnerf/{scene_name}.yaml",
    }
    write_json(out_root / "prep_summary.json", summary)
    logger.log(f"Finished successfully. Summary: {summary}")
    return 0
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
