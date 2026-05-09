#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
DEFAULT_TAGS = [
    "test/loss_viewpoint - psnr",  # tag written by this repo's train.py
    "test/psnr",
    "eval/psnr",
    "metrics/psnr",
]


def to_yamlable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): to_yamlable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_yamlable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().tolist()
    if hasattr(x, "item") and callable(x.item):
        try:
            return x.item()
        except Exception:
            pass
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def checkpoint_sort_key(p: Path) -> tuple[int, int, float]:
    if p.name == "chkpnt_best.pth":
        return (2, sys.maxsize, p.stat().st_mtime)
    m = re.search(r"chkpnt(\d+)\.pth$", p.name)
    if m:
        return (1, int(m.group(1)), p.stat().st_mtime)
    return (0, -1, p.stat().st_mtime)


def find_checkpoint(run_dir: Path, glob: str) -> Path | None:
    ckpts = [p for p in run_dir.glob(glob) if p.is_file()]
    return max(ckpts, key=checkpoint_sort_key) if ckpts else None


def read_tb_psnr(run_dir: Path, metric_tag: str | None, mode: str) -> tuple[float, int, str] | None:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise SystemExit("Install TensorBoard first: pip install tensorboard") from exc

    if not list(run_dir.glob("events.out.tfevents.*")):
        return None

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])

    if metric_tag:
        psnr_tags = [metric_tag] if metric_tag in tags else []
    else:
        psnr_tags = [t for t in DEFAULT_TAGS if t in tags]
        psnr_tags = psnr_tags or [t for t in tags if "psnr" in t.lower()]
    if not psnr_tags:
        return None

    picked: tuple[float, int, str] | None = None
    for tag in psnr_tags:
        events = acc.Scalars(tag)
        if not events:
            continue
        if mode == "last":
            ev = max(events, key=lambda e: e.step)
        elif mode == "min":
            ev = min(events, key=lambda e: e.value)
        else:
            ev = max(events, key=lambda e: e.value)
        candidate = (float(ev.value), int(ev.step), tag)
        if picked is None:
            picked = candidate
        elif mode == "min" and candidate[0] < picked[0]:
            picked = candidate
        elif mode != "min" and candidate[0] > picked[0]:
            picked = candidate
    return picked


def load_checkpoint_config(ckpt_path: Path) -> dict[str, Any] | None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, Mapping):
        return None
    run_config = ckpt.get("run_config", {})
    args = run_config.get("args", {}) if isinstance(run_config, Mapping) else {}
    if not isinstance(args, Mapping):
        return None
    cfg = dict(to_yamlable(args))

    # Make the generated YAML start a clean rerun. In this codebase, a non-empty
    # start_checkpoint causes train.py to ignore the YAML and load immutable args
    # from that checkpoint instead.
    cfg["config"] = None
    cfg["start_checkpoint"] = None
    return cfg


def run_name(run_dir: Path, root: Path) -> str:
    rel = run_dir.resolve().relative_to(root.resolve())
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "__".join(rel.parts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Folder containing training run folders")
    ap.add_argument("--threshold", type=float, default=15.0)
    ap.add_argument("--out-dir", type=Path, default=Path("rerun_cfg"))
    ap.add_argument("--metric-tag", default=None, help="Default auto-detects a PSNR scalar")
    ap.add_argument("--mode", choices=["best", "last", "min"], default="best")
    ap.add_argument("--checkpoint-glob", default="chkpnt*.pth")
    ap.add_argument("--model-root", type=Path, default=None, help="Optional new model_path root")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted({p.parent for p in root.rglob(args.checkpoint_glob) if p.is_file()})
    wrote = skipped = 0

    for rd in run_dirs:
        ckpt = find_checkpoint(rd, args.checkpoint_glob)
        psnr = read_tb_psnr(rd, args.metric_tag, args.mode)
        if ckpt is None or psnr is None:
            skipped += 1
            print(f"SKIP {rd}: missing checkpoint or TensorBoard PSNR")
            continue

        value, step, tag = psnr
        if value >= args.threshold:
            print(f"OK   {rd}: {tag}={value:.3f} at step {step}")
            continue

        cfg = load_checkpoint_config(ckpt)
        if cfg is None:
            skipped += 1
            print(f"SKIP {rd}: no checkpoint['run_config']['args'] in {ckpt}")
            continue

        name = run_name(rd, root)
        if args.model_root is not None:
            cfg["model_path"] = str(args.model_root / name)

        out_path = args.out_dir / f"{name}.yml"
        print(f"LOW  {rd}: {tag}={value:.3f}; {'would write' if args.dry_run else 'writing'} {out_path}")
        if not args.dry_run:
            header = f"# from: {ckpt}\n# {tag}={value:.6f} at step {step}\n"
            out_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            wrote += 1

    print(f"done: wrote {wrote}, skipped {skipped}, scanned {len(run_dirs)} run dirs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
