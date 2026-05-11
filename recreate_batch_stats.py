#!/usr/bin/env python3
"""
Recreate batch_train.py-style metrics from existing 4DGS checkpoints.

Put this file in the repo root or pass --repo-root. It scans a checkpoint folder,
evaluates the final/best checkpoint in each run directory, evaluates all chkpnt*.pth
history checkpoints, writes per-run run_metrics.json plus checkpoint_eval_metrics.*,
and writes combined ablation_metrics.* at --out-root.

This intentionally calls batch_train.evaluate_checkpoint() and the same CSV/JSONL
writers, so PSNR/SSIM/LPIPS/FPS/gaussian-count fields are computed the same way as
batch_train metrics-only runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def add_repo_to_path(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_checkpoint_cpu(path: Path) -> Dict[str, Any]:
    from utils.checkpoint_utils import load_checkpoint

    return load_checkpoint(path, map_location="cpu")


def plain_run_args(checkpoint_payload: Mapping[str, Any]) -> Dict[str, Any]:
    run_config = checkpoint_payload.get("run_config", {})
    args = run_config.get("args", {}) if isinstance(run_config, Mapping) else {}
    return dict(args) if isinstance(args, Mapping) else {}


def infer_scene_name(run_dir: Path, checkpoint_payload: Mapping[str, Any], layout: str) -> str:
    run_args = plain_run_args(checkpoint_payload)
    source_path = str(run_args.get("source_path", "") or "")
    if layout == "scene_variant" and run_dir.parent.name:
        return run_dir.parent.name
    if source_path:
        return Path(source_path).name
    return run_dir.parent.name or run_dir.name


def infer_variant_name(run_dir: Path, checkpoint_path: Path, checkpoint_payload: Mapping[str, Any], layout: str) -> str:
    run_args = plain_run_args(checkpoint_payload)
    model_path = str(run_args.get("model_path", "") or "")
    if layout in {"scene_variant", "variant"}:
        return run_dir.name
    if model_path:
        return Path(model_path).name
    return checkpoint_path.stem


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def infer_variant_tags(checkpoint_payload: Mapping[str, Any], variant_name: str) -> Dict[str, str]:
    run_args = plain_run_args(checkpoint_payload)
    run_config = checkpoint_payload.get("run_config", {})
    gaussian_kwargs = run_config.get("gaussian_kwargs", {}) if isinstance(run_config, Mapping) else {}
    if not isinstance(gaussian_kwargs, Mapping):
        gaussian_kwargs = {}

    sh_degree = int(run_args.get("sh_degree", gaussian_kwargs.get("sh_degree", 3)) or 0)
    force_sh_3d = truthy(run_args.get("force_sh_3d", False))
    eval_shfs_4d = truthy(run_args.get("eval_shfs_4d", not force_sh_3d))

    if sh_degree <= 0:
        appearance = "rgb"
    elif sh_degree == 1 and force_sh_3d:
        appearance = "sh1"
    elif sh_degree == 3 and (not force_sh_3d) and eval_shfs_4d:
        appearance = "sh3"
    elif sh_degree == 3 and force_sh_3d:
        appearance = "sh3_3d"
    else:
        appearance = f"sh{sh_degree}"

    tags: Dict[str, str] = {
        "isotropy": "isotropic" if truthy(run_args.get("isotropic_gaussians", gaussian_kwargs.get("isotropic_gaussians", False))) else "anisotropic",
        "appearance": appearance,
        "sorting": "sort_free" if truthy(run_args.get("sort_free_render", False)) else "sort",
        "usplat": "use_usplat" if truthy(run_args.get("use_usplat", False)) else "no_usplat",
        "dropout": "dropout" if float(run_args.get("random_dropout_prob", 0.0) or 0.0) > 0.0 else "no_dropout",
        "ess": "ess" if truthy(run_args.get("enable_edge_guided_splitting", False)) else "no_ess",
    }

    prune_ratio = float(run_args.get("spacetime_prune_ratio", run_args.get("prune_ratio", 0.0)) or 0.0)
    prune_from = int(run_args.get("spacetime_prune_from_iter", run_args.get("prune_from_iter", -1)) or -1)
    tags["pruning"] = "interleaved_prune_densify" if prune_ratio > 0.0 and prune_from >= 0 else "no_pruning"

    lower_name = variant_name.lower()
    if lower_name.startswith("paper_"):
        tags["matrix_preset"] = "paper"
    elif lower_name.startswith("essential_"):
        tags["matrix_preset"] = "essential"
    elif lower_name.startswith("compact_"):
        tags["matrix_preset"] = "compact"

    if "mobilegs" in lower_name:
        tags["method_family"] = "mobilegs"
    elif "instant4d" in lower_name:
        tags["method_family"] = "instant4d"
    elif "usplat" in lower_name:
        tags["method_family"] = "usplat4d"
    elif "dropout" in lower_name:
        tags["method_family"] = "dropoutgs"
    elif "4dgs1k" in lower_name or "st_prune" in lower_name:
        tags["method_family"] = "4dgs_1k"
    elif "hybrid" in lower_name:
        tags["method_family"] = "hybrid"
    else:
        tags["method_family"] = "4dgs"

    return tags


def checkpoint_iteration(checkpoint_payload: Mapping[str, Any], checkpoint_path: Path) -> int | None:
    try:
        return int(checkpoint_payload.get("iteration"))
    except Exception:
        pass
    stem = checkpoint_path.stem
    if stem.startswith("chkpnt") and stem[len("chkpnt"):].isdigit():
        return int(stem[len("chkpnt"):])
    return None


def guess_iterations(run_dir: Path, checkpoints: Sequence[Path], payloads: Mapping[Path, Mapping[str, Any]]) -> int:
    vals: List[int] = []
    for ckpt in checkpoints:
        payload = payloads.get(ckpt)
        if payload is not None:
            args = plain_run_args(payload)
            try:
                vals.append(int(args.get("iterations")))
            except Exception:
                pass
            it = checkpoint_iteration(payload, ckpt)
            if it is not None:
                vals.append(it)
    return max(vals) if vals else 0


def checkpoint_name_iteration(path: Path) -> int | None:
    stem = path.stem
    if stem == "chkpnt_best":
        return None
    if stem.startswith("chkpnt") and stem[len("chkpnt"):].isdigit():
        return int(stem[len("chkpnt"):])
    return None


def final_checkpoint_for_run(run_dir: Path, checkpoints: Sequence[Path], iterations: int, payloads: Mapping[Path, Mapping[str, Any]]) -> Path:
    exact = run_dir / f"chkpnt{iterations}.pth"
    if exact.exists():
        return exact
    best = run_dir / "chkpnt_best.pth"
    if best.exists():
        return best

    numeric: List[tuple[int, Path]] = []
    for ckpt in checkpoints:
        it = checkpoint_name_iteration(ckpt)
        if it is not None:
            numeric.append((it, ckpt))
    if numeric:
        return max(numeric, key=lambda x: x[0])[1]

    # Fallback for nonstandard names: choose the checkpoint with the largest saved iteration.
    scored: List[tuple[int, Path]] = []
    for ckpt in checkpoints:
        payload = payloads.get(ckpt)
        if payload is None:
            continue
        it = checkpoint_iteration(payload, ckpt)
        if it is not None:
            scored.append((it, ckpt))
    if scored:
        return max(scored, key=lambda x: x[0])[1]
    return sorted(checkpoints)[-1]


def history_checkpoints(run_dir: Path, checkpoints: Sequence[Path]) -> List[Path]:
    numeric: List[tuple[int, Path]] = []
    for ckpt in checkpoints:
        it = checkpoint_name_iteration(ckpt)
        if it is not None:
            numeric.append((it, ckpt))
    out = [p for _, p in sorted(numeric, key=lambda x: x[0])]
    best = run_dir / "chkpnt_best.pth"
    if best.exists() and all(p.resolve() != best.resolve() for p in out):
        out.append(best)
    if out:
        return out
    return sorted(checkpoints)


def stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def find_checkpoint_files(root: Path, checkpoint_glob: str) -> List[Path]:
    files = [p for p in root.glob(checkpoint_glob) if p.is_file()]
    return sorted({p.resolve() for p in files})


def group_by_parent(checkpoints: Sequence[Path]) -> Dict[Path, List[Path]]:
    grouped: Dict[Path, List[Path]] = {}
    for ckpt in checkpoints:
        grouped.setdefault(ckpt.parent.resolve(), []).append(ckpt.resolve())
    return dict(sorted(grouped.items(), key=lambda kv: str(kv[0])))


def build_metric_args(parsed: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        eval_split=parsed.eval_split,
        checkpoint_eval_split=parsed.checkpoint_eval_split,
        render_fps_warmup=parsed.render_fps_warmup,
        checkpoint_metrics_filename=parsed.checkpoint_metrics_filename,
        checkpoint_metrics_jsonl_filename=parsed.checkpoint_metrics_jsonl_filename,
        skip_checkpoint_metrics=parsed.skip_checkpoint_metrics,
        mobilegs_report=parsed.mobilegs_report,
        mobilegs_report_scope=parsed.mobilegs_report_scope,
        require_mobilegs_report=False,
        mobilegs_benchmark_render_mode=parsed.mobilegs_benchmark_render_mode,
        mobilegs_force_first_order_sh=parsed.mobilegs_force_first_order_sh,
        mobilegs_mobile_filename=parsed.mobilegs_mobile_filename,
        mobilegs_metrics_filename=parsed.mobilegs_metrics_filename,
        mobilegs_codebook_size=parsed.mobilegs_codebook_size,
        mobilegs_block_size=parsed.mobilegs_block_size,
        mobilegs_kmeans_iters=parsed.mobilegs_kmeans_iters,
        mobilegs_uniform_bits=parsed.mobilegs_uniform_bits,
        mobilegs_build_visibility_filter=parsed.mobilegs_build_visibility_filter,
        mobilegs_temporal_keyframes=parsed.mobilegs_temporal_keyframes,
        mobilegs_temporal_mask_window=parsed.mobilegs_temporal_mask_window,
        mobilegs_temporal_mask_threshold=parsed.mobilegs_temporal_mask_threshold,
        mobilegs_views_per_keyframe=parsed.mobilegs_views_per_keyframe,
        mobilegs_benchmark_split=parsed.mobilegs_benchmark_split,
        mobilegs_benchmark_warmup=parsed.mobilegs_benchmark_warmup,
        mobilegs_benchmark_repeats=parsed.mobilegs_benchmark_repeats,
        mobilegs_quality_samples=parsed.mobilegs_quality_samples,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Recreate batch_train.py metrics from existing checkpoints.")
    ap.add_argument("checkpoint_root", type=Path, help="Folder containing external run directories/checkpoints.")
    ap.add_argument("--repo-root", type=Path, default=Path.cwd(), help="4DGS repo root containing batch_train.py.")
    ap.add_argument("--out-root", type=Path, default=None, help="Where combined ablation_metrics.* are written. Default: checkpoint_root.")
    ap.add_argument("--checkpoint-glob", default="**/chkpnt*.pth", help="Glob relative to checkpoint_root.")
    ap.add_argument("--layout", choices=["scene_variant", "variant", "flat"], default="scene_variant", help="How to infer scene/variant names from paths.")
    ap.add_argument("--eval-split", choices=["test", "train"], default="test")
    ap.add_argument("--checkpoint-eval-split", choices=["test", "train"], default="test")
    ap.add_argument("--render-fps-warmup", type=int, default=3)
    ap.add_argument("--summary-filename", default="ablation_metrics.csv")
    ap.add_argument("--summary-jsonl-filename", default="ablation_metrics.jsonl")
    ap.add_argument("--checkpoint-metrics-filename", default="checkpoint_eval_metrics.csv")
    ap.add_argument("--checkpoint-metrics-jsonl-filename", default="checkpoint_eval_metrics.jsonl")
    ap.add_argument("--combined-checkpoint-filename", default="checkpoint_eval_metrics_all.csv")
    ap.add_argument("--combined-checkpoint-jsonl-filename", default="checkpoint_eval_metrics_all.jsonl")
    ap.add_argument("--skip-checkpoint-metrics", action="store_true")
    ap.add_argument("--mobilegs-report", action=argparse.BooleanOptionalAction, default=False, help="Also run batch_train Mobile-GS export/compression/FPS stats.")
    ap.add_argument("--mobilegs-report-scope", choices=["all", "sort_free"], default="all")
    ap.add_argument("--mobilegs-benchmark-render-mode", choices=["match", "sort_free", "sorted"], default="match")
    ap.add_argument("--mobilegs-force-first-order-sh", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--mobilegs-mobile-filename", default="mobilegs_quantized.mobile.pt")
    ap.add_argument("--mobilegs-metrics-filename", default="mobilegs_metrics.json")
    ap.add_argument("--mobilegs-codebook-size", type=int, default=256)
    ap.add_argument("--mobilegs-block-size", type=int, default=8)
    ap.add_argument("--mobilegs-kmeans-iters", type=int, default=16)
    ap.add_argument("--mobilegs-uniform-bits", type=int, default=8)
    ap.add_argument("--mobilegs-build-visibility-filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mobilegs-temporal-keyframes", type=int, default=32)
    ap.add_argument("--mobilegs-temporal-mask-window", type=int, default=1)
    ap.add_argument("--mobilegs-temporal-mask-threshold", type=float, default=0.05)
    ap.add_argument("--mobilegs-views-per-keyframe", type=int, default=0)
    ap.add_argument("--mobilegs-benchmark-split", choices=["test", "train"], default="test")
    ap.add_argument("--mobilegs-benchmark-warmup", type=int, default=20)
    ap.add_argument("--mobilegs-benchmark-repeats", type=int, default=200)
    ap.add_argument("--mobilegs-quality-samples", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="Recompute even if run_metrics.json already exists.")
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    add_repo_to_path(repo_root)

    import batch_train as bt

    checkpoint_root = args.checkpoint_root.resolve()
    out_root = (args.out_root or checkpoint_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    metric_args = build_metric_args(args)

    checkpoints = find_checkpoint_files(checkpoint_root, args.checkpoint_glob)
    if not checkpoints:
        print(f"No checkpoints matched {args.checkpoint_glob!r} under {checkpoint_root}", file=sys.stderr)
        return 2

    grouped = group_by_parent(checkpoints)
    print(f"Found {len(checkpoints)} checkpoints in {len(grouped)} run directories.")

    summary_rows: List[Dict[str, Any]] = []
    all_checkpoint_rows: List[Dict[str, Any]] = []

    for run_index, (run_dir, run_ckpts) in enumerate(grouped.items()):
        metrics_path = run_dir / "run_metrics.json"
        if metrics_path.exists() and not args.force:
            existing = bt.load_json_if_exists(metrics_path)
            if isinstance(existing, dict):
                print(f"[SKIP] existing run_metrics.json: {run_dir}")
                summary_rows.append(dict(existing))
                jsonl_path = run_dir / args.checkpoint_metrics_jsonl_filename
                all_checkpoint_rows.extend(bt.read_jsonl_rows(jsonl_path))
                continue

        payloads: Dict[Path, Mapping[str, Any]] = {}
        valid_ckpts: List[Path] = []
        for ckpt in run_ckpts:
            try:
                payload = load_checkpoint_cpu(ckpt)
                payloads[ckpt] = payload
                valid_ckpts.append(ckpt)
            except Exception as exc:
                print(f"[WARN] skipping non self-contained checkpoint {ckpt}: {exc}")
        if not valid_ckpts:
            continue

        iterations = guess_iterations(run_dir, valid_ckpts, payloads)
        final_ckpt = final_checkpoint_for_run(run_dir, valid_ckpts, iterations, payloads)
        final_payload = payloads.get(final_ckpt) or load_checkpoint_cpu(final_ckpt)
        scene_name = infer_scene_name(run_dir, final_payload, args.layout)
        variant_name = infer_variant_name(run_dir, final_ckpt, final_payload, args.layout)
        variant_tags = infer_variant_tags(final_payload, variant_name)
        run_hash = stable_hash({
            "model_path": str(run_dir),
            "scene_name": scene_name,
            "variant_name": variant_name,
            "variant_tags": variant_tags,
            "index": run_index,
        })
        generated_config_path = str(run_dir / "external_checkpoint_config_from_payload.yaml")

        row: Dict[str, Any] = {
            "status": "ok",
            "scene_name": scene_name,
            "variant_name": variant_name,
            "generated_config_path": generated_config_path,
            "model_path": str(run_dir),
            "run_hash": run_hash,
            "returncode": 0,
            "action": "metrics_only",
            "training_wall_clock_sec": None,
            "peak_vram_mb": None,
        }
        row.update(variant_tags)

        print(f"[METRICS] {scene_name}/{variant_name} -> {final_ckpt.name}")
        try:
            eval_result = bt.evaluate_checkpoint(
                repo_root=repo_root,
                generated_config_path=Path(generated_config_path),
                model_path=run_dir,
                checkpoint_path=final_ckpt,
                split=args.eval_split,
                render_fps_warmup=args.render_fps_warmup,
            )
            row.update(eval_result)
            row["peak_vram_mb"] = row.get("peak_eval_vram_mb")
        except Exception as exc:
            row["status"] = "metrics_failed"
            row["metrics_error"] = str(exc)
            try:
                row["checkpoint_size_bytes"] = final_ckpt.stat().st_size
                row["checkpoint_path"] = str(final_ckpt.resolve())
            except FileNotFoundError:
                pass

        if not args.skip_checkpoint_metrics:
            checkpoint_rows: List[Dict[str, Any]] = []
            for checkpoint_index, ckpt in enumerate(history_checkpoints(run_dir, valid_ckpts)):
                ckpt_payload = payloads.get(ckpt) or load_checkpoint_cpu(ckpt)
                ckpt_name_iteration = checkpoint_name_iteration(ckpt)
                c_row: Dict[str, Any] = {
                    "status": "ok",
                    "run_hash": run_hash,
                    "scene_name": scene_name,
                    "variant_name": variant_name,
                    "generated_config_path": generated_config_path,
                    "model_path": str(run_dir),
                    "checkpoint_index": checkpoint_index,
                    "checkpoint_filename": ckpt.name,
                    "checkpoint_path": str(ckpt.resolve()),
                    "checkpoint_name_iteration": ckpt_name_iteration,
                    "training_wall_clock_sec_at_checkpoint": None,
                    "eval_requested_split": args.checkpoint_eval_split,
                }
                c_row.update(variant_tags)
                try:
                    c_eval = bt.evaluate_checkpoint(
                        repo_root=repo_root,
                        generated_config_path=Path(generated_config_path),
                        model_path=run_dir,
                        checkpoint_path=ckpt,
                        split=args.checkpoint_eval_split,
                        render_fps_warmup=args.render_fps_warmup,
                    )
                    c_row.update(c_eval)
                except Exception as exc:
                    c_row["status"] = "metrics_failed"
                    c_row["metrics_error"] = str(exc)
                    try:
                        c_row["checkpoint_size_bytes"] = ckpt.stat().st_size
                    except FileNotFoundError:
                        pass
                checkpoint_rows.append(c_row)
            bt.write_checkpoint_metric_files(run_dir, metric_args, checkpoint_rows)
            row["checkpoint_metrics_rows"] = len(checkpoint_rows)
            row["checkpoint_metrics_csv_path"] = str((run_dir / args.checkpoint_metrics_filename).resolve())
            row["checkpoint_metrics_jsonl_path"] = str((run_dir / args.checkpoint_metrics_jsonl_filename).resolve())
            all_checkpoint_rows.extend(checkpoint_rows)
        else:
            row["checkpoint_metrics_status"] = "skipped"

        if args.mobilegs_report:
            try:
                mobile_result = bt.run_mobilegs_export_benchmark(
                    repo_root=repo_root,
                    model_path=run_dir,
                    checkpoint_path=final_ckpt,
                    args=metric_args,
                )
                row.update(mobile_result)
            except Exception as exc:
                row["mobile_status"] = "failed"
                row["mobile_error"] = str(exc)
        else:
            row["mobile_status"] = "skipped"

        try:
            row.update(bt.load_training_diagnostics(run_dir))
        except Exception:
            pass
        row["model_path_size_bytes"] = bt.directory_size_bytes(run_dir)
        metrics_json_path = bt.write_run_metrics_json(run_dir, row)
        row["metrics_json_path"] = str(metrics_json_path.resolve())
        # Write it again after metrics_json_path has been populated, matching batch_train behavior.
        bt.write_run_metrics_json(run_dir, row)
        summary_rows.append(row)

    # De-duplicate like batch_train collect_summary_rows.
    dedup: Dict[str, Dict[str, Any]] = {}
    for row in summary_rows:
        key = str(row.get("model_path", ""))
        if key:
            dedup[key] = row
    summary_rows = [dedup[k] for k in sorted(dedup)]

    summary_csv = out_root / args.summary_filename
    summary_jsonl = out_root / args.summary_jsonl_filename
    bt.write_csv_summary(summary_csv, summary_rows)
    bt.append_jsonl(summary_jsonl, summary_rows)

    if all_checkpoint_rows:
        dedup_ckpt: Dict[tuple[str, str], Dict[str, Any]] = {}
        for row in all_checkpoint_rows:
            dedup_ckpt[(str(row.get("run_hash", "")), str(row.get("checkpoint_path", "")))] = row
        checkpoint_rows = [dedup_ckpt[k] for k in sorted(dedup_ckpt)]
        bt.write_csv_summary(out_root / args.combined_checkpoint_filename, checkpoint_rows)
        bt.append_jsonl(out_root / args.combined_checkpoint_jsonl_filename, checkpoint_rows)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_jsonl}")
    if all_checkpoint_rows:
        print(f"Wrote {out_root / args.combined_checkpoint_filename}")
        print(f"Wrote {out_root / args.combined_checkpoint_jsonl_filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
