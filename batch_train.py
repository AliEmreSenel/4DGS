from __future__ import annotations

"""Generate, schedule, run, and log ablation training runs.

Features:
- Cartesian-product ablations across scenes/configs
- Metrics logging per run: PSNR, SSIM, LPIPS, final Gaussian count,
  checkpoint size, render FPS, peak VRAM, training wall-clock
- USplat on/off as an ablation axis
- Robust failure handling: failed runs are recorded and skipped for metrics,
  but the overall orchestration continues
- Slurm integration with one main batch job and 4 long-lived `srun` worker steps
- Auto-detection of already completed runs
- Auto-detection of trained-but-unmeasured runs (metrics-only mode)
- Greedy worker assignment so each GPU gets a queue of runs

Typical usage:
    python train_ablations_slurm.py configs/dnerf/*.yaml --submit-slurm

Important runtime behavior:
- Supports both `uv run` and direct Python execution
- Prefers `uv run` automatically when `uv` is available in the submitting environment
- Falls back to the exact submitting interpreter (`sys.executable`) when `uv` is unavailable
- Exports the current environment into the batch job explicitly
"""

import argparse
import ast
import csv
import fcntl
import itertools
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import yaml


DEFAULT_ISOTROPY = {
    "anisotropic": {"isotropic_gaussians": False},
    "isotropic": {"isotropic_gaussians": True},
}

DEFAULT_APPEARANCE = {
    "rgb": {"sh_degree": 0, "force_sh_3d": True, "eval_shfs_4d": False},
    "sh3": {"sh_degree": 3, "force_sh_3d": False, "eval_shfs_4d": True},
}

DEFAULT_SORTING = {
    "sort": {"sort_free_render": False},
    "sort_free": {"sort_free_render": True},
}

DEFAULT_USPLAT = {
    "no_usplat": {"use_usplat": False},
    "use_usplat": {"use_usplat": True},
}

SUPPORTED_AXES = ("isotropy", "appearance", "sorting", "pruning", "usplat")


@dataclass(frozen=True)
class AblationVariant:
    name: str
    tags: Dict[str, str]
    overrides: Dict[str, Any]


@dataclass
class ScheduleOptions:
    one_shot_prune_step: int = 5001
    one_shot_prune_ratio: float = 0.85
    one_shot_densify_from_iter: int = 500
    one_shot_densify_until_iter: int = 5000
    one_shot_densification_interval: int = 100

    interleaved_prune_from_iter: int = 2000
    interleaved_prune_until_iter: int = 7500
    interleaved_prune_ratio: float = 0.10
    interleaved_prune_interval: int = 2000
    interleaved_prune_min_points: int = 1000
    interleaved_densify_from_iter: int = 500
    interleaved_densify_until_iter: int = 7500
    interleaved_densification_interval: int = 100


@dataclass(frozen=True)
class RunSpec:
    config_path: str
    generated_config_path: str
    model_path: str
    command: List[str]
    scene_name: str
    variant_name: str
    variant_tags: Dict[str, str]
    index: int
    iterations: int


@dataclass(frozen=True)
class PendingRun:
    run_spec: RunSpec
    action: str  # train_metrics | metrics_only
    estimated_cost: float


@dataclass
class WorkerAssignment:
    worker_index: int
    runs: List[PendingRun]


@dataclass
class ExistingState:
    status: str  # complete | metrics_only | pending
    metrics_path: str | None
    checkpoint_path: str | None
    metrics_payload: Dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch train.py ablation sweeps locally or through a Slurm driver job."
    )
    parser.add_argument("configs", nargs="+", help="Base config files.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used when the runner mode resolves to python.")
    parser.add_argument("--runner", choices=["auto", "uv", "python"], default="auto", help="Command runner for train/eval/worker launch. auto prefers `uv run` when uv is available.")
    parser.add_argument("--uv-binary", default="uv", help="uv executable to use when --runner=uv or auto resolves to uv.")
    parser.add_argument("--train-script", default="train.py", help="Training entrypoint to invoke.")
    parser.add_argument("--repo-root", default=None, help="Repository root. Inferred from --train-script when omitted.")
    parser.add_argument("--output-root", default=None, help="Optional root directory for generated model outputs.")
    parser.add_argument("--generated-config-root", default=None, help="Optional root directory for generated YAML configs.")
    parser.add_argument("--scene-name", default=None, help="Optional explicit scene name prefix.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--write-configs-only", action="store_true", help="Only write generated configs.")
    parser.add_argument("--print-only-paths", action="store_true", help="Only print generated model paths.")
    parser.add_argument("--limit", type=int, default=None, help="Limit runs per input config.")
    parser.add_argument(
        "--extra-arg",
        dest="extra_args",
        action="append",
        default=[],
        help="Additional raw CLI fragment appended to every training command.",
    )
    parser.add_argument(
        "--set",
        dest="global_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra flat override applied to every run.",
    )
    parser.add_argument("--axes", default="isotropy,appearance", help="Comma-separated ablation axes. The default is intentionally small and safe; add sorting,pruning,usplat explicitly for larger sweeps.")
    parser.add_argument("--isotropy-options", default="anisotropic,isotropic")
    parser.add_argument("--appearance-options", default="rgb,sh3")
    parser.add_argument("--sorting-options", default="sort")
    parser.add_argument("--pruning-options", default="no_pruning,densify_then_prune_once")
    parser.add_argument("--usplat-options", default="no_usplat")
    parser.add_argument("--include-invalid-combinations", action="store_true", help="Do not filter known incompatible ablation combinations.")

    parser.add_argument("--one-shot-prune-step", type=int, default=5001)
    parser.add_argument("--one-shot-prune-ratio", type=float, default=0.85)
    parser.add_argument("--one-shot-densify-from-iter", type=int, default=500)
    parser.add_argument("--one-shot-densify-until-iter", type=int, default=5000)
    parser.add_argument("--one-shot-densification-interval", type=int, default=100)

    parser.add_argument("--interleaved-prune-from-iter", type=int, default=2000)
    parser.add_argument("--interleaved-prune-until-iter", type=int, default=7500)
    parser.add_argument("--interleaved-prune-ratio", type=float, default=0.10)
    parser.add_argument("--interleaved-prune-interval", type=int, default=2000)
    parser.add_argument("--interleaved-prune-min-points", type=int, default=1000)
    parser.add_argument("--interleaved-densify-from-iter", type=int, default=500)
    parser.add_argument("--interleaved-densify-until-iter", type=int, default=7500)
    parser.add_argument("--interleaved-densification-interval", type=int, default=100)

    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--eval-split", choices=["test", "train"], default="test")
    parser.add_argument("--render-fps-warmup", type=int, default=3)
    parser.add_argument("--vram-poll-interval", type=float, default=1.0)
    parser.add_argument("--summary-filename", default="ablation_metrics.csv")
    parser.add_argument("--summary-jsonl-filename", default="ablation_metrics.jsonl")
    parser.add_argument("--retry-failed-existing", action=argparse.BooleanOptionalAction, default=True, help="Retry runs whose run_metrics.json says failed. Use --no-retry-failed-existing to keep old skip-failed behavior.")

    # Slurm controls
    parser.add_argument("--submit-slurm", action="store_true", help="Submit one main Slurm job that will launch srun workers.")
    parser.add_argument("--slurm-driver", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--slurm-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--assignment-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--slurm-partition", default="stud")
    parser.add_argument("--slurm-account", default="3221337")
    parser.add_argument("--slurm-qos", default="stud")
    parser.add_argument("--slurm-time", default="1-00:00:00")
    parser.add_argument("--slurm-mem", default="400G")
    parser.add_argument("--slurm-gpus", type=int, default=4)
    parser.add_argument("--slurm-tasks", type=int, default=4)
    parser.add_argument("--slurm-total-cpus", type=int, default=8)
    parser.add_argument("--slurm-job-name", default="4dgs-ablations")
    parser.add_argument("--slurm-log-dir", default=None)
    parser.add_argument("--slurm-export", default="ALL", help="Value passed to sbatch --export. Default keeps the full submitting environment.")
    parser.add_argument("--slurm-chdir", default=None, help="Working directory for the batch job. Defaults to the repo root.")
    parser.add_argument("--slurm-extra-sbatch-arg", dest="slurm_extra_sbatch_args", action="append", default=[])
    parser.add_argument("--slurm-srun-extra-arg", dest="slurm_srun_extra_args", action="append", default=[])
    parser.add_argument("--no-auto-submit-from-driver", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--quota-command", default="lquota")
    parser.add_argument("--quota-fallback-root", default=str(Path.home()), help="Directory measured with du --apparent-size when lquota is unavailable. Defaults to the user home.")
    parser.add_argument("--quota-limit-gb", type=float, default=50.0, help="Logical effective quota limit in GB.")
    parser.add_argument("--quota-reserve-gb", type=float, default=2.0, help="Always keep at least this much quota free.")
    parser.add_argument("--train-run-peak-storage-gb", type=float, default=2.5, help="Reserved temporary quota per active training run.")
    parser.add_argument("--quota-poll-interval", type=float, default=30.0, help="Seconds between quota checks while waiting.")
    parser.add_argument("--cleanup-existing-artifacts", action=argparse.BooleanOptionalAction, default=True, help="Prune bulky artifacts from existing run directories before scheduling.")
    parser.add_argument("--cleanup-after-run", action=argparse.BooleanOptionalAction, default=True, help="Delete bulky artifacts after each run, keeping only the best available checkpoint and metadata.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    return data


def flatten_cfg(cfg: Mapping[str, Any], out: MutableMapping[str, Any] | None = None) -> Dict[str, Any]:
    if out is None:
        out = {}
    for key, value in cfg.items():
        if isinstance(value, Mapping):
            flatten_cfg(value, out)
        else:
            out[key] = value
    return dict(out)


def parse_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def parse_key_value_list(items: Sequence[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = parse_scalar(value.strip())
    return overrides


def normalize_choice_list(raw: str) -> List[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def build_pruning_registry(total_iterations: int, options: ScheduleOptions) -> Dict[str, Dict[str, Any]]:
    if total_iterations <= 0:
        raise ValueError(f"iterations must be positive, got {total_iterations}")

    one_shot_prune_step = min(max(int(options.one_shot_prune_step), 1), total_iterations)
    one_shot_densify_from_iter = min(max(int(options.one_shot_densify_from_iter), 0), total_iterations)
    one_shot_densify_until_iter = min(
        max(int(options.one_shot_densify_until_iter), one_shot_densify_from_iter),
        total_iterations,
    )

    interleaved_prune_from_iter = min(max(int(options.interleaved_prune_from_iter), 1), total_iterations)
    interleaved_prune_until_iter = min(
        max(int(options.interleaved_prune_until_iter), interleaved_prune_from_iter),
        total_iterations,
    )
    interleaved_densify_from_iter = min(max(int(options.interleaved_densify_from_iter), 0), total_iterations)
    interleaved_densify_until_iter = min(
        max(int(options.interleaved_densify_until_iter), interleaved_densify_from_iter),
        total_iterations,
    )

    return {
        "no_pruning": {
            "enable_spatio_temporal_pruning": False,
            "final_prune_from_iter": -1,
            "final_prune_ratio": 0.0,
        },
        "densify_then_prune_once": {
            "enable_spatio_temporal_pruning": False,
            "densify_from_iter": one_shot_densify_from_iter,
            "densify_until_iter": one_shot_densify_until_iter,
            "densification_interval": options.one_shot_densification_interval,
            "final_prune_from_iter": one_shot_prune_step,
            "final_prune_ratio": options.one_shot_prune_ratio,
        },
        "interleaved_prune_densify": {
            "enable_spatio_temporal_pruning": True,
            "spatio_temporal_pruning_ratio": options.interleaved_prune_ratio,
            "spatio_temporal_pruning_from_iter": interleaved_prune_from_iter,
            "spatio_temporal_pruning_until_iter": interleaved_prune_until_iter,
            "spatio_temporal_pruning_interval": options.interleaved_prune_interval,
            "spatio_temporal_pruning_min_points": options.interleaved_prune_min_points,
            "spatio_temporal_pruning_max_total_ratio": 0.50,
            "densify_from_iter": interleaved_densify_from_iter,
            "densify_until_iter": interleaved_densify_until_iter,
            "densification_interval": options.interleaved_densification_interval,
            "final_prune_from_iter": -1,
            "final_prune_ratio": 0.0,
        },
    }


def build_axis_registry(flat_cfg: Mapping[str, Any], schedule_options: ScheduleOptions) -> Dict[str, Dict[str, Dict[str, Any]]]:
    total_iterations = int(flat_cfg.get("iterations", 0))
    return {
        "isotropy": DEFAULT_ISOTROPY,
        "appearance": DEFAULT_APPEARANCE,
        "sorting": DEFAULT_SORTING,
        "pruning": build_pruning_registry(total_iterations, schedule_options),
        "usplat": DEFAULT_USPLAT,
    }


def build_axis_variants(axis_name: str, requested_options: Sequence[str], registry: Mapping[str, Dict[str, Dict[str, Any]]]) -> List[AblationVariant]:
    axis_registry = registry[axis_name]
    variants: List[AblationVariant] = []
    for option in requested_options:
        if option not in axis_registry:
            valid = ", ".join(sorted(axis_registry))
            raise ValueError(f"Unknown option '{option}' for axis '{axis_name}'. Valid: {valid}")
        variants.append(AblationVariant(name=option, tags={axis_name: option}, overrides=dict(axis_registry[option])))
    return variants


def build_cartesian_variants(
    axes: Sequence[str],
    option_map: Mapping[str, Sequence[str]],
    flat_cfg: Mapping[str, Any],
    schedule_options: ScheduleOptions,
) -> List[AblationVariant]:
    registry = build_axis_registry(flat_cfg=flat_cfg, schedule_options=schedule_options)
    per_axis: List[List[AblationVariant]] = []
    for axis_name in axes:
        per_axis.append(build_axis_variants(axis_name, option_map[axis_name], registry))

    variants: List[AblationVariant] = []
    for combo in itertools.product(*per_axis):
        tags: Dict[str, str] = {}
        overrides: Dict[str, Any] = {}
        names: List[str] = []
        for item in combo:
            tags.update(item.tags)
            overrides.update(item.overrides)
            names.append(item.name)
        variants.append(AblationVariant(name="__".join(names), tags=tags, overrides=overrides))
    return variants


def invalid_variant_reason(flat_cfg: Mapping[str, Any], variant: AblationVariant) -> str | None:
    merged = dict(flat_cfg)
    merged.update(variant.overrides)
    if bool(merged.get("sort_free_render", False)) and int(merged.get("env_map_res", 0) or 0) > 0:
        return "sort_free_render does not support env_map_res"
    if bool(merged.get("sort_free_render", False)) and float(merged.get("lambda_depth", 0.0) or 0.0) > 0.0:
        return "sort_free_render does not return depth for lambda_depth"
    if bool(merged.get("sort_free_render", False)) and float(merged.get("lambda_opa_mask", 0.0) or 0.0) > 0.0:
        return "sort_free_render alpha proxy is not compatible with lambda_opa_mask"
    return None


def filter_valid_variants(flat_cfg: Mapping[str, Any], variants: Sequence[AblationVariant], include_invalid: bool) -> List[AblationVariant]:
    if include_invalid:
        return list(variants)
    valid: List[AblationVariant] = []
    skipped: List[tuple[str, str]] = []
    for variant in variants:
        reason = invalid_variant_reason(flat_cfg, variant)
        if reason is None:
            valid.append(variant)
        else:
            skipped.append((variant.name, reason))
    for name, reason in skipped:
        print(f"[SKIP INVALID] {name}: {reason}")
    return valid


def apply_dependent_overrides(run_overrides: MutableMapping[str, Any], global_overrides: Mapping[str, Any], flat_cfg: Mapping[str, Any] | None = None) -> None:
    flat_cfg = flat_cfg or {}
    sort_free = bool(run_overrides.get("sort_free_render", flat_cfg.get("sort_free_render", False)))
    user_set_mobile_lr = "mobilegs_opacity_phi_lr" in global_overrides
    if not sort_free:
        # Keep the MobileGS MLP and optimizer out of ordinary sorted-render runs.
        run_overrides["mobilegs_opacity_phi_lr"] = 0.0
    elif not user_set_mobile_lr and float(run_overrides.get("mobilegs_opacity_phi_lr", 0.0) or 0.0) <= 0.0:
        # Sort-free training needs the view-dependent opacity/phi MLP.
        run_overrides["mobilegs_opacity_phi_lr"] = 1e-3


def infer_scene_name(config_path: Path, flat_cfg: Mapping[str, Any], explicit_scene_name: str | None) -> str:
    if explicit_scene_name:
        return explicit_scene_name
    source_path = flat_cfg.get("source_path")
    if isinstance(source_path, str) and source_path:
        return Path(source_path).name
    model_path = flat_cfg.get("model_path")
    if isinstance(model_path, str) and model_path:
        return Path(model_path).name
    return config_path.stem


def get_output_root(flat_cfg: Mapping[str, Any], explicit_output_root: str | None) -> Path:
    if explicit_output_root:
        return Path(explicit_output_root)
    base_model_path = Path(str(flat_cfg.get("model_path", "output")))
    parent = base_model_path.parent if base_model_path.parent != Path("") else Path("output")
    return parent / "ablations"


def build_model_path(output_root: Path, scene_name: str, variant: AblationVariant) -> Path:
    variant_suffix = variant.name.replace("__", "--")
    return output_root / scene_name / variant_suffix


def generated_config_root(output_root: Path, explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root)
    return output_root / "generated_configs"


PIPELINE_OVERRIDE_KEYS = {
    "convert_SHs_python",
    "compute_cov3D_python",
    "debug",
    "use_usplat",
    "sort_free_render",
    "temporal_mask_threshold",
    "temporal_mask_keyframes",
    "temporal_mask_window",
    "random_dropout_prob",
    "env_map_res",
    "env_optimize_until",
    "env_optimize_from",
    "eval_shfs_4d",
}

OPTIMIZATION_OVERRIDE_KEYS = {
    "iterations", "position_lr_init", "position_t_lr_init", "position_lr_final",
    "position_lr_delay_mult", "position_lr_max_steps", "feature_lr", "opacity_lr",
    "scaling_lr", "rotation_lr", "percent_dense", "lambda_dssim", "lambda_rdr",
    "thresh_opa_prune", "densification_interval", "opacity_reset_interval",
    "densify_from_iter", "densify_until_iter", "densify_grad_threshold",
    "densify_grad_t_threshold", "densify_until_num_points", "final_prune_from_iter",
    "final_prune_ratio", "sh_increase_interval", "lambda_opa_mask", "lambda_rigid",
    "lambda_motion", "lambda_depth", "mobilegs_opacity_phi_lr",
    "enable_spatio_temporal_pruning", "spatio_temporal_pruning_ratio",
    "spatio_temporal_pruning_min_points", "spatio_temporal_pruning_random",
    "spatio_temporal_pruning_from_iter", "spatio_temporal_pruning_until_iter",
    "spatio_temporal_pruning_interval", "spatio_temporal_pruning_max_total_ratio",
    "lambda_key", "lambda_non_key", "usplat_start_iter", "usplat_eta_c",
    "usplat_phi", "usplat_key_ratio", "usplat_spt_threshold", "usplat_knn_k",
    "usplat_u_tau_percentile", "usplat_max_key_nodes", "usplat_assignment_chunk_size",
    "usplat_key_assignment_chunk_size", "usplat_motion_window",
    "usplat_nonkey_loss_chunk_size", "usplat_quat_chunk_size",
    "usplat_cov_eigengap_eps", "record_training_diagnostics",
    "diagnostics_short_lifespan_threshold",
}

MODEL_OVERRIDE_KEYS = {
    "sh_degree", "source_path", "model_path", "images", "resolution",
    "white_background", "data_device", "eval", "extension", "num_extra_pts",
    "loaded_pth", "frame_ratio", "dataloader", "prefilter_var",
}


def _deep_copy_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_copy_mapping(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy_mapping(v) for v in value]
    return value


def _find_nested_key_paths(config: Mapping[str, Any], key: str, prefix: tuple[str, ...] = ()) -> List[tuple[str, ...]]:
    paths: List[tuple[str, ...]] = []
    for cur_key, cur_value in config.items():
        cur_path = prefix + (str(cur_key),)
        if cur_key == key:
            paths.append(cur_path)
        if isinstance(cur_value, Mapping):
            paths.extend(_find_nested_key_paths(cur_value, key, cur_path))
    return paths


def _set_path(config: MutableMapping[str, Any], path: Sequence[str], value: Any) -> None:
    cur: MutableMapping[str, Any] = config
    for piece in path[:-1]:
        next_value = cur.setdefault(piece, {})
        if not isinstance(next_value, MutableMapping):
            next_value = {}
            cur[piece] = next_value
        cur = next_value
    cur[path[-1]] = value


def _preferred_override_path(config: Mapping[str, Any], key: str) -> tuple[str, ...]:
    paths = _find_nested_key_paths(config, key)
    if len(paths) == 1:
        return paths[0]
    # Prefer the canonical config groups when the key is not already present.
    if key in PIPELINE_OVERRIDE_KEYS:
        return ("PipelineParams", key)
    if key in OPTIMIZATION_OVERRIDE_KEYS:
        return ("OptimizationParams", key)
    if key in MODEL_OVERRIDE_KEYS:
        return ("ModelParams", key)
    if paths:
        # Avoid writing a duplicate top-level key when a nested key already exists.
        return min(paths, key=len)
    return (key,)


def apply_flat_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    derived = _deep_copy_mapping(config)
    for key, value in overrides.items():
        _set_path(derived, _preferred_override_path(derived, key), value)
    return derived


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def assemble_command(args: argparse.Namespace, train_script: str, generated_config_path: Path, extra_args: Sequence[str]) -> List[str]:
    cmd = [*command_prefix(args), train_script, "--config", str(generated_config_path)]
    for fragment in extra_args:
        cmd.extend(shlex.split(fragment))
    return cmd


def shell_join(tokens: Sequence[str]) -> str:
    return shlex.join(list(tokens))


def effective_python(args: argparse.Namespace) -> str:
    python_value = str(args.python).strip()
    return python_value if python_value else sys.executable


def uv_binary(args: argparse.Namespace) -> str:
    value = str(getattr(args, "uv_binary", "uv")).strip()
    return value if value else "uv"


def effective_runner(args: argparse.Namespace) -> str:
    runner = str(getattr(args, "runner", "auto")).strip().lower() or "auto"
    if runner == "auto":
        return "uv" if shutil.which(uv_binary(args)) else "python"
    return runner


def command_prefix(args: argparse.Namespace) -> List[str]:
    runner = effective_runner(args)
    if runner == "uv":
        return [uv_binary(args), "run"]
    if runner == "python":
        return [effective_python(args)]
    raise ValueError(f"Unsupported runner mode: {runner}")


def wrap_script_command(args: argparse.Namespace, script_path: Path, script_args: Sequence[str]) -> List[str]:
    return [*command_prefix(args), str(script_path), *script_args]


def quota_cli_args(args: argparse.Namespace) -> List[str]:
    resolved_quota = resolve_quota_command(args) or str(getattr(args, "quota_command", "lquota"))
    return [
        "--quota-command",
        resolved_quota,
        "--quota-fallback-root",
        str(getattr(args, "quota_fallback_root", str(Path.home()))),
        "--quota-limit-gb",
        str(args.quota_limit_gb),
        "--quota-reserve-gb",
        str(args.quota_reserve_gb),
        "--train-run-peak-storage-gb",
        str(args.train_run_peak_storage_gb),
        "--quota-poll-interval",
        str(args.quota_poll_interval),
    ]


def build_option_map(args: argparse.Namespace) -> Dict[str, List[str]]:
    return {
        "isotropy": normalize_choice_list(args.isotropy_options),
        "appearance": normalize_choice_list(args.appearance_options),
        "sorting": normalize_choice_list(args.sorting_options),
        "pruning": normalize_choice_list(args.pruning_options),
        "usplat": normalize_choice_list(args.usplat_options),
    }


def build_run_specs(args: argparse.Namespace, schedule_options: ScheduleOptions) -> List[RunSpec]:
    requested_axes = normalize_choice_list(args.axes)
    unknown_axes = [axis for axis in requested_axes if axis not in SUPPORTED_AXES]
    if unknown_axes:
        raise ValueError(f"Unknown axes: {', '.join(unknown_axes)}")

    option_map = build_option_map(args)
    global_overrides = parse_key_value_list(args.global_overrides)
    run_specs: List[RunSpec] = []

    for config_raw in args.configs:
        config_path = Path(config_raw)
        cfg = load_yaml(config_path)
        flat_cfg = flatten_cfg(cfg)
        variants = build_cartesian_variants(requested_axes, option_map, flat_cfg, schedule_options)
        variants = filter_valid_variants(flat_cfg, variants, args.include_invalid_combinations)
        if args.limit is not None:
            variants = variants[: args.limit]

        scene_name = infer_scene_name(config_path, flat_cfg, args.scene_name)
        output_root = get_output_root(flat_cfg, args.output_root)
        config_root = generated_config_root(output_root, args.generated_config_root) / scene_name
        base_seed = int(flat_cfg.get("seed", 6666))
        iterations = int(flat_cfg.get("iterations", 0))

        for index, variant in enumerate(variants):
            model_path = build_model_path(output_root, scene_name, variant)
            run_overrides = dict(variant.overrides)
            run_overrides.update(global_overrides)
            apply_dependent_overrides(run_overrides, global_overrides, flat_cfg)
            run_overrides["model_path"] = str(model_path)
            if args.seed_offset:
                run_overrides["seed"] = base_seed + args.seed_offset + index

            derived_cfg = apply_flat_overrides(cfg, run_overrides)
            generated_config_path = config_root / f"{variant.name}.yaml"
            write_yaml(generated_config_path, derived_cfg)

            command = assemble_command(args, args.train_script, generated_config_path, args.extra_args)
            run_specs.append(
                RunSpec(
                    config_path=str(config_path.resolve()),
                    generated_config_path=str(generated_config_path.resolve()),
                    model_path=str(model_path.resolve()),
                    command=command,
                    scene_name=scene_name,
                    variant_name=variant.name,
                    variant_tags=variant.tags,
                    index=index,
                    iterations=iterations,
                )
            )
    return run_specs


class GPUPeakMonitor:
    def __init__(self, pid: int, poll_interval: float = 1.0):
        self.pid = pid
        self.poll_interval = max(0.2, float(poll_interval))
        self.peak_mb: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self._enabled:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if not self._enabled:
            return None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        return self.peak_mb

    def _run(self) -> None:
        while not self._stop_event.is_set():
            current_mb = self._query_pid_memory_mb()
            if current_mb is not None:
                if self.peak_mb is None or current_mb > self.peak_mb:
                    self.peak_mb = current_mb
            self._stop_event.wait(self.poll_interval)

    def _process_tree_pids(self) -> set[int]:
        """Return the monitored PID plus descendants.

        `uv run` and shell wrappers often spawn a child Python process that owns
        the CUDA context. Monitoring only the original Popen PID then reports
        zero VRAM. This lightweight ps-based tree walk keeps the monitor wrapper
        agnostic to uv/python/srun launch details.
        """
        pids = {int(self.pid)}
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,ppid="],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return pids
        if result.returncode != 0:
            return pids
        children: Dict[int, List[int]] = {}
        for raw_line in result.stdout.splitlines():
            parts = raw_line.strip().split()
            if len(parts) != 2:
                continue
            try:
                pid, ppid = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            children.setdefault(ppid, []).append(pid)
        stack = [int(self.pid)]
        while stack:
            parent = stack.pop()
            for child in children.get(parent, []):
                if child not in pids:
                    pids.add(child)
                    stack.append(child)
        return pids

    def _query_pid_memory_mb(self) -> float | None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        monitored_pids = self._process_tree_pids()
        total = 0.0
        found = False
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [piece.strip() for piece in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                used_mb = float(parts[1])
            except ValueError:
                continue
            if pid in monitored_pids:
                total += used_mb
                found = True
        return total if found else 0.0


def run_subprocess_with_monitor(command: Sequence[str], cwd: Path, poll_interval: float) -> Dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.Popen(list(command), cwd=str(cwd))
    monitor = GPUPeakMonitor(proc.pid, poll_interval=poll_interval)
    monitor.start()
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    peak_vram_mb = monitor.stop()
    elapsed = time.perf_counter() - start
    return {
        "returncode": returncode,
        "training_wall_clock_sec": elapsed,
        "peak_vram_mb": peak_vram_mb,
    }


def repo_root_from_args(args: argparse.Namespace) -> Path:
    if args.repo_root:
        return Path(args.repo_root).resolve()
    train_script_path = Path(args.train_script)
    if train_script_path.exists():
        return train_script_path.resolve().parent
    return Path.cwd().resolve()


def expected_final_checkpoint(model_path: Path, iterations: int) -> Path:
    return model_path / f"chkpnt{iterations}.pth"


def find_best_available_checkpoint(model_path: Path, iterations: int) -> Path | None:
    exact = expected_final_checkpoint(model_path, iterations)
    if exact.exists():
        return exact
    best = model_path / "chkpnt_best.pth"
    if best.exists():
        return best
    numeric: List[tuple[int, Path]] = []
    for path in model_path.glob("chkpnt*.pth"):
        stem = path.stem.replace("chkpnt", "")
        if stem.isdigit():
            numeric.append((int(stem), path))
    if numeric:
        return max(numeric)[1]
    return None


def load_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def detect_existing_state(run_spec: RunSpec, retry_failed_existing: bool) -> ExistingState:
    model_path = Path(run_spec.model_path)
    metrics_path = model_path / "run_metrics.json"
    metrics_payload = load_json_if_exists(metrics_path)
    checkpoint_path = find_best_available_checkpoint(model_path, run_spec.iterations)

    if metrics_payload is not None:
        status = str(metrics_payload.get("status", ""))
        if status == "ok":
            return ExistingState("complete", str(metrics_path), str(checkpoint_path) if checkpoint_path else None, metrics_payload)
        if status == "metrics_failed" and checkpoint_path is not None:
            return ExistingState("metrics_only", str(metrics_path), str(checkpoint_path), metrics_payload)
        if status == "failed" and not retry_failed_existing:
            return ExistingState("complete", str(metrics_path), str(checkpoint_path) if checkpoint_path else None, metrics_payload)

    if checkpoint_path is not None:
        return ExistingState("metrics_only", str(metrics_path) if metrics_path.exists() else None, str(checkpoint_path), metrics_payload)

    return ExistingState("pending", str(metrics_path) if metrics_path.exists() else None, None, metrics_payload)


def estimated_run_cost(run_spec: RunSpec, action: str) -> float:
    cost = float(max(run_spec.iterations, 1))
    tags = run_spec.variant_tags
    if tags.get("usplat") == "use_usplat":
        cost *= 1.75
    if tags.get("sorting") == "sort_free":
        cost *= 1.15
    if tags.get("appearance") == "sh3":
        cost *= 1.10
    if tags.get("pruning") == "interleaved_prune_densify":
        cost *= 1.20
    elif tags.get("pruning") == "densify_then_prune_once":
        cost *= 1.05
    if action == "metrics_only":
        cost *= 0.08
    return cost


def build_pending_runs(run_specs: Sequence[RunSpec], retry_failed_existing: bool) -> tuple[List[PendingRun], List[Dict[str, Any]]]:
    pending: List[PendingRun] = []
    existing_rows: List[Dict[str, Any]] = []
    for run_spec in run_specs:
        existing_state = detect_existing_state(run_spec, retry_failed_existing)
        if existing_state.status == "complete":
            if existing_state.metrics_payload is not None:
                existing_rows.append(dict(existing_state.metrics_payload))
            continue
        if existing_state.status == "metrics_only":
            action = "metrics_only"
        else:
            action = "train_metrics"
        pending.append(PendingRun(run_spec, action, estimated_run_cost(run_spec, action)))
    return pending, existing_rows


def partition_pending_runs(pending_runs: Sequence[PendingRun], num_workers: int) -> List[WorkerAssignment]:
    assignments = [WorkerAssignment(worker_index=i, runs=[]) for i in range(num_workers)]
    if not pending_runs:
        return assignments

    loads = [0.0 for _ in range(num_workers)]
    for item in sorted(pending_runs, key=lambda x: x.estimated_cost, reverse=True):
        idx = min(range(num_workers), key=lambda i: loads[i])
        assignments[idx].runs.append(item)
        loads[idx] += item.estimated_cost
    return assignments


def summary_output_root(args: argparse.Namespace, run_specs: Sequence[RunSpec]) -> Path:
    if args.output_root:
        return Path(args.output_root)
    if run_specs:
        return Path(run_specs[0].model_path).parents[1]
    return Path("ablations")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_run_metrics_json(model_path: Path, payload: Mapping[str, Any]) -> Path:
    metrics_path = model_path / "run_metrics.json"
    model_path.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return metrics_path


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    preferred_fields = [
        "status",
        "scene_name",
        "variant_name",
        "isotropy",
        "appearance",
        "sorting",
        "pruning",
        "usplat",
        "generated_config_path",
        "model_path",
        "checkpoint_path",
        "eval_split_used",
        "eval_checkpoint_iteration",
        "psnr",
        "ssim",
        "lpips",
        "final_gaussian_count",
        "checkpoint_size_bytes",
        "model_path_size_bytes",
        "render_fps",
        "peak_vram_mb",
        "peak_eval_vram_mb",
        "quota_used_gb",
        "quota_limit_gb",
        "quota_free_gb",
        "training_wall_clock_sec",
        "action",
        "returncode",
        "metrics_json_path",
    ]
    fieldnames: List[str] = []
    seen = set()
    for name in preferred_fields:
        if any(name in row for row in rows):
            fieldnames.append(name)
            seen.add(name)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_summary_rows(run_specs: Sequence[RunSpec], existing_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [dict(row) for row in existing_rows]
    for run_spec in run_specs:
        metrics_path = Path(run_spec.model_path) / "run_metrics.json"
        payload = load_json_if_exists(metrics_path)
        if payload is None:
            continue
        rows.append(dict(payload))

    dedup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("model_path", ""))
        if not key:
            continue
        dedup[key] = row
    return [dedup[key] for key in sorted(dedup)]


def robust_torch_load(torch_module: Any, path: Path, map_location: str) -> Any:
    try:
        return torch_module.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=map_location)


def load_generated_namespace(cfg_path: Path, repo_root: Path) -> argparse.Namespace:
    flat_cfg = flatten_cfg(load_yaml(cfg_path))
    ns = argparse.Namespace(**flat_cfg)
    source_path = getattr(ns, "source_path", None)
    if isinstance(source_path, str) and source_path and not os.path.isabs(source_path):
        ns.source_path = str((repo_root / source_path).resolve())
    model_path = getattr(ns, "model_path", None)
    if isinstance(model_path, str) and model_path and not os.path.isabs(model_path):
        ns.model_path = str((repo_root / model_path).resolve())
    return ns


def extract_final_gaussian_count(model_params: Any) -> int | None:
    if isinstance(model_params, dict):
        model_params = model_params.get("gaussians")
    if isinstance(model_params, (tuple, list)) and len(model_params) > 1:
        xyz = model_params[1]
        if hasattr(xyz, "shape") and len(xyz.shape) >= 1:
            return int(xyz.shape[0])
    return None


def scalar_float(value: Any) -> float:
    """Accept either a scalar tensor or a Python numeric metric value."""
    return float(value.item()) if hasattr(value, "item") else float(value)


def evaluate_checkpoint(
    repo_root: Path,
    generated_config_path: Path,
    model_path: Path,
    checkpoint_path: Path,
    split: str,
    render_fps_warmup: int,
) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from utils.checkpoint_utils import checkpoint_args, load_checkpoint
    from utils.image_utils import psnr
    from utils.loss_utils import lpips as lpips_metric
    from utils.loss_utils import ssim

    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_payload = load_checkpoint(checkpoint_path, map_location)
    cfg_ns = checkpoint_args(checkpoint_payload)

    dummy_parser = argparse.ArgumentParser()
    lp = ModelParams(dummy_parser)
    pp = PipelineParams(dummy_parser)
    defaults = dummy_parser.parse_args([])
    merged = argparse.Namespace(**vars(defaults))
    for key, value in vars(cfg_ns).items():
        setattr(merged, key, value)

    merged.model_path = str(model_path.resolve())
    dataset = lp.extract(merged)
    pipe = pp.extract(merged)

    gaussian_kwargs = dict(checkpoint_payload["run_config"].get("gaussian_kwargs", {}))
    if int(gaussian_kwargs.get("gaussian_dim", 4)) != 4:
        raise ValueError("Only 4D Gaussian checkpoints are supported.")
    time_duration = list(gaussian_kwargs.get("time_duration", [-0.5, 0.5]))
    num_pts = int(getattr(merged, "num_pts", 100000))
    num_pts_ratio = float(getattr(merged, "num_pts_ratio", 1.0))

    gaussians = GaussianModel(**gaussian_kwargs)
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

    model_params = checkpoint_payload["gaussians"]
    loaded_iter = int(checkpoint_payload["iteration"])
    gaussians.restore(model_params, None)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    if hasattr(gaussians, "active_sh_degree_t"):
        gaussians.active_sh_degree_t = gaussians.max_sh_degree_t

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if split == "test" and len(scene.test_cameras[1.0]) > 0:
        camera_dataset = scene.getTestCameras()
        split_used = "test"
    else:
        camera_dataset = scene.getTrainCameras()
        split_used = "train"

    camera_items = list(camera_dataset)
    if not camera_items:
        raise RuntimeError(f"No cameras available for split={split_used}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    with torch.no_grad():
        for gt_image, viewpoint in camera_items:
            gt_image = gt_image.to("cuda", non_blocking=True)
            viewpoint = viewpoint.cuda(non_blocking=True, copy=False)
            render_pkg = render(viewpoint, scene.gaussians, pipe, background)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            totals["psnr"] += scalar_float(psnr(image, gt_image).mean())
            totals["ssim"] += scalar_float(ssim(image, gt_image).mean())
            totals["lpips"] += scalar_float(lpips_metric(image[None].cpu(), gt_image[None].cpu()))

    count = float(len(camera_items))
    metric_results = {
        "psnr": totals["psnr"] / count,
        "ssim": totals["ssim"] / count,
        "lpips": totals["lpips"] / count,
        "eval_split_used": split_used,
        "eval_checkpoint_iteration": int(loaded_iter),
    }

    warmup_count = min(max(render_fps_warmup, 0), len(camera_items))
    with torch.no_grad():
        for idx in range(warmup_count):
            _, viewpoint = camera_items[idx]
            viewpoint = viewpoint.cuda(non_blocking=True, copy=False)
            _ = render(viewpoint, scene.gaussians, pipe, background)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _, viewpoint in camera_items:
            viewpoint = viewpoint.cuda(non_blocking=True, copy=False)
            _ = render(viewpoint, scene.gaussians, pipe, background)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    metric_results["render_fps"] = len(camera_items) / max(elapsed, 1e-9)
    metric_results["peak_eval_vram_mb"] = (
        torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        if torch.cuda.is_available()
        else None
    )
    metric_results["final_gaussian_count"] = extract_final_gaussian_count(checkpoint_payload)
    metric_results["checkpoint_size_bytes"] = checkpoint_path.stat().st_size
    metric_results["checkpoint_path"] = str(checkpoint_path.resolve())
    return metric_results



def flatten_metric_payload(prefix: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_metric_payload(name, value))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[name] = value
        elif isinstance(value, list):
            # Keep short histograms/percentile lists as compact JSON strings for CSV.
            flat[name] = json.dumps(value)
    return flat


def load_training_diagnostics(model_path: Path) -> Dict[str, Any]:
    payload = load_json_if_exists(model_path / "training_diagnostics.json")
    if not isinstance(payload, dict):
        return {}
    return flatten_metric_payload("diag", payload)


def run_one_pending(pending: PendingRun, args: argparse.Namespace, repo_root: Path) -> Dict[str, Any]:
    run_spec = pending.run_spec
    model_path = Path(run_spec.model_path)
    generated_config_path = Path(run_spec.generated_config_path)
    summary_root = summary_output_root(args, [run_spec])

    row: Dict[str, Any] = {
        "status": "ok",
        "scene_name": run_spec.scene_name,
        "variant_name": run_spec.variant_name,
        "generated_config_path": str(generated_config_path),
        "model_path": str(model_path),
        "returncode": 0,
        "action": pending.action,
        "training_wall_clock_sec": 0.0,
        "peak_vram_mb": None,
    }
    row.update(run_spec.variant_tags)

    checkpoint_path = find_best_available_checkpoint(model_path, run_spec.iterations)
    reservation_id: str | None = None

    try:
        if pending.action == "train_metrics":
            reservation_id = acquire_quota_reservation(args, summary_root, model_path)
            print(f"[RUN] {run_spec.variant_name} -> {model_path}")
            train_result = run_subprocess_with_monitor(run_spec.command, repo_root, args.vram_poll_interval)
            row["returncode"] = train_result["returncode"]
            row["training_wall_clock_sec"] = train_result["training_wall_clock_sec"]
            row["peak_vram_mb"] = train_result["peak_vram_mb"]
            if train_result["returncode"] != 0:
                row["status"] = "failed"
                row["model_path_size_bytes"] = cleanup_run_directory(model_path, run_spec.iterations, args.cleanup_after_run)
                metrics_json_path = write_run_metrics_json(model_path, row)
                row["metrics_json_path"] = str(metrics_json_path.resolve())
                return row
            checkpoint_path = find_best_available_checkpoint(model_path, run_spec.iterations)
            if checkpoint_path is None:
                row["status"] = "failed"
                row["error"] = "Training finished without any checkpoint file."
                row["model_path_size_bytes"] = cleanup_run_directory(model_path, run_spec.iterations, args.cleanup_after_run)
                metrics_json_path = write_run_metrics_json(model_path, row)
                row["metrics_json_path"] = str(metrics_json_path.resolve())
                return row
        else:
            print(f"[METRICS] {run_spec.variant_name} -> {model_path}")
            if checkpoint_path is None:
                row["status"] = "failed"
                row["error"] = "metrics_only requested but no checkpoint was found."
                row["model_path_size_bytes"] = cleanup_run_directory(model_path, run_spec.iterations, args.cleanup_after_run)
                metrics_json_path = write_run_metrics_json(model_path, row)
                row["metrics_json_path"] = str(metrics_json_path.resolve())
                return row

        if not args.skip_metrics:
            try:
                eval_result = evaluate_checkpoint(
                    repo_root=repo_root,
                    generated_config_path=generated_config_path,
                    model_path=model_path,
                    checkpoint_path=checkpoint_path,
                    split=args.eval_split,
                    render_fps_warmup=args.render_fps_warmup,
                )
                row.update(eval_result)
                peak_vram_mb = row.get("peak_vram_mb")
                peak_eval_vram_mb = row.get("peak_eval_vram_mb")
                if peak_vram_mb is None:
                    row["peak_vram_mb"] = peak_eval_vram_mb
                elif peak_eval_vram_mb is not None:
                    row["peak_vram_mb"] = max(float(peak_vram_mb), float(peak_eval_vram_mb))
            except Exception as exc:
                row["status"] = "metrics_failed"
                row["metrics_error"] = str(exc)

        row.update(load_training_diagnostics(model_path))
        row["model_path_size_bytes"] = cleanup_run_directory(model_path, run_spec.iterations, args.cleanup_after_run)
        try:
            used_gb, limit_gb = query_lquota_gb(args)
            row["quota_used_gb"] = used_gb
            row["quota_limit_gb"] = limit_gb
            row["quota_free_gb"] = max(0.0, limit_gb - used_gb)
        except Exception as exc:
            row["quota_error"] = str(exc)

        metrics_json_path = write_run_metrics_json(model_path, row)
        row["metrics_json_path"] = str(metrics_json_path.resolve())
        return row
    finally:
        release_quota_reservation(summary_root, reservation_id)


def slurm_cpus_per_task(args: argparse.Namespace) -> int:
    return max(1, args.slurm_total_cpus // max(1, args.slurm_tasks))


def parse_mem_to_mb(mem_value: str) -> int:
    text = str(mem_value).strip()
    match = re.fullmatch(r"(?i)([0-9]+(?:\.[0-9]+)?)([kmgt]?)b?", text)
    if not match:
        raise ValueError(f"Could not parse memory value: {mem_value}")
    value = float(match.group(1))
    unit = match.group(2).upper()
    multipliers = {"": 1.0 / (1024 * 1024), "K": 1.0 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}
    return max(1, int(value * multipliers[unit]))


def format_mem_from_mb(mem_mb: int) -> str:
    if mem_mb % 1024 == 0:
        return f"{mem_mb // 1024}G"
    return f"{mem_mb}M"


def slurm_mem_per_worker(args: argparse.Namespace, num_workers: int | None = None) -> str:
    workers = max(1, num_workers if num_workers is not None else args.slurm_tasks)
    total_mb = parse_mem_to_mb(args.slurm_mem)
    per_worker_mb = max(1, total_mb // workers)
    return format_mem_from_mb(per_worker_mb)


def slurm_workspace(summary_root: Path, job_id: str | None = None) -> Path:
    name = job_id or "manual"
    return summary_root / ".slurm_orchestrator" / name


def parse_capacity_to_gb(text: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGTP]?)(?:i?B|B)?", text.strip(), re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).upper()
    scale = {
        "": 1.0 / (1024.0**3),
        "K": 1.0 / (1024.0**2),
        "M": 1.0 / 1024.0,
        "G": 1.0,
        "T": 1024.0,
        "P": 1024.0 * 1024.0,
    }
    return value * scale.get(unit, 1.0)


def resolve_quota_command(args: argparse.Namespace) -> str | None:
    raw = str(getattr(args, "quota_command", "lquota")).strip() or "lquota"

    candidate_paths = []
    raw_path = Path(raw)
    if raw_path.is_absolute():
        candidate_paths.append(raw_path)
    else:
        found = shutil.which(raw)
        if found:
            candidate_paths.append(Path(found))
        env_found = os.environ.get("LQUOTA_CMD")
        if env_found:
            candidate_paths.append(Path(env_found))
        candidate_paths.append(Path("/userScripts/lquota"))
        try:
            result = subprocess.run(
                ["bash", "-lc", f"command -v {shlex.quote(raw)}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            shell_lines = (result.stdout or "").strip().splitlines()
            if shell_lines:
                candidate_paths.append(Path(shell_lines[-1].strip()))
        except Exception:
            pass

    seen = set()
    for candidate in candidate_paths:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            resolved = candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists() and os.access(str(resolved), os.X_OK):
            return str(resolved)

    return None


def query_lquota_gb(args: argparse.Namespace) -> tuple[float, float]:
    quota_cmd = resolve_quota_command(args)
    if quota_cmd is not None:
        result = subprocess.run([quota_cmd], check=False, capture_output=True, text=True, timeout=15)
        raw_text = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        used_match = re.search(r"USED\s*=\s*([^\s]+)", raw_text, re.IGNORECASE)
        limit_match = re.search(r"LIMIT\s*=\s*([^\s]+)", raw_text, re.IGNORECASE)
        if used_match and limit_match:
            used_gb = parse_capacity_to_gb(used_match.group(1))
            limit_gb = parse_capacity_to_gb(limit_match.group(1))
            if used_gb is not None and limit_gb is not None:
                return used_gb, min(limit_gb, float(args.quota_limit_gb))
        print(
            f"[QUOTA] Could not parse lquota output from {quota_cmd}; "
            f"falling back to du on {args.quota_fallback_root}"
        )

    fallback_root = Path(str(getattr(args, "quota_fallback_root", str(Path.home())))).expanduser()
    if not fallback_root.exists():
        raise RuntimeError(
            f"Quota command '{args.quota_command}' could not be resolved and fallback root "
            f"'{fallback_root}' does not exist."
        )

    result = subprocess.run(
        ["du", "-sb", "--apparent-size", str(fallback_root)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Quota command '{args.quota_command}' could not be resolved and fallback du "
            f"measurement failed for '{fallback_root}': {result.stderr.strip()}"
        )

    fields = (result.stdout or "").strip().split()
    if not fields:
        raise RuntimeError(
            f"Quota command '{args.quota_command}' could not be resolved and fallback du "
            f"measurement returned no output for '{fallback_root}'."
        )

    used_bytes = int(fields[0])
    used_gb = used_bytes / (1024.0 ** 3)
    limit_gb = float(args.quota_limit_gb)
    print(
        f"[QUOTA] Falling back to du --apparent-size for {fallback_root}: "
        f"used={used_gb:.2f}GB limit={limit_gb:.2f}GB"
    )
    return used_gb, limit_gb


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def quota_paths(summary_root: Path) -> tuple[Path, Path]:
    workspace = slurm_workspace(summary_root, os.environ.get("SLURM_JOB_ID"))
    return workspace / "quota.lock", workspace / "quota_reservations.json"


def load_quota_reservations(path: Path) -> list[dict[str, Any]]:
    payload = load_json_if_exists(path)
    if not isinstance(payload, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        pid = int(item.get("pid", -1))
        if pid > 0 and process_exists(pid):
            cleaned.append(item)
    return cleaned


def save_quota_reservations(path: Path, payload: Sequence[Mapping[str, Any]]) -> None:
    write_json(path, list(payload))


def acquire_quota_reservation(args: argparse.Namespace, summary_root: Path, model_path: Path) -> str:
    lock_path, reservations_path = quota_paths(summary_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    reservation_id = f"{os.getpid()}::{model_path}"
    while True:
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            reservations = load_quota_reservations(reservations_path)
            used_gb, limit_gb = query_lquota_gb(args)
            reserved_gb = sum(float(item.get("gb", 0.0)) for item in reservations)
            available_gb = limit_gb - used_gb - float(args.quota_reserve_gb) - reserved_gb
            if available_gb >= float(args.train_run_peak_storage_gb):
                reservations.append(
                    {
                        "reservation_id": reservation_id,
                        "pid": os.getpid(),
                        "gb": float(args.train_run_peak_storage_gb),
                        "model_path": str(model_path),
                        "created_at": time.time(),
                    }
                )
                save_quota_reservations(reservations_path, reservations)
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                return reservation_id
            print(
                f"[QUOTA] Waiting for storage slot for {model_path.name}: "
                f"used={used_gb:.2f}GB limit={limit_gb:.2f}GB reserved={reserved_gb:.2f}GB "
                f"need={float(args.train_run_peak_storage_gb):.2f}GB reserve={float(args.quota_reserve_gb):.2f}GB"
            )
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        time.sleep(max(5.0, float(args.quota_poll_interval)))


def release_quota_reservation(summary_root: Path, reservation_id: str | None) -> None:
    if reservation_id is None:
        return
    lock_path, reservations_path = quota_paths(summary_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        reservations = load_quota_reservations(reservations_path)
        reservations = [item for item in reservations if str(item.get("reservation_id")) != reservation_id]
        save_quota_reservations(reservations_path, reservations)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except FileNotFoundError:
            pass
    return total


def cleanup_model_artifacts(model_path: Path, keep_checkpoint: Path | None) -> None:
    if not model_path.exists():
        return
    keep_resolved = keep_checkpoint.resolve() if keep_checkpoint and keep_checkpoint.exists() else None
    for ckpt in model_path.glob("chkpnt*.pth"):
        try:
            if keep_resolved is not None and ckpt.resolve() == keep_resolved:
                continue
        except FileNotFoundError:
            continue
        remove_path(ckpt)
    for bulky in ["point_cloud", "renders", "test", "train"]:
        remove_path(model_path / bulky)


def cleanup_run_directory(model_path: Path, iterations: int, enabled: bool) -> int:
    checkpoint_path = find_best_available_checkpoint(model_path, iterations)
    if enabled:
        cleanup_model_artifacts(model_path, checkpoint_path)
    return directory_size_bytes(model_path)


def cleanup_existing_artifacts(run_specs: Sequence[RunSpec], enabled: bool) -> None:
    if not enabled:
        return
    for run_spec in run_specs:
        cleanup_run_directory(Path(run_spec.model_path), run_spec.iterations, True)


def dump_assignments(workspace: Path, assignments: Sequence[WorkerAssignment]) -> List[Path]:
    paths: List[Path] = []
    workspace.mkdir(parents=True, exist_ok=True)
    for assignment in assignments:
        path = workspace / f"worker_{assignment.worker_index:02d}.json"
        payload = {
            "worker_index": assignment.worker_index,
            "runs": [
                {
                    "run_spec": asdict(item.run_spec),
                    "action": item.action,
                    "estimated_cost": item.estimated_cost,
                }
                for item in assignment.runs
            ],
        }
        write_json(path, payload)
        paths.append(path)
    return paths


def run_worker_from_assignment(args: argparse.Namespace, repo_root: Path) -> int:
    if not args.assignment_file:
        raise ValueError("--assignment-file is required in --slurm-worker mode")
    payload = load_json(Path(args.assignment_file))
    worker_index = int(payload.get("worker_index", -1))
    runs_payload = payload.get("runs", [])
    print(f"[WORKER {worker_index}] Loaded {len(runs_payload)} runs from {args.assignment_file}")

    for item in runs_payload:
        run_spec = RunSpec(**item["run_spec"])
        pending = PendingRun(run_spec=run_spec, action=item["action"], estimated_cost=float(item.get("estimated_cost", 0.0)))
        try:
            row = run_one_pending(pending, args, repo_root)
            print(f"[WORKER {worker_index}] Finished {run_spec.variant_name} status={row.get('status')}")
        except Exception as exc:
            fallback = {
                "status": "failed",
                "scene_name": run_spec.scene_name,
                "variant_name": run_spec.variant_name,
                "generated_config_path": run_spec.generated_config_path,
                "model_path": run_spec.model_path,
                "action": pending.action,
                "returncode": -1,
                "error": str(exc),
            }
            metrics_json_path = write_run_metrics_json(Path(run_spec.model_path), fallback)
            print(f"[WORKER {worker_index}] Exception in {run_spec.variant_name}: {exc}")
            print(f"[WORKER {worker_index}] Wrote failure record to {metrics_json_path}")
    return 0


def gather_and_write_summaries(args: argparse.Namespace, run_specs: Sequence[RunSpec], existing_rows: Sequence[Dict[str, Any]]) -> None:
    summary_root = summary_output_root(args, run_specs)
    rows = collect_summary_rows(run_specs, existing_rows)
    summary_csv_path = summary_root / args.summary_filename
    summary_jsonl_path = summary_root / args.summary_jsonl_filename
    write_csv_summary(summary_csv_path, rows)
    append_jsonl(summary_jsonl_path, rows)
    print(f"Wrote CSV summary to {summary_csv_path}")
    print(f"Wrote JSONL summary to {summary_jsonl_path}")
    print(f"Collected {len(rows)} rows")


def run_slurm_driver(args: argparse.Namespace, run_specs: Sequence[RunSpec], existing_rows: Sequence[Dict[str, Any]], pending_runs: Sequence[PendingRun]) -> int:
    summary_root = summary_output_root(args, run_specs)
    job_id = os.environ.get("SLURM_JOB_ID")
    workspace = slurm_workspace(summary_root, job_id)
    if not pending_runs:
        print("No pending runs. Only refreshing summaries.")
        gather_and_write_summaries(args, run_specs, existing_rows)
        return 0

    num_workers = min(max(1, args.slurm_tasks), len(pending_runs))
    assignments = partition_pending_runs(pending_runs, num_workers)
    assignment_paths = dump_assignments(workspace, assignments)
    cpus_per_task = slurm_cpus_per_task(args)
    mem_per_worker = slurm_mem_per_worker(args, num_workers)

    processes: List[subprocess.Popen[Any]] = []
    for worker_index, assignment_path in enumerate(assignment_paths):
        if not assignments[worker_index].runs:
            continue
        step_cmd = [
            "srun",
            "--exclusive",
            "--exact",
            "-N1",
            "-n1",
            f"--gres=gpu:1",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem={mem_per_worker}",
        ]
        for extra in args.slurm_srun_extra_args:
            step_cmd.extend(shlex.split(extra))
        step_cmd.extend(
            wrap_script_command(
                args,
                Path(__file__).resolve(),
                [
                    *args.configs,
                    "--slurm-worker",
                    "--assignment-file",
                    str(assignment_path),
                    "--repo-root",
                    str(repo_root_from_args(args)),
                    "--train-script",
                    args.train_script,
                    "--eval-split",
                    args.eval_split,
                    "--render-fps-warmup",
                    str(args.render_fps_warmup),
                    "--vram-poll-interval",
                    str(args.vram_poll_interval),
                    "--runner",
                    effective_runner(args),
                    "--uv-binary",
                    uv_binary(args),
                    "--python",
                    effective_python(args),
                    *quota_cli_args(args),
                ],
            )
        )
        if not args.cleanup_after_run:
            step_cmd.append("--no-cleanup-after-run")
        if not args.cleanup_existing_artifacts:
            step_cmd.append("--no-cleanup-existing-artifacts")
        if args.skip_metrics:
            step_cmd.append("--skip-metrics")
        print(f"[SLURM STEP] worker={worker_index} cpus={cpus_per_task} mem={mem_per_worker} cmd={shell_join(step_cmd)}")
        processes.append(subprocess.Popen(step_cmd))

    for proc in processes:
        proc.wait()

    gather_and_write_summaries(args, run_specs, existing_rows)
    return 0


def local_serial_driver(args: argparse.Namespace, run_specs: Sequence[RunSpec], existing_rows: Sequence[Dict[str, Any]], pending_runs: Sequence[PendingRun]) -> int:
    repo_root = repo_root_from_args(args)
    for pending in pending_runs:
        try:
            run_one_pending(pending, args, repo_root)
        except Exception as exc:
            run_spec = pending.run_spec
            fallback = {
                "status": "failed",
                "scene_name": run_spec.scene_name,
                "variant_name": run_spec.variant_name,
                "generated_config_path": run_spec.generated_config_path,
                "model_path": run_spec.model_path,
                "action": pending.action,
                "returncode": -1,
                "error": str(exc),
            }
            write_run_metrics_json(Path(run_spec.model_path), fallback)
            print(f"[LOCAL] Exception in {run_spec.variant_name}: {exc}")
    gather_and_write_summaries(args, run_specs, existing_rows)
    return 0


def build_submit_command(args: argparse.Namespace) -> List[str]:
    script_path = Path(__file__).resolve()
    log_dir = Path(args.slurm_log_dir).resolve() if args.slurm_log_dir else (summary_output_root(args, []) / "slurm_logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    cpus_per_task = slurm_cpus_per_task(args)

    batch_chdir = Path(args.slurm_chdir).resolve() if args.slurm_chdir else repo_root_from_args(args).resolve()

    submit_cmd = [
        "sbatch",
        f"--job-name={args.slurm_job_name}",
        "--nodes=1",
        f"--ntasks={args.slurm_tasks}",
        f"--gres=gpu:{args.slurm_gpus}",
        f"--cpus-per-task={cpus_per_task}",
        f"--mem={args.slurm_mem}",
        f"--time={args.slurm_time}",
        f"--chdir={str(batch_chdir)}",
        f"--export={args.slurm_export}",
        f"--output={str(log_dir / '%x-%j.out')}",
        f"--error={str(log_dir / '%x-%j.err')}",
    ]
    if args.slurm_partition:
        submit_cmd.append(f"--partition={args.slurm_partition}")
    if args.slurm_account:
        submit_cmd.append(f"--account={args.slurm_account}")
    if args.slurm_qos:
        submit_cmd.append(f"--qos={args.slurm_qos}")
    for extra in args.slurm_extra_sbatch_args:
        submit_cmd.extend(shlex.split(extra))

    print(f"[SLURM SUBMIT] runner={effective_runner(args)} python={effective_python(args)} uv={uv_binary(args)} quota={resolve_quota_command(args) or args.quota_command} quota_fallback_root={args.quota_fallback_root} chdir={batch_chdir} export={args.slurm_export} step_cpus={cpus_per_task} step_mem={slurm_mem_per_worker(args)}")

    wrapped_args = [arg for arg in sys.argv[1:] if arg != "--submit-slurm"]
    wrapped_args.append("--slurm-driver")
    if args.repo_root is None:
        wrapped_args.extend(["--repo-root", str(repo_root_from_args(args))])

    wrap_cmd = wrap_script_command(
        args,
        script_path,
        [
            *wrapped_args,
            "--runner",
            effective_runner(args),
            "--uv-binary",
            uv_binary(args),
            "--python",
            effective_python(args),
            *quota_cli_args(args),
        ],
    )
    submit_cmd.extend(["--wrap", shell_join(wrap_cmd)])
    return submit_cmd


def main(args: argparse.Namespace) -> int:
    if args.slurm_worker:
        return run_worker_from_assignment(args, repo_root_from_args(args))

    schedule_options = ScheduleOptions(
        one_shot_prune_step=args.one_shot_prune_step,
        one_shot_prune_ratio=args.one_shot_prune_ratio,
        one_shot_densify_from_iter=args.one_shot_densify_from_iter,
        one_shot_densify_until_iter=args.one_shot_densify_until_iter,
        one_shot_densification_interval=args.one_shot_densification_interval,
        interleaved_prune_from_iter=args.interleaved_prune_from_iter,
        interleaved_prune_until_iter=args.interleaved_prune_until_iter,
        interleaved_prune_ratio=args.interleaved_prune_ratio,
        interleaved_prune_interval=args.interleaved_prune_interval,
        interleaved_prune_min_points=args.interleaved_prune_min_points,
        interleaved_densify_from_iter=args.interleaved_densify_from_iter,
        interleaved_densify_until_iter=args.interleaved_densify_until_iter,
        interleaved_densification_interval=args.interleaved_densification_interval,
    )

    run_specs = build_run_specs(args, schedule_options)

    if args.print_only_paths:
        for run_spec in run_specs:
            print(run_spec.model_path)
        return 0

    if args.dry_run or args.write_configs_only:
        for run_spec in run_specs:
            print(shell_join(run_spec.command))
        return 0

    if args.cleanup_existing_artifacts:
        cleanup_existing_artifacts(run_specs, True)

    pending_runs, existing_rows = build_pending_runs(run_specs, args.retry_failed_existing)

    if args.submit_slurm:
        submit_cmd = build_submit_command(args)
        print(shell_join(submit_cmd))
        result = subprocess.run(submit_cmd, check=False)
        return int(result.returncode)

    if args.slurm_driver:
        return run_slurm_driver(args, run_specs, existing_rows, pending_runs)

    return local_serial_driver(args, run_specs, existing_rows, pending_runs)


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
