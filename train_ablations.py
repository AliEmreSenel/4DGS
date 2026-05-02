from __future__ import annotations

"""Generate, launch, and log ablation training runs.

This script expands one or more base YAML configs across a Cartesian product of
ablation axes, writes a derived YAML config for each run, launches ``train.py``,
and logs a summary row per run.

Default ablation grid:
    {Isotropic, Anisotropic}
    x {RGB, SH(3)}
    x {Sort, Sort-free}
    x {No Pruning, Densify then Prune Once, Interleaved Prune+Densify}

Logged per run:
    PSNR
    SSIM
    LPIPS
    final Gaussian count
    checkpoint size
    render FPS
    peak VRAM
    training wall-clock
"""

import argparse
import ast
import csv
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
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

SUPPORTED_AXES = ("isotropy", "appearance", "sorting", "pruning")


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

    interleaved_prune_from_iter: int = 500
    interleaved_prune_until_iter: int = 7500
    interleaved_prune_ratio: float = 0.50
    interleaved_prune_interval: int = 500
    interleaved_densify_from_iter: int = 500
    interleaved_densify_until_iter: int = 7500
    interleaved_densification_interval: int = 100


@dataclass(frozen=True)
class RunSpec:
    config_path: Path
    generated_config_path: Path
    model_path: Path
    command: List[str]
    scene_name: str
    variant: AblationVariant
    index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch train.py ablation sweeps from one or more base YAML configs."
    )
    parser.add_argument("configs", nargs="+", help="Base config files.")
    parser.add_argument("--python", default="python", help="Python executable used to launch training.")
    parser.add_argument("--train-script", default="train.py", help="Training entrypoint to invoke.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional root directory for generated model outputs.",
    )
    parser.add_argument(
        "--generated-config-root",
        default=None,
        help="Optional root directory for generated YAML configs.",
    )
    parser.add_argument("--scene-name", default=None, help="Optional explicit scene name prefix.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--write-configs-only", action="store_true", help="Only write generated configs.")
    parser.add_argument("--print-only-paths", action="store_true", help="Only print generated model paths.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of generated runs per input config.")
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
    parser.add_argument(
        "--axes",
        default="isotropy,appearance,sorting,pruning",
        help="Comma-separated axis subset to use.",
    )
    parser.add_argument("--isotropy-options", default="anisotropic,isotropic")
    parser.add_argument("--appearance-options", default="rgb,sh3")
    parser.add_argument("--sorting-options", default="sort,sort_free")
    parser.add_argument(
        "--pruning-options",
        default="no_pruning,densify_then_prune_once,interleaved_prune_densify",
    )

    parser.add_argument("--one-shot-prune-step", type=int, default=5001)
    parser.add_argument("--one-shot-prune-ratio", type=float, default=0.85)
    parser.add_argument("--one-shot-densify-from-iter", type=int, default=500)
    parser.add_argument("--one-shot-densify-until-iter", type=int, default=5000)
    parser.add_argument("--one-shot-densification-interval", type=int, default=100)

    parser.add_argument("--interleaved-prune-from-iter", type=int, default=500)
    parser.add_argument("--interleaved-prune-until-iter", type=int, default=7500)
    parser.add_argument("--interleaved-prune-ratio", type=float, default=0.50)
    parser.add_argument("--interleaved-prune-interval", type=int, default=500)
    parser.add_argument("--interleaved-densify-from-iter", type=int, default=500)
    parser.add_argument("--interleaved-densify-until-iter", type=int, default=7500)
    parser.add_argument("--interleaved-densification-interval", type=int, default=100)

    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--skip-metrics", action="store_true", help="Do not run post-training metric collection.")
    parser.add_argument("--eval-split", choices=["test", "train"], default="test")
    parser.add_argument("--render-fps-warmup", type=int, default=3)
    parser.add_argument("--vram-poll-interval", type=float, default=1.0)
    parser.add_argument("--summary-filename", default="ablation_metrics.csv")
    parser.add_argument("--summary-jsonl-filename", default="ablation_metrics.jsonl")
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
        },
        "densify_then_prune_once": {
            "enable_spatio_temporal_pruning": True,
            "spatio_temporal_pruning_ratio": options.one_shot_prune_ratio,
            "spatio_temporal_pruning_from_iter": one_shot_prune_step,
            "spatio_temporal_pruning_until_iter": one_shot_prune_step,
            "spatio_temporal_pruning_interval": 1,
            "densify_from_iter": one_shot_densify_from_iter,
            "densify_until_iter": one_shot_densify_until_iter,
            "densification_interval": options.one_shot_densification_interval,
        },
        "interleaved_prune_densify": {
            "enable_spatio_temporal_pruning": True,
            "spatio_temporal_pruning_ratio": options.interleaved_prune_ratio,
            "spatio_temporal_pruning_from_iter": interleaved_prune_from_iter,
            "spatio_temporal_pruning_until_iter": interleaved_prune_until_iter,
            "spatio_temporal_pruning_interval": options.interleaved_prune_interval,
            "densify_from_iter": interleaved_densify_from_iter,
            "densify_until_iter": interleaved_densify_until_iter,
            "densification_interval": options.interleaved_densification_interval,
        },
    }


def build_axis_registry(flat_cfg: Mapping[str, Any], schedule_options: ScheduleOptions) -> Dict[str, Dict[str, Dict[str, Any]]]:
    total_iterations = int(flat_cfg.get("iterations", 0))
    return {
        "isotropy": DEFAULT_ISOTROPY,
        "appearance": DEFAULT_APPEARANCE,
        "sorting": DEFAULT_SORTING,
        "pruning": build_pruning_registry(total_iterations, schedule_options),
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


def apply_flat_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    derived = dict(config)
    for key, value in overrides.items():
        derived[key] = value
    return derived


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def assemble_command(python_exe: str, train_script: str, generated_config_path: Path, extra_args: Sequence[str]) -> List[str]:
    cmd = [python_exe, train_script, "--config", str(generated_config_path)]
    for fragment in extra_args:
        cmd.extend(shlex.split(fragment))
    return cmd


def shell_join(tokens: Sequence[str]) -> str:
    return shlex.join(list(tokens))


def build_option_map(args: argparse.Namespace) -> Dict[str, List[str]]:
    return {
        "isotropy": normalize_choice_list(args.isotropy_options),
        "appearance": normalize_choice_list(args.appearance_options),
        "sorting": normalize_choice_list(args.sorting_options),
        "pruning": normalize_choice_list(args.pruning_options),
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
        if args.limit is not None:
            variants = variants[: args.limit]

        scene_name = infer_scene_name(config_path, flat_cfg, args.scene_name)
        output_root = get_output_root(flat_cfg, args.output_root)
        config_root = generated_config_root(output_root, args.generated_config_root) / scene_name
        base_seed = int(flat_cfg.get("seed", 6666))

        for index, variant in enumerate(variants):
            model_path = build_model_path(output_root, scene_name, variant)
            run_overrides = dict(variant.overrides)
            run_overrides.update(global_overrides)
            run_overrides["model_path"] = str(model_path)
            if args.seed_offset:
                run_overrides["seed"] = base_seed + args.seed_offset + index

            derived_cfg = apply_flat_overrides(cfg, run_overrides)
            generated_config_path = config_root / f"{variant.name}.yaml"
            write_yaml(generated_config_path, derived_cfg)

            command = assemble_command(args.python, args.train_script, generated_config_path, args.extra_args)
            run_specs.append(
                RunSpec(
                    config_path=config_path,
                    generated_config_path=generated_config_path,
                    model_path=model_path,
                    command=command,
                    scene_name=scene_name,
                    variant=variant,
                    index=index,
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
            if pid == self.pid:
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
    return {"returncode": returncode, "training_wall_clock_sec": elapsed, "peak_vram_mb": peak_vram_mb}


def repo_root_from_train_script(train_script: str) -> Path:
    train_script_path = Path(train_script)
    if train_script_path.exists():
        if train_script_path.is_dir():
            return train_script_path.resolve()
        return train_script_path.resolve().parent
    return Path.cwd().resolve()


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


def robust_torch_load(torch_module: Any, path: Path, map_location: str) -> Any:
    try:
        return torch_module.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=map_location)


def find_final_checkpoint(model_path: Path, expected_iteration: int | None) -> Path:
    if expected_iteration is not None and expected_iteration > 0:
        exact = model_path / f"chkpnt{expected_iteration}.pth"
        if exact.exists():
            return exact
    ckpts = sorted(model_path.glob("chkpnt*.pth"))
    numeric: List[tuple[int, Path]] = []
    for p in ckpts:
        stem = p.stem.replace("chkpnt", "")
        if stem.isdigit():
            numeric.append((int(stem), p))
    if numeric:
        return max(numeric)[1]
    best = model_path / "chkpnt_best.pth"
    if best.exists():
        return best
    raise FileNotFoundError(f"No checkpoint found in {model_path}")


def extract_final_gaussian_count(model_params: Any) -> int | None:
    if isinstance(model_params, (tuple, list)) and len(model_params) > 1:
        xyz = model_params[1]
        if hasattr(xyz, "shape") and len(xyz.shape) >= 1:
            return int(xyz.shape[0])
    return None


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
    from utils.image_utils import psnr
    from utils.loss_utils import lpips as lpips_metric
    from utils.loss_utils import ssim

    cfg_ns = load_generated_namespace(generated_config_path, repo_root)

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

    gaussian_dim = int(getattr(merged, "gaussian_dim", 3))
    time_duration = list(getattr(merged, "time_duration", [-0.5, 0.5]))
    rot_4d = bool(getattr(merged, "rot_4d", False))
    force_sh_3d = bool(getattr(merged, "force_sh_3d", False))
    isotropic_gaussians = bool(getattr(merged, "isotropic_gaussians", False))
    num_pts = int(getattr(merged, "num_pts", 100000))
    num_pts_ratio = float(getattr(merged, "num_pts_ratio", 1.0))
    sh_degree_t = 2 if bool(getattr(pipe, "eval_shfs_4d", False)) else 0

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

    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    model_params, loaded_iter = robust_torch_load(torch, checkpoint_path, map_location)
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
            totals["psnr"] += float(psnr(image, gt_image).mean().item())
            totals["ssim"] += float(ssim(image, gt_image).mean().item())
            totals["lpips"] += float(lpips_metric(image[None].cpu(), gt_image[None].cpu()).item())

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
    metric_results["final_gaussian_count"] = extract_final_gaussian_count(model_params)
    metric_results["checkpoint_size_bytes"] = checkpoint_path.stat().st_size
    metric_results["checkpoint_path"] = str(checkpoint_path.resolve())
    return metric_results


def write_run_metrics_json(model_path: Path, payload: Mapping[str, Any]) -> Path:
    metrics_path = model_path / "run_metrics.json"
    model_path.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return metrics_path


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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
        "render_fps",
        "peak_vram_mb",
        "peak_eval_vram_mb",
        "training_wall_clock_sec",
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


def summary_output_root(args: argparse.Namespace, run_specs: Sequence[RunSpec]) -> Path:
    if args.output_root:
        return Path(args.output_root)
    if run_specs:
        return run_specs[0].model_path.parents[1]
    return Path("ablations")


def main(args: argparse.Namespace) -> int:
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
        interleaved_densify_from_iter=args.interleaved_densify_from_iter,
        interleaved_densify_until_iter=args.interleaved_densify_until_iter,
        interleaved_densification_interval=args.interleaved_densification_interval,
    )

    run_specs = build_run_specs(args, schedule_options)

    if args.print_only_paths:
        for run_spec in run_specs:
            print(str(run_spec.model_path))
        return 0

    for run_spec in run_specs:
        print(shell_join(run_spec.command))

    if args.dry_run or args.write_configs_only:
        return 0

    repo_root = repo_root_from_train_script(args.train_script)
    summary_root = summary_output_root(args, run_specs)
    summary_csv_path = summary_root / args.summary_filename
    summary_jsonl_path = summary_root / args.summary_jsonl_filename

    rows: List[Dict[str, Any]] = []
    overall_returncode = 0

    for run_spec in run_specs:
        print(f"\n=== Running {run_spec.variant.name} -> {run_spec.model_path} ===")
        train_result = run_subprocess_with_monitor(run_spec.command, repo_root, args.vram_poll_interval)

        row: Dict[str, Any] = {
            "status": "ok" if train_result["returncode"] == 0 else "failed",
            "scene_name": run_spec.scene_name,
            "variant_name": run_spec.variant.name,
            "generated_config_path": str(run_spec.generated_config_path.resolve()),
            "model_path": str(run_spec.model_path.resolve()),
            "returncode": train_result["returncode"],
            "training_wall_clock_sec": train_result["training_wall_clock_sec"],
            "peak_vram_mb": train_result["peak_vram_mb"],
        }
        row.update(run_spec.variant.tags)

        if train_result["returncode"] == 0 and not args.skip_metrics:
            try:
                cfg_flat = flatten_cfg(load_yaml(run_spec.generated_config_path))
                checkpoint_path = find_final_checkpoint(
                    run_spec.model_path,
                    expected_iteration=int(cfg_flat.get("iterations", 0)),
                )
                eval_result = evaluate_checkpoint(
                    repo_root=repo_root,
                    generated_config_path=run_spec.generated_config_path,
                    model_path=run_spec.model_path,
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
                overall_returncode = overall_returncode or 1
        elif train_result["returncode"] != 0:
            overall_returncode = overall_returncode or int(train_result["returncode"])

        metrics_json_path = write_run_metrics_json(run_spec.model_path, row)
        row["metrics_json_path"] = str(metrics_json_path.resolve())

        rows.append(row)
        append_jsonl(summary_jsonl_path, row)
        write_csv_summary(summary_csv_path, rows)

    print(f"\nWrote CSV summary to {summary_csv_path}")
    print(f"Wrote JSONL summary to {summary_jsonl_path}")
    return overall_returncode


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
