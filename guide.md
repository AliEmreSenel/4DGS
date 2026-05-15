# Scripts

The following are instructions for running the scripts.

Run all commands from the repository root so that relative paths, configuration files, and project-local modules resolve correctly.

This repository uses `uv` because it provides fast, reproducible Python environment management with a lockfile-based workflow. Compared with ad-hoc `pip` or manually managed virtual environments, `uv` makes it easier to install the same dependency set across machines and avoid accidentally running scripts with the wrong Python environment.

Install and synchronize the project dependencies with: `uv sync`. This creates or updates the local `uv` environment using the dependency versions defined by the repository.

For running scripts, use `uv run ...` ensures that the script is executed with the Python interpreter and dependencies managed by `uv`, rather than accidentally using a different system Python environment.

CUDA is required for this repository. Training, rendering, benchmarking, and the interactive viewer all rely on Gaussian rasterizers implemented as CUDA extensions. These components will not run correctly on CPU-only systems or on systems without a working CUDA-capable GPU, compatible NVIDIA drivers, and a usable CUDA/PyTorch setup.


## Training

Train a single scene from a config:

```bash
uv run python train.py --config configs/dnerf/bouncingballs.yaml
```

The config supplies defaults for model, pipeline, and optimization parameters. CLI flags override config values:

```bash
uv run python train.py \
  --config configs/dnerf/bouncingballs.yaml \
  --model_path output/debug/bouncingballs_10k \
  --iterations 10000 \
  --test_iterations 1000 5000 10000 \
  --save_iterations 10000
```

Useful training flags:

| Flag                    | Meaning                                                |
| ----------------------- | ------------------------------------------------------ |
| `--config`              | YAML config to load.                                   |
| `--model_path`          | Output directory for checkpoints, logs, and metadata.  |
| `--source_path`         | Dataset path. Usually set in the config.               |
| `--iterations`          | Number of optimization steps.                          |
| `--test_iterations`     | Iterations at which test metrics are evaluated.        |
| `--save_iterations`     | Iterations at which checkpoints are saved.             |
| `--start_checkpoint`    | Resume from a saved checkpoint.                        |
| `--seed`                | Random seed. Default is `6666`.                        |
| `--deterministic`       | Enable deterministic seeding where supported.          |
| `--exhaust_test`        | Add dense evaluation checkpoints every 500 iterations. |
| `--isotropic_gaussians` | Use isotropic instead of anisotropic Gaussian scales.  |
| `--force_sh_3d`         | Disable 4D SH/SCH evaluation and use spatial SH only.  |
| `--batch_size`          | Number of training cameras per optimization step.      |

Training writes checkpoints under `model_path`. The best checkpoint is saved as:

```text
chkpnt_best.pth
```

Scheduled checkpoints are saved as:

```text
chkpnt<iteration>.pth
```

When resuming with `--start_checkpoint`, the script reloads the immutable run configuration stored inside the checkpoint. External config files are ignored for resumed runs.

---

## Rendering

Render a trained checkpoint:

```bash
uv run python render.py \
  --model_file output/path/to/chkpnt_best.pth \
  --out_dir output/renders/bouncingballs \
  --frames 600 \
  --fps 30
```

By default, `render.py` writes:

```text
orbit_time.mp4
render_info.txt
```

Use `--save_png` to also export individual frames:

```bash
uv run python render.py \
  --model_file output/path/to/chkpnt_best.pth \
  --out_dir output/renders/bouncingballs_frames \
  --frames 120 \
  --fps 30 \
  --save_png
```

Use `--no_video` to write frames only:

```bash
uv run python render.py \
  --model_file output/path/to/chkpnt_best.pth \
  --out_dir output/renders/frames_only \
  --frames 120 \
  --no_video
```

Render modes:

| `--time_mode`          | Behavior                                                |
| ---------------------- | ------------------------------------------------------- |
| `orbit-time`           | Orbit around the scene while sweeping time. Default.    |
| `orbit-only`           | Orbit around the scene at a fixed timestamp.            |
| `time-only`            | Keep the camera fixed and sweep time.                   |
| `sync_arc_time`        | Move along the captured camera arc while sweeping time. |
| `sync_arc_freeze`      | Move along the captured camera arc at a fixed time.     |
| `bounded_novel_time`   | Use bounded novel camera motion while sweeping time.    |
| `bounded_novel_freeze` | Use bounded novel camera motion at a fixed time.        |

Example fixed-time orbit:

```bash
uv run python render.py \
  --model_file output/path/to/chkpnt_best.pth \
  --out_dir output/renders/frozen_orbit \
  --time_mode orbit-only \
  --freeze_time 0.5 \
  --frames 300 \
  --fps 30
```

Common render flags:

| Flag                         | Meaning                                                      |
| ---------------------------- | ------------------------------------------------------------ |
| `--width`, `--height`        | Override render resolution. `0` keeps checkpoint resolution. |
| `--split`                    | Camera split used to infer paths: `train`, `test`, or `all`. |
| `--time_start`, `--time_end` | Override animation time range.                               |
| `--freeze_time`              | Fixed timestamp for freeze modes.                            |
| `--video_codec`              | Codec passed to ffmpeg/imageio. Default is `libx264`.        |
| `--video_quality`            | Encoder quality. Default is `8`.                             |
| `--ffmpeg_params`            | Extra ffmpeg parameters, for example `-preset ultrafast`.    |
| `--temporal_mask_threshold`  | Override temporal active-mask threshold.                     |
| `--temporal_mask_keyframes`  | Enable cached temporal keyframe masks when greater than 1.   |
| `--temporal_mask_window`     | Number of neighboring temporal keyframes to union.           |

---

## Interactive viewer

Open a checkpoint in the CUDA viewer:

```bash
uv run python interactive_viewer.py \
  --model_file output/path/to/chkpnt_best.pth \
  --show_info
```

Useful viewer flags:

| Flag                    | Meaning                                             |
| ----------------------- | --------------------------------------------------- |
| `--split`               | Initial camera split: `train`, `test`, or `all`.    |
| `--start_camera`        | Initial camera index in the selected split.         |
| `--width`, `--height`   | Window/render size. `0` uses checkpoint resolution. |
| `--fps`                 | Target viewer frame rate. Default is `60`.          |
| `--vsync`               | Request SDL/pygame vsync.                           |
| `--move_speed`          | Manual movement speed.                              |
| `--boost`               | Shift-key movement multiplier.                      |
| `--freeze_time`         | Start at a fixed timestamp.                         |
| `--pause_time`          | Start with time playback paused.                    |
| `--sort_free_render`    | Force sort-free rendering.                          |
| `--no_sort_free_render` | Force sorted rendering.                             |

Controls:

| Input          | Action                                          |
| -------------- | ----------------------------------------------- |
| Mouse          | Look around.                                    |
| `W/S`, arrows  | Move forward/back.                              |
| `A/D`, arrows  | Strafe left/right.                              |
| `Space/Ctrl`   | Move up/down.                                   |
| `Q/E`          | Scrub time backward/forward.                    |
| `R/T`          | Roll camera.                                    |
| `Shift`        | Speed boost.                                    |
| `+/-`          | Adjust movement speed.                          |
| `P`            | Pause/resume time playback.                     |
| `I`            | Toggle FPS/VRAM/Gaussian-count overlay.         |
| `F`            | Toggle fullscreen.                              |
| `C`, `Shift+C` | Snap to next/previous camera in selected split. |
| `F1/F2`        | Previous/next train camera.                     |
| `F3/F4`        | Previous/next test camera.                      |
| `Home`         | Reset pose.                                     |
| `Tab`          | Toggle mouse capture.                           |
| `H`            | Print controls.                                 |
| `Esc`          | Quit.                                           |

---

## Ablation sweeps

Use `batch_train.py` for reproducible experiment matrices. It generates per-variant YAML configs, launches training, records final metrics, evaluates checkpoint histories, optionally exports Mobile-GS payloads, and can run either locally or through one Slurm allocation with worker tasks.

Run from the repository root so generated configs, dataset paths, and local imports resolve correctly. CUDA is required for training, metrics, rendering, and Mobile-GS benchmarking.

Check the environment before launching a sweep:

```bash
uv run python batch_train.py --preflight
```

For a Slurm run, include the Slurm flags in the preflight command so it also checks `sbatch`/`srun` availability and prints the capacity plan:

```bash
uv run python batch_train.py \
  --preflight \
  --submit-slurm \
  --slurm-partition gpuh200 \
  --slurm-gpus 4 \
  --slurm-tasks 4
```

Always inspect generated commands first:

```bash
uv run python batch_train.py configs/dnerf_ablation/*.yaml --dry-run
```

Write generated YAML configs without training:

```bash
uv run python batch_train.py configs/dnerf_ablation/*.yaml \
  --write-configs-only \
  --output-root output/ablations
```

Print only the generated output directories:

```bash
uv run python batch_train.py configs/dnerf_ablation/*.yaml \
  --print-only-paths \
  --output-root output/ablations
```

Run the default curated paper-style matrix locally:

```bash
uv run python batch_train.py configs/dnerf_ablation/*.yaml \
  --output-root output/ablations \
  --no-quota-reservation
```

Run on Slurm:

```bash
uv run python batch_train.py configs/dnerf_ablation/*.yaml \
  --submit-slurm \
  --slurm-partition gpuh200 \
  --slurm-gpus 4 \
  --slurm-tasks 4 \
  --slurm-total-cpus 8 \
  --slurm-mem 160G \
  --output-root output/ablations
```

By default, `--runner auto` prefers `uv run python` when `uv` is available and falls back to the current Python interpreter. Use `--runner python` for a direct Python launch, or `--runner uv --uv-binary /path/to/uv` to force a specific `uv`.

### Matrix presets

If no explicit axis flags are supplied, the default preset is `paper`. If any `--axes` or `--*-options` flag is supplied without `--matrix-preset`, the script switches to `cartesian`.

| Preset      | Meaning |
| ----------- | ------- |
| `paper`     | Curated method-style matrix: native 4DGS, 4DGS-1K-style pruning, DropoutGS RDR, ESS, RDR+ESS, USplat, Instant4D-lite, Mobile-GS sort-free, and selected hybrids. |
| `essential` | Exhaustive on/off combinations of implemented method families for a single scene. This can be much larger than a smoke test. |
| `compact`   | Smaller curated subset of key paper-style rows. |
| `full`      | Paper rows plus extra controls and hybrids. |
| `cartesian` | Explicit product of `--axes` and the matching `--*-options`. |

Supported Cartesian axes:

| Axis         | Options |
| ------------ | ------- |
| `isotropy`   | `anisotropic`, `isotropic` |
| `appearance` | `rgb`, `sh1`, `sh3`, `sh3_3d` |
| `sorting`    | `sort`, `sort_free` |
| `pruning`    | `no_pruning`, `early_init_pruning`, `final_pruning`, `densify_then_prune_once`, `interleaved_prune_densify` |
| `usplat`     | `no_usplat`, `use_usplat` |
| `dropout`    | `no_dropout`, `dropout`, `use_dropout` |
| `ess`        | `no_ess`, `ess`, `use_ess` |

Known invalid combinations are filtered by default. Notably, `sort_free_render` is incompatible with env maps, depth loss, opacity-mask loss, and USplat uncertainty scoring. Use `--include-invalid-combinations` only when deliberately debugging those cases.

### Iteration and schedule defaults

`batch_train.py` retargets generated runs to `--max-iters 30000` by default. It scales iteration-like schedule values such as densification, pruning, ESS, opacity reset, SH increase, and USplat start from the source config to the target iteration count.

Use this to preserve each input config's original iteration count:

```bash
uv run python batch_train.py configs/dnerf_ablation/*.yaml \
  --max-iters 0 \
  --output-root output/ablations
```

Useful schedule controls:

| Flag | Meaning |
| ---- | ------- |
| `--max-iters N` | Target iteration count for generated runs. Default `30000`; `0` keeps config iterations unchanged. |
| `--schedule-reference-iters N` | Reference iteration count for schedule CLI defaults. Default `15000`. |
| `--scale-schedule-intervals` / `--no-scale-schedule-intervals` | Enable/disable scaling of interval-style values. |
| `--seed-offset N` | Offset seeds across generated variants. |
| `--set KEY=VALUE` | Apply a flat override to every generated run. Values are parsed as Python literals when possible. |
| `--extra-arg '...'` | Append raw CLI args to every `train.py` command. |

Pruning schedule controls:

| Flag | Meaning |
| ---- | ------- |
| `--early-init-prune-step`, `--early-init-prune-ratio` | Configure `early_init_pruning`. |
| `--final-prune-fraction`, `--final-prune-ratio` | Configure `final_pruning`. |
| `--one-shot-prune-step`, `--one-shot-prune-ratio` | Configure `densify_then_prune_once`. |
| `--one-shot-densify-from-iter`, `--one-shot-densify-until-iter`, `--one-shot-densification-interval` | Densification schedule before one-shot pruning. |
| `--interleaved-prune-from-iter`, `--interleaved-prune-until-iter`, `--interleaved-prune-ratio`, `--interleaved-prune-interval` | Interleaved spatio-temporal pruning schedule. |
| `--interleaved-densify-from-iter`, `--interleaved-densify-until-iter`, `--interleaved-densification-interval` | Densification schedule paired with interleaved pruning. |

### Local 8 GB laptop mode

Use `--laptop-8gb` for local low-memory runs:

```bash
uv run python batch_train.py configs/dnerf_ablation/bouncingballs.yaml \
  --laptop-8gb \
  --limit 4 \
  --output-root output/ablations_laptop
```

Laptop mode changes defaults unless you explicitly override them:

| Default changed by `--laptop-8gb` | Value |
| --------------------------------- | ----- |
| Runner | `python` |
| Quota reservation | disabled |
| Matrix preset | `essential` if no matrix flags were supplied |
| Max iterations | `10000` |
| Concurrent ablations per GPU | `1` |
| Batch size | `1` |
| Initial points | `50000` |
| Save/test iterations | `[10000]` |
| Resolution | `4` |
| Densification cap | `150000` points |

For a tiny command-generation smoke test, override iterations explicitly:

```bash
uv run python batch_train.py configs/dnerf_ablation/bouncingballs.yaml \
  --laptop-8gb \
  --matrix-preset cartesian \
  --axes isotropy,appearance,sorting \
  --isotropy-options anisotropic,isotropic \
  --appearance-options rgb,sh3 \
  --sorting-options sort,sort_free \
  --limit 4 \
  --output-root output/ablations_smoke \
  --set iterations=1000 \
  --set position_lr_max_steps=1000 \
  --set 'test_iterations=[1000]' \
  --set 'save_iterations=[1000]'
```

### Cartesian examples

Fixed sort-free Bouncing Balls sweep:

```bash
uv run python batch_train.py configs/dnerf_ablation/bouncingballs.yaml \
  --laptop-8gb \
  --matrix-preset cartesian \
  --axes isotropy,usplat,appearance,sorting,pruning,dropout,ess \
  --isotropy-options anisotropic,isotropic \
  --usplat-options no_usplat \
  --appearance-options sh3 \
  --sorting-options sort_free \
  --pruning-options no_pruning \
  --dropout-options no_dropout \
  --ess-options no_ess \
  --output-root output/ablations_fixed_longer_sortfree \
  --set 'save_iterations=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]' \
  --set 'test_iterations=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]' \
  --set iterations=10000 \
  --set position_lr_max_steps=10000 \
  --set ess_max_splits=2500 \
  --set use_usplat=False \
  --no-quota-reservation
```

TRex sweep with dropout enabled globally instead of as an axis:

```bash
uv run python batch_train.py configs/dnerf_ablation/trex.yaml \
  --laptop-8gb \
  --matrix-preset cartesian \
  --axes isotropy,appearance,sorting,pruning,ess \
  --sorting-options sort,sort_free \
  --output-root output/ablations_no_usplat_dropout_on \
  --set 'save_iterations=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]' \
  --set 'test_iterations=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]' \
  --set use_usplat=False \
  --set random_dropout_prob=0.2 \
  --set lambda_rdr=1.0 \
  --no-quota-reservation
```

### Metrics, resume behavior, and outputs

The runner auto-detects completed runs. If `run_metrics.json` is already `ok`, the run is skipped. If a checkpoint exists but metrics are missing or failed, it runs metrics-only. Failed existing runs are retried by default; use `--no-retry-failed-existing` to keep old failure records.

Metrics flags:

| Flag | Meaning |
| ---- | ------- |
| `--skip-metrics` | Train only; skip final metrics and Mobile-GS quality comparisons. |
| `--eval-split test|train` | Split for final metrics. Default `test`, falling back to train if no test cameras exist. |
| `--render-fps-warmup N` | Warmup renders before FPS timing. |
| `--vram-poll-interval SEC` | `nvidia-smi` polling interval for peak training VRAM. |
| `--summary-filename NAME` | Summary CSV filename. Default `ablation_metrics.csv`. |
| `--summary-jsonl-filename NAME` | Summary JSONL filename. Default `ablation_metrics.jsonl`. |
| `--skip-checkpoint-metrics` | Disable per-checkpoint training-curve metrics. |
| `--checkpoint-eval-split test|train` | Split for checkpoint-history metrics. Default `test`. |
| `--checkpoint-metrics-filename NAME` | Per-run checkpoint-history CSV filename. |
| `--checkpoint-metrics-jsonl-filename NAME` | Per-run checkpoint-history JSONL filename. |

Batch output files include:

| File | Contents |
| ---- | -------- |
| `run_metrics.json` | Per-run status, final metrics, paths, timing, VRAM, quota, Mobile-GS fields, and diagnostics. |
| `ablation_metrics.csv` | Summary table across runs. |
| `ablation_metrics.jsonl` | JSONL version of the summary. |
| `checkpoint_eval_metrics.csv` | Aggregated checkpoint-history metrics across runs. |
| `checkpoint_eval_metrics.jsonl` | JSONL version of checkpoint-history metrics. |
| `<run>/checkpoint_eval_metrics.csv` | Per-run checkpoint-history metrics. |
| `<run>/checkpoint_eval_metrics.jsonl` | Per-run checkpoint-history metrics in JSONL. |
| `mobilegs_metrics.json` | Per-run Mobile-GS export/benchmark metrics when enabled. |
| `mobilegs_quantized.mobile.pt` | Per-run quantized Mobile-GS payload when enabled. |

### Mobile-GS reporting

Mobile-GS export, compression, and benchmark reporting is enabled by default for all rows, not only sort-free rows. It stores compressed payloads and post-quantization speed/quality metrics next to each run.

Common Mobile-GS reporting flags:

| Flag | Meaning |
| ---- | ------- |
| `--mobilegs-report` / `--no-mobilegs-report` | Enable/disable post-training Mobile-GS reporting. Default enabled. |
| `--mobilegs-report-scope all|sort_free` | Report all rows or only sort-free rows. Default `all`. |
| `--require-mobilegs-report` / `--no-require-mobilegs-report` | Fail the run if Mobile-GS reporting fails. Default does not fail the run. |
| `--mobilegs-benchmark-render-mode match|sort_free|sorted` | Renderer used for compressed-payload benchmarking. Default `match`. |
| `--mobilegs-force-first-order-sh` | Train/export first-order SH for Mobile-GS reporting when applicable. |
| `--mobilegs-teacher-checkpoint PATH` | Optional sorted-render teacher checkpoint for Mobile-GS distillation. |
| `--mobilegs-sh-distill-lambda`, `--mobilegs-depth-distill-lambda` | Teacher RGB/depth distillation weights. |
| `--mobilegs-codebook-size`, `--mobilegs-block-size`, `--mobilegs-kmeans-iters`, `--mobilegs-uniform-bits` | NVQ/uniform quantization settings. |
| `--mobilegs-build-visibility-filter` / `--no-mobilegs-build-visibility-filter` | Enable/disable temporal visibility-mask export. Default enabled. |
| `--mobilegs-temporal-keyframes`, `--mobilegs-temporal-mask-window`, `--mobilegs-temporal-mask-threshold`, `--mobilegs-views-per-keyframe` | Temporal visibility-mask controls. |
| `--mobilegs-benchmark-split test|train` | Split used for Mobile-GS benchmark. Default `test`. |
| `--mobilegs-benchmark-warmup`, `--mobilegs-benchmark-repeats`, `--mobilegs-quality-samples` | Benchmark and quality-sampling controls. |
| `--mobilegs-mobile-filename`, `--mobilegs-metrics-filename` | Per-run output filenames. |

Example: train all rows but only run Mobile-GS reporting for sort-free rows:

```bash
uv run python batch_train.py configs/dnerf_ablation/*.yaml \
  --output-root output/ablations \
  --mobilegs-report-scope sort_free
```

### Slurm, quota, and cleanup

`--submit-slurm` submits one allocation and launches long-lived `srun` worker steps. Runs are greedily assigned to workers by estimated cost. Each worker can run multiple ablations concurrently with `--ablations-per-gpu`; the default is `3` for large GPUs, while `--laptop-8gb` lowers it to `1`.

Useful Slurm flags:

| Flag | Meaning |
| ---- | ------- |
| `--submit-slurm` | Submit an `sbatch` allocation. With `--dry-run`, prints the `sbatch` command only. |
| `--slurm-partition`, `--slurm-account`, `--slurm-qos`, `--slurm-time` | Standard Slurm allocation controls. |
| `--slurm-gpus`, `--slurm-gpus-per-node`, `--slurm-nodes` | GPU/node allocation controls. |
| `--slurm-tasks`, `--slurm-tasks-per-node` | Worker task layout. |
| `--slurm-total-cpus`, `--slurm-mem` | CPU and memory allocation. |
| `--slurm-gres`, `--slurm-worker-gres` | Exact GRES strings for `sbatch`/worker `srun`. |
| `--ablations-per-gpu`, `--runs-per-gpu` | Concurrent ablations per worker/GPU. |
| `--slurm-log-dir`, `--slurm-job-name` | Slurm log and job naming. |
| `--slurm-export`, `--slurm-chdir` | Batch environment export and working directory. |
| `--slurm-extra-sbatch-arg`, `--slurm-srun-extra-arg` | Append raw Slurm args. |

Quota reservation is enabled by default for local and Slurm runs. Disable it on local machines or clusters without the expected quota tooling:

```bash
uv run python batch_train.py configs/dnerf_ablation/*.yaml \
  --output-root output/ablations \
  --no-quota-reservation
```

Quota and cleanup flags:

| Flag | Meaning |
| ---- | ------- |
| `--quota-reservation` / `--no-quota-reservation` | Reserve quota before active training runs. |
| `--quota-command` | Quota command. Default `lquota`. |
| `--quota-fallback-root` | Directory measured with `du` if quota command is unavailable. |
| `--quota-limit-gb`, `--quota-reserve-gb`, `--train-run-peak-storage-gb` | Quota accounting controls. |
| `--quota-poll-interval` | Seconds between quota checks while waiting. |
| `--cleanup-existing-artifacts` / `--no-cleanup-existing-artifacts` | Prune bulky artifacts from existing run dirs before scheduling when safe. |
| `--cleanup-after-run` / `--no-cleanup-after-run` | Delete bulky artifacts after each run while preserving metadata and the best available checkpoint. |

---

## Rendering ablation videos

`ablation_script.sh` batch-renders ablation checkpoints into consistently named videos.

```bash
bash ablation_script.sh
```

Common environment overrides:

```bash
OUTPUT_ROOT=output/ablations_fixed_longer_sortfree \
OUT_ROOT=output/ablation_vids \
FRAMES=300 \
FPS=30 \
TIME_MODE=orbit-time \
SPLIT=all \
bash ablation_script.sh
```

Dry run:

```bash
DRY_RUN=1 bash ablation_script.sh
```

The script also includes a black-video check. Disable it only when debugging:

```bash
BLACK_CHECK=0 bash ablation_script.sh
```

---

## HTML result browser

Two scripts generate HTML pages from rendered output. `html_export.py` is the general-purpose browser - it scans a directory recursively and visualises the full range of ablation experiments. `html_poster.py` is a hardcoded, narrowed version built specifically for the poster figures; it shows a fixed set of scenes and configurations and does not accept a directory argument.

Create a local HTML overview of rendered ablation videos and images:

```bash
uv run python html_export.py output/ablation_vids -o ablations.html
uv run python html_poster.py output/ablation_vids -o ablations.html

```

Disable recursive search:

```bash
uv run python html_export.py output/ablation_vids \
  --no-recursive \
  -o ablations.html
```

Open `ablations.html` in a browser to compare generated videos/images and any matching checkpoint metrics.

---

## Compression

### Simple checkpoint compression

Use `compress.py` for a compact compression sanity check and round-trip validation:

```bash
uv run python compress.py \
  --ckpt-path output/path/to/chkpnt_best.pth \
  --compressed-path output/path/to/chkpnt_best_compressed.pth
```

Useful options:

```bash
uv run python compress.py \
  --ckpt-path output/path/to/chkpnt_best.pth \
  --compressed-path output/path/to/chkpnt_best_compressed.pth \
  --attr-bits 8 \
  --device auto
```

`compress.py` reports raw checkpoint size, Gaussian payload size, compressed size, reduction ratios, and reconstruction error.

### Universal postprocess compression

Use `compression_postprocess.py` for full post-training compression, optional pruning, evaluation, and visual diagnostics:

```bash
uv run python compression_postprocess.py \
  --ckpt-path output/path/to/chkpnt_best.pth \
  --output-dir output/compression_eval/bouncingballs \
  --codec mobilegs \
  --eval-samples 8
```

Common codecs:

| Codec      | Meaning                                                                |
| ---------- | ---------------------------------------------------------------------- |
| `mobilegs` | GPCC xyz compression plus NVQ attribute codebooks and uniform opacity. |
| `nvq`      | NVQ attribute compression.                                             |
| `uniform`  | Uniform quantization.                                                  |
| `float16`  | Half-precision payload.                                                |
| `float32`  | Uncompressed float payload baseline.                                   |

Useful options:

| Flag                        | Meaning                                                                |
| --------------------------- | ---------------------------------------------------------------------- |
| `--target-gaussians`        | Contribution-prune to a target Gaussian count.                         |
| `--target-size-mb`          | Estimate pruning needed for a target tensor budget.                    |
| `--sh-degree-cap`           | Cap SH degree during export.                                           |
| `--mobilegs-first-order-sh` | Export Mobile-GS-style first-order SH.                                 |
| `--build-temporal-filter`   | Build keyframe visibility masks for temporal acceleration.             |
| `--render-mode`             | Use `checkpoint`, `sorted`, or `sort_free` renderer during evaluation. |
| `--save-renders`            | Save rendered comparison images.                                       |
| `--save-difference-plots`   | Save visual error plots.                                               |
| `--save-gaussian-plots`     | Save Gaussian distribution diagnostics.                                |

The main metrics file is:

```text
compression_postprocess_metrics.json
```

---

## Mobile-GS export and benchmarking

Export a Mobile-GS/NVQ payload:

```bash
uv run python mobile_export.py \
  --ckpt-path output/path/to/chkpnt_best.pth \
  --output output/path/to/model.mobile.pt
```

Benchmark the exported payload:

```bash
uv run python mobile_benchmark.py \
  --ckpt-path output/path/to/chkpnt_best.pth \
  --mobile-path output/path/to/model.mobile.pt \
  --split test \
  --warmup 20 \
  --repeats 200 \
  --quality-samples 16 \
  --output-json output/path/to/mobile_benchmark.json
```

Useful export flags:

| Flag                        | Meaning                                              |
| --------------------------- | ---------------------------------------------------- |
| `--first-order-sh`          | Export first-order SH. Enabled by default.           |
| `--keep-full-sh`            | Preserve full SH instead of slicing to first order.  |
| `--codebook-size`           | NVQ codebook size. Default is `256`.                 |
| `--block-size`              | NVQ block size. Default is `8`.                      |
| `--uniform-bits`            | Uniform quantization bits. Default is `8`.           |
| `--build-visibility-filter` | Build temporal visibility masks. Enabled by default. |
| `--temporal-keyframes`      | Number of temporal keyframes. Default is `32`.       |

---

## Utility scripts

Dataset and profiling helpers live in `scripts/`.

Convert synchronized videos into a 4DGS dataset:

```bash
uv run python scripts/video_to_4dgs_dataset.py \
  --videos cam0.mp4 cam1.mp4 cam2.mp4 \
  --out data/custom/my_scene \
  --scene-name my_scene \
  --fps 10 \
  --colmap-fps 6
```

Convert a NeRF Capture export zip into D-NeRF-style layout:

```bash
uv run python scripts/nerfcapture_zip_to_dnerf.py \
  capture.zip \
  data/custom/capture_dnerf
```

Profile sorted vs sort-free checkpoint FPS:

```bash
uv run python scripts/fps_profile_ckpts.py \
  --sorted-ckpt output/sorted/chkpnt_best.pth \
  --sort-free-ckpt output/sort_free/chkpnt_best.pth \
  --split test \
  --warmup 50 \
  --repeats 1000 \
  --output-json fps_profile_results.json \
  --output-csv fps_profile_results.csv
```

Collect low-PSNR runs for reruns:

```bash
uv run python scripts/collect_low_psnr_reruns_small.py \
  output/ablations \
  --threshold 15.0 \
  --out-dir rerun_cfg \
  --dry-run
```

## Gaussian visualizer

Export an interactive 3D Plotly visualization of the Gaussians from a checkpoint:

```bash
uv run python export_4dgs_plotly_html.py \
  output/best_checkpoints/trex800.pth \
  --toggle-checkpoint output/best_checkpoints/bouncingballs800.pth \
  --checkpoint-labels trex bouncingballs \
  --t 0.5 \
  --out plotly_gaussians.html \
  --max-gaussians 2000 \
  --gray-opacity-knee 0.06 \
  --gray-opacity-power 2.5 \
  --gray-min-opacity 0.015 \
  --leftover-opacity 1.0 \
  --include-plotlyjs cdn \
  --sigma 2.0
```

Open `plotly_gaussians.html` in a browser to inspect the Gaussian ellipsoids at a chosen timestamp. When `--toggle-checkpoint` is supplied, a **Dataset** button appears on the page to switch between the two checkpoints without reloading.

The script also writes a companion data file:

```text
plotly_data/<output-stem>.js
```

This file must stay alongside the HTML when sharing or moving the output.

### Flags

| Flag                       | Meaning                                                                                   |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| `checkpoint`               | Path to the primary `.pth` checkpoint to visualize.                                       |
| `--toggle-checkpoint`      | Optional second checkpoint toggled by a button in the page.                               |
| `--checkpoint-labels`      | Two labels for the dataset toggle button, e.g. `trex bouncingballs`.                      |
| `--t`                      | Timestamp to evaluate the Gaussians at. Required.                                         |
| `--out`                    | Output HTML path. Default `gaussians_snippet.html`.                                       |
| `--max-gaussians`          | Maximum number of Gaussians to render. `0` exports all. Default `1000`.                   |
| `--opacity-min`            | Minimum alpha to include a Gaussian. Default `0.0`.                                       |
| `--sigma`                  | Ellipsoid radius in standard deviations. Larger = bigger ellipsoids. Default `1.0`.       |
| `--ellipsoid-res`          | Sphere mesh resolution per ellipsoid. Default `6`.                                        |
| `--gray-opacity-knee`      | Transition knee for de-emphasising gray Gaussians during selection. Default `0.08`.       |
| `--gray-opacity-power`     | Sharpness of the color/white opacity transition during selection. Default `2.0`.          |
| `--white-opacity-start`    | RGB min-channel threshold where near-white Gaussians are favored. Default `0.82`.         |
| `--gray-min-opacity`       | Minimum selection weight for gray or low-chroma Gaussians. Default `0.015`.               |
| `--render-opacity`         | Final rendered opacity applied uniformly to every selected Gaussian. Default `1.0`.       |
| `--leftover-opacity`       | Alias for `--render-opacity`, kept for backwards compatibility.                           |
| `--camera-center`          | Camera origin used for SH color evaluation, as three floats. Default `0.0 0.0 0.0`.      |
| `--include-plotlyjs`       | `cdn` for a standalone file with a CDN script tag; `false` for an embeddable snippet.     |
