# Scripts

The following are instructions for running the scripts.

Run all commands from the repository root so that relative paths, configuration files, and project-local modules resolve correctly.

This repository uses `uv` because it provides fast, reproducible Pythonenvironment management with a lockfile-based workflow. Compared with ad-hoc `pip` or manually managed virtual environments, `uv` makes it easier to install the same dependency set across machines and avoid accidentally running scripts with the wrong Python environment.

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

Use `batch_train.py` for reproducible experiment matrices. It generates per-variant configs, launches training, records metrics, and can optionally submit a Slurm job.

Always inspect generated commands first:

```bash
uv run python batch_train.py configs/dnerf_ablation/*.yaml --dry-run
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
  --output-root output/ablations
```

Recommended matrix presets:

| Preset      | Meaning                                                                  |
| ----------- | ------------------------------------------------------------------------ |
| `paper`     | Curated method-style matrix. Default when no explicit axes are supplied. |
| `essential` | Broader on/off coverage of implemented method families.                  |
| `compact`   | Smaller curated sweep.                                                   |
| `full`      | Larger curated sweep with extra controls/hybrids.                        |
| `cartesian` | Explicit product of `--axes` and the matching `--*-options`.             |

Supported Cartesian axes:

| Axis         | Example options                                                      |
| ------------ | -------------------------------------------------------------------- |
| `isotropy`   | `anisotropic`, `isotropic`                                           |
| `appearance` | `rgb`, `sh1`, `sh3`, `sh3_3d`                                        |
| `sorting`    | `sort`, `sort_free`                                                  |
| `pruning`    | `no_pruning`, `densify_then_prune_once`, `interleaved_prune_densify` |
| `usplat`     | `no_usplat`, `use_usplat`                                            |
| `dropout`    | `no_dropout`, `dropout`                                              |
| `ess`        | `no_ess`, `ess`                                                      |

Example Cartesian smoke test:

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
  --no-quota-reservation
```

Example fixed sort-free Bouncing Balls sweep:

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
  --set 'save_iterations=[]' \
  --set 'test_iterations=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]' \
  --set iterations=10000 \
  --set position_lr_max_steps=10000 \
  --set ess_max_splits=2500 \
  --set use_usplat=False \
  --no-quota-reservation
```

Example TRex sweep with dropout enabled for every variant:

```bash
uv run python batch_train.py configs/dnerf_ablation/trex.yaml \
  --laptop-8gb \
  --matrix-preset cartesian \
  --axes isotropy,appearance,sorting,pruning,ess \
  --sorting-options sort,sort_free \
  --output-root output/ablations_no_usplat_dropout_on \
  --set 'save_iterations=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]' \
  --set use_usplat=False \
  --set random_dropout_prob=0.2 \
  --set lambda_rdr=1.0 \
  --no-quota-reservation
```

In the second command, dropout is not an axis. It is enabled globally through `random_dropout_prob` and `lambda_rdr`.

Important batch flags:

| Flag                                             | Meaning                                                                              |
| ------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `--dry-run`                                      | Print generated commands without running.                                            |
| `--write-configs-only`                           | Generate YAML configs only.                                                          |
| `--print-only-paths`                             | Print generated output paths only.                                                   |
| `--limit`                                        | Limit variants per input config. Useful for smoke tests.                             |
| `--set KEY=VALUE`                                | Apply a flat override to every generated run.                                        |
| `--extra-arg`                                    | Append raw CLI arguments to each training command.                                   |
| `--output-root`                                  | Root directory for generated model outputs.                                          |
| `--generated-config-root`                        | Root directory for generated YAML configs.                                           |
| `--skip-metrics`                                 | Train only; do not run post-training metrics.                                        |
| `--skip-checkpoint-metrics`                      | Disable per-checkpoint training-curve metrics.                                       |
| `--mobilegs-report` / `--no-mobilegs-report`     | Enable/disable Mobile-GS export and benchmark after each run.                        |
| `--mobilegs-report-scope`                        | Run Mobile-GS reporting for `all` rows or only `sort_free` rows.                     |
| `--submit-slurm`                                 | Submit one Slurm allocation with worker tasks.                                       |
| `--no-quota-reservation`                         | Disable quota checks, recommended for local machines.                                |
| `--cleanup-after-run` / `--no-cleanup-after-run` | Delete bulky artifacts after each run while preserving metadata and best checkpoint. |

Batch output files include:

| File                            | Contents                                         |
| ------------------------------- | ------------------------------------------------ |
| `run_metrics.json`              | Per-run final metrics and status.                |
| `ablation_metrics.csv`          | Summary table across runs.                       |
| `ablation_metrics.jsonl`        | JSONL version of the summary.                    |
| `checkpoint_eval_metrics.csv`   | Metrics for evaluated checkpoints over training. |
| `checkpoint_eval_metrics.jsonl` | JSONL version of per-checkpoint metrics.         |
| `mobilegs_metrics.json`         | Mobile-GS export/benchmark metrics when enabled. |
| `mobilegs_quantized.mobile.pt`  | Quantized Mobile-GS payload when enabled.        |

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

Create a local HTML overview of rendered ablation videos and images:

```bash
uv run python html_export.py output/ablation_vids -o ablations.html
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
