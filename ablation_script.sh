#!/usr/bin/env bash
set -euo pipefail

# Batch-render all ablation checkpoints to consistently named videos.
#
# Usage:
#   chmod +x render_ablation_videos.sh
#   ./render_ablation_videos.sh
#
# Common overrides:
#   RENDER_SCRIPT=render.py ./render_ablation_videos.sh
#   OUTPUT_ROOT=output OUT_ROOT=output/ablation_vids FRAMES=300 FPS=30 ./render_ablation_videos.sh
#   TIME_MODE=bounded_novel_time SPLIT=all ./render_ablation_videos.sh
#   DRY_RUN=1 ./render_ablation_videos.sh
#   BLACK_CHECK=0 ./render_ablation_videos.sh
#
# The render script must be the attached checkpoint video renderer. It accepts:
#   --repo_root --model_file --out_dir --frames --fps --time_mode --split ...

REPO_ROOT="${REPO_ROOT:-.}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
OUT_ROOT="${OUT_ROOT:-output/ablation_vids}"
RENDER_SCRIPT="${RENDER_SCRIPT:-render.py}"

FRAMES="${FRAMES:-600}"
FPS="${FPS:-30}"
WIDTH="${WIDTH:-0}"
HEIGHT="${HEIGHT:-0}"
TIME_MODE="${TIME_MODE:-orbit-time}"
SPLIT="${SPLIT:-all}"
VIDEO_CODEC="${VIDEO_CODEC:-libx264}"
VIDEO_QUALITY="${VIDEO_QUALITY:-8}"
MACRO_BLOCK_SIZE="${MACRO_BLOCK_SIZE:-16}"
WRITER_QUEUE="${WRITER_QUEUE:-8}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"

# Black-video protection.
BLACK_CHECK="${BLACK_CHECK:-1}"
KEEP_BLACK_TMP="${KEEP_BLACK_TMP:-1}"
BLACK_MEAN_THRESHOLD="${BLACK_MEAN_THRESHOLD:-2.0}"
BLACK_NONBLACK_FRACTION_THRESHOLD="${BLACK_NONBLACK_FRACTION_THRESHOLD:-0.001}"

# Optional renderer tuning. Leave empty to omit.
TEMPORAL_MASK_THRESHOLD="${TEMPORAL_MASK_THRESHOLD:-}"
TEMPORAL_MASK_KEYFRAMES="${TEMPORAL_MASK_KEYFRAMES:-}"
TEMPORAL_MASK_WINDOW="${TEMPORAL_MASK_WINDOW:-}"
FFMPEG_PARAMS="${FFMPEG_PARAMS:-}"
EXTRA_RENDER_ARGS="${EXTRA_RENDER_ARGS:-}"

mkdir -p "$OUT_ROOT"

if [[ ! -f "$RENDER_SCRIPT" ]]; then
  echo "Missing renderer: $RENDER_SCRIPT" >&2
  echo "Set RENDER_SCRIPT=/path/to/the/attached/python/file" >&2
  exit 2
fi

video_is_black() {
  local video="$1"

  python3 - "$video" "$BLACK_MEAN_THRESHOLD" "$BLACK_NONBLACK_FRACTION_THRESHOLD" <<'PY'
import subprocess
import sys

video = sys.argv[1]
mean_threshold = float(sys.argv[2])
nonblack_fraction_threshold = float(sys.argv[3])

cmd = [
    "ffmpeg", "-v", "error",
    "-i", video,
    "-vf", "fps=1,scale=64:64:flags=bilinear,format=gray",
    "-f", "rawvideo",
    "-"
]

p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if p.returncode != 0:
    sys.stderr.write(p.stderr.decode(errors="replace"))
    sys.exit(2)

buf = p.stdout
if not buf:
    print("black-check: no sampled frames", file=sys.stderr)
    sys.exit(2)

mean_luma = sum(buf) / len(buf)
nonblack_fraction = sum(1 for b in buf if b > 8) / len(buf)

print(
    f"black-check: mean_luma={mean_luma:.3f} "
    f"nonblack_fraction={nonblack_fraction:.6f}",
    file=sys.stderr,
)

# Exit 0 means black.
# Exit 1 means not black.
if mean_luma < mean_threshold and nonblack_fraction < nonblack_fraction_threshold:
    sys.exit(0)

sys.exit(1)
PY
}

metadata_for_checkpoint() {
  local ckpt="$1"
  uv run - "$ckpt" <<'PY'
import csv
import re
import sys
from pathlib import Path

MATRIX_PRESET = "cartesian"
USE_USPLAT = False
DROPOUT_DEFAULT = "no_dropout"

ckpt = Path(sys.argv[1]).resolve()
variant_dir = ckpt.parent
scene_dir = variant_dir.parent
ablation_dir = scene_dir.parent
scene_name = scene_dir.name
variant_name_dir = variant_dir.name
ablation_name = ablation_dir.name
ckpt_name = ckpt.name


def parse_variant_dirname(dirname: str) -> dict:
    parts = dirname.split("--")
    use_usplat = USE_USPLAT
    if len(parts) == 5:
        isotropy, appearance, sorting, pruning, ess = parts
        dropout = DROPOUT_DEFAULT
    elif len(parts) == 6:
        isotropy, appearance, sorting, pruning, a, b = parts
        ess_tokens = {"ess", "no_ess"}
        dropout_tokens = {"dropout", "no_dropout", "yes_dropout"}
        if a in ess_tokens and b in dropout_tokens:
            ess, dropout = a, b
        else:
            dropout, ess = a, b
    elif len(parts) == 7:
        isotropy, appearance, sorting, pruning, ess, dropout, usplat = parts
        use_usplat = usplat != "no_usplat"
    else:
        raise ValueError(f"Unexpected variant folder name: {dirname}")
    return {
        "scene_name": scene_name,
        "isotropy": isotropy,
        "appearance": appearance,
        "sorting": sorting,
        "pruning": pruning,
        "dropout": dropout,
        "ess": ess,
        "use_usplat": use_usplat,
    }


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def truthy(value):
    return clean(value).lower() in {"1", "true", "yes", "y", "use_usplat"}


def infer_use_usplat_from_tokens(row):
    tokens = []
    for key in ("variant_name", "model_path", "generated_config_path"):
        v = clean(row.get(key))
        if not v:
            continue
        if key == "variant_name":
            tokens.extend(v.split("__"))
        elif key == "model_path":
            tokens.extend(Path(v).name.split("--"))
        else:
            tokens.extend(Path(v).stem.split("__"))
    return "use_usplat" in tokens


def checkpoint_iteration_from_name(name):
    m = re.search(r"(\d+)", name)
    return m.group(1) if m else ""


try:
    meta = parse_variant_dirname(variant_name_dir)
except ValueError:
    base = re.sub(r"[^A-Za-z0-9_.+-]+", "-", variant_name_dir).strip("-")
    print(base or "NA")
    sys.exit(0)

csv_path = variant_dir / "checkpoint_eval_metrics.csv"
matched_row = None

if csv_path.exists():
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        rows = []
        for row in reader:
            row = {clean(k): clean(v) for k, v in row.items() if k is not None}
            rows.append(row)

        ckpt_resolved = str(ckpt)
        ckpt_iter = checkpoint_iteration_from_name(ckpt_name)

        for row in rows:
            candidates = [
                clean(row.get("checkpoint_path")),
                clean(row.get("checkpoint_filename")),
            ]
            candidate_names = {Path(c).name for c in candidates if c}
            if (
                ckpt_resolved in candidates
                or ckpt_name in candidates
                or ckpt_name in candidate_names
            ):
                matched_row = row
                break

        if matched_row is None and ckpt_iter:
            for row in rows:
                row_iters = {
                    clean(row.get("eval_checkpoint_iteration")),
                    clean(row.get("checkpoint_name_iteration")),
                }
                if ckpt_iter in row_iters:
                    matched_row = row
                    break

        if matched_row is None and len(rows) == 1:
            matched_row = rows[0]

if matched_row:
    for key in ("scene_name", "isotropy", "appearance", "sorting", "pruning", "dropout", "ess"):
        if clean(matched_row.get(key)):
            meta[key] = clean(matched_row[key])

    if clean(matched_row.get("use_usplat")):
        meta["use_usplat"] = truthy(matched_row.get("use_usplat"))

    if clean(matched_row.get("eval_checkpoint_iteration")):
        meta["eval_checkpoint_iteration"] = clean(matched_row["eval_checkpoint_iteration"])

key = f"{ablation_name}/{scene_name}"

if key == "ablations_usplat_sort_dropout_off/trex":
    meta["use_usplat"] = True

if key == "ablations_no_usplat_dropout_on/trex":
    meta["dropout"] = "yes_dropout"

if key == "ablations_bouncing_balls_usplat_vs_no_usplat_7k/bouncingballs":
    meta["use_usplat"] = infer_use_usplat_from_tokens(
        matched_row
        or {
            "variant_name": "__".join(
                [
                    meta["isotropy"],
                    meta["appearance"],
                    meta["sorting"],
                    meta["pruning"],
                    meta["dropout"],
                    meta["ess"],
                ]
            ),
            "model_path": str(variant_dir),
            "generated_config_path": "",
        }
    )

if "eval_checkpoint_iteration" not in meta or not clean(meta["eval_checkpoint_iteration"]):
    meta["eval_checkpoint_iteration"] = checkpoint_iteration_from_name(ckpt_name) or ckpt.stem

ordered = [
    "scene_name",
    "isotropy",
    "use_usplat",
    "appearance",
    "sorting",
    "pruning",
    "dropout",
    "ess",
    "eval_checkpoint_iteration",
]

values = []

for col in ordered:
    value = meta[col]
    if col == "use_usplat":
        value = "use_usplat" if bool(value) else "no_usplat"

    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "-", value).strip("-")
    values.append(value or "NA")

print("__".join(values))
PY
}

find "$OUTPUT_ROOT" -mindepth 4 -maxdepth 4 -type f \( -name 'chkpnt*.pth' -o -name 'checkpoint*.pth' \) -print0 |
  sort -z |
  while IFS= read -r -d '' ckpt; do
    base_name="$(metadata_for_checkpoint "$ckpt")"
    final_video="$OUT_ROOT/${base_name}.mp4"
    render_dir="$OUT_ROOT/.render_tmp/${base_name}"
    log_file="$OUT_ROOT/${base_name}.render.log"

    if [[ -f "$final_video" && "$OVERWRITE" != "1" ]]; then
      echo "Skip existing: $final_video"
      continue
    fi

    rm -rf "$render_dir"
    mkdir -p "$render_dir"

    cmd=(
      python3 "$RENDER_SCRIPT"
      --repo_root "$REPO_ROOT"
      --model_file "$ckpt"
      --out_dir "$render_dir"
      --frames "$FRAMES"
      --fps "$FPS"
      --width "$WIDTH"
      --height "$HEIGHT"
      --time_mode "$TIME_MODE"
      --split "$SPLIT"
      --video_codec "$VIDEO_CODEC"
      --video_quality "$VIDEO_QUALITY"
      --macro_block_size "$MACRO_BLOCK_SIZE"
      --writer_queue "$WRITER_QUEUE"
    )

    if [[ -n "$TEMPORAL_MASK_THRESHOLD" ]]; then
      cmd+=(--temporal_mask_threshold "$TEMPORAL_MASK_THRESHOLD")
    fi

    if [[ -n "$TEMPORAL_MASK_KEYFRAMES" ]]; then
      cmd+=(--temporal_mask_keyframes "$TEMPORAL_MASK_KEYFRAMES")
    fi

    if [[ -n "$TEMPORAL_MASK_WINDOW" ]]; then
      cmd+=(--temporal_mask_window "$TEMPORAL_MASK_WINDOW")
    fi

    if [[ -n "$FFMPEG_PARAMS" ]]; then
      # shellcheck disable=SC2206
      ffmpeg_array=( $FFMPEG_PARAMS )
      cmd+=(--ffmpeg_params "${ffmpeg_array[@]}")
    fi

    if [[ -n "$EXTRA_RENDER_ARGS" ]]; then
      # shellcheck disable=SC2206
      extra_array=( $EXTRA_RENDER_ARGS )
      cmd+=("${extra_array[@]}")
    fi

    echo "Render: $ckpt"
    echo "Video:  $final_video"
    echo "Log:    $log_file"

    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'Command:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
      continue
    fi

    printf 'Command:' > "$log_file"
    printf ' %q' "${cmd[@]}" >> "$log_file"
    printf '\n' >> "$log_file"

    "${cmd[@]}" 2>&1 | tee -a "$log_file"

    case "$TIME_MODE" in
      orbit-time|orbit-only|time-only)
        produced="$render_dir/orbit_time.mp4"
        ;;
      *)
        produced="$render_dir/bounded_novel.mp4"
        ;;
    esac

    if [[ ! -f "$produced" ]]; then
      echo "Renderer finished, but expected video not found: $produced" >&2
      exit 1
    fi

    if [[ "$BLACK_CHECK" == "1" ]]; then
      black_check_status=0

      if video_is_black "$produced" 2>&1 | tee -a "$log_file"; then
        black_check_status=1
      else
        black_check_status=0
      fi

      if [[ "$black_check_status" == "1" ]]; then
        echo "BLACK VIDEO DETECTED: $produced" >&2
        echo "Checkpoint: $ckpt" >&2
        echo "Final name would have been: $final_video" >&2

        {
          echo "BLACK VIDEO DETECTED"
          echo "Checkpoint: $ckpt"
          echo "Produced: $produced"
          echo "Final name would have been: $final_video"
        } >> "$log_file"

        if [[ "$KEEP_BLACK_TMP" == "1" ]]; then
          bad_dir="$OUT_ROOT/.black_tmp/${base_name}"
          rm -rf "$bad_dir"
          mkdir -p "$(dirname "$bad_dir")"
          mv "$render_dir" "$bad_dir"
          echo "Kept failed render at: $bad_dir" >&2
          echo "Kept failed render at: $bad_dir" >> "$log_file"
        fi

        continue
      fi
    fi

    mv -f "$produced" "$final_video"

    if [[ -f "$render_dir/render_info.txt" ]]; then
      mv -f "$render_dir/render_info.txt" "$OUT_ROOT/${base_name}.render_info.txt"
    fi

    rm -rf "$render_dir"
  done