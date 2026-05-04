# D-NeRF ablation bases

This folder mirrors `configs/dnerf/` but resets every scene to a clean native-4DGS baseline for the ablation runner. The runner writes generated configs and outputs under `output/dnerf_ablation/<scene>/ablations/...` unless another output root is provided.

Default matrix:

```bash
python batch_train.py configs/dnerf_ablation/*.yaml --dry-run
python batch_train.py configs/dnerf_ablation/*.yaml --submit-slurm
```

The default `--matrix-preset paper` is curated rather than Cartesian. It includes native 4DGS, 4DGS-1K-style spatio-temporal pruning, DropoutGS RDR, USplat4D graph regularization, Instant4D-lite, Mobile-GS sort-free rendering, and compatible hybrid rows. Use `--matrix-preset compact` for a smaller sweep, `--matrix-preset full` for extra controls/hybrids, or `--matrix-preset cartesian --axes ...` for a custom product.

## V2 matrix updates

- Mobile-GS compression is a post-training evaluation for every ablation row by default. Each run exports a quantized Mobile-GS payload, benchmarks the compressed model, and records `mobile_*` speed, size, and quality-vs-raw metrics in the per-run JSON and summary CSV.
- DropoutGS RDR and ESS are independent mechanisms in the Cartesian runner. Use the `dropout` axis for RDR and the `ess` axis for edge-guided splitting.

Full Cartesian product with ESS separated:

```bash
python batch_train.py configs/dnerf_ablation/*.yaml \
  --matrix-preset cartesian \
  --axes isotropy,appearance,sorting,pruning,usplat,dropout,ess \
  --isotropy-options anisotropic,isotropic \
  --appearance-options rgb,sh1,sh3,sh3_3d \
  --sorting-options sort,sort_free \
  --pruning-options no_pruning,densify_then_prune_once,interleaved_prune_densify \
  --usplat-options no_usplat,use_usplat \
  --dropout-options no_dropout,dropout \
  --ess-options no_ess,ess \
  --mobilegs-report --mobilegs-report-scope all \
  --mobilegs-benchmark-render-mode match \
  --dry-run
```

The raw product is 384 variants per scene. With the default compatibility filter, `sort_free x use_usplat` is skipped, leaving 288 trainable variants per scene. Mobile-GS compression/benchmarking then runs after each trainable variant without creating an additional training axis.

Resource-aware launch:

```bash
python scripts/probe_resources_and_plan.py configs/dnerf_ablation/*.yaml --matrix-preset paper
bash scripts/run_recommended_ablations.sh
```

Or use the wrapper:

```bash
MATRIX_PRESET=paper bash scripts/train_ablation_plan.sh
```
