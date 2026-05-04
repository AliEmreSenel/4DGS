# D-NeRF ablation bases

This folder mirrors `configs/dnerf/` but resets every scene to a clean native-4DGS baseline for the ablation runner. The runner writes generated configs and outputs under `output/dnerf_ablation/<scene>/ablations/...` unless another output root is provided.

Default matrix:

```bash
python batch_train.py configs/dnerf_ablation/*.yaml --dry-run
python batch_train.py configs/dnerf_ablation/*.yaml --submit-slurm
```

The default `--matrix-preset paper` is curated rather than Cartesian. It includes native 4DGS, 4DGS-1K-style spatio-temporal pruning, DropoutGS RDR, USplat4D graph regularization, Instant4D-lite, Mobile-GS sort-free rendering, and compatible hybrid rows. Use `--matrix-preset compact` for a smaller sweep, `--matrix-preset full` for extra controls/hybrids, or `--matrix-preset cartesian --axes ...` for a custom product.
