# splat4d-webgpu-rendercontract

Production-grade WebGPU/Rust/WASM 4D Gaussian Splatting renderer package with **locked render contracts**.

Format compatibility fix: Rust now accepts legacy `sort-free-mobile-gs` but serializes and expects the pack/exporter canonical string `sort-free-mobilegs`.

The renderer no longer treats `sorted-alpha` and `sort-free` as quality presets. They are different checkpoint families:

- A `sorted-alpha` checkpoint is rendered only by the sorted alpha-compositing path.
- A `sort-free-mobilegs` checkpoint is rendered only by the MobileGS sort-free path, including its opacity/phi MLP.
- A `sort-free-weighted-oit` checkpoint is rendered only by its weighted-OIT contract.

There is no `?quality=sortfree` and no UI switch that changes render families. Device-specific reductions are not automatic and are not part of this renderer contract. Any future reduced-quality mode must be an explicit pack-supported variant, not a cross-family substitution.

## What the pack contains

`.splat4dpack` v3 stores all render-affecting checkpoint state:

- `META.render_policy.required_render_type`, `allowed_render_types`, `forbidden_render_types`, and `render_type_locked`.
- Canonical 4D Gaussian tensors: mean4, scale4, left/right 4D quaternions, opacity, flags, appearance offsets.
- Appearance model metadata: RGB, SH, 4DSH/Spherindrical, or auxiliary features.
- Temporal masks: `TMSK`, `KEYF`, `MASK` chunks.
- Sort-free opacity/phi MLP: `MLPM`, `MLPW` chunks.
- Compression metadata and optional preserved original payloads.
- Environment-map hooks and provenance.

## Pipeline

```text
raw checkpoint (.pt/.pth/.ckpt/.chkpnt)
  -> tools/export_checkpoint.py
  -> scene.splat4dpack
  -> WebGPU renderer locked to scene render contract
```

The browser intentionally does not unpickle raw PyTorch checkpoints.

## Export

```bash
python tools/export_checkpoint.py model.chkpnt -o scene.splat4dpack --preserve-original
```

The exporter resolves common 3DGS/4DGS layouts, temporal masks, compressed layouts, RGB/isotropic variants, and MobileGS sort-free MLP weights. If a checkpoint declares or contains a sort-free MobileGS renderer, missing MLP weights are a hard error.

## Build

```bash
cargo build --workspace
wasm-pack build crates/splat-wasm --target web --out-dir ../../web/pkg
python -m http.server -d web 8080
```

Open `http://localhost:8080` and load a `.splat4dpack`.

## Renderer contracts

- `sorted-alpha`: 4D conditioning -> temporal cull/mask -> tile binning -> per-tile depth sort -> alpha blending.
- `sort-free-mobilegs`: 4D conditioning -> temporal cull/mask -> opacity/phi MLP -> depth-aware weighted accumulation -> compositing.
- `sort-free-weighted-oit`: only for checkpoints explicitly exported for this contract.

Attempting to render a pack with a forbidden render type fails loudly. The renderer never silently substitutes a different family.

## Checkpoint exporter compatibility update

`tools/export_checkpoint.py` supports the repository checkpoint format:

```python
{
  "format": "4dgs-self-contained-v1",
  "iteration": int,
  "gaussians": GaussianModel.capture(include_mobilegs=...),
  "run_config": {"args": ..., "gaussian_kwargs": ...},
  "scene": ...,
  "requires_mobilegs": bool,
}
```

For this format, Gaussian tensors are read from `checkpoint["gaussians"]`, not from top-level `_xyz` keys.  The exporter applies the correct 4DGS activations from `GaussianModel.capture()`:

- `_scaling` and `_scaling_t` are exponentiated.
- `_opacity` is passed through sigmoid.
- left/right quaternions are normalized.
- `run_config.args.sort_free_render` and `requires_mobilegs` lock the render contract.
- `run_config` and `scene` metadata are preserved in the pack metadata.

Example:

```bash
uv run tools/export_checkpoint.py ../output/dnerf/bouncingballs_simple_prune/chkpnt_best.pth \
  -o bouncingballs.splat4dpack \
  --preserve-original \
  --name bouncingballs
```

## CUDA-parity WebGPU renderer

This package contains a CUDA-forward-pipeline WebGPU renderer. See `docs/CUDA_PARITY_WEBGPU.md`.
