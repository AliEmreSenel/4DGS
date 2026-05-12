#!/usr/bin/env python3
"""Export a 4DGS self-contained checkpoint to a Plotly HTML snippet.

Example:
  python export_4dgs_plotly_html.py chkpnt_best.pth --t 0.5 --out gaussians.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().float()


def identity_quat(n: int) -> torch.Tensor:
    q = torch.zeros((n, 4), dtype=torch.float32)
    q[:, 0] = 1.0
    return q


def normalize_quat(q: torch.Tensor, fallback_n: int | None = None) -> torch.Tensor:
    if q.numel() == 0:
        return identity_quat(int(fallback_n or 0))
    q = as_float_tensor(q)
    return q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)


def build_rotation(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = normalize_quat(q).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def build_rotation_4d(l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    a, b, c, d = normalize_quat(l).unbind(-1)
    p, q, rr, s = normalize_quat(r).unbind(-1)

    def mat(items):
        return torch.stack(items, dim=0).view(4, 4, -1).permute(2, 0, 1)

    ml = mat([a, -b, -c, -d, b, a, -d, c, c, d, a, -b, d, -c, b, a])
    mr = mat([p, q, rr, s, -q, p, -s, rr, -rr, s, p, -q, -s, -rr, q, p])
    return (ml @ mr).flip(1, 2)


def spatial_covariance(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    # Matches GaussianModel.get_covariance(): L = diag(s) @ R; cov = L.T @ L.
    L = torch.diag_embed(scales) @ build_rotation(rotations)
    return L.transpose(1, 2) @ L


def conditional_4d(scales_xyzt, rot_l, rot_r, xyz, center_t, t):
    # Matches GaussianModel.get_current_covariance_and_mean_offset().
    L = build_rotation_4d(rot_l, rot_r) @ torch.diag_embed(scales_xyzt)
    cov4 = L @ L.transpose(1, 2)
    cov_11, cov_12 = cov4[:, :3, :3], cov4[:, :3, 3:4]
    cov_t = cov4[:, 3:4, 3:4].clamp_min(1e-8)
    return (
        xyz + cov_12.squeeze(-1) / cov_t.squeeze(-1) * (float(t) - center_t),
        cov_11 - cov_12 @ cov_12.transpose(1, 2) / cov_t,
        cov_t.squeeze(-1),
    )


def temporal_marginal(center_t, cov_t, t: float, prefilter_var: float) -> torch.Tensor:
    sigma = cov_t.clone() + (float(prefilter_var) if prefilter_var > 0.0 else 0.0)
    sigma = torch.nan_to_num(sigma, nan=1e-8, posinf=1e8, neginf=1e-8).clamp_min(1e-8)
    exponent = -0.5 * (center_t - float(t)) ** 2 / sigma
    return torch.exp(torch.nan_to_num(exponent, nan=-1e8, posinf=0.0, neginf=-1e8))


def sh_basis(dirs: torch.Tensor) -> list[torch.Tensor]:
    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    return [
        torch.ones_like(x) * C0,
        -C1 * y,
        C1 * z,
        -C1 * x,
        C2[0] * xy,
        C2[1] * yz,
        C2[2] * (2.0 * zz - xx - yy),
        C2[3] * xz,
        C2[4] * (xx - yy),
        C3[0] * y * (3 * xx - yy),
        C3[1] * xy * z,
        C3[2] * y * (4 * zz - xx - yy),
        C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
        C3[4] * x * (4 * zz - xx - yy),
        C3[5] * z * (xx - yy),
        C3[6] * x * (xx - 3 * yy),
    ]


def eval_sh(deg: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    return sum(b * sh[..., k] for k, b in enumerate(sh_basis(dirs)[: min(max((deg + 1) ** 2, 1), 16)]))


def eval_shfs_4d(deg: int, deg_t: int, sh: torch.Tensor, dirs: torch.Tensor, dirs_t: torch.Tensor, duration: float) -> torch.Tensor:
    # Exact layout used by utils/sh_utils.py for the common 4DGS SH3+time basis.
    if deg < 3 or deg_t <= 0 or sh.shape[-1] < 33:
        return eval_sh(min(deg, 3), sh, dirs)

    b = sh_basis(dirs)
    out = sum(b[k] * sh[..., k] for k in range(16))
    out += sum(torch.cos(2 * torch.pi * dirs_t / float(duration)) * b[k] * sh[..., 16 + k] for k in range(16))
    if deg_t > 1 and sh.shape[-1] >= 48:
        out += sum(torch.cos(4 * torch.pi * dirs_t / float(duration)) * b[k] * sh[..., 32 + k] for k in range(16))
    return out


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("format") != "4dgs-self-contained-v1" or "gaussians" not in ckpt:
        raise ValueError(f"{path} is not a 4dgs-self-contained-v1 checkpoint")
    return ckpt


def unpack_checkpoint(ckpt: dict, t: float, camera_center: np.ndarray):
    g = ckpt["gaussians"]
    if len(g) != 20:
        raise ValueError(f"Unsupported Gaussian tuple length {len(g)}; expected 20")

    active_sh_degree, active_sh_degree_t = int(g[0]), int(g[18])
    xyz, features_dc, features_rest, scaling, rotation = map(as_float_tensor, g[1:6])
    opacity_raw = as_float_tensor(g[6]).reshape(-1)
    center_t, scaling_t = as_float_tensor(g[13]).reshape(-1, 1), as_float_tensor(g[14]).reshape(-1, 1)
    rotation_r, rot_4d = as_float_tensor(g[15]), bool(g[16])

    kwargs = dict(ckpt.get("run_config", {}).get("gaussian_kwargs", {}))
    duration = float(kwargs.get("time_duration", [-0.5, 0.5])[1]) - float(kwargs.get("time_duration", [-0.5, 0.5])[0])

    n, scales, scales_t = xyz.shape[0], torch.exp(scaling), torch.exp(scaling_t)
    if bool(kwargs.get("isotropic_gaussians", False)) and scales.shape[1] == 1:
        scales, rotation, rotation_r = scales.repeat(1, 3), identity_quat(n), identity_quat(n)
    else:
        rotation, rotation_r = normalize_quat(rotation, n), normalize_quat(rotation_r, n)

    if rot_4d:
        xyz_t, cov3, cov_t = conditional_4d(torch.cat([scales, scales_t], dim=1), rotation, rotation_r, xyz, center_t, t)
    else:
        xyz_t, cov3, cov_t = xyz, spatial_covariance(scales, rotation), scales_t

    alpha = torch.sigmoid(opacity_raw).reshape(-1, 1) * temporal_marginal(center_t, cov_t, t, float(kwargs.get("prefilter_var", -1.0)))
    features = torch.cat([features_dc, features_rest], dim=1).transpose(1, 2).contiguous()
    dirs = xyz_t - torch.tensor(camera_center, dtype=torch.float32).reshape(1, 3)
    dirs /= dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)

    rgb = (
        eval_sh(active_sh_degree, features, dirs)
        if bool(kwargs.get("force_sh_3d", False))
        else eval_shfs_4d(active_sh_degree, active_sh_degree_t, features, dirs, center_t - float(t), duration)
    )
    return xyz_t.numpy(), cov3.numpy(), torch.clamp(rgb + 0.5, 0.0, 1.0).numpy(), alpha.reshape(-1).numpy(), kwargs, int(ckpt.get("iteration", -1))


def sphere_mesh(res: int):
    u, v = np.linspace(0.0, 2.0 * np.pi, 2 * res + 1), np.linspace(0.0, np.pi, res + 1)
    uu, vv = np.meshgrid(u, v)
    verts = np.stack([np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)], axis=-1).reshape(-1, 3)
    cols = 2 * res + 1
    faces = [
        (a, a + cols, a + 1) if f == 0 else (a + 1, a + cols, a + cols + 1)
        for i in range(res)
        for j in range(2 * res)
        for f, a in enumerate([i * cols + j] * 2)
    ]
    return verts.astype(np.float32), np.asarray(faces, dtype=np.int64)


def rgba(rgb, alpha):
    r, g, b = np.clip(np.round(np.asarray(rgb) * 255.0), 0, 255).astype(int)
    return f"rgba({r},{g},{b},{float(np.clip(alpha, 0.0, 1.0)):.4f})"


def color_opacity_multiplier(
    rgb: np.ndarray,
    knee: float = 0.08,
    power: float = 2.0,
    white_start: float = 0.82,
    min_multiplier: float = 0.28,
) -> np.ndarray:
    """Return opacity multiplier used only for filtering.

    Gray/black Gaussians receive a smaller selection weight.
    Saturated or near-white Gaussians receive a larger selection weight.
    This does not control final rendered opacity.
    """
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    signal = np.maximum(
        rgb.max(axis=1) - rgb.min(axis=1),
        np.clip((rgb.min(axis=1) - float(white_start)) / max(1e-6, 1.0 - float(white_start)), 0.0, 1.0),
    )
    ramp = 1.0 - np.exp(-np.power(signal / max(float(knee), 1e-6), float(power)))
    return np.clip(float(min_multiplier) + (1.0 - float(min_multiplier)) * ramp, 0.0, 1.0)


def make_ellipsoid_mesh(xyz, cov, rgb, alpha, sigma: float, res: int):
    unit, unit_faces = sphere_mesh(res)
    xs, ys, zs, ii, jj, kk, facecolors, offset = [], [], [], [], [], [], [], 0

    for c, C, color, a in zip(xyz, cov, rgb, alpha):
        vals, vecs = np.linalg.eigh(0.5 * (C + C.T))
        verts = unit @ (vecs @ np.diag(float(sigma) * np.sqrt(np.clip(vals, 1e-12, None)))).T + c[None, :]
        faces = unit_faces + offset
        for out, values in zip((xs, ys, zs), verts.T):
            out.extend(values.tolist())
        for out, values in zip((ii, jj, kk), faces.T):
            out.extend(values.tolist())
        facecolors.extend([rgba(color, a)] * len(unit_faces))
        offset += len(verts)

    return xs, ys, zs, ii, jj, kk, facecolors


def choose_points(alpha: np.ndarray, opacity_min: float, max_gaussians: int) -> np.ndarray:
    idx = np.flatnonzero(alpha >= float(opacity_min))
    if idx.size == 0:
        idx = np.arange(alpha.size)
    return idx[np.argsort(alpha[idx])[-max_gaussians:]] if max_gaussians > 0 and idx.size > max_gaussians else idx


def build_gaussian_mesh_trace(checkpoint_path: Path, args: argparse.Namespace, label: str, visible: bool = True) -> tuple[go.Mesh3d, str, int, int, int]:
    xyz, cov, rgb, alpha, kwargs, iteration = unpack_checkpoint(load_checkpoint(checkpoint_path), args.t, np.asarray(args.camera_center))
    total_gaussians = len(alpha)
    selection_alpha = np.clip(
        alpha * color_opacity_multiplier(rgb, args.gray_opacity_knee, args.gray_opacity_power, args.white_opacity_start, args.gray_min_opacity),
        0.0,
        1.0,
    )
    idx = choose_points(selection_alpha, args.opacity_min, args.max_gaussians)
    visual_alpha = np.full(len(idx), float(np.clip(args.render_opacity, 0.0, 1.0)), dtype=np.float32)
    mx, my, mz, mi, mj, mk, facecolors = make_ellipsoid_mesh(
        xyz[idx],
        cov[idx],
        rgb[idx],
        visual_alpha,
        sigma=args.sigma,
        res=max(3, int(args.ellipsoid_res)),
    )

    trace = go.Mesh3d(
        x=mx,
        y=my,
        z=mz,
        i=mi,
        j=mj,
        k=mk,
        facecolor=facecolors,
        flatshading=True,
        opacity=1.0,
        name=label,
        hoverinfo="skip",
        showscale=False,
        visible=visible,
    )
    title = f"4DGS gaussians: {label} | t={args.t} | iter={iteration} | shown={len(idx)}/{total_gaussians}"
    return trace, title, len(idx), total_gaussians, iteration


def write_plotly_data_file(fig: go.Figure, out_path: Path, plot_div_id: str, config: dict) -> tuple[str, str, Path]:
    data_dir = out_path.parent / "plotly_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    data_key = f"{out_path.stem}:{plot_div_id}"
    data_filename = f"{out_path.stem}.js"
    data_path = data_dir / data_filename

    fig_json = fig.to_plotly_json()
    payload = {
        "data": fig_json["data"],
        "layout": fig_json["layout"],
        "config": config,
    }

    data_path.write_text(
        "window.__GAUSSIANS_PLOTLY_PAYLOADS__ = window.__GAUSSIANS_PLOTLY_PAYLOADS__ || {};\n"
        f"window.__GAUSSIANS_PLOTLY_PAYLOADS__[{json.dumps(data_key)}] = "
        f"{json.dumps(payload, cls=PlotlyJSONEncoder)};\n",
        encoding="utf-8",
    )

    return f"plotly_data/{data_filename}", data_key, data_path


def plotly_loader_html(plot_div_id: str, data_url: str, data_key: str, include_plotlyjs: str) -> str:
    plotly_script = ""
    if include_plotlyjs == "cdn":
        plotly_script = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n'

    return f"""
{plotly_script}<div id="{plot_div_id}" style="width:100%; height:100vh;"></div>

<script src="{data_url}"></script>
<script>
(function () {{
  const plotId = {json.dumps(plot_div_id)};
  const dataKey = {json.dumps(data_key)};
  let attempts = 0;

  function init() {{
    const plot = document.getElementById(plotId);
    const payloads = window.__GAUSSIANS_PLOTLY_PAYLOADS__ || {{}};
    const payload = payloads[dataKey];

    if (!plot || !window.Plotly || !payload) {{
      if (++attempts < 400) window.setTimeout(init, 50);
      return;
    }}

    Plotly.newPlot(plot, payload.data, payload.layout, payload.config || {{}});
  }}

  init();
}})();
</script>
"""


def mobile_toggle_controls(plot_div_id: str, dataset_labels: tuple[str, str] | None = None, dataset_titles: tuple[str, str] | None = None) -> str:
    has_dataset_toggle = dataset_labels is not None and dataset_titles is not None
    dataset_button_html = ""
    if has_dataset_toggle:
        dataset_button_html = (
            '\n  <button id="dataset-toggle" class="gaussians-control-button" type="button" aria-pressed="false">\n'
            f'    Dataset: {dataset_labels[0]}\n'
            '  </button>'
        )

    return f"""
<style>
  #{plot_div_id} {{ touch-action: none; }}
  .gaussians-controls {{
    position: fixed; left: 50%; bottom: 18px; transform: translateX(-50%); z-index: 9999;
    display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; pointer-events: none;
  }}
  .gaussians-control-button {{
    pointer-events: auto; min-width: 190px; min-height: 58px; padding: 14px 26px; border: 0;
    border-radius: 999px; background: rgba(255, 255, 255, 0.94); color: #111; font-size: 18px;
    font-weight: 800; line-height: 1; box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
    touch-action: manipulation; user-select: none; -webkit-user-select: none;
  }}
  .gaussians-control-button:active {{ transform: scale(0.97); }}
</style>

<div class="gaussians-controls">
  <button id="dragmode-toggle" class="gaussians-control-button" type="button" aria-pressed="false">
    Mode: Rotate
  </button>{dataset_button_html}
</div>

<script>
(function () {{
  const plotId = "{plot_div_id}";
  const dragButton = document.getElementById("dragmode-toggle");
  const datasetButton = document.getElementById("dataset-toggle");
  const hasDatasetToggle = {json.dumps(has_dataset_toggle)};
  const datasetLabels = {json.dumps(list(dataset_labels or ("", "")))};
  const datasetTitles = {json.dumps(list(dataset_titles or ("", "")))};
  let mode = "orbit", dataset = 0, attempts = 0;

  function setDragButton() {{
    dragButton.textContent = "Mode: " + (mode === "orbit" ? "Rotate" : "Pan");
    dragButton.setAttribute("aria-pressed", mode === "orbit" ? "false" : "true");
  }}

  function setDatasetButton() {{
    if (!datasetButton) return;
    datasetButton.textContent = "Dataset: " + datasetLabels[dataset];
    datasetButton.setAttribute("aria-pressed", dataset === 1 ? "true" : "false");
  }}

  function setMode(nextMode) {{
    const plot = document.getElementById(plotId);
    if (!plot || !window.Plotly || !plot._fullLayout) return;
    mode = nextMode;
    Plotly.relayout(plot, {{"scene.dragmode": mode}});
    setDragButton();
  }}

  function setDataset(nextDataset) {{
    const plot = document.getElementById(plotId);
    if (!plot || !window.Plotly || !plot._fullLayout || !hasDatasetToggle) return;
    dataset = nextDataset;
    Plotly.restyle(plot, {{visible: [dataset === 0, dataset === 1]}}, [0, 1]);
    Plotly.relayout(plot, {{"title.text": datasetTitles[dataset]}});
    setDatasetButton();
  }}

  function init() {{
    const plot = document.getElementById(plotId);
    if (!dragButton || !plot || !window.Plotly || !plot._fullLayout) {{
      if (++attempts < 400) window.setTimeout(init, 50);
      return;
    }}

    dragButton.addEventListener("click", function () {{
      setMode(mode === "orbit" ? "pan" : "orbit");
    }});

    if (hasDatasetToggle && datasetButton) {{
      datasetButton.addEventListener("click", function () {{
        setDataset(dataset === 0 ? 1 : 0);
      }});
      setDataset(0);
    }}

    setMode("orbit");
  }}

  init();
}})();
</script>
"""


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add = ap.add_argument
    add("checkpoint", type=Path)
    add("--toggle-checkpoint", type=Path, default=None, help="optional second checkpoint shown by the dataset toggle button")
    add("--checkpoint-labels", nargs=2, metavar=("FIRST", "SECOND"), default=None, help="labels used by the dataset toggle button")
    add("--t", type=float, required=True, help="timestamp to visualize")
    add("--out", type=Path, default=Path("gaussians_snippet.html"))
    add("--max-gaussians", type=int, default=1000, help="0 means export all visible gaussians")
    add("--opacity-min", type=float, default=0.0)
    add("--sigma", type=float, default=1.0, help="ellipsoid radius in standard deviations")
    add("--ellipsoid-res", type=int, default=6)
    add("--point-size", type=float, default=1.5, help=argparse.SUPPRESS)
    add("--gray-opacity-knee", type=float, default=0.08, help="smaller = gray-to-opaque transition happens faster during filtering")
    add("--gray-opacity-power", type=float, default=2.0, help="larger = sharper color/white opacity transition during filtering")
    add("--white-opacity-start", type=float, default=0.82, help="RGB min-channel value where near-white Gaussians start being favored during filtering")
    add("--gray-min-opacity", type=float, default=0.015, help="minimum opacity multiplier used only for selecting gray/low-chroma leftovers")
    add("--render-opacity", type=float, default=1.0, help="final rendered opacity for every selected Gaussian")
    add("--leftover-opacity", type=float, default=None, help=argparse.SUPPRESS)  # Backward-compatible old flag name.
    add("--camera-center", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="view origin used for SH color evaluation")
    add("--include-plotlyjs", choices=("false", "cdn"), default="false", help="use 'false' for an embeddable snippet, 'cdn' for an easier standalone file")
    return ap


def main():
    args = parser().parse_args()
    if args.leftover_opacity is not None:
        args.render_opacity = args.leftover_opacity

    labels = (
        tuple(args.checkpoint_labels)
        if args.checkpoint_labels
        else (args.checkpoint.stem, args.toggle_checkpoint.stem if args.toggle_checkpoint else "")
    )

    fig = go.Figure()

    trace_a, title_a, shown_a, total_a, _ = build_gaussian_mesh_trace(args.checkpoint, args, labels[0], visible=True)
    fig.add_trace(trace_a)

    stats, dataset_titles = [f"{labels[0]}: {shown_a}/{total_a} gaussians"], None

    if args.toggle_checkpoint is not None:
        trace_b, title_b, shown_b, total_b, _ = build_gaussian_mesh_trace(args.toggle_checkpoint, args, labels[1], visible=False)
        fig.add_trace(trace_b)
        stats.append(f"{labels[1]}: {shown_b}/{total_b} gaussians")
        dataset_titles = (title_a, title_b)

    axis = dict(backgroundcolor="black", gridcolor="#333", zerolinecolor="#555", color="white")
    fig.update_layout(
        title=title_a,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        scene=dict(
            dragmode="orbit",
            aspectmode="data",
            bgcolor="black",
            xaxis=axis,
            yaxis=axis,
            zaxis=axis,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    plot_div_id = "gaussians-plot"
    config = {
        "displayModeBar": False,
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)

    data_url, data_key, data_path = write_plotly_data_file(fig, args.out, plot_div_id, config)

    html = plotly_loader_html(
        plot_div_id=plot_div_id,
        data_url=data_url,
        data_key=data_key,
        include_plotlyjs=args.include_plotlyjs,
    )

    html += mobile_toggle_controls(
        plot_div_id,
        dataset_labels=labels if args.toggle_checkpoint is not None else None,
        dataset_titles=dataset_titles,
    )

    args.out.write_text(html, encoding="utf-8")

    print(
        f"wrote {args.out} and {data_path} "
        f"({', '.join(stats)}, t={args.t}, render_opacity={args.render_opacity})"
    )


if __name__ == "__main__":
    main()