import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from gaussian_renderer import render

"""Dynamic Uncertainty Estimation (Section 4.1 of USplat4D paper).

For Gaussian i at frame t, the scalar uncertainty is:
    u_{i,t} = (sum_h (T_i^h * alpha_i)^2)^{-1}   if all covered pixels converge
    u_{i,t} = phi                                otherwise

The renderer's return_gaussian_scores=True path is intentionally left as the
linear sum_h T_i^h alpha_i score used by spatio-temporal pruning. USplat calls
return_gaussian_scores_sq=True and supplies a per-pixel RGB-L1 error map so the
CUDA rasterizer can accumulate both Eq. (3) and the exact contributed-pixel
maximum for Eq. (4).

Depth-aware anisotropic 3D uncertainty (Eq. 8):
    U_{i,t} = R_wc * diag(rx*u, ry*u, rz*u) * R_wc^T
with default [rx, ry, rz] = [1, 1, 100].
"""


def _current_covariance_and_mean_offset(gaussians, timestamp):
    if getattr(gaussians, "gaussian_dim", 4) == 4 and getattr(gaussians, "rot_4d", False):
        return gaussians.get_current_covariance_and_mean_offset(1.0, float(timestamp))
    cov = gaussians.get_covariance(1.0)
    delta = torch.zeros_like(gaussians.get_xyz)
    return cov, delta


def _disk_max_error(
    err_map: Tensor,
    px: Tensor,
    py: Tensor,
    radii: Tensor,
    W: int,
    H: int,
    chunk_size: int = 4096,
) -> Tensor:
    """For each disk footprint, return the maximum pixel error inside it."""
    if px.numel() == 0:
        return err_map.new_empty((0,))

    max_r = int(radii.max().item())
    offs = torch.arange(-max_r, max_r + 1, device=err_map.device)
    dy, dx = torch.meshgrid(offs, offs, indexing="ij")
    dx = dx.reshape(1, -1)
    dy = dy.reshape(1, -1)
    dist2 = dx.square() + dy.square()

    err_flat = err_map.contiguous().view(-1)
    out = err_map.new_empty((px.numel(),))
    neg_inf = torch.tensor(float("-inf"), dtype=err_map.dtype, device=err_map.device)

    for s in range(0, px.numel(), chunk_size):
        e = min(s + chunk_size, px.numel())
        px_c = px[s:e].unsqueeze(1)
        py_c = py[s:e].unsqueeze(1)
        r2 = radii[s:e].unsqueeze(1).square()

        xx = px_c + dx
        yy = py_c + dy
        valid = (dist2 <= r2) & (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        lin = (yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)).long()
        vals = err_flat[lin]
        vals = torch.where(valid, vals, neg_inf)
        out[s:e] = vals.max(dim=1).values

    return out


@torch.no_grad()
def compute_uncertainty_single_frame(
    means_t: Tensor,
    opacities_raw: Tensor,
    w2c: Tensor,
    K: Tensor,
    img_wh: Tuple[int, int],
    gt_img: Tensor,
    full_rendered_rgb: Tensor,
    radii: Tensor,
    depth: Tensor,
    blend_weight_sum_sq: Optional[Tensor] = None,
    contrib_max_error: Optional[Tensor] = None,
    eta_c: float = 0.5,
    phi: float = 1e6,
    depth_margin_rel: float = 0.05,
    chunk_size: int = 4096,
) -> Tuple[Tensor, Tensor]:
    device = means_t.device
    W, H = img_wh
    G = means_t.shape[0]
    alpha = opacities_raw.sigmoid().flatten()

    R = w2c[:3, :3]
    t = w2c[:3, 3]
    means_cam = means_t @ R.T + t
    g_depths = means_cam[:, 2]
    z = g_depths.clamp_min(1e-6)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    g_means2d = torch.stack((fx * means_cam[:, 0] / z + cx, fy * means_cam[:, 1] / z + cy), dim=-1)

    g_radii = radii.reshape(-1)
    u = torch.full((G,), phi, dtype=alpha.dtype, device=device)
    visible_mask = g_radii > 0
    vis_idx = visible_mask.nonzero(as_tuple=True)[0]
    if vis_idx.numel() == 0:
        return u, g_radii

    if blend_weight_sum_sq is None or blend_weight_sum_sq.numel() == 0:
        raise RuntimeError(
            "USplat uncertainty requires squared renderer scores. "
            "Call render(..., return_gaussian_scores_sq=True) after rebuilding the CUDA extension."
        )
    if blend_weight_sum_sq.shape[0] != G:
        raise RuntimeError(
            f"Expected squared gaussian_scores with shape ({G},), got {tuple(blend_weight_sum_sq.shape)}."
        )
    if contrib_max_error is None or contrib_max_error.numel() == 0:
        raise RuntimeError(
            "USplat convergence requires gaussian_score_max_error from the CUDA renderer. "
            "Call render(..., return_gaussian_scores_sq=True, gaussian_score_error_map=err_map)."
        )
    if contrib_max_error.shape[0] != G:
        raise RuntimeError(
            f"Expected gaussian_score_max_error with shape ({G},), got {tuple(contrib_max_error.shape)}."
        )

    score_vis = blend_weight_sum_sq.to(device=device, dtype=alpha.dtype)[vis_idx]
    max_err_vis = contrib_max_error.to(device=device, dtype=alpha.dtype)[vis_idx]

    supported = score_vis > 0
    converged = max_err_vis < eta_c
    u_vis = torch.where(
        supported & converged,
        1.0 / score_vis.clamp_min(1e-8),
        torch.full_like(score_vis, phi),
    )

    u[vis_idx] = u_vis
    return u, g_radii


@torch.no_grad()
def compute_uncertainty_all_frames(
    gaussians,
    train_cameras,
    pipe,
    background,
    device,
    eta_c=0.5,
    phi=1e6,
):
    G = gaussians.get_xyz.shape[0]
    T = len(train_cameras)
    u_all = torch.full((G, T), fill_value=phi, device=device)

    for t, (gt_img_raw, viewpoint_cam) in enumerate(train_cameras):
        # First render produces the RGB image used by the convergence test. The
        # second render receives the resulting error map and accumulates scores
        # only over the exact pixels that each Gaussian actually contributes to.
        render_rgb_pkg = render(viewpoint_cam, gaussians, pipe, background)
        gt_img = gt_img_raw.to(device)
        err_map = (render_rgb_pkg["render"].permute(1, 2, 0) - gt_img.permute(1, 2, 0)).abs().sum(dim=-1).contiguous()
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            return_gaussian_scores_sq=True,
            gaussian_score_error_map=err_map,
        )
        _, delta_mean = _current_covariance_and_mean_offset(gaussians, viewpoint_cam.timestamp)
        means_t = gaussians.get_xyz + delta_mean
        K = torch.tensor(
            [
                [viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx * 0.5)), 0, viewpoint_cam.image_width / 2],
                [0, viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy * 0.5)), viewpoint_cam.image_height / 2],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        u_t, _ = compute_uncertainty_single_frame(
            means_t=means_t,
            opacities_raw=gaussians._opacity,
            w2c=viewpoint_cam.world_view_transform,
            K=K,
            img_wh=(viewpoint_cam.image_width, viewpoint_cam.image_height),
            gt_img=gt_img.permute(1, 2, 0),
            full_rendered_rgb=render_rgb_pkg["render"].permute(1, 2, 0),
            radii=render_pkg["radii"],
            depth=render_pkg["depth"].squeeze(),
            blend_weight_sum_sq=render_pkg["gaussian_scores"],
            contrib_max_error=render_pkg["gaussian_score_max_error"],
            eta_c=eta_c,
            phi=phi,
        )
        u_all[:, t] = u_t

    return u_all


def build_uncertainty_3d_matrices(
    u_scalar: Tensor,
    w2cs: Tensor,
    r_scale: Tuple[float, float, float] = (1.0, 1.0, 100.0),
) -> Tensor:
    """Build anisotropic 3D uncertainty matrices (Eq. 8 in paper)."""
    G, T = u_scalar.shape
    device = u_scalar.device
    rx, ry, rz = r_scale
    R_wc = w2cs[:, :3, :3].transpose(-1, -2)
    r = torch.tensor([rx, ry, rz], device=device, dtype=torch.float32)
    diag_vals = r[None, None, :] * u_scalar[:, :, None]
    R = R_wc[None, :, :, :]
    D = torch.diag_embed(diag_vals)
    return R @ D @ R.transpose(-1, -2)


def mahalanobis_sq(
    delta: Tensor,
    u_scalar: Tensor,
    R_wc: Tensor,
    r_scale: Tuple[float, float, float] = (1.0, 1.0, 100.0),
) -> Tensor:
    """Compute Mahalanobis squared distance ||delta||^2_{U^{-1}}.

    USplat is a regularizer, not the primary rendering objective.  Treat
    singular uncertainty/pose values as low-confidence residuals rather than
    returning NaN and stopping all training.
    """
    rx, ry, rz = r_scale
    device = delta.device
    delta = torch.nan_to_num(delta, nan=0.0, posinf=1e6, neginf=-1e6)
    R_wc = torch.nan_to_num(R_wc, nan=0.0, posinf=0.0, neginf=0.0)
    u_scalar = torch.nan_to_num(u_scalar, nan=1e6, posinf=1e6, neginf=1e6).clamp_min(1e-8)
    delta_cam = (R_wc.transpose(-1, -2) @ delta.unsqueeze(-1)).squeeze(-1)
    r = torch.tensor([rx, ry, rz], device=device, dtype=delta.dtype)
    inv_ru = 1.0 / (r * u_scalar.unsqueeze(-1)).clamp_min(1e-8)
    out = (delta_cam ** 2 * inv_ru).sum(dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0).clamp_min(0.0)
