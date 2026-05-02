import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from gaussian_renderer import render

"""Dynamic Uncertainty Estimation (Section 4.1 of USplat4D paper).

For Gaussian i at frame t, the scalar uncertainty is:
    u_{i,t} = (sum_h (T_i^h * alpha_i)^2)^{-1}   if pixel residuals < eta_c
    u_{i,t} = phi                                   otherwise

We approximate sum_h (T_i^h * alpha_i)^2 using:
    - alpha_i^2 * pi * r_i^2  for visible (non-occluded) Gaussians
    - 0  (=> phi)             for occluded or out-of-frustum Gaussians

Occlusion is detected by comparing the Gaussian's camera-space depth against
the rendered depth map at the projected 2D pixel. Convergence is checked by
comparing rendered color vs. ground-truth.

Depth-aware anisotropic 3D uncertainty (Eq. 8):
    U_{i,t} = R_wc * diag(rx*u, ry*u, rz*u) * R_wc^T
with default [rx, ry, rz] = [1, 1, 0.01]  (depth axis down-weighted so that
depth deviations are penalized 100x more in the Mahalanobis loss).
"""

def _disk_max_error(
    err_map: Tensor,   # (H, W)
    px: Tensor,        # (N,)
    py: Tensor,        # (N,)
    radii: Tensor,     # (N,) integer radii
    W: int,
    H: int,
    chunk_size: int = 4096,
) -> Tensor:
    """
    For each center (px[i], py[i]) and radius radii[i], return:
        max(err_map[y, x]) over all valid pixels inside the disk.
    This is vectorized over each chunk of Gaussians.
    """
    if px.numel() == 0:
        return err_map.new_empty((0,))

    max_r = int(radii.max().item())
    offs = torch.arange(-max_r, max_r + 1, device=err_map.device)

    dy, dx = torch.meshgrid(offs, offs, indexing="ij")  # (Kside, Kside)
    dx = dx.reshape(1, -1)                              # (1, K)
    dy = dy.reshape(1, -1)                              # (1, K)
    dist2 = dx.square() + dy.square()                   # (1, K)

    err_flat = err_map.contiguous().view(-1)
    out = err_map.new_empty((px.numel(),))
    neg_inf = torch.tensor(float("-inf"), dtype=err_map.dtype, device=err_map.device)

    for s in range(0, px.numel(), chunk_size):
        e = min(s + chunk_size, px.numel())

        px_c = px[s:e].unsqueeze(1)         # (B, 1)
        py_c = py[s:e].unsqueeze(1)         # (B, 1)
        r2 = radii[s:e].unsqueeze(1).square()  # (B, 1)

        xx = px_c + dx                      # (B, K)
        yy = py_c + dy                      # (B, K)

        valid = (
            (dist2 <= r2) &
            (xx >= 0) & (xx < W) &
            (yy >= 0) & (yy < H)
        )

        # Clamp only for safe indexing; invalid entries are removed by masking.
        lin = (yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)).long()  # (B, K)
        vals = err_flat[lin]                                         # (B, K)
        vals = torch.where(valid, vals, neg_inf)

        out[s:e] = vals.max(dim=1).values

    return out


@torch.no_grad()
def compute_uncertainty_single_frame(
    means_t: Tensor,           # (G, 3)
    opacities_raw: Tensor,     # (G,)
    w2c: Tensor,               # (4, 4)
    K: Tensor,                 # (3, 3)
    img_wh: Tuple[int, int],   # (W, H)
    gt_img: Tensor,            # (H, W, 3)
    full_rendered_rgb: Tensor, # (H, W, 3)
    radii: Tensor,             # (G,)
    depth: Tensor,             # (H, W)
    eta_c: float = 0.5,
    phi: float = 1e6,
    depth_margin_rel: float = 0.05,
    chunk_size: int = 4096,
) -> Tuple[Tensor, Tensor]:

    device = means_t.device
    W, H = img_wh
    G = means_t.shape[0]

    alpha = opacities_raw.sigmoid().flatten()  # (G,)

    # Faster than homogeneous multiplication for rigid/affine camera transforms.
    # If your w2c is not affine, revert to the homogeneous version.
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    means_cam = means_t @ R.T + t              # (G, 3)

    g_depths = means_cam[:, 2]                 # (G,)
    z = g_depths.clamp_min(1e-6)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    g_means2d = torch.stack(
        (
            fx * means_cam[:, 0] / z + cx,
            fy * means_cam[:, 1] / z + cy,
        ),
        dim=-1,
    )  # (G, 2)

    g_radii = radii.reshape(-1)
    u = torch.full((G,), phi, dtype=alpha.dtype, device=device)

    visible_mask = g_radii > 0
    vis_idx = visible_mask.nonzero(as_tuple=True)[0]
    if vis_idx.numel() == 0:
        return u, g_radii

    vis_depths = g_depths[vis_idx]
    vis_alpha = alpha[vis_idx]
    vis_radii = g_radii[vis_idx]

    # Keep raw rounded centers for convergence logic.
    vis_px = g_means2d[vis_idx, 0].round().long()
    vis_py = g_means2d[vis_idx, 1].round().long()

    # Use clamped centers only for depth lookup.
    px_clamped = vis_px.clamp(0, W - 1)
    py_clamped = vis_py.clamp(0, H - 1)

    d_rendered = depth[py_clamped, px_clamped]
    margin = depth_margin_rel * d_rendered.clamp_min(1e-3)
    not_occluded = vis_depths <= d_rendered + margin

    area = torch.pi * vis_radii.clamp_min(0.5).square()
    u_vis = 1.0 / (vis_alpha.square() * area + 1e-8)

    # Precompute per-pixel color error once.
    err_map = (full_rendered_rgb - gt_img).abs().mean(dim=-1)  # (H, W)

    # Match original behavior: integer footprint radius.
    r_int = vis_radii.long()

    # Original code only checked convergence when the rounded center was in-bounds.
    center_in_bounds = (
        (vis_px >= 0) & (vis_px < W) &
        (vis_py >= 0) & (vis_py < H)
    )

    # No reason to test convergence for already-occluded Gaussians.
    to_check = center_in_bounds & not_occluded & (r_int > 0)

    not_converged = torch.zeros_like(to_check)
    if to_check.any():
        max_err = _disk_max_error(
            err_map=err_map,
            px=vis_px[to_check],
            py=vis_py[to_check],
            radii=r_int[to_check],
            W=W,
            H=H,
            chunk_size=chunk_size,
        )
        not_converged[to_check] = max_err > eta_c

    keep = not_occluded & ~not_converged
    u_vis = torch.where(keep, u_vis, torch.full_like(u_vis, phi))

    u[vis_idx] = u_vis
    return u, g_radii

@torch.no_grad()
def compute_uncertainty_all_frames(
    gaussians,           # your GaussianModel
    train_cameras,       # your list of training cameras
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
        # Get rendered outputs from YOUR renderer
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)

        # Get GT image
        gt_img = gt_img_raw.to(device)

        # Get positions at this timestamp
        _, delta_mean = gaussians.get_current_covariance_and_mean_offset(
            1.0, viewpoint_cam.timestamp
        )
        means_t = gaussians.get_xyz + delta_mean

        # Get K matrix from camera
        K = torch.tensor([
        [viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx * 0.5)), 0, viewpoint_cam.image_width / 2],
        [0, viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy * 0.5)), viewpoint_cam.image_height / 2],
        [0, 0, 1]
        ], dtype=torch.float32, device=device)
        u_t, _ = compute_uncertainty_single_frame(
            means_t=means_t,
            opacities_raw=gaussians._opacity,
            w2c=viewpoint_cam.world_view_transform,
            K=K,
            img_wh=(viewpoint_cam.image_width, viewpoint_cam.image_height),
            gt_img=gt_img.permute(1, 2, 0),
            full_rendered_rgb=render_pkg["render"].permute(1, 2, 0),
            radii=render_pkg["radii"],
            depth=render_pkg["depth"].squeeze(),
            eta_c=eta_c,
            phi=phi,
        )
        u_all[:, t] = u_t

    return u_all

def build_uncertainty_3d_matrices(
    u_scalar: Tensor,    # (G, T)  scalar uncertainties
    w2cs: Tensor,        # (T, 4, 4)  world-to-camera transforms
    r_scale: Tuple[float, float, float] = (1.0, 1.0, 0.01),
) -> Tensor:
    """Build anisotropic 3D uncertainty matrices (Eq. 8 in paper).

    U_{i,t} = R_wc * diag(rx*u, ry*u, rz*u) * R_wc^T

    where R_wc is the camera-to-world rotation at frame t,
    and [rx, ry, rz] = r_scale (default: depth axis down-weighted at 0.01).

    Returns:
        U: (G, T, 3, 3)  per-Gaussian per-frame uncertainty matrices
    """
    G, T = u_scalar.shape
    device = u_scalar.device
    rx, ry, rz = r_scale

    # Extract camera-to-world rotations: R_wc = w2c^{-1}[:3, :3] = w2c[:3,:3]^T
    # w2c[:3,:3] is R_cw, so R_wc = R_cw^T
    R_wc = w2cs[:, :3, :3].transpose(-1, -2)  # (T, 3, 3) = R cam-to-world

    # Build diagonal scale vectors: (3,) -> scaled by (rx, ry, rz)
    r = torch.tensor([rx, ry, rz], device=device, dtype=torch.float32)  # (3,)

    # U_{i,t} = R_wc[t] * diag(r * u[i,t]) * R_wc[t]^T
    # Shape maneuver: (G, T, 3, 3)
    # u_scalar: (G, T)
    # r: (3,)
    # diag values: (G, T, 3) = r[None,None,:] * u_scalar[:,:,None]
    diag_vals = r[None, None, :] * u_scalar[:, :, None]  # (G, T, 3)

    # U = R_wc @ diag(diag_vals) @ R_wc^T
    # Expand R_wc to (1, T, 3, 3) and diag to (G, T, 3, 3)
    R = R_wc[None, :, :, :]           # (1, T, 3, 3)
    D = torch.diag_embed(diag_vals)   # (G, T, 3, 3)

    U = R @ D @ R.transpose(-1, -2)   # (G, T, 3, 3)
    return U


def mahalanobis_sq(
    delta: Tensor,       # (..., 3)  displacement vectors in world space
    u_scalar: Tensor,    # (...,)    scalar uncertainty (per Gaussian per frame)
    R_wc: Tensor,        # (..., 3, 3)  camera-to-world rotation
    r_scale: Tuple[float, float, float] = (1.0, 1.0, 0.01),
) -> Tensor:
    """Compute Mahalanobis squared distance ||delta||^2_{U^{-1}}.

    U = R_wc * diag(rx*u, ry*u, rz*u) * R_wc^T
    U^{-1} = R_wc * diag(1/(rx*u), 1/(ry*u), 1/(rz*u)) * R_wc^T

    ||delta||^2_{U^{-1}} = delta_cam^T * diag(1/(r*u)) * delta_cam
                         = sum_d (delta_cam[d])^2 / (r[d] * u)

    where delta_cam = R_wc^T @ delta (transform to camera space).

    For [rx, ry, rz] = [1, 1, 0.01] and small u:
        x, y deviations penalized by 1/u
        z (depth) deviations penalized by 100/u   => depth errors corrected harder

    Returns:
        loss: (...)  per-element Mahalanobis squared distance
    """
    rx, ry, rz = r_scale
    device = delta.device

    # Transform delta to camera space: delta_cam = R_wc^T @ delta
    # R_wc^T has shape (..., 3, 3), delta has shape (..., 3)
    delta_cam = (R_wc.transpose(-1, -2) @ delta.unsqueeze(-1)).squeeze(-1)  # (..., 3)

    # Scale for U^{-1}: 1 / (r * u)  — large u → small penalty
    r = torch.tensor([rx, ry, rz], device=device, dtype=delta.dtype)
    inv_ru = 1.0 / (r * u_scalar.unsqueeze(-1).clamp(min=1e-8))  # (..., 3)

    # ||delta||^2_{U^{-1}} = sum_d (delta_cam_d)^2 * inv_ru_d
    return (delta_cam ** 2 * inv_ru).sum(dim=-1)  # (...,)
