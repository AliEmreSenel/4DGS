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


@torch.no_grad()
def compute_uncertainty_single_frame(
    means_t: Tensor,       # (G, 3)  Gaussian positions in world space at frame t
    opacities_raw: Tensor, # (G,)    pre-sigmoid opacities
    w2c: Tensor,           # (4, 4)  world-to-camera
    K: Tensor,             # (3, 3)  camera intrinsics
    img_wh: Tuple[int, int],  # (W, H)
    gt_img: Tensor,        # (H, W, 3)  ground-truth RGB in [0,1]
    full_rendered_rgb: Tensor, # (H, W, 3) full scene rendered RGB in [0,1]
    radii: Tensor,    
    depth: Tensor,  
    eta_c: float = 0.5,    # color-error threshold for convergence check
    phi: float = 1e6,      # large constant for high-uncertainty Gaussians
    depth_margin_rel: float = 0.05,  # relative depth margin for occlusion test
) -> Tuple[Tensor, Tensor]:
   
    device = means_t.device
    W, H = img_wh
    G = means_t.shape[0]

    # ---------- 1. Render scene to get depth + RGB + per-Gaussian projections ----------
    # Activate Gaussian parameters
    alpha = torch.sigmoid(opacities_raw)          # (G,)
   
    # Project Gaussian centers to camera space analytically
    means3D_h = torch.cat(
        [means_t, torch.ones(G, 1, device=device)], dim=-1
    )  # (G, 4)
    means_cam = (w2c @ means3D_h.T).T  # (G, 4)

    # Per-Gaussian camera-space depth
    g_depths = means_cam[:, 2]  # (G,)

    # Per-Gaussian projected 2D pixel centers
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = means_cam[:, 2].clamp(min=1e-6)
    g_means2d = torch.stack([
        fx * means_cam[:, 0] / z + cx,
        fy * means_cam[:, 1] / z + cy,
    ], dim=-1)  # (G, 2)

    # radii and rendered_depth come from outside — pass them in as arguments
    g_radii = radii           # (G,)  passed in
    rendered_depth = depth    # (H, W) passed in

    u = torch.full((G,), fill_value=phi, dtype=torch.float32, device=device)

    # Gaussians with radii > 0 are within the camera frustum
    visible_mask = g_radii > 0  # (G,)

    if visible_mask.any():
        vis_idx = visible_mask.nonzero(as_tuple=True)[0]  # (N_vis,)

        # Projected pixel coords (integer, clamped to image bounds)
        px = g_means2d[vis_idx, 0].round().long().clamp(0, W - 1)  # (N_vis,)
        py = g_means2d[vis_idx, 1].round().long().clamp(0, H - 1)  # (N_vis,)

        # Depth comparison for occlusion detection
        d_gauss    = g_depths[vis_idx]                    # (N_vis,)
        d_rendered = rendered_depth[py, px]               # (N_vis,)
        margin     = depth_margin_rel * d_rendered.clamp(min=1e-3)
        not_occluded = d_gauss <= d_rendered + margin     # (N_vis,)

        # Projected 2D area × opacity² (approximation of sum_h (v_i^h)^2)
        r = g_radii[vis_idx]                              # (N_vis,)
        a = alpha[vis_idx]                                # (N_vis,)
        area = math.pi * r.clamp(min=0.5) ** 2           # (N_vis,) at least 1px circle
        inv_sum_sq_v = 1.0 / (a * a * area + 1e-8)       # (N_vis,)

        # Apply large constant where occluded
        u_vis = torch.where(not_occluded, inv_sum_sq_v,
                            torch.full_like(inv_sum_sq_v, phi))
        u[vis_idx] = u_vis


    # ---------- 3. Convergence filter (Eq. 6–7) ----------
    # If any pixel covered by Gaussian i has color error > eta_c, set u_i = phi.
    # We check all pixels within the Gaussian's 2D footprint (circle of radius r).
    if visible_mask.any():
        vis_idx = visible_mask.nonzero(as_tuple=True)[0]
        px = g_means2d[vis_idx, 0].round().long()
        py = g_means2d[vis_idx, 1].round().long()
        radii_vis = g_radii[vis_idx].long()

        # Check convergence: for each Gaussian, check if ANY pixel in its footprint
        # has color error > eta_c. If so, mark as unconverged.
        not_converged = torch.zeros(len(vis_idx), dtype=torch.bool, device=device)
        max_color_err = torch.zeros(len(vis_idx), device=device)
        
        for i in range(len(vis_idx)):
            r = radii_vis[i].item()
            if r <= 0:
                continue
            
            # Get bounding box of Gaussian footprint
            x_min = max(0, int(px[i].item()) - r)
            x_max = min(W - 1, int(px[i].item()) + r)
            y_min = max(0, int(py[i].item()) - r)
            y_max = min(H - 1, int(py[i].item()) + r)
            
            # Check if Gaussian center is within image bounds
            if x_min <= int(px[i].item()) <= x_max and y_min <= int(py[i].item()) <= y_max:
                # Get all pixels within the circular footprint
                yy, xx = torch.meshgrid(
                    torch.arange(y_min, y_max + 1, device=device),
                    torch.arange(x_min, x_max + 1, device=device),
                    indexing='ij'
                )
                
                # Check which pixels are within the circular radius
                dist_sq = (xx.float() - px[i].float()) ** 2 + (yy.float() - py[i].float()) ** 2
                in_circle = dist_sq <= (r ** 2)
                
                if in_circle.any():
                    # Get color errors for pixels in the footprint
                    pixels_in_circle = yy[in_circle], xx[in_circle]
                    color_at_pixels = full_rendered_rgb[pixels_in_circle]  # (N_pixels, 3)
                    gt_at_pixels = gt_img[pixels_in_circle]  # (N_pixels, 3)
                    pixel_errors = (color_at_pixels - gt_at_pixels).abs().mean(dim=-1)  # (N_pixels,)
                    
                    # Mark as unconverged if ANY pixel exceeds threshold
                    max_err = pixel_errors.max().item()
                    max_color_err[i] = max_err
                    not_converged[i] = max_err > eta_c
        
        u[vis_idx] = torch.where(not_converged, torch.full_like(u[vis_idx], phi), u[vis_idx])


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
