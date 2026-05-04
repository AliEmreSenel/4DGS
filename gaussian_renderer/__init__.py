#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.nn import functional as F
import math
from .diff_gaussian_rasterization import GaussianRasterizationSettings as SortedGaussianRasterizationSettings
from .diff_gaussian_rasterization import GaussianRasterizer as SortedGaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, eval_shfs_4d


def _pipeline_float(pipe, name, default):
    return float(getattr(pipe, name, default))


def _pipeline_int(pipe, name, default):
    return int(getattr(pipe, name, default))


def _tensor_version(obj, name):
    tensor = getattr(obj, name, None)
    return -1 if tensor is None else int(getattr(tensor, "_version", -1))


def _select_temporal_active_mask(pc, timestamp, marginal_t, pipe, scaling_modifier):
    """Return the temporal render mask for inference/training.

    The default is the original per-frame marginal threshold.  During no-grad
    rendering, optional key-frame masks approximate the 4DGS-1K temporal
    filtering idea: active sets for nearby timestamps are re-used instead of
    admitting every Gaussian.  We intentionally disable the cache when gradients
    are enabled so training never uses stale masks after parameter updates.
    """
    threshold = max(0.0, _pipeline_float(pipe, "temporal_mask_threshold", 0.05))
    current_mask = marginal_t[:, 0] > threshold

    keyframes = _pipeline_int(pipe, "temporal_mask_keyframes", 0)
    if keyframes <= 1 or torch.is_grad_enabled():
        return current_mask

    if pc.get_t.numel() == 0:
        return current_mask

    t0 = float(pc.time_duration[0])
    t1 = float(pc.time_duration[1])
    if not math.isfinite(t0) or not math.isfinite(t1) or abs(t1 - t0) < 1e-8:
        return current_mask

    # Keep the cache conservative.  It is only valid for the same point count,
    # threshold, keyframe layout and render-time scale modifier.
    cache_key = (
        int(pc.get_xyz.shape[0]),
        pc.get_xyz.device.type,
        str(pc.get_xyz.device),
        float(threshold),
        int(keyframes),
        float(t0),
        float(t1),
        float(scaling_modifier),
        bool(getattr(pc, "rot_4d", False)),
        float(getattr(pc, "prefilter_var", -1.0)),
        _tensor_version(pc, "_t"),
        _tensor_version(pc, "_scaling_t"),
        _tensor_version(pc, "_scaling"),
        _tensor_version(pc, "_rotation"),
        _tensor_version(pc, "_rotation_r"),
    )
    cache = getattr(pc, "_temporal_active_mask_cache", None)
    if cache is None or cache.get("key") != cache_key:
        masks = []
        for i in range(keyframes):
            kt = t0 + (t1 - t0) * (float(i) / max(1, keyframes - 1))
            masks.append((pc.get_marginal_t(kt, scaling_modifier)[:, 0] > threshold).detach())
        cache = {"key": cache_key, "masks": torch.stack(masks, dim=0)}
        setattr(pc, "_temporal_active_mask_cache", cache)

    masks = cache["masks"]
    frac = (float(timestamp) - t0) / (t1 - t0)
    pos = int(math.ceil(max(0.0, min(1.0, frac)) * max(0, keyframes - 1)))
    window = max(1, _pipeline_int(pipe, "temporal_mask_window", 2))
    lo = max(0, pos - window)
    hi = min(keyframes, pos + window + 1)
    if lo >= hi:
        return current_mask
    key_mask = masks[lo:hi].any(dim=0)
    # Avoid pathological all-empty frames caused by too few keyframes.
    return key_mask if bool(key_mask.any()) else current_mask


def render(
    viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,
    scaling_modifier = 1.0, override_color = None,
    return_gaussian_scores = False,
    return_gaussian_scores_sq = False,
    gaussian_score_error_map = None,
    apply_random_dropout = False,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    if pc.gaussian_dim != 4:
        raise ValueError("Only 4D Gaussian models are supported by this renderer.")

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    use_mobilegs_sort_free = bool(getattr(pipe, "sort_free_render", False))

    if use_mobilegs_sort_free and getattr(pipe, "env_map_res", 0):
        raise ValueError(
            "sort_free_render does not support env_map_res yet because the no-sort "
            "rasterizer only returns a detached transmittance/alpha proxy."
        )

    if use_mobilegs_sort_free:
        if (not torch.is_grad_enabled()) and pc.mobilegs_opacity_phi_nn is None:
            raise RuntimeError(
                "MobileGS sort-free rendering requires a trained opacity/phi MLP in the checkpoint. "
                "Re-train with --sort_free_render enabled and render from that checkpoint."
            )

        from .diff_gaussian_rasterization_ms_nosorting import (
            GaussianRasterizationSettings as MobileGSRasterizationSettings,
            GaussianRasterizer as MobileGSRasterizer,
        )

    if use_mobilegs_sort_free:
        raster_settings = MobileGSRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
        )
        rasterizer = MobileGSRasterizer(raster_settings=raster_settings)
    else:
        raster_settings = SortedGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color if not pipe.env_map_res else torch.zeros(3, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            sh_degree_t=pc.active_sh_degree_t,
            campos=viewpoint_camera.camera_center,
            timestamp=viewpoint_camera.timestamp,
            time_duration=float(pc.time_duration[1]) - float(pc.time_duration[0]),
            rot_4d=pc.rot_4d,
            gaussian_dim=4,
            force_sh_3d=pc.force_sh_3d,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = SortedGaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    scales_t = None
    rotations = None
    rotations_r = None
    ts = None
    cov3D_precomp = None
    marginal_t = None
    prefilter_var = -1.0
    mask = None
    if use_mobilegs_sort_free:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if use_mobilegs_sort_free:
        # Apply the temporal Gaussian marginal in the sort-free path as well.
        # rot_4d Gaussians also have a timestamp-dependent spatial mean/covariance;
        # non-rotated 4D Gaussians keep their spatial covariance and use the temporal
        # marginal as an opacity gate.
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
        opacity = opacity * marginal_t
        if pc.rot_4d:
            cov3D_precomp, delta_mean = pc.get_current_covariance_and_mean_offset(
                scaling_modifier, viewpoint_camera.timestamp
            )
            means3D = means3D + delta_mean
    elif pipe.compute_cov3D_python:
        if pc.rot_4d:
            cov3D_precomp, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
            means3D = means3D + delta_mean
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
        opacity = opacity * marginal_t
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        scales_t = pc.get_scaling_t
        ts = pc.get_t
        if pc.rot_4d:
            rotations_r = pc.get_rotation_r
        if pc.prefilter_var > 0.0:
            prefilter_var = pc.prefilter_var

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
            if pipe.compute_cov3D_python or use_mobilegs_sort_free:
                sh_means = means3D
            elif pc.rot_4d:
                _, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
                sh_means = means3D + delta_mean
            else:
                sh_means = means3D
            dir_pp = (sh_means - viewpoint_camera.camera_center.expand_as(sh_means)).detach()
            dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)
            if pc.force_sh_3d:
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            else:
                dir_t = (pc.get_t - viewpoint_camera.timestamp).detach()
                sh2rgb = eval_shfs_4d(pc.active_sh_degree, pc.active_sh_degree_t, shs_view, dir_pp_normalized, dir_t, pc.time_duration[1] - pc.time_duration[0])
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            if ts is None:
                ts = pc.get_t
    else:
        colors_precomp = override_color

    if use_mobilegs_sort_free and colors_precomp is None and not pc.force_sh_3d:
        # The no-sort CUDA path can evaluate ordinary 3D SH in-kernel.  Only
        # precompute 4D SHFS colors here because that rasterizer intentionally
        # has no timestamp/SCH interface.
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
        dir_pp = (means3D - viewpoint_camera.camera_center.expand_as(means3D)).detach()
        dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)
        dir_t = (pc.get_t - viewpoint_camera.timestamp).detach()
        sh2rgb = eval_shfs_4d(
            pc.active_sh_degree,
            pc.active_sh_degree_t,
            shs_view,
            dir_pp_normalized,
            dir_t,
            pc.time_duration[1] - pc.time_duration[0],
        )
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        shs = None
    
    # The CUDA rasterizer expects a dense per-Gaussian flow buffer.
    flow_2d = torch.zeros_like(pc.get_xyz[:, :2])

    dropout_prob = float(getattr(pipe, "random_dropout_prob", 0.0))
    dropout_mask = None
    dropout_total = int(means3D.shape[0])
    dropout_kept = None
    if apply_random_dropout and dropout_prob > 0.0 and not (return_gaussian_scores or return_gaussian_scores_sq):
        dropout_prob = min(max(dropout_prob, 0.0), 1.0)
        dropout_mask = torch.rand(
            (means3D.shape[0],), device=means3D.device, dtype=torch.float32
        ) > dropout_prob
        if dropout_mask.numel() > 0 and not dropout_mask.any():
            dropout_mask[torch.randint(dropout_mask.shape[0], (1,), device=dropout_mask.device)] = True
        dropout_kept = int(dropout_mask.sum().detach().item())
    
    mask = None
    # Temporal active filtering. The threshold is configurable, and inference can
    # use cached key-frame masks to skip Gaussians that are inactive around the
    # requested timestamp.
    if marginal_t is not None and not (return_gaussian_scores or return_gaussian_scores_sq):
        mask = _select_temporal_active_mask(
            pc, viewpoint_camera.timestamp, marginal_t, pipe, scaling_modifier
        )
    if dropout_mask is not None:
        mask = dropout_mask if mask is None else (mask & dropout_mask)
    if mask is not None:
        if means2D is not None:
            means2D = means2D[mask]
        if means3D is not None:
            means3D = means3D[mask]
        if ts is not None:
            ts = ts[mask]
        if shs is not None:
            shs = shs[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]
        if opacity is not None:
            opacity = opacity[mask]
        if scales is not None:
            scales = scales[mask]
        if scales_t is not None:
            scales_t = scales_t[mask]
        if rotations is not None:
            rotations = rotations[mask]
        if rotations_r is not None:
            rotations_r = rotations_r[mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[mask]
        if flow_2d is not None:
            flow_2d = flow_2d[mask]
    
    gaussian_score_max_error = None
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)

    if means3D.shape[0] == 0:
        # Empty temporal/key-frame masks are valid; render the configured
        # background color instead of hard-coded black, while keeping a zero
        # gradient anchor for training-time bookkeeping.
        grad_anchor = screenspace_points.sum() * 0.0
        if pc.get_t.numel() > 0:
            grad_anchor = grad_anchor + pc.get_t.sum() * 0.0
        rendered_image = (
            bg_color.to(device=means3D.device, dtype=means3D.dtype)
            .view(3, 1, 1)
            .expand(3, image_height, image_width)
            .clone()
            + grad_anchor
        )
        radii = torch.empty((0,), device=means3D.device, dtype=torch.int32)
        depth = rendered_image.new_zeros((1, image_height, image_width))
        alpha = rendered_image.new_zeros((1, image_height, image_width))
        flow = rendered_image.new_zeros((2, image_height, image_width))
        gaussian_scores = means3D.new_empty((0,)) if (return_gaussian_scores or return_gaussian_scores_sq) else None
        gaussian_score_max_error = means3D.new_empty((0,)) if gaussian_score_error_map is not None else None
    elif use_mobilegs_sort_free:
        cam_center = viewpoint_camera.camera_center.expand_as(means3D)
        dir_pp = (means3D - cam_center).detach()
        dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)
        shs_mlp = pc.get_features
        scales_mlp = pc.get_scaling
        rotations_mlp = pc.get_rotation
        if mask is not None:
            # The raster inputs have already been masked above. Apply the same
            # mask to the MLP inputs for dropout and temporal filtering so
            # tensor lengths stay aligned.
            shs_mlp = shs_mlp[mask]
            scales_mlp = scales_mlp[mask]
            rotations_mlp = rotations_mlp[mask]

        t_values = pc.get_t
        if mask is not None:
            t_values = t_values[mask]
            marginal_t_mlp = marginal_t[mask]
        else:
            # Do not replace the temporal marginal by ones here. The
            # return_gaussian_scores path disables masking and still needs
            # the true timestamp support for scoring/pruning.
            marginal_t_mlp = marginal_t
        time_span = max(float(pc.time_duration[1] - pc.time_duration[0]), 1e-6)
        time_mid = 0.5 * float(pc.time_duration[0] + pc.time_duration[1])
        t_norm = (t_values - time_mid) / time_span
        time_features = torch.cat([
            (float(viewpoint_camera.timestamp) - t_values) / time_span,
            t_norm,
            marginal_t_mlp,
        ], dim=1)

        phi, mlp_opacity = pc.get_mobilegs_opacity_phi(
            shs_mlp,
            scales_mlp,
            dir_pp_normalized,
            rotations_mlp,
            time_features=time_features,
        )
        # Preserve the learned base opacity instead of replacing it entirely.
        # This keeps sort-free rendering compatible with existing 4DGS-style
        # checkpoints and lets the MLP act as a view-dependent visibility gate.
        opacity = opacity * mlp_opacity

        rendered_image, radii, _kernel_time, transmittance, gaussian_scores, gaussian_score_max_error = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            theta=torch.zeros_like(phi),
            phi=phi,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            compute_scores=(return_gaussian_scores or return_gaussian_scores_sq),
            compute_score_squares=return_gaussian_scores_sq,
            score_error_map=gaussian_score_error_map,
        )
        depth = rendered_image.new_zeros((1, rendered_image.shape[1], rendered_image.shape[2]))
        alpha = (1.0 - transmittance).unsqueeze(0).clamp(0.0, 1.0)
        flow = rendered_image.new_zeros((2, rendered_image.shape[1], rendered_image.shape[2]))
        covs_com = means3D.new_zeros((means3D.shape[0], 6))
    else:
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, depth, alpha, flow, covs_com, gaussian_scores, gaussian_score_max_error = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            flow_2d = flow_2d,
            opacities = opacity,
            ts = ts,
            scales = scales,
            scales_t = scales_t,
            rotations = rotations,
            rotations_r = rotations_r,
            cov3D_precomp = cov3D_precomp,
            prefilter_var = prefilter_var,
            compute_scores=(return_gaussian_scores or return_gaussian_scores_sq),
            compute_score_squares=return_gaussian_scores_sq,
            score_error_map=gaussian_score_error_map)
    
    if pipe.env_map_res:
        assert pc.env_map is not None
        R = 60
        rays_o, rays_d = viewpoint_camera.get_rays()
        delta = ((rays_o*rays_d).sum(-1))**2 - (rays_d**2).sum(-1)*((rays_o**2).sum(-1)-R**2)
        assert (delta > 0).all()
        t_inter = -(rays_o*rays_d).sum(-1)+torch.sqrt(delta)/(rays_d**2).sum(-1)
        xyz_inter = rays_o + rays_d * t_inter.unsqueeze(-1)
        tu = torch.atan2(xyz_inter[...,1:2], xyz_inter[...,0:1]) / (2 * torch.pi) + 0.5 # theta
        tv = torch.acos(xyz_inter[...,2:3] / R) / torch.pi
        texcoord = torch.cat([tu, tv], dim=-1) * 2 - 1
        bg_color_from_envmap = F.grid_sample(pc.env_map[None], texcoord[None])[0] # 3,H,W
        # mask2 = (0 < xyz_inter[...,0]) & (xyz_inter[...,1] > 0) # & (xyz_inter[...,2] > -19)
        rendered_image = rendered_image + (1 - alpha) * bg_color_from_envmap # * mask2[None]
    
    if mask is not None:
        radii_all = radii.new_zeros(mask.shape)
        radii_all[mask] = radii
    else:
        radii_all = radii

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    dropout_survival_rate = None
    if dropout_kept is not None and dropout_total > 0:
        dropout_survival_rate = float(dropout_kept) / float(dropout_total)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii_all > 0,
            "radii": radii_all,
            "depth": depth,
            "alpha": alpha,
            "flow": flow,
            "gaussian_scores": gaussian_scores if (return_gaussian_scores or return_gaussian_scores_sq) else None,
            "gaussian_score_max_error": gaussian_score_max_error,
            "dropout_kept": dropout_kept,
            "dropout_total": dropout_total if dropout_kept is not None else None,
            "dropout_survival_rate": dropout_survival_rate}
