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

import os
import json
import random
import torch
from torch import nn
from utils.loss_utils import l1_loss, ssim, msssim, lpips as lpips_metric
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from scene.gaussian_model import coerce_time_duration
from utils.general_utils import safe_state, knn
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
from utils.uncertainty import compute_uncertainty_all_frames
from utils.graph import build_graph, USplat4DGraph
from utils.usplat_losses import key_node_loss, non_key_node_loss
from utils.dqb import quat_to_rotmat, rotmat_to_quat
from utils.checkpoint_utils import build_run_config, checkpoint_args, load_checkpoint

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def save_gaussian_args(model_path, gaussian_kwargs):
    with open(os.path.join(model_path, "gaussian_args"), "w") as f:
        f.write(str(Namespace(**gaussian_kwargs)))



def _to_float(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _safe_percentiles(
    tensor,
    percentiles=(0, 1, 5, 25, 50, 75, 95, 99, 100),
    max_samples=1_000_000,
):
    """Return diagnostic percentiles without launching fragile CUDA kernels."""
    if tensor is None:
        return {}
    try:
        values = tensor.detach().reshape(-1)
        total = values.numel()
        if total == 0:
            return {}
        # Move first, then filter/sort/quantile on CPU.  This keeps diagnostics
        # from launching a CUDA gather after graph/loss indexing has gone bad.
        if values.is_cuda:
            if total > max_samples:
                idx = torch.linspace(0, total - 1, steps=max_samples, dtype=torch.long, device="cpu")
                values = values.detach().cpu().float().index_select(0, idx)
            else:
                values = values.detach().cpu().float()
        else:
            if total > max_samples:
                idx = torch.linspace(0, total - 1, steps=max_samples, dtype=torch.long, device=values.device)
                values = values.index_select(0, idx)
            values = values.float().cpu()
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return {}
        qs = torch.tensor([p / 100.0 for p in percentiles], dtype=values.dtype)
        try:
            out = torch.quantile(values, qs)
        except RuntimeError:
            ordered = torch.sort(values).values
            if ordered.numel() == 0:
                return {}
            ranks = (qs * (ordered.numel() - 1)).round().long().clamp_(0, ordered.numel() - 1)
            out = ordered.index_select(0, ranks)
        return {f"p{p}": _to_float(v) for p, v in zip(percentiles, out)}
    except Exception as exc:
        return {"error": str(exc)}


def _safe_mean(value):
    if value is None or value.numel() == 0:
        return None
    value = value.detach().float()
    value = value[torch.isfinite(value)]
    return None if value.numel() == 0 else float(value.mean().item())



def scale_invariant_depth_distill_loss(depth, teacher_depth, eps=1e-8):
    valid = torch.isfinite(depth) & torch.isfinite(teacher_depth) & (depth > 0) & (teacher_depth > 0)
    if not bool(valid.any()):
        return depth.new_tensor(0.0)
    log_diff = torch.log(depth[valid].clamp_min(eps)) - torch.log(teacher_depth[valid].clamp_min(eps))
    return (log_diff * log_diff).mean() - log_diff.mean().pow(2)


def _clone_pipe_for_teacher(pipe):
    teacher_pipe = deepcopy(pipe)
    teacher_pipe.sort_free_render = False
    teacher_pipe.temporal_mask_keyframes = 0
    teacher_pipe.temporal_mask_mode = "marginal"
    return teacher_pipe


def _load_mobilegs_teacher(path, device="cuda"):
    checkpoint_payload = load_checkpoint(path, map_location=device)
    teacher_kwargs = dict(checkpoint_payload["run_config"].get("gaussian_kwargs", {}))
    teacher = GaussianModel(**teacher_kwargs)
    teacher.restore(checkpoint_payload["gaussians"], training_args=None)
    if teacher.mobilegs_opacity_phi_nn is not None:
        teacher.mobilegs_opacity_phi_nn.eval()
    return teacher


@torch.no_grad()
def compute_4dgs_1k_diagnostics(gaussians, scene, pipe, background, render_fn, short_lifespan_threshold=0.25):
    diagnostics = {}
    if getattr(gaussians, "gaussian_dim", 4) != 4 or gaussians.get_xyz.shape[0] == 0:
        return diagnostics

    G = int(gaussians.get_xyz.shape[0])
    sigma_t = gaussians.get_cov_t(scaling_modifier=1.0).detach().reshape(-1)
    diagnostics["sigma_t_percentiles"] = _safe_percentiles(sigma_t)
    diagnostics["percent_short_lifespan_gaussians"] = float((sigma_t < float(short_lifespan_threshold)).float().mean().item()) if sigma_t.numel() else None
    diagnostics["short_lifespan_threshold"] = float(short_lifespan_threshold)

    training_cameras = scene.getTrainCameras()
    masks_by_timestamp = {}
    active_ratios = []
    for _gt, cam in training_cameras:
        render_pkg = render_fn(cam, gaussians, pipe, background)
        radii = render_pkg.get("radii")
        if radii is None or radii.numel() == 0:
            mask = torch.zeros(G, dtype=torch.bool, device=gaussians.get_xyz.device)
        else:
            mask = (radii.reshape(-1) > 0).to(device=gaussians.get_xyz.device)
            if mask.numel() != G:
                fixed = torch.zeros(G, dtype=torch.bool, device=gaussians.get_xyz.device)
                fixed[: min(G, mask.numel())] = mask[: min(G, mask.numel())]
                mask = fixed
        key = float(cam.timestamp)
        masks_by_timestamp[key] = mask if key not in masks_by_timestamp else (masks_by_timestamp[key] | mask)
        active_ratios.append(mask.float().mean())

    if active_ratios:
        ar = torch.stack(active_ratios)
        diagnostics["active_ratio_mean"] = float(ar.mean().item())
        diagnostics["active_ratio_percentiles"] = _safe_percentiles(ar)

    ious = []
    sorted_ts = sorted(masks_by_timestamp)
    for a, b in zip(sorted_ts[:-1], sorted_ts[1:]):
        ma = masks_by_timestamp[a]
        mb = masks_by_timestamp[b]
        union = (ma | mb).sum().float()
        if union.item() > 0:
            ious.append(((ma & mb).sum().float() / union).detach())
    if ious:
        iou = torch.stack(ious)
        diagnostics["activation_iou_adjacent_mean"] = float(iou.mean().item())
        diagnostics["activation_iou_adjacent_percentiles"] = _safe_percentiles(iou)

    try:
        prune_scores = gaussians.compute_spatio_temporal_variation_score(scene, pipe, background, render_fn)
        diagnostics["prune_score_percentiles"] = _safe_percentiles(prune_scores)
    except Exception as exc:
        diagnostics["prune_score_error"] = str(exc)
    return diagnostics



def _sobel_edge_map(image: torch.Tensor) -> torch.Tensor:
    """Return a normalized single-channel edge probability map in [0, 1]."""
    if image.ndim == 3:
        gray = image.mean(dim=0, keepdim=True).unsqueeze(0)
    elif image.ndim == 4:
        gray = image.mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Expected CHW or NCHW image, got shape {tuple(image.shape)}")
    dtype = gray.dtype
    device = gray.device
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    gx = torch.nn.functional.conv2d(gray, kx, padding=1)
    gy = torch.nn.functional.conv2d(gray, ky, padding=1)
    edge = torch.sqrt(gx.square() + gy.square()).squeeze(0).squeeze(0)
    edge = edge - edge.amin()
    denom = edge.amax().clamp_min(1e-8)
    return (edge / denom).clamp(0.0, 1.0)


@torch.no_grad()
def compute_edge_guided_split_mask(gaussians, scene, pipe, background, render_fn, opt):
    """Approximate DropoutGS ESS with contribution-weighted image edges.

    The paper's ESS aggregates edge probabilities over the Gaussian footprint and
    splits high-edge, large-scale primitives.  The renderer already exposes the
    alpha-transmittance contribution score and a per-Gaussian max value over an
    auxiliary pixel map; we combine these to get a stable, implementation-light
    edge score that works for native 4D and all ablation interactions.
    """
    G = int(gaussians.get_xyz.shape[0])
    if G == 0:
        return torch.zeros((0,), dtype=torch.bool, device=background.device), {}

    edge_scores = torch.zeros((G,), device=background.device)
    coverage_scores = torch.zeros((G,), device=background.device)
    for gt_image, viewpoint_camera in scene.getTrainCameras():
        gt_image = gt_image.to(background.device, non_blocking=True)
        viewpoint_camera = viewpoint_camera.cuda(non_blocking=True, copy=False)
        edge_map = _sobel_edge_map(gt_image).contiguous()
        render_pkg = render_fn(
            viewpoint_camera,
            gaussians,
            pipe,
            background,
            return_gaussian_scores=True,
            gaussian_score_error_map=edge_map.reshape(-1),
        )
        contribution = render_pkg.get("gaussian_scores")
        max_edge = render_pkg.get("gaussian_score_max_error")
        if contribution is None or contribution.numel() == 0:
            continue
        contribution = contribution.to(edge_scores.device)
        if max_edge is None or max_edge.numel() == 0:
            weighted = contribution
        else:
            weighted = contribution * max_edge.to(edge_scores.device).clamp(0.0, 1.0)
        n = min(G, weighted.numel())
        edge_scores[:n] += weighted[:n]
        coverage_scores[:n] += contribution[:n]

    normalized_edge = edge_scores / coverage_scores.clamp_min(1e-8)
    scales = gaussians.get_scaling.max(dim=1).values
    finite_edge = normalized_edge[torch.isfinite(normalized_edge) & (coverage_scores > 0)]
    finite_scale = scales[torch.isfinite(scales)]
    if finite_edge.numel() == 0 or finite_scale.numel() == 0:
        return torch.zeros((G,), dtype=torch.bool, device=background.device), {
            "edge_score_percentiles": _safe_percentiles(normalized_edge),
            "scale_percentiles": _safe_percentiles(scales),
            "selected": 0,
        }

    edge_q = max(0.0, min(1.0, float(getattr(opt, "ess_edge_percentile", 0.90))))
    scale_q = max(0.0, min(1.0, float(getattr(opt, "ess_scale_percentile", 0.70))))
    edge_thr = torch.quantile(finite_edge, torch.tensor(edge_q, device=finite_edge.device, dtype=finite_edge.dtype))
    scale_thr = torch.quantile(finite_scale, torch.tensor(scale_q, device=finite_scale.device, dtype=finite_scale.dtype))
    selected = (normalized_edge >= edge_thr) & (scales >= scale_thr) & (coverage_scores > 0)

    max_splits = int(getattr(opt, "ess_max_splits", 5000) or 0)
    if max_splits > 0 and int(selected.sum().item()) > max_splits:
        score = normalized_edge * scales
        idx = torch.nonzero(selected, as_tuple=True)[0]
        top = torch.topk(score[idx], k=max_splits, largest=True).indices
        limited = torch.zeros_like(selected)
        limited[idx[top]] = True
        selected = limited

    stats = {
        "edge_score_percentiles": _safe_percentiles(normalized_edge),
        "scale_percentiles": _safe_percentiles(scales),
        "edge_threshold": _to_float(edge_thr),
        "scale_threshold": _to_float(scale_thr),
        "selected": int(selected.sum().item()),
    }
    return selected, stats


def _write_training_diagnostics(model_path, diagnostics):
    path = os.path.join(model_path, "training_diagnostics.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _merge_graph_stats(base, graph, u_scalar=None):
    stats = dict(getattr(graph, "stats", {}) or {})
    if u_scalar is not None:
        stats["uncertainty_percentiles"] = _safe_percentiles(u_scalar)
    base.update(stats)


def _indexed_xyz(gaussians, indices=None):
    if indices is None:
        return gaussians.get_xyz
    indices = indices.to(device=gaussians.get_xyz.device, dtype=torch.long)
    return gaussians.get_xyz[indices]


def _indexed_scaling(gaussians, indices=None):
    """Return activated spatial scaling without materializing all Gaussians."""
    if indices is None:
        scaling_param = gaussians._scaling
    else:
        indices = indices.to(device=gaussians.get_xyz.device, dtype=torch.long)
        scaling_param = gaussians._scaling[indices]
    scaling = gaussians.scaling_activation(scaling_param)
    if (
        getattr(gaussians, "isotropic_gaussians", False)
        and scaling.numel() > 0
        and scaling.shape[-1] == 1
    ):
        scaling = scaling.repeat(1, 3)
    return scaling


def _indexed_scaling_t(gaussians, indices=None):
    if indices is None:
        scaling_t_param = gaussians._scaling_t
    else:
        indices = indices.to(device=gaussians.get_xyz.device, dtype=torch.long)
        scaling_t_param = gaussians._scaling_t[indices]
    return gaussians.scaling_activation(scaling_t_param)


def _identity_quat(n, device, dtype):
    quat = torch.zeros((n, 4), device=device, dtype=dtype)
    quat[:, 0] = 1.0
    return quat


def _normalize_quat(quat, eps=1e-8):
    # Optimizer explosions can briefly introduce non-finite rotations.  Keep
    # downstream graph losses from crashing; invalid quaternions are converted
    # to the identity-like finite vector and then normalized.
    quat = torch.nan_to_num(quat, nan=0.0, posinf=0.0, neginf=0.0)
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)


def _indexed_rotation(gaussians, indices=None, right=False):
    """Return activated quaternion(s) without normalizing the whole model."""
    xyz = _indexed_xyz(gaussians, indices)
    if getattr(gaussians, "isotropic_gaussians", False):
        return _identity_quat(xyz.shape[0], xyz.device, xyz.dtype)

    rot_param = gaussians._rotation_r if right else gaussians._rotation
    if rot_param.numel() == 0:
        return _identity_quat(xyz.shape[0], xyz.device, xyz.dtype)
    if indices is not None:
        indices = indices.to(device=gaussians.get_xyz.device, dtype=torch.long)
        rot_param = rot_param[indices]
    return gaussians.rotation_activation(rot_param)


def _current_covariance_and_mean_offset(gaussians, timestamp, indices=None):
    """Return current covariance and mean offset for every ablation.

    If ``indices`` is provided, only those Gaussians are evaluated. This avoids
    materializing activated scaling/rotation for the full scene when USplat
    computes full-sequence losses on small key/non-key chunks.
    """
    if indices is not None:
        indices = indices.to(device=gaussians.get_xyz.device, dtype=torch.long)

    xyz = _indexed_xyz(gaussians, indices)
    scaling = _indexed_scaling(gaussians, indices)
    rotation = _indexed_rotation(gaussians, indices)

    if getattr(gaussians, "gaussian_dim", 4) == 4 and getattr(gaussians, "rot_4d", False):
        scaling_xyzt = torch.cat([scaling, _indexed_scaling_t(gaussians, indices)], dim=1)
        rotation_r = _indexed_rotation(gaussians, indices, right=True)
        t_values = gaussians.get_t if indices is None else gaussians.get_t[indices]
        return gaussians.covariance_activation(
            scaling_xyzt,
            1.0,
            rotation,
            rotation_r,
            dt=float(timestamp) - t_values,
        )

    cov = gaussians.covariance_activation(scaling, 1.0, rotation)
    delta = torch.zeros_like(xyz)
    return cov, delta


def _cov6_to_matrix(cov6):
    """Convert symmetric 3D covariance stored as xx,xy,xz,yy,yz,zz to matrices."""
    cov = cov6.new_zeros(cov6.shape[:-1] + (3, 3))
    cov[..., 0, 0] = cov6[..., 0]
    cov[..., 0, 1] = cov[..., 1, 0] = cov6[..., 1]
    cov[..., 0, 2] = cov[..., 2, 0] = cov6[..., 2]
    cov[..., 1, 1] = cov6[..., 3]
    cov[..., 1, 2] = cov[..., 2, 1] = cov6[..., 4]
    cov[..., 2, 2] = cov6[..., 5]
    return cov


def _covariance_quat_from_cov6(cov6, reference_quat, eigengap_eps=1e-4):
    """Estimate a time-varying orientation from the current 3D covariance.

    4DGS does not expose an explicit per-frame quaternion. We therefore infer
    one from the covariance eigenbasis, but that basis is undefined when two
    eigenvalues are nearly equal. In those cases eigendecomposition can produce
    arbitrary axis flips across frames, which then destabilizes USplat rotation
    and DQB losses. Fall back to the learned/base rotation whenever the
    covariance is too close to isotropic or has an ambiguous adjacent eigengap.
    """
    reference_quat = _normalize_quat(reference_quat)

    # ``torch.linalg.eigh`` can fail to converge when a covariance is slightly
    # non-symmetric, non-finite, or numerically close to singular.  That can
    # happen during early/unstable USplat iterations even though those same
    # Gaussians should fall back to their learned/base rotations.  Sanitize and
    # regularize the matrices before eigendecomposition, and retain a robust
    # all-reference fallback if the CUDA solver still rejects a batch.
    raw_cov = _cov6_to_matrix(cov6)
    raw_finite = torch.isfinite(raw_cov).all(dim=(-1, -2))
    cov = torch.nan_to_num(raw_cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.transpose(-1, -2))

    eye = torch.eye(3, device=cov.device, dtype=cov.dtype).expand(cov.shape[:-2] + (3, 3))
    abs_cov = cov.abs()
    diag = cov.diagonal(dim1=-2, dim2=-1)
    diag_abs = abs_cov.diagonal(dim1=-2, dim2=-1)
    off_diag_sum = abs_cov.sum(dim=-1) - diag_abs
    min_gershgorin = (diag - off_diag_sum).min(dim=-1).values
    scale = diag_abs.mean(dim=-1).clamp_min(1e-8)
    jitter = scale * max(float(eigengap_eps), 1e-6)
    shift = (-min_gershgorin).clamp_min(0.0) + jitter
    cov = cov + shift[..., None, None] * eye

    try:
        evals, evecs = torch.linalg.eigh(cov)
    except RuntimeError:
        # Degenerate chunks are safest as reference rotations.  The zero-valued
        # anchor preserves autograd connectivity without routing unstable
        # eigensolver gradients into ill-conditioned covariance parameters.
        return reference_quat + cov6.sum(dim=-1, keepdim=True) * 0.0
    order = evals.argsort(dim=-1, descending=True)
    sorted_evals = torch.gather(evals, dim=-1, index=order)

    gather_idx = order.unsqueeze(-2).expand(*order.shape[:-1], 3, 3)
    R = torch.gather(evecs, dim=-1, index=gather_idx)
    det = torch.linalg.det(R)
    flip = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    R = R.clone()
    R[..., :, 2] = R[..., :, 2] * flip.unsqueeze(-1)

    quat = _normalize_quat(rotmat_to_quat(R))
    align = (quat * reference_quat).sum(dim=-1, keepdim=True) < 0
    quat = torch.where(align, -quat, quat)

    # Relative adjacent eigengap. If either gap is tiny, the corresponding
    # eigenspace is ambiguous and the eigenvectors should not define rotation.
    scale = sorted_evals[..., :2].abs().clamp_min(1e-12)
    adjacent_gap = (sorted_evals[..., :2] - sorted_evals[..., 1:]).abs() / scale
    anisotropy = (
        (sorted_evals[..., 0] - sorted_evals[..., 2]).abs()
        / sorted_evals[..., 0].abs().clamp_min(1e-12)
    )
    finite = raw_finite & torch.isfinite(sorted_evals).all(dim=-1) & torch.isfinite(quat).all(dim=-1)
    ambiguous = (adjacent_gap.min(dim=-1).values < float(eigengap_eps)) | (
        anisotropy < float(eigengap_eps)
    ) | (~finite)

    return torch.where(ambiguous.unsqueeze(-1), reference_quat, quat)


def _current_means_and_quats_for_timestamps(
    gaussians,
    timestamps,
    indices=None,
    quat_chunk_size=64,
    cov_eigengap_eps=1e-4,
):
    """Return differentiable positions and quaternions for selected timestamps.

    The function is index-aware and internally chunked. Full-sequence USplat
    loss usually touches tens of thousands of non-key Gaussians; constructing
    covariance/eigen tensors for the whole scene at once is not viable on
    commodity GPUs.
    """
    device = gaussians.get_xyz.device
    if indices is None:
        indices = torch.arange(gaussians.get_xyz.shape[0], device=device)
    else:
        indices = indices.to(device=device, dtype=torch.long)

    chunk_size = max(1, int(quat_chunk_size))
    xyz_all = gaussians.get_xyz
    use_time_varying_rot = (
        getattr(gaussians, "gaussian_dim", 4) == 4
        and getattr(gaussians, "rot_4d", False)
        and not getattr(gaussians, "isotropic_gaussians", False)
    )

    means_by_t = []
    quats_by_t = []
    for timestamp in timestamps:
        mean_chunks = []
        quat_chunks = []
        for start in range(0, indices.numel(), chunk_size):
            idx = indices[start:start + chunk_size]
            cov, delta = _current_covariance_and_mean_offset(
                gaussians,
                float(timestamp),
                indices=idx,
            )
            mean = xyz_all[idx] + delta
            mean = torch.where(torch.isfinite(mean), mean, xyz_all[idx].detach())
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)
            mean_chunks.append(mean)
            ref_quat = torch.nan_to_num(_indexed_rotation(gaussians, idx), nan=0.0, posinf=0.0, neginf=0.0)
            zero_ref = ref_quat.norm(dim=-1, keepdim=True) < 1e-8
            if zero_ref.any():
                identity = torch.zeros_like(ref_quat)
                identity[:, 0] = 1.0
                ref_quat = torch.where(zero_ref, identity, ref_quat)
            if use_time_varying_rot:
                quat_chunks.append(
                    _covariance_quat_from_cov6(
                        cov, ref_quat, eigengap_eps=cov_eigengap_eps
                    )
                )
            else:
                quat_chunks.append(ref_quat)
            del cov, delta
        means_by_t.append(torch.cat(mean_chunks, dim=0))
        quats_by_t.append(torch.cat(quat_chunks, dim=0))
        del mean_chunks, quat_chunks

    return torch.stack(means_by_t, dim=1), torch.stack(quats_by_t, dim=1)


def _make_rigid_transforms(pos_t, pos_o, quats_t):
    """Build canonical-to-current SE(3) transforms."""
    N, B = pos_t.shape[:2]
    pos_t = torch.nan_to_num(pos_t, nan=0.0, posinf=1e6, neginf=-1e6)
    pos_o = torch.nan_to_num(pos_o, nan=0.0, posinf=1e6, neginf=-1e6)
    quats_t = torch.nan_to_num(quats_t, nan=0.0, posinf=0.0, neginf=0.0)
    q_flat = quats_t.reshape(-1, 4)
    zero_q = q_flat.norm(dim=-1, keepdim=True) < 1e-8
    if zero_q.any():
        identity = torch.zeros_like(q_flat)
        identity[:, 0] = 1.0
        q_flat = torch.where(zero_q, identity, q_flat)
    R = torch.nan_to_num(quat_to_rotmat(q_flat), nan=0.0, posinf=0.0, neginf=0.0).reshape(N, B, 3, 3)
    Rp0 = (R @ pos_o[:, None, :, None]).squeeze(-1)
    t = torch.nan_to_num(pos_t - Rp0, nan=0.0, posinf=1e6, neginf=-1e6)
    T_mat = torch.zeros(N, B, 3, 4, device=pos_t.device, dtype=pos_t.dtype)
    T_mat[..., :3, :3] = R
    T_mat[..., 3] = t
    return T_mat


def _select_usplat_frame_indices(center_timestamp, timestamps, window):
    T = timestamps.numel()
    if int(window) <= 0 or int(window) >= T:
        return torch.arange(T, device=timestamps.device)
    window = max(1, int(window))
    center = torch.argmin(torch.abs(timestamps - float(center_timestamp))).item()
    start = max(0, min(center - window // 2, T - window))
    return torch.arange(start, start + window, device=timestamps.device)


def rebuild_usplat_state(gaussians, training_dataset, pipe, background, opt, iteration):
    print(f"[USplat4D] Rebuilding graph at iteration {iteration}...")
    with torch.no_grad():
        u_scalar = compute_uncertainty_all_frames(
            gaussians=gaussians,
            train_cameras=training_dataset,
            pipe=pipe,
            background=background,
            device=background.device,
            eta_c=opt.usplat_eta_c,
            phi=opt.usplat_phi,
        )
        T_all = len(training_dataset)
        G = gaussians.get_xyz.shape[0]
        means_t_all = torch.zeros(G, T_all, 3, device=background.device)
        w2cs_all = torch.zeros(T_all, 4, 4, device=background.device)
        timestamps_all = torch.zeros(T_all, device=background.device)
        for t_idx, (_, cam) in enumerate(training_dataset):
            cam = cam.cuda(non_blocking=True, copy=False)
            _, delta = _current_covariance_and_mean_offset(gaussians, cam.timestamp)
            means_t_all[:, t_idx] = gaussians.get_xyz + delta
            w2cs_all[t_idx] = cam.world_view_transform
            timestamps_all[t_idx] = float(cam.timestamp)

        graph = build_graph(
            means_t=means_t_all,
            u_scalar=u_scalar,
            w2cs=w2cs_all,
            key_ratio=getattr(opt, "usplat_key_ratio", 0.02),
            spt_threshold=getattr(opt, "usplat_spt_threshold", 5),
            knn_k=getattr(opt, "usplat_knn_k", 8),
            u_tau_percentile=getattr(opt, "usplat_u_tau_percentile", -1.0),
            max_key_nodes=getattr(opt, "usplat_max_key_nodes", 2000),
            assignment_chunk_size=getattr(opt, "usplat_assignment_chunk_size", 128),
            key_assignment_chunk_size=getattr(opt, "usplat_key_assignment_chunk_size", 512),
            device=background.device,
        )
        p_pretrained_base = gaussians.get_xyz.detach().clone()
        p_pretrained_t = means_t_all.detach().clone()

    print(
        f"[USplat4D] Graph rebuilt: {graph.num_key} key nodes, "
        f"{graph.num_nonkey} non-key nodes"
    )
    return u_scalar, graph, p_pretrained_base, p_pretrained_t, w2cs_all, timestamps_all

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size, isotropic_gaussians,
             use_usplat=False, usplat_start_iter=10000, run_config_args=None):
    
    time_duration = coerce_time_duration(time_duration)

    if gaussian_dim != 4:
        raise ValueError("Only 4D Gaussian training is supported.")

    if dataset.frame_ratio > 1:
        time_duration = [
            time_duration[0] / dataset.frame_ratio,
            time_duration[1] / dataset.frame_ratio,
        ]

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    mobilegs_first_order = bool(getattr(opt, "mobilegs_force_first_order_sh", False))
    if mobilegs_first_order:
        # Mobile-GS first-order export serializes DC + SH1, i.e. 4 SH
        # coefficients per RGB channel. Force sh_degree=1 instead of merely
        # capping it so RGB/DC-only runs do not train an incompatible MLP
        # input layout (16 trained inputs vs 25 exported inputs).
        dataset.sh_degree = 1
        force_sh_3d = True
        pipe.eval_shfs_4d = False

    gaussian_kwargs = {
        "sh_degree": dataset.sh_degree,
        "gaussian_dim": gaussian_dim,
        "time_duration": time_duration,
        "rot_4d": rot_4d,
        "force_sh_3d": force_sh_3d,
        "sh_degree_t": 0 if mobilegs_first_order else (2 if pipe.eval_shfs_4d else 0),
        "prefilter_var": dataset.prefilter_var,
        "isotropic_gaussians": isotropic_gaussians,
    }
    gaussians = GaussianModel(**gaussian_kwargs)
    scene = Scene(
        dataset,
        gaussians,
        num_pts=num_pts,
        num_pts_ratio=num_pts_ratio,
        time_duration=time_duration,
    )
    save_gaussian_args(scene.model_path, gaussian_kwargs)
    # Self-contained checkpoint saves require the full immutable CLI/config
    # namespace plus the Gaussian constructor kwargs.  This must be set before
    # the first possible best-checkpoint save, which can happen at an early test
    # iteration such as 500.
    if run_config_args is None:
        merged_args = {}
        for group in (dataset, opt, pipe):
            merged_args.update(vars(group))
        run_config_args = Namespace(**merged_args)
    scene.checkpoint_run_config = build_run_config(run_config_args, gaussian_kwargs)
    scene.checkpoint_include_mobilegs = bool(getattr(pipe, "sort_free_render", False))

    if getattr(pipe, "sort_free_render", False) and bool(use_usplat):
        raise ValueError(
            "USplat uncertainty requires sorted alpha-blending scores. "
            "Disable --sort_free_render or --use_usplat for this run."
        )

    if getattr(pipe, "sort_free_render", False):
        if getattr(pipe, "env_map_res", 0):
            raise ValueError(
                "env_map_res is not supported with sort_free_render: the no-sort "
                "rasterizer does not produce the sorted alpha needed for env-map compositing."
            )
        if getattr(opt, "lambda_depth", 0.0) > 0.0:
            raise ValueError(
                "lambda_depth is not supported with sort_free_render: the MobileGS "
                "sort-free rasterizer currently returns only a zero depth placeholder."
            )
        if getattr(opt, "lambda_opa_mask", 0.0) > 0.0:
            raise ValueError(
                "lambda_opa_mask is not supported with sort_free_render: the MobileGS "
                "sort-free alpha output is a detached transmittance proxy."
            )
        if float(getattr(opt, "mobilegs_opacity_phi_lr", 0.0) or 0.0) <= 0.0:
            opt.mobilegs_opacity_phi_lr = 1e-3
    else:
        # Avoid allocating/training the MobileGS opacity/phi MLP in ordinary sorted runs.
        opt.mobilegs_opacity_phi_lr = 0.0

    mobilegs_teacher = None
    mobilegs_teacher_pipe = None
    teacher_ckpt = str(getattr(opt, "mobilegs_teacher_checkpoint", "") or "")
    lambda_mobilegs_sh_distill = float(getattr(opt, "lambda_mobilegs_sh_distill", 0.0) or 0.0)
    lambda_mobilegs_depth_distill = float(getattr(opt, "lambda_mobilegs_depth_distill", 0.0) or 0.0)
    if lambda_mobilegs_sh_distill > 0.0 or lambda_mobilegs_depth_distill > 0.0:
        if not teacher_ckpt:
            raise ValueError(
                "Mobile-GS distillation requested but mobilegs_teacher_checkpoint is empty."
            )
        mobilegs_teacher = _load_mobilegs_teacher(teacher_ckpt, device="cuda")
        mobilegs_teacher_pipe = _clone_pipe_for_teacher(pipe)
        print(f"[Mobile-GS] Loaded teacher checkpoint for distillation: {teacher_ckpt}")

    gaussians.training_setup(opt)
    
    # USplat4D setup
    graph = None
    p_pretrained_base = None
    p_pretrained_t = None
    usplat_w2cs = None
    usplat_timestamps = None
    u_scalar = None
    usplat_state_dirty = False
    usplat_graph_rebuild_count = 0
    usplat_diag_stats = {}
    skipped_nonfinite_losses = 0
    dropout_stats_accum = {"count": 0.0}
    ess_stats_accum = {"runs": 0, "selected_total": 0}
    st_prune_score_percentiles = {}
    
    if checkpoint:
        checkpoint_payload = load_checkpoint(checkpoint, map_location="cuda")
        first_iter = int(checkpoint_payload["iteration"])
        gaussians.restore(checkpoint_payload["gaussians"], opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    best_psnr = 0.0
    ema_loss_for_log = torch.tensor(0.0, device="cuda")
    ema_l1loss_for_log = torch.tensor(0.0, device="cuda")
    ema_ssimloss_for_log = torch.tensor(0.0, device="cuda")
    lambda_all = [
        key
        for key in opt.__dict__.keys()
        if key.startswith("lambda") and key != "lambda_dssim"
    ]
    lambda_ema_for_log = {name: torch.tensor(0.0, device="cuda") for name in lambda_all}

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if pipe.env_map_res:
        env_map = nn.Parameter(
            torch.zeros(
                (3, pipe.env_map_res, pipe.env_map_res),
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    else:
        env_map = None

    gaussians.env_map = env_map

    training_dataset = scene.getTrainCameras()
    spatio_temporal_pruning_from_iter = (
        opt.spatio_temporal_pruning_from_iter
        if opt.spatio_temporal_pruning_from_iter >= 0
        else opt.densify_from_iter
    )
    spatio_temporal_pruning_until_iter = (
        opt.spatio_temporal_pruning_until_iter
        if opt.spatio_temporal_pruning_until_iter >= 0
        else opt.densify_until_iter
    )
    spatio_temporal_pruning_interval = (
        opt.spatio_temporal_pruning_interval
        if opt.spatio_temporal_pruning_interval > 0
        else opt.densification_interval
    )
    final_prune_ratio = float(getattr(opt, "final_prune_ratio", 0.0))
    if final_prune_ratio > 1.0:
        final_prune_ratio = final_prune_ratio / 100.0
    final_prune_ratio = max(0.0, min(final_prune_ratio, 1.0))

    train_len = len(training_dataset)
    if train_len <= 0:
        raise RuntimeError("Training dataset is empty; cannot optimize.")
    requested_batch_size = max(1, int(batch_size))
    effective_batch_size = min(requested_batch_size, train_len)
    if effective_batch_size < requested_batch_size:
        print(
            f"[WARN] Requested batch_size={requested_batch_size} but the training split has "
            f"only {train_len} item(s). Using batch_size={effective_batch_size} and drop_last=False."
        )
    num_workers = 12 if dataset.dataloader else 0
    dataloader_kwargs = {
        "batch_size": effective_batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": lambda x: x,
        "drop_last": False,
        "pin_memory": dataset.dataloader,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2
    training_dataloader = DataLoader(training_dataset, **dataloader_kwargs)

    iteration = first_iter
    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
            current_batch_size = len(batch_data)
            if current_batch_size <= 0:
                continue
            iteration += 1
            if iteration > opt.iterations:
                break

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            should_densify = iteration < opt.densify_until_iter and (
                opt.densify_until_num_points < 0
                or gaussians.get_xyz.shape[0] < opt.densify_until_num_points
            )

            if should_densify:
                batch_point_grad = []
                batch_visibility_filter = []
                batch_radii = []

            if use_usplat and iteration >= usplat_start_iter and (
                graph is None or usplat_state_dirty
            ):
                (
                    u_scalar,
                    graph,
                    p_pretrained_base,
                    p_pretrained_t,
                    usplat_w2cs,
                    usplat_timestamps,
                ) = rebuild_usplat_state(
                    gaussians=gaussians,
                    training_dataset=training_dataset,
                    pipe=pipe,
                    background=background,
                    opt=opt,
                    iteration=iteration,
                )
                usplat_state_dirty = False
                usplat_graph_rebuild_count += 1
                _merge_graph_stats(usplat_diag_stats, graph, u_scalar)

            batch_Ll1 = torch.tensor(0.0, device="cuda")
            batch_Lssim = torch.tensor(0.0, device="cuda")
            batch_loss = torch.tensor(0.0, device="cuda")
            batch_Ldepth = torch.tensor(0.0, device="cuda")
            batch_Lopa_mask = torch.tensor(0.0, device="cuda")
            batch_Lrigid = torch.tensor(0.0, device="cuda")
            batch_Lmotion = torch.tensor(0.0, device="cuda")
            batch_Lrdr = torch.tensor(0.0, device="cuda")
            batch_Lmobile_sh_distill = torch.tensor(0.0, device="cuda")
            batch_Lmobile_depth_distill = torch.tensor(0.0, device="cuda")
            batch_psnr = torch.tensor(0.0, device="cuda")
            reg_loss = torch.tensor(0.0, device="cuda")
            Ldepth = torch.tensor(0.0, device="cuda")
            Lopa_mask = torch.tensor(0.0, device="cuda")
            Lrigid = torch.tensor(0.0, device="cuda")
            Lmotion = torch.tensor(0.0, device="cuda")
            
            Lkey = torch.tensor(0.0, device=background.device)
            Lnon_key = torch.tensor(0.0, device=background.device)
            skip_optimizer_step = False

            for batch_idx in range(current_batch_size):
                gt_image, viewpoint_cam = batch_data[batch_idx]
                gt_image = gt_image.to(background.device, non_blocking=True)
                viewpoint_cam = viewpoint_cam.cuda(non_blocking=True, copy=False)

                # Full-model render drives the photometric loss and density-control
                # statistics. DropoutGS-style RDR, when enabled, is added below as a
                # consistency loss from a dropout submodel to this full render.
                render_pkg = render(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    background,
                    apply_random_dropout=False,
                )
                image, viewspace_point_tensor, visibility_filter, radii = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )
                depth = render_pkg["depth"]
                alpha = render_pkg["alpha"]

                # Loss
                Ll1_i = l1_loss(image, gt_image)
                Lssim_i = 1.0 - ssim(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * Ll1_i + opt.lambda_dssim * Lssim_i

                if mobilegs_teacher is not None:
                    with torch.no_grad():
                        teacher_pkg = render(
                            viewpoint_cam,
                            mobilegs_teacher,
                            mobilegs_teacher_pipe,
                            background,
                        )
                        teacher_image = torch.clamp(teacher_pkg["render"], 0.0, 1.0)
                        teacher_depth = teacher_pkg.get("depth")
                    if lambda_mobilegs_sh_distill > 0.0:
                        Lmobile_sh_i = l1_loss(image, teacher_image)
                        loss = loss + lambda_mobilegs_sh_distill * Lmobile_sh_i
                        batch_Lmobile_sh_distill += Lmobile_sh_i.detach()
                    if lambda_mobilegs_depth_distill > 0.0 and teacher_depth is not None:
                        Lmobile_depth_i = scale_invariant_depth_distill_loss(
                            depth, teacher_depth.to(depth.device, dtype=depth.dtype)
                        )
                        loss = loss + lambda_mobilegs_depth_distill * Lmobile_depth_i
                        batch_Lmobile_depth_distill += Lmobile_depth_i.detach()

                # DropoutGS RDR: train a random low-complexity submodel to match
                # the full model render, rather than using the ground truth as the
                # dropout target. This keeps gradients local to neighbors of dropped
                # Gaussians and records the dropout-vs-full agreement.
                Lrdr_i = torch.tensor(0.0, device=background.device)
                dropout_prob = float(getattr(pipe, "random_dropout_prob", 0.0) or 0.0)
                lambda_rdr = float(getattr(opt, "lambda_rdr", 1.0) or 0.0)
                if dropout_prob > 0.0 and lambda_rdr > 0.0:
                    dropout_pkg = render(
                        viewpoint_cam,
                        gaussians,
                        pipe,
                        background,
                        apply_random_dropout=True,
                    )
                    dropout_image = dropout_pkg["render"]
                    # RDR is a consistency loss between the full render and a
                    # dropout submodel.  Do not detach the full render by
                    # default: the official DropoutGS training code optimizes
                    # both sides of this consistency loss, and this lets the
                    # full-render screen-space tensor receive RDR gradients for
                    # densification.  Detaching turns RDR into a one-sided
                    # teacher-student loss where dropped Gaussians get no RDR
                    # signal on that iteration, which tends to starve
                    # densification and encourages opacity pruning.
                    full_teacher = image.detach() if bool(getattr(opt, "rdr_detach_full_render", False)) else image
                    Lrdr_l1 = l1_loss(dropout_image, full_teacher)
                    Lrdr_ssim = 1.0 - ssim(dropout_image, full_teacher)
                    Lrdr_i = (1.0 - opt.lambda_dssim) * Lrdr_l1 + opt.lambda_dssim * Lrdr_ssim
                    loss = loss + lambda_rdr * Lrdr_i
                    batch_Lrdr += Lrdr_i.detach()
                    with torch.no_grad():
                        dropout_stats_accum["count"] += 1.0
                        dropout_stats_accum["rdr_loss_sum"] = dropout_stats_accum.get("rdr_loss_sum", 0.0) + float(Lrdr_i.detach().item())
                        full_teacher_detached = full_teacher.detach()
                        dropout_stats_accum["full_vs_dropout_psnr_sum"] = dropout_stats_accum.get("full_vs_dropout_psnr_sum", 0.0) + float(psnr(dropout_image.detach(), full_teacher_detached).mean().item())
                        dropout_stats_accum["full_vs_dropout_ssim_sum"] = dropout_stats_accum.get("full_vs_dropout_ssim_sum", 0.0) + float(ssim(dropout_image.detach(), full_teacher_detached).mean().item())
                        survival = dropout_pkg.get("dropout_survival_rate")
                        if survival is not None:
                            dropout_stats_accum["survival_rate_sum"] = dropout_stats_accum.get("survival_rate_sum", 0.0) + float(survival)

                ###### opa mask Loss ######
                if opt.lambda_opa_mask > 0:
                    gt_alpha_mask = viewpoint_cam.gt_alpha_mask.to(
                        alpha.device, dtype=alpha.dtype
                    )
                    o = alpha.clamp(1e-6, 1 - 1e-6)
                    sky = 1 - gt_alpha_mask

                    Lopa_mask_i = (-sky * torch.log(1 - o)).mean()

                    # lambda_opa_mask = opt.lambda_opa_mask * (1 - 0.99 * min(1, iteration/opt.iterations))
                    lambda_opa_mask = opt.lambda_opa_mask
                    loss = loss + lambda_opa_mask * Lopa_mask_i
                    batch_Lopa_mask += Lopa_mask_i.detach()
                ########################

                ###### depth loss ######
                if opt.lambda_depth > 0:
                    gt_depth = viewpoint_cam.gt_depth.to(
                        depth.device, dtype=depth.dtype
                    )
                    valid = torch.isfinite(gt_depth)
                    if valid.any():
                        Ldepth_i = (depth[valid] - gt_depth[valid]).abs().mean()
                        loss = loss + opt.lambda_depth * Ldepth_i
                        batch_Ldepth += Ldepth_i.detach()
                ########################
                
                usplat_loss_for_log = torch.tensor(0.0, device=background.device)

                # --- Compute USplat losses every iteration after graph state is ready ---
                if use_usplat and graph is not None and iteration >= usplat_start_iter:

                    # Density control schedule
                    total_usplat = opt.iterations - usplat_start_iter
                    in_first_10 = iteration < usplat_start_iter + int(0.1 * total_usplat)
                    in_last_20  = iteration > opt.iterations - int(0.2 * total_usplat)
                    if in_first_10 or in_last_20:
                        should_densify = False

                    frame_idx = _select_usplat_frame_indices(
                        viewpoint_cam.timestamp,
                        usplat_timestamps,
                        getattr(opt, "usplat_motion_window", -1),
                    )
                    timestamps_window = usplat_timestamps[frame_idx]
                    quat_chunk_size = max(1, int(getattr(opt, "usplat_quat_chunk_size", 64)))
                    cov_eigengap_eps = max(
                        0.0,
                        float(getattr(opt, "usplat_cov_eigengap_eps", 1e-4)),
                    )
                    nonkey_chunk_size = max(
                        1,
                        int(getattr(opt, "usplat_nonkey_loss_chunk_size", 64)),
                    )

                    N_k = graph.num_key
                    N_n = graph.num_nonkey

                    # Key nodes are few enough to evaluate over the selected temporal
                    # sequence. Non-key nodes are streamed below and backpropagated per
                    # chunk, so their graphs are freed immediately instead of being kept
                    # for all 49k+ nodes until the image backward pass.
                    pos_key_t, quats_key = _current_means_and_quats_for_timestamps(
                        gaussians,
                        timestamps_window,
                        indices=graph.key_idx,
                        quat_chunk_size=quat_chunk_size,
                        cov_eigengap_eps=cov_eigengap_eps,
                    )

                    pos_key_pre = p_pretrained_t[graph.key_idx][:, frame_idx]
                    pos_o_key = p_pretrained_base[graph.key_idx]
                    transforms_key = _make_rigid_transforms(pos_key_t, pos_o_key, quats_key)

                    u_key = u_scalar[graph.key_idx][:, frame_idx]
                    R_wc_t = usplat_w2cs[frame_idx, :3, :3].transpose(-1, -2)

                    R_key_t = transforms_key[:, :, :, :3]
                    t_key_t = transforms_key[:, :, :, 3]

                    # --- Non-key node loss, exact selected sequence but memory streamed ---
                    if N_n > 0 and opt.lambda_non_key > 0:
                        Lnon_key_accum = torch.tensor(0.0, device=background.device)
                        # Non-key losses use key motion as a target/neighbor scaffold.
                        # Detaching key tensors avoids retaining and replaying the key
                        # graph for every non-key chunk, which is the source of the OOM.
                        R_key_target = R_key_t.detach()
                        t_key_target = t_key_t.detach()
                        pos_key_target = pos_key_t.detach()
                        quats_key_target = quats_key.detach()
                        transforms_key_target = transforms_key.detach()

                        for nk_start in range(0, N_n, nonkey_chunk_size):
                            nk_end = min(nk_start + nonkey_chunk_size, N_n)
                            nk_global_idx = graph.nonkey_idx[nk_start:nk_end]
                            nk_count = nk_end - nk_start

                            pos_nk_t, quats_nk = _current_means_and_quats_for_timestamps(
                                gaussians,
                                timestamps_window,
                                indices=nk_global_idx,
                                quat_chunk_size=quat_chunk_size,
                                cov_eigengap_eps=cov_eigengap_eps,
                            )
                            pos_nk_pre = p_pretrained_t[nk_global_idx][:, frame_idx]
                            pos_o_nk = p_pretrained_base[nk_global_idx]
                            transforms_nk = _make_rigid_transforms(pos_nk_t, pos_o_nk, quats_nk)
                            u_nk = u_scalar[nk_global_idx][:, frame_idx]

                            Lnon_key_chunk = non_key_node_loss(
                                pos_nk_t=pos_nk_t,
                                quats_nk_t=quats_nk,
                                transforms_nk_t=transforms_nk,
                                pos_nk_pretrained=pos_nk_pre,
                                u_nk=u_nk,
                                R_wc_t=R_wc_t,
                                pos_o_nk=pos_o_nk,
                                R_key_t=R_key_target,
                                t_key_t=t_key_target,
                                pos_key_t=pos_key_target,
                                quats_key_t=quats_key_target,
                                transforms_key_t=transforms_key_target,
                                pos_o_key=pos_o_key,
                                nonkey_nbrs_local=graph.nonkey_nbrs[nk_start:nk_end],
                                nonkey_nbr_weights=graph.nonkey_nbr_weights[nk_start:nk_end],
                                nonkey_nbrs_global=graph.nonkey_nbrs[nk_start:nk_end],
                            )
                            chunk_weight = float(nk_count) / float(N_n)
                            if torch.isfinite(Lnon_key_chunk):
                                (opt.lambda_non_key * chunk_weight * Lnon_key_chunk / current_batch_size).backward()
                                Lnon_key_accum = Lnon_key_accum + Lnon_key_chunk.detach() * nk_count
                            else:
                                skipped_nonfinite_losses += 1
                                usplat_state_dirty = True
                                if skipped_nonfinite_losses <= 5 or iteration % 100 == 0:
                                    print(
                                        f"[WARN] Skipping non-finite USplat non-key chunk at iter {iteration}; "
                                        "continuing with RGB/RDR gradients."
                                    )

                            del (
                                pos_nk_t,
                                quats_nk,
                                pos_nk_pre,
                                pos_o_nk,
                                transforms_nk,
                                u_nk,
                                Lnon_key_chunk,
                            )

                        Lnon_key = Lnon_key_accum / float(N_n)
                        usplat_loss_for_log = usplat_loss_for_log + opt.lambda_non_key * Lnon_key.detach()
                    else:
                        Lnon_key = torch.tensor(0.0, device=background.device)

                    # --- Key node loss ---
                    if opt.lambda_key > 0:
                        Lkey = key_node_loss(
                            pos_key_t=pos_key_t,
                            quats_key_t=quats_key,
                            transforms_key_t=transforms_key,
                            pos_key_pretrained=pos_key_pre,
                            u_key=u_key,
                            R_wc_t=R_wc_t,
                            pos_o=pos_o_key,
                            key_nbrs_local=graph.key_nbrs,
                            key_nbr_weights=graph.key_nbr_weights,
                        )
                        if torch.isfinite(Lkey):
                            (opt.lambda_key * Lkey / current_batch_size).backward()
                            usplat_loss_for_log = usplat_loss_for_log + opt.lambda_key * Lkey.detach()
                        else:
                            skipped_nonfinite_losses += 1
                            usplat_state_dirty = True
                            if skipped_nonfinite_losses <= 5 or iteration % 100 == 0:
                                print(
                                    f"[WARN] Skipping non-finite USplat key loss at iter {iteration}; "
                                    "continuing with RGB/RDR gradients."
                                )
                    else:
                        Lkey = torch.tensor(0.0, device=background.device)

                    del pos_key_t, quats_key, transforms_key, R_key_t, t_key_t
                
                batch_Ll1 += Ll1_i.detach()
                batch_Lssim += Lssim_i.detach()
                batch_psnr += psnr(image, gt_image).mean().detach()

                loss = loss / current_batch_size
                if torch.isfinite(loss):
                    batch_loss += loss.detach() + usplat_loss_for_log / current_batch_size
                    loss.backward()
                else:
                    skipped_nonfinite_losses += 1
                    skip_optimizer_step = True
                if should_densify and not skip_optimizer_step:
                    batch_point_grad.append(
                        torch.norm(viewspace_point_tensor.grad[:, :2], dim=-1)
                    )
                    batch_radii.append(radii)
                    batch_visibility_filter.append(visibility_filter)

            ###### rigid loss ######
            if opt.lambda_rigid > 0:
                k = min(20, gaussians.get_xyz.shape[0] - 1)
                if k > 0:
                    # cur_time = viewpoint_cam.timestamp
                    # _, delta_mean = gaussians.get_current_covariance_and_mean_offset(1.0, cur_time)
                    xyz_mean = gaussians.get_xyz
                    xyz_cur = xyz_mean  #  + delta_mean
                    idx, dist = knn(
                        xyz_cur[None].contiguous().detach(),
                        xyz_cur[None].contiguous().detach(),
                        k,
                    )
                    _, velocity = gaussians.get_current_covariance_and_mean_offset(
                        1.0, gaussians.get_t + 0.1
                    )
                    weight = torch.exp(-100 * dist)
                    # cur_marginal_t = gaussians.get_marginal_t(cur_time).detach().squeeze(-1)
                    # marginal_weights = cur_marginal_t[idx] * cur_marginal_t[None,:,None]
                    # weight *= marginal_weights

                    # mean_t, cov_t = gaussians.get_t, gaussians.get_cov_t(scaling_modifier=1)
                    # mean_t_nn, cov_t_nn = mean_t[idx], cov_t[idx]
                    # weight *= torch.exp(-0.5*(mean_t[None, :, None]-mean_t_nn)**2/cov_t[None, :, None]/cov_t_nn*(cov_t[None, :, None]+cov_t_nn)).squeeze(-1).detach()
                    vel_dist = torch.norm(
                        velocity[idx] - velocity[None, :, None], p=2, dim=-1
                    )
                    Lrigid = (weight * vel_dist).sum() / k / xyz_cur.shape[0]
                    reg_loss = reg_loss + opt.lambda_rigid * Lrigid
                    batch_Lrigid = Lrigid.detach()
                else:
                    Lrigid = torch.tensor(0.0, device="cuda")
                    batch_Lrigid = Lrigid.detach()
            ########################

            ###### motion loss ######
            if opt.lambda_motion > 0:
                _, velocity = gaussians.get_current_covariance_and_mean_offset(
                    1.0, gaussians.get_t + 0.1
                )
                if velocity.shape[0] > 0:
                    Lmotion = velocity.norm(p=2, dim=1).mean()
                    reg_loss = reg_loss + opt.lambda_motion * Lmotion
                    batch_Lmotion = Lmotion.detach()
                else:
                    Lmotion = torch.tensor(0.0, device="cuda")
                    batch_Lmotion = Lmotion.detach()
            ########################

            if opt.lambda_rigid > 0 or opt.lambda_motion > 0:
                if torch.isfinite(reg_loss):
                    batch_loss += reg_loss.detach()
                else:
                    skipped_nonfinite_losses += 1
                    skip_optimizer_step = True
                if reg_loss.requires_grad and torch.isfinite(reg_loss):
                    reg_loss.backward()

            Ll1 = batch_Ll1 / current_batch_size
            Lssim = batch_Lssim / current_batch_size
            loss = batch_loss
            if opt.lambda_depth > 0:
                Ldepth = batch_Ldepth / current_batch_size
            if opt.lambda_opa_mask > 0:
                Lopa_mask = batch_Lopa_mask / current_batch_size
            if opt.lambda_rigid > 0:
                Lrigid = batch_Lrigid
            if opt.lambda_motion > 0:
                Lmotion = batch_Lmotion
            Lrdr = batch_Lrdr / current_batch_size
            Lmobile_sh_distill = batch_Lmobile_sh_distill / current_batch_size
            Lmobile_depth_distill = batch_Lmobile_depth_distill / current_batch_size
            if should_densify and not skip_optimizer_step:
                if current_batch_size > 1:
                    visibility_count = torch.stack(batch_visibility_filter, 1).sum(1)
                    visibility_filter = visibility_count > 0
                    radii = torch.stack(batch_radii, 1).max(1)[0]

                    batch_viewspace_point_grad = torch.stack(batch_point_grad, 1).sum(1)
                    batch_viewspace_point_grad[visibility_filter] = (
                        batch_viewspace_point_grad[visibility_filter]
                        * current_batch_size
                        / visibility_count[visibility_filter]
                    )
                    batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)

                    if gaussians.gaussian_dim == 4:
                        t_grad = gaussians._t.grad
                        if t_grad is None:
                            # Sort-free 4D rendering can legitimately produce a
                            # constant background frame when all Gaussians are
                            # outside the timestamp support, or a frame whose
                            # visible contribution does not touch t. Densification
                            # still expects a dense per-Gaussian temporal gradient.
                            t_grad = torch.zeros_like(gaussians._t)
                        batch_t_grad = t_grad[:, 0].detach().clone()
                        batch_t_grad[visibility_filter] = (
                            batch_t_grad[visibility_filter]
                            * current_batch_size
                            / visibility_count[visibility_filter]
                        )
                        batch_t_grad = batch_t_grad.unsqueeze(1)
                else:
                    if gaussians.gaussian_dim == 4:
                        t_grad = gaussians._t.grad
                        if t_grad is None:
                            t_grad = torch.zeros_like(gaussians._t)
                        batch_t_grad = t_grad.detach().clone()

            loss_dict = {"Ll1": Ll1, "Lssim": Lssim}
            if float(getattr(opt, "lambda_rdr", 0.0) or 0.0) > 0.0 and float(getattr(pipe, "random_dropout_prob", 0.0) or 0.0) > 0.0:
                loss_dict["Lrdr"] = Lrdr
            if lambda_mobilegs_sh_distill > 0.0:
                loss_dict["Lmobile_sh_distill"] = Lmobile_sh_distill
            if lambda_mobilegs_depth_distill > 0.0:
                loss_dict["Lmobile_depth_distill"] = Lmobile_depth_distill
            if opt.lambda_depth > 0:
                loss_dict["Ldepth"] = Ldepth
            if opt.lambda_opa_mask > 0:
                loss_dict["Lopa"] = Lopa_mask
            if opt.lambda_rigid > 0:
                loss_dict["Lrigid"] = Lrigid
            if opt.lambda_motion > 0:
                loss_dict["Lmotion"] = Lmotion

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.detach() + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.detach() + 0.6 * ema_l1loss_for_log
                ema_ssimloss_for_log = 0.4 * Lssim.detach() + 0.6 * ema_ssimloss_for_log

                lambda_loss_values = {
                    "lambda_opa_mask": Lopa_mask,
                    "lambda_depth": Ldepth,
                    "lambda_rigid": Lrigid,
                    "lambda_motion": Lmotion,
                    "lambda_rdr": Lrdr,
                    "lambda_mobilegs_sh_distill": Lmobile_sh_distill,
                    "lambda_mobilegs_depth_distill": Lmobile_depth_distill,
                }

                for lambda_name in lambda_all:
                    if (
                        opt.__dict__[lambda_name] > 0
                        and lambda_name in lambda_loss_values
                    ):
                        lambda_ema_for_log[lambda_name] = (
                            0.4 * lambda_loss_values[lambda_name].detach()
                            + 0.6 * lambda_ema_for_log[lambda_name]
                        )

                if iteration % 10 == 0:
                    psnr_for_log = (batch_psnr / current_batch_size).item()
                    postfix = {
                        "Loss": f"{ema_loss_for_log.item():.{7}f}",
                        "PSNR": f"{psnr_for_log:.{2}f}",
                        "Ll1": f"{ema_l1loss_for_log.item():.{4}f}",
                        "Lssim": f"{ema_ssimloss_for_log.item():.{4}f}",
                    }

                    for lambda_name in lambda_all:
                        if (
                            opt.__dict__[lambda_name] > 0
                            and lambda_name in lambda_loss_values
                        ):
                            postfix[lambda_name.replace("lambda_", "L")] = (
                                f"{lambda_ema_for_log[lambda_name].item():.{4}f}"
                            )

                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                test_psnr = training_report(
                    tb_writer,
                    iteration,
                    Ll1,
                    Lssim,
                    loss,
                    l1_loss,
                    testing_iterations,
                    scene,
                    render,
                    (pipe, background),
                    loss_dict,
                )
                if iteration in testing_iterations:
                    if test_psnr >= best_psnr:
                        best_psnr = test_psnr
                        print("\n[ITER {}] Saving best checkpoint".format(iteration))
                        scene.save(iteration, filename="chkpnt_best.pth")

                if iteration in saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if should_densify and not skip_optimizer_step:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    if current_batch_size == 1:
                        gaussians.add_densification_stats(
                            viewspace_point_tensor,
                            visibility_filter,
                            batch_t_grad if gaussians.gaussian_dim == 4 else None,
                        )
                    else:
                        gaussians.add_densification_stats_grad(
                            batch_viewspace_point_grad,
                            visibility_filter,
                            batch_t_grad if gaussians.gaussian_dim == 4 else None,
                        )

                    if (
                        iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0
                    ):
                        size_threshold = (
                            20 if iteration > opt.opacity_reset_interval else None
                        )
                        # In MobileGS/sort-free mode the learned per-Gaussian
                        # opacity used for rendering comes from the MobileGS MLP,
                        # not from GaussianModel._opacity. The stored opacity tensor
                        # would therefore be stale/untrained and can delete the entire
                        # scene after opacity resets. Keep densification and size
                        # pruning, but disable that opacity criterion here.
                        min_opacity_for_prune = (
                            0.0 if getattr(pipe, "sort_free_render", False)
                            else opt.thresh_opa_prune
                        )
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            min_opacity_for_prune,
                            scene.cameras_extent,
                            size_threshold,
                            opt.densify_grad_t_threshold,
                        )
                        if use_usplat and iteration >= usplat_start_iter:
                            graph = None
                            p_pretrained_base = None
                            p_pretrained_t = None
                            usplat_w2cs = None
                            usplat_timestamps = None
                            u_scalar = None
                            usplat_state_dirty = True

                    # The sort-free renderer predicts opacity with the MobileGS
                    # MLP. Resetting GaussianModel._opacity is not used by the
                    # sort-free forward pass and makes the opacity-pruning signal
                    # misleading, so skip it in this mode.
                    if (
                        not getattr(pipe, "sort_free_render", False)
                        and (
                            iteration % opt.opacity_reset_interval == 0
                            or (
                                dataset.white_background
                                and iteration == opt.densify_from_iter
                            )
                        )
                    ):
                        gaussians.reset_opacity()
                        if use_usplat and iteration >= usplat_start_iter:
                            # USplat uncertainty/key selection depends on opacity via
                            # alpha-transmittance blending weights. After an opacity
                            # reset the cached uncertainty and graph are stale, so force
                            # a rebuild before the next USplat loss is evaluated.
                            graph = None
                            p_pretrained_base = None
                            p_pretrained_t = None
                            usplat_w2cs = None
                            usplat_timestamps = None
                            u_scalar = None
                            usplat_state_dirty = True

                # Optimizer step
                if iteration < opt.iterations:
                    if skip_optimizer_step:
                        if skipped_nonfinite_losses <= 5 or iteration % 100 == 0:
                            print(
                                f"[WARN] Skipping optimizer step at iter {iteration} "
                                "because a non-finite loss was detected."
                            )
                        gaussians.optimizer.zero_grad(set_to_none=True)
                        if gaussians.mobilegs_opacity_phi_optimizer is not None:
                            gaussians.mobilegs_opacity_phi_optimizer.zero_grad(
                                set_to_none=True
                            )
                        if pipe.env_map_res and iteration < pipe.env_optimize_until:
                            env_map_optimizer.zero_grad(set_to_none=True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)
                        if gaussians.mobilegs_opacity_phi_optimizer is not None:
                            gaussians.mobilegs_opacity_phi_optimizer.step()
                            gaussians.mobilegs_opacity_phi_optimizer.zero_grad(
                                set_to_none=True
                            )
                        if pipe.env_map_res and iteration < pipe.env_optimize_until:
                            env_map_optimizer.step()
                            env_map_optimizer.zero_grad(set_to_none=True)

                should_final_st_prune = (
                    opt.final_prune_from_iter >= 0
                    and final_prune_ratio > 0.0
                    and iteration == opt.final_prune_from_iter
                )

                if (
                    opt.enable_spatio_temporal_pruning
                    and not should_final_st_prune
                    and iteration >= spatio_temporal_pruning_from_iter
                    and iteration <= spatio_temporal_pruning_until_iter
                    and (iteration - spatio_temporal_pruning_from_iter)
                    % spatio_temporal_pruning_interval
                    == 0
                ):
                    print(
                        f"\n[ITER {iteration}] Computing spatio-temporal pruning scores"
                    )
                    spatio_temporal_scores = (
                        gaussians.compute_spatio_temporal_variation_score(
                            scene, pipe, background, render
                        )
                    )
                    st_prune_score_percentiles = _safe_percentiles(spatio_temporal_scores)
                    min_remaining_points = int(
                        getattr(opt, "spatio_temporal_pruning_min_points", 1)
                    )
                    print(
                        f"[ITER {iteration}] Pruning Gaussians with ratio {opt.spatio_temporal_pruning_ratio} "
                        f"and min_remaining_points={min_remaining_points}"
                    )
                    gaussians.prune_with_spatio_temporal_score(
                        spatio_temporal_scores,
                        opt.spatio_temporal_pruning_ratio,
                        reset_optimizer_state=True,
                        probabilistic=getattr(
                            opt, "spatio_temporal_pruning_random", False
                        ),
                        min_remaining_points=min_remaining_points,
                    )
                    if use_usplat and iteration >= usplat_start_iter:
                        graph = None
                        p_pretrained_base = None
                        p_pretrained_t = None
                        usplat_w2cs = None
                        usplat_timestamps = None
                        u_scalar = None
                        usplat_state_dirty = True

                ess_from_iter = int(getattr(opt, "ess_from_iter", -1))
                if ess_from_iter < 0:
                    ess_from_iter = int(getattr(opt, "densify_from_iter", 0))
                ess_until_iter = int(getattr(opt, "ess_until_iter", -1))
                if ess_until_iter < 0:
                    ess_until_iter = int(getattr(opt, "densify_until_iter", opt.iterations))
                ess_interval = max(1, int(getattr(opt, "ess_interval", 2000)))
                if (
                    bool(getattr(opt, "enable_edge_guided_splitting", False))
                    and iteration < opt.iterations
                    and iteration >= ess_from_iter
                    and iteration <= ess_until_iter
                    and (iteration - ess_from_iter) % ess_interval == 0
                    and int(gaussians.get_xyz.shape[0]) > 0
                ):
                    print(f"\n[ITER {iteration}] Computing edge-guided splitting mask")
                    ess_mask, ess_stats = compute_edge_guided_split_mask(
                        gaussians, scene, pipe, background, render, opt
                    )
                    ess_children = max(1, int(getattr(opt, "ess_split_children", 2)))
                    ess_selected = int(ess_mask.sum().item())
                    if ess_selected > 0:
                        print(
                            f"[ITER {iteration}] ESS splitting {ess_selected} Gaussians "
                            f"into {ess_children} children each"
                        )
                        gaussians.split_points_by_mask(ess_mask, N=ess_children)
                        if use_usplat and iteration >= usplat_start_iter:
                            graph = None
                            p_pretrained_base = None
                            p_pretrained_t = None
                            usplat_w2cs = None
                            usplat_timestamps = None
                            u_scalar = None
                            usplat_state_dirty = True
                    else:
                        print(f"[ITER {iteration}] ESS selected no Gaussians")
                    ess_stats_accum["runs"] = int(ess_stats_accum.get("runs", 0)) + 1
                    ess_stats_accum["selected_total"] = (
                        int(ess_stats_accum.get("selected_total", 0)) + ess_selected
                    )
                    ess_stats_accum["last"] = ess_stats

                if should_final_st_prune:
                    print(
                        f"\n[ITER {iteration}] Computing final spatio-temporal pruning scores"
                    )
                    final_spatio_temporal_scores = (
                        gaussians.compute_spatio_temporal_variation_score(
                            scene, pipe, background, render
                        )
                    )
                    st_prune_score_percentiles = _safe_percentiles(final_spatio_temporal_scores)
                    min_remaining_points = int(getattr(opt, "spatio_temporal_pruning_min_points", 1))
                    print(
                        f"[ITER {iteration}] Final ST-pruning Gaussians with ratio {final_prune_ratio} "
                        f"and min_remaining_points={min_remaining_points}"
                    )
                    gaussians.prune_with_spatio_temporal_score(
                        final_spatio_temporal_scores,
                        final_prune_ratio,
                        reset_optimizer_state=True,
                        probabilistic=getattr(
                            opt, "spatio_temporal_pruning_random", False
                        ),
                        min_remaining_points=min_remaining_points,
                    )
                    if use_usplat and iteration >= usplat_start_iter:
                        graph = None
                        p_pretrained_base = None
                        p_pretrained_t = None
                        usplat_w2cs = None
                        usplat_timestamps = None
                        u_scalar = None
                        usplat_state_dirty = True
                    scene.save(iteration)

                if use_usplat and iteration >= usplat_start_iter and usplat_state_dirty:
                    (
                        u_scalar,
                        graph,
                        p_pretrained_base,
                        p_pretrained_t,
                        usplat_w2cs,
                        usplat_timestamps,
                    ) = rebuild_usplat_state(
                        gaussians=gaussians,
                        training_dataset=training_dataset,
                        pipe=pipe,
                        background=background,
                        opt=opt,
                        iteration=iteration,
                    )
                    usplat_state_dirty = False
                    usplat_graph_rebuild_count += 1
                    _merge_graph_stats(usplat_diag_stats, graph, u_scalar)

    diagnostics = {
        "final_gaussian_count": int(gaussians.get_xyz.shape[0]),
        "dropout": {},
        "ess": {},
        "usplat": dict(usplat_diag_stats),
        "pruning": {},
    }
    dropout_count = float(dropout_stats_accum.get("count", 0.0))
    if dropout_count > 0:
        diagnostics["dropout"] = {
            "samples": dropout_count,
            "rdr_loss_mean": dropout_stats_accum.get("rdr_loss_sum", 0.0) / dropout_count,
            "full_vs_dropout_psnr_mean": dropout_stats_accum.get("full_vs_dropout_psnr_sum", 0.0) / dropout_count,
            "full_vs_dropout_ssim_mean": dropout_stats_accum.get("full_vs_dropout_ssim_sum", 0.0) / dropout_count,
        }
        if "survival_rate_sum" in dropout_stats_accum:
            diagnostics["dropout"]["survival_rate_mean"] = dropout_stats_accum["survival_rate_sum"] / dropout_count
    ess_runs = int(ess_stats_accum.get("runs", 0))
    if ess_runs > 0:
        ess_selected_total = int(ess_stats_accum.get("selected_total", 0))
        diagnostics["ess"] = {
            "runs": ess_runs,
            "selected_total": ess_selected_total,
            "selected_mean": float(ess_selected_total) / float(max(1, ess_runs)),
            "last": ess_stats_accum.get("last", {}),
        }
    if use_usplat:
        diagnostics["usplat"]["graph_rebuild_count"] = int(usplat_graph_rebuild_count)
    if st_prune_score_percentiles:
        diagnostics["pruning"]["last_prune_score_percentiles"] = st_prune_score_percentiles

    if bool(getattr(opt, "record_training_diagnostics", True)):
        try:
            diagnostics["four_dgs_1k"] = compute_4dgs_1k_diagnostics(
                gaussians,
                scene,
                pipe,
                background,
                render,
                short_lifespan_threshold=float(getattr(opt, "diagnostics_short_lifespan_threshold", 0.25)),
            )
        except Exception as exc:
            diagnostics["four_dgs_1k"] = {"error": str(exc)}
        _write_training_diagnostics(scene.model_path, diagnostics)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    Lssim,
    loss,
    l1_loss,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    loss_dict=None,
):
    should_log_train = (iteration % 10 == 0) or (iteration in testing_iterations)
    if tb_writer and should_log_train:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/ssim_loss", Lssim.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar(
            "total_points", scene.gaussians.get_xyz.shape[0], iteration
        )
        if iteration % 500 == 0:
            opacity_values = scene.gaussians.get_opacity
            if opacity_values.numel() > 0:
                tb_writer.add_histogram(
                    "scene/opacity_histogram", opacity_values, iteration
                )
        if loss_dict is not None:
            if "Lrigid" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/rigid_loss",
                    loss_dict["Lrigid"].item(),
                    iteration,
                )
            if "Ldepth" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/depth_loss",
                    loss_dict["Ldepth"].item(),
                    iteration,
                )
            if "Lrdr" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/rdr_loss",
                    loss_dict["Lrdr"].item(),
                    iteration,
                )
            if "Lmobile_sh_distill" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/mobilegs_sh_distill_loss",
                    loss_dict["Lmobile_sh_distill"].item(),
                    iteration,
                )
            if "Lmobile_depth_distill" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/mobilegs_depth_distill_loss",
                    loss_dict["Lmobile_depth_distill"].item(),
                    iteration,
                )
            if "Ltv" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/tv_loss", loss_dict["Ltv"].item(), iteration
                )
            if "Lopa" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/opa_loss", loss_dict["Lopa"].item(), iteration
                )
            if "Lptsopa" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/pts_opa_loss",
                    loss_dict["Lptsopa"].item(),
                    iteration,
                )
            if "Lsmooth" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/smooth_loss",
                    loss_dict["Lsmooth"].item(),
                    iteration,
                )
            if "Llaplacian" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/laplacian_loss",
                    loss_dict["Llaplacian"].item(),
                    iteration,
                )

    psnr_test_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        validation_configs = (
            {
                "name": "train",
                "cameras": [
                    train_cameras[idx % len(train_cameras)] for idx in range(5, 30, 5)
                ],
            },
            {
                "name": "test",
                "cameras": [test_cameras[idx] for idx in range(len(test_cameras))],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                msssim_test = 0.0
                lpips_test = 0.0
                for idx, batch_data in enumerate(tqdm(config["cameras"])):
                    gt_image, viewpoint = batch_data
                    gt_image = gt_image.to("cuda", non_blocking=True)
                    viewpoint = viewpoint.cuda(non_blocking=True, copy=False)

                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)

                    depth = easy_cmap(render_pkg["depth"][0])
                    alpha = torch.clamp(render_pkg["alpha"], 0.0, 1.0).repeat(3, 1, 1)
                    if tb_writer and (idx < 5):
                        grid = [gt_image, image, alpha, depth]
                        grid = make_grid(grid, nrow=2)
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/gt_vs_render".format(viewpoint.image_name),
                            grid[None],
                            global_step=iteration,
                        )

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    msssim_test += msssim(image[None].cpu(), gt_image[None].cpu())
                    lpips_test += lpips_metric(image[None].cpu(), gt_image[None].cpu())
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])
                msssim_test /= len(config["cameras"])
                lpips_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS {}".format(
                        iteration, config["name"], l1_test, psnr_test, lpips_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - msssim",
                        msssim_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - lpips",
                        lpips_test,
                        iteration,
                    )
                if config["name"] == "test":
                    psnr_test_iter = psnr_test.item()

        torch.cuda.empty_cache()
    return psnr_test_iter


def setup_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def flatten_cfg(cfg, out=None):
    if out is None:
        out = {}
    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            flatten_cfg(v, out)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gaussian_dim", type=int, default=4, choices=[4])
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--isotropic_gaussians", action="store_true")
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--exhaust_test", action="store_true")
    
    # First pass: only get --config
    pre_args, _ = parser.parse_known_args(sys.argv[1:])

    # Load config and use it as parser defaults. Resume checkpoints carry an
    # immutable run config, so external config files are ignored for resumed runs.
    if pre_args.start_checkpoint:
        pass
    elif pre_args.config is not None:
        cfg = OmegaConf.load(pre_args.config)
        cfg_defaults = flatten_cfg(cfg)

        for k in cfg_defaults:
            if not hasattr(pre_args, k):
                raise ValueError(f"Unknown config key: {k}")

        parser.set_defaults(**cfg_defaults)

    # Second pass: real parse, CLI now overrides config
    args = parser.parse_args(sys.argv[1:])
    if args.start_checkpoint:
        resume_checkpoint = args.start_checkpoint
        checkpoint_payload = load_checkpoint(resume_checkpoint, map_location="cpu")
        args = checkpoint_args(checkpoint_payload)
        args.start_checkpoint = resume_checkpoint
        print(f"Loaded immutable run config from checkpoint: {resume_checkpoint}")
    args.time_duration = coerce_time_duration(args.time_duration)
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [
            i for i in range(0, args.iterations, 500)
        ]
    setup_seed(args.seed, deterministic=args.deterministic)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)


    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from,
             args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.batch_size, args.isotropic_gaussians,
             args.use_usplat, args.usplat_start_iter, args)

    print("\nTraining complete.")
