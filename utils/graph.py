from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class USplat4DGraph:
    key_idx: Tensor
    nonkey_idx: Tensor
    key_nbrs: Tensor
    key_nbr_weights: Tensor
    nonkey_key_idx: Tensor
    nonkey_nbrs: Tensor
    nonkey_nbr_weights: Tensor
    stats: dict | None = None

    @property
    def num_key(self) -> int:
        return self.key_idx.shape[0]

    @property
    def num_nonkey(self) -> int:
        return self.nonkey_idx.shape[0]



def _percentiles_dict(values: Tensor, percentiles=(0, 1, 5, 25, 50, 75, 95, 99, 100)) -> dict:
    values = values.detach().float().reshape(-1)
    values = values[values.isfinite()]
    if values.numel() == 0:
        return {}
    qs = torch.tensor([p / 100.0 for p in percentiles], device=values.device, dtype=values.dtype)
    out = torch.quantile(values, qs)
    return {f"p{p}": float(v.item()) for p, v in zip(percentiles, out)}


def _safe_quantile(
    values: Tensor,
    q: float,
    max_samples: int = 1_000_000,
) -> Tensor:
    """Compute an approximate quantile without feeding huge tensors to torch.quantile."""
    values = values.reshape(-1)
    values = values[values.isfinite()]
    if values.numel() == 0:
        raise RuntimeError("All values are non-finite; check graph inputs.")

    if values.numel() > max_samples:
        sample_idx = torch.randint(values.numel(), (max_samples,), device=values.device)
        values = values[sample_idx]

    return torch.quantile(values, q)


def _camera_uncertainty_weighted_sq(
    delta_world: Tensor,
    u_sum: Tensor,
    R_cw: Tensor,
    r_scale: Tuple[float, float, float],
) -> Tensor:
    """Uncertainty-aware graph distance under U_i,t + U_j,t.

    The graph should avoid unreliable anchors. Therefore high uncertainty must
    make an edge less attractive. This is intentionally the opposite direction
    from the training loss, which uses U^{-1} to down-weight uncertain residuals.
    """
    r = torch.tensor(r_scale, dtype=delta_world.dtype, device=delta_world.device)
    delta_cam = torch.einsum("...ij,...j->...i", R_cw, delta_world)
    weight = (r * u_sum.unsqueeze(-1)).clamp_min(1e-8)
    return (delta_cam.square() * weight).sum(dim=-1)


def build_graph(
    means_t: Tensor,
    u_scalar: Tensor,
    w2cs: Optional[Tensor] = None,
    r_scale: Tuple[float, float, float] = (1.0, 1.0, 100.0),
    key_ratio: float = 0.02,
    spt_threshold: int = 5,
    voxel_size: Optional[float] = None,
    knn_k: int = 8,
    u_tau_percentile: Optional[float] = None,
    device: Optional[torch.device] = None,
    max_key_nodes: int = 2000,
    assignment_chunk_size: int = 16,
    key_assignment_chunk_size: int = 512,
) -> USplat4DGraph:
    if device is None:
        device = means_t.device

    G, T, _ = means_t.shape
    means_t = means_t.to(device)
    u_scalar = u_scalar.to(device)
    if w2cs is None:
        w2cs = torch.eye(4, device=device, dtype=means_t.dtype).unsqueeze(0).repeat(T, 1, 1)
    else:
        w2cs = w2cs.to(device=device, dtype=means_t.dtype)

    finite_u = u_scalar[u_scalar.isfinite()]
    if finite_u.numel() == 0:
        raise RuntimeError("All uncertainties are non-finite; check rendering output.")
    if u_tau_percentile is None or u_tau_percentile < 0:
        u_tau_percentile = key_ratio
    tau = _safe_quantile(finite_u, float(u_tau_percentile)).item()
    low_u_mask = u_scalar < tau

    if voxel_size is None:
        target_n_voxels = G * key_ratio * 5
        means_flat = means_t.reshape(-1, 3)
        p2 = torch.stack([_safe_quantile(means_flat[:, dim], 0.02) for dim in range(3)])
        p98 = torch.stack([_safe_quantile(means_flat[:, dim], 0.98) for dim in range(3)])
        scene_range = (p98 - p2).clamp(min=1e-3)
        scene_vol = scene_range.prod().item()
        voxel_size = float((scene_vol / max(target_n_voxels, 1)) ** (1.0 / 3.0))
        voxel_size = max(voxel_size, 1e-4)

    # Paper-style candidate sampling: per-frame voxelization, keep one low-u Gaussian per occupied voxel.
    D = 100003
    candidates = []
    for t_idx in range(T):
        frame_pos = means_t[:, t_idx]
        low_frame = low_u_mask[:, t_idx]
        if not low_frame.any():
            continue
        vox_min = frame_pos.min(dim=0).values
        vox_ids = ((frame_pos - vox_min) / voxel_size).long()
        vox_hash = vox_ids[:, 0] + D * vox_ids[:, 1] + D * D * vox_ids[:, 2]
        low_idx = low_frame.nonzero(as_tuple=True)[0]
        unique_vox, vox_inv = torch.unique(vox_hash[low_frame], return_inverse=True)
        for vid in range(unique_vox.shape[0]):
            in_vox = low_idx[(vox_inv == vid).nonzero(as_tuple=True)[0]]
            if in_vox.numel() > 0:
                candidates.append(in_vox[torch.randint(in_vox.numel(), (1,), device=device)])

    if len(candidates) == 0:
        raise RuntimeError("No key node candidates found; try relaxing u_tau_percentile.")
    candidates = torch.unique(torch.cat(candidates))

    sig_period = low_u_mask[candidates].sum(dim=1)
    candidates = candidates[sig_period >= spt_threshold]
    if candidates.numel() == 0:
        raise RuntimeError(
            f"No key nodes survive SPT={spt_threshold} filter. Try reducing spt_threshold."
        )

    max_key = min(max(int(G * key_ratio), 1), max(int(max_key_nodes), 1))
    if candidates.numel() > max_key:
        c_sig_period = low_u_mask[candidates].sum(dim=1)
        c_mean_u = u_scalar[candidates].mean(dim=1)
        score = -c_sig_period.float() + c_mean_u * 1e-8
        top = score.topk(max_key, largest=False).indices
        candidates = candidates[top]

    key_idx = candidates
    is_key = torch.zeros(G, dtype=torch.bool, device=device)
    is_key[key_idx] = True
    nonkey_idx = (~is_key).nonzero(as_tuple=True)[0]

    N_k = key_idx.shape[0]
    k_eff = min(knn_k, N_k - 1)
    key_means_all = means_t[key_idx]
    key_u_all = u_scalar[key_idx]
    best_t = key_u_all.argmin(dim=1)

    graph_dist = torch.empty(N_k, N_k, device=device, dtype=means_t.dtype)
    KEY_CHUNK = max(int(key_assignment_chunk_size), 1)
    for start in range(0, N_k, KEY_CHUNK):
        end = min(start + KEY_CHUNK, N_k)
        row = torch.arange(start, end, device=device)
        t_hat = best_t[row]
        p_i = key_means_all[row, t_hat].unsqueeze(1)
        u_i = key_u_all[row, t_hat].unsqueeze(1)
        p_j = key_means_all[:, t_hat, :].permute(1, 0, 2)
        u_j = key_u_all[:, t_hat].T
        R_cw = w2cs[t_hat, :3, :3].unsqueeze(1).expand(-1, N_k, -1, -1)
        graph_dist[start:end] = _camera_uncertainty_weighted_sq(p_i - p_j, u_i + u_j, R_cw, r_scale)

    graph_dist.fill_diagonal_(float("inf"))
    if k_eff > 0:
        _, key_nbr_local = graph_dist.topk(k_eff, dim=-1, largest=False)
        nbr_dists = graph_dist.gather(1, key_nbr_local)
        key_nbr_weights = F.softmax(-nbr_dists.clamp(min=1e-8), dim=-1)
    else:
        # Single-key-node scenes cannot have a non-self key edge. Keep a benign
        # self-neighbor with unit weight so downstream tensor shapes remain valid.
        key_nbr_local = torch.zeros(N_k, 1, dtype=torch.long, device=device)
        key_nbr_weights = torch.ones(N_k, 1, dtype=means_t.dtype, device=device)
        k_eff = 1

    N_n = nonkey_idx.shape[0]
    nk_means_all = means_t[nonkey_idx]
    nk_u_all = u_scalar[nonkey_idx]

    CHUNK = max(int(assignment_chunk_size), 1)
    KEY_CHUNK = max(int(key_assignment_chunk_size), 1)
    nonkey_key_local = torch.empty(N_n, dtype=torch.long, device=device)
    nonkey_assignment_best_dist = torch.empty(N_n, dtype=means_t.dtype, device=device)
    for start in range(0, N_n, CHUNK):
        end = min(start + CHUNK, N_n)
        best_dist = torch.full((end - start,), float("inf"), device=device)
        best_key = torch.zeros((end - start,), dtype=torch.long, device=device)
        nk_means_chunk = nk_means_all[start:end]
        nk_u_chunk = nk_u_all[start:end]
        for key_start in range(0, N_k, KEY_CHUNK):
            key_end = min(key_start + KEY_CHUNK, N_k)
            dist_chunk = torch.zeros((end - start, key_end - key_start), device=device, dtype=means_t.dtype)
            for t_idx in range(T):
                d = nk_means_chunk[:, None, t_idx, :] - key_means_all[None, key_start:key_end, t_idx, :]
                u_s = nk_u_chunk[:, None, t_idx] + key_u_all[None, key_start:key_end, t_idx]
                dist_chunk = dist_chunk + _camera_uncertainty_weighted_sq(d, u_s, w2cs[t_idx, :3, :3], r_scale)
            local_dist, local_key = dist_chunk.min(dim=-1)
            update = local_dist < best_dist
            best_dist[update] = local_dist[update]
            best_key[update] = local_key[update] + key_start
        nonkey_key_local[start:end] = best_key
        nonkey_assignment_best_dist[start:end] = best_dist

    assigned_key_nbrs = key_nbr_local[nonkey_key_local]
    extra = nonkey_key_local.unsqueeze(-1)
    nonkey_nbrs = torch.cat([assigned_key_nbrs, extra], dim=-1)

    nk_best_t = nk_u_all.argmin(dim=1)
    nk_pos_best = nk_means_all[torch.arange(N_n, device=device), nk_best_t]
    nk_u_best = nk_u_all[torch.arange(N_n, device=device), nk_best_t]
    nbr_key_global = key_idx[nonkey_nbrs.reshape(-1)].reshape(N_n, k_eff + 1)
    nbr_t = nk_best_t.unsqueeze(-1).expand(-1, k_eff + 1)
    nbr_pos = means_t[nbr_key_global.reshape(-1), nbr_t.reshape(-1)].reshape(N_n, k_eff + 1, 3)
    nbr_u = u_scalar[nbr_key_global.reshape(-1), nbr_t.reshape(-1)].reshape(N_n, k_eff + 1)
    diff_nk = nk_pos_best.unsqueeze(1) - nbr_pos
    u_sum_nk = nk_u_best.unsqueeze(1) + nbr_u
    R_cw_nk = w2cs[nk_best_t, :3, :3].unsqueeze(1).expand(-1, k_eff + 1, -1, -1)
    graph_dist_nk = _camera_uncertainty_weighted_sq(diff_nk, u_sum_nk, R_cw_nk, r_scale)
    nonkey_nbr_weights = F.softmax(-graph_dist_nk.clamp(min=1e-8), dim=-1)

    # Diagnostics for ablation summaries. Keep tensors out of the dataclass payload.
    key_row = torch.arange(N_k, device=device)
    key_edge_lengths = torch.empty((0,), device=device, dtype=means_t.dtype)
    if N_k > 0 and key_nbr_local.numel() > 0:
        t_hat = best_t[key_row].unsqueeze(1).expand_as(key_nbr_local)
        p_i = key_means_all[key_row, best_t].unsqueeze(1)
        p_j = key_means_all[key_nbr_local.reshape(-1), t_hat.reshape(-1)].reshape(N_k, k_eff, 3)
        key_edge_lengths = torch.linalg.norm(p_i - p_j, dim=-1).reshape(-1)
    nonkey_assignment_dist = nonkey_assignment_best_dist.detach() if N_n > 0 else torch.empty((0,), device=device)
    sig_period_all = low_u_mask.sum(dim=1)
    key_sig_period = low_u_mask[key_idx].sum(dim=1)
    hist_bins = torch.bincount(key_sig_period.clamp_min(0).to(torch.long), minlength=T + 1)
    stats = {
        "num_key": int(key_idx.numel()),
        "num_nonkey": int(nonkey_idx.numel()),
        "key_fraction": float(key_idx.numel()) / float(max(G, 1)),
        "candidate_count_after_spt": int(candidates.numel()),
        "key_coverage_ratio": float(key_idx.numel()) / float(max(G, 1)),
        "significant_period_threshold": int(spt_threshold),
        "low_uncertainty_tau": float(tau),
        "key_significant_period_histogram": [int(x) for x in hist_bins.detach().cpu().tolist()],
        "all_significant_period_percentiles": _percentiles_dict(sig_period_all),
        "key_edge_length_percentiles": _percentiles_dict(key_edge_lengths),
        "key_edge_weight_percentiles": _percentiles_dict(key_nbr_weights),
        "nonkey_edge_weight_percentiles": _percentiles_dict(nonkey_nbr_weights),
        "nonkey_assignment_distance_percentiles": _percentiles_dict(nonkey_assignment_dist),
    }

    return USplat4DGraph(
        key_idx=key_idx,
        nonkey_idx=nonkey_idx,
        key_nbrs=key_nbr_local,
        key_nbr_weights=key_nbr_weights,
        nonkey_key_idx=nonkey_key_local,
        nonkey_nbrs=nonkey_nbrs,
        nonkey_nbr_weights=nonkey_nbr_weights,
        stats=stats,
    )
