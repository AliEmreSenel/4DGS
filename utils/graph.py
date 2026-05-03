from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

@dataclass
class USplat4DGraph:
    key_idx:            Tensor   # (N_k,)
    nonkey_idx:         Tensor   # (N_n,)
    key_nbrs:           Tensor   # (N_k, K)
    key_nbr_weights:    Tensor   # (N_k, K)
    nonkey_key_idx:     Tensor   # (N_n,)
    nonkey_nbrs:        Tensor   # (N_n, K+1)
    nonkey_nbr_weights: Tensor   # (N_n, K+1)

    @property
    def num_key(self) -> int:
        return self.key_idx.shape[0]

    @property
    def num_nonkey(self) -> int:
        return self.nonkey_idx.shape[0]


def _safe_quantile(
    values: Tensor,
    q: float,
    max_samples: int = 1_000_000,
) -> Tensor:
    """Compute an approximate quantile without feeding huge tensors to torch.quantile.

    Some PyTorch builds fail for very large inputs with:
        RuntimeError: quantile() input tensor is too large

    U-Splat uses quantiles only as robust thresholds, so a random sample is
    sufficient and avoids the hard failure for large G x T uncertainty tensors.
    """
    values = values.reshape(-1)
    values = values[values.isfinite()]
    if values.numel() == 0:
        raise RuntimeError("All values are non-finite; check graph inputs.")

    if values.numel() > max_samples:
        sample_idx = torch.randint(
            values.numel(),
            (max_samples,),
            device=values.device,
        )
        values = values[sample_idx]

    return torch.quantile(values, q)


def build_graph(
    means_t: Tensor,      # (G, T, 3)   per-Gaussian per-frame world positions
    u_scalar: Tensor,     # (G, T)      per-Gaussian per-frame uncertainties
    key_ratio: float = 0.02,       # fraction of Gaussians treated as key nodes (~top 2%)
    spt_threshold: int = 5,        # significant period threshold (min #frames with u < tau)
    voxel_size: Optional[float] = None,  # 3D voxel size; auto if None
    knn_k: int = 8,                # number of key-key neighbors per key node
    u_tau_percentile: float = 0.50, # uncertainty percentile below which a Gaussian is "low-u"
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

    # ---- 1. Key node selection ----
    # Low-uncertainty threshold tau: below this, a Gaussian is "reliably observed"
    finite_u = u_scalar[u_scalar.isfinite()]
    if finite_u.numel() == 0:
        raise RuntimeError("All uncertainties are non-finite; check rendering output.")
    tau = _safe_quantile(finite_u, u_tau_percentile).item()
    
    
    low_u_mask = u_scalar < tau  # (G, T)  True where Gaussian is reliable

    # Stage 1: Candidate sampling via 3D voxel gridization
    # Use canonical positions (mean over all frames) for voxelization
    means_canon = means_t.mean(dim=1)   # (G, 3) approximate canonical position

    # Auto voxel size: scale scene so that ~target number of voxels are filled
    if voxel_size is None:
        # Aim for approximately G * key_ratio voxels, spread over the scene extent
        target_n_voxels = G * key_ratio * 5   # slightly larger to account for empty voxels
        # Robust scene range using 2nd and 98th percentiles to ignore outliers
        p2 = torch.stack([_safe_quantile(means_canon[:, dim], 0.02) for dim in range(3)])
        p98 = torch.stack([_safe_quantile(means_canon[:, dim], 0.98) for dim in range(3)])
        scene_range = (p98 - p2).clamp(min=1e-3)
        scene_vol = scene_range.prod().item()
        voxel_size = float((scene_vol / max(target_n_voxels, 1)) ** (1.0 / 3.0))
        voxel_size = max(voxel_size, 1e-4)

    # Assign each Gaussian to a voxel by discretizing its canonical position
    vox_min = means_canon.min(dim=0).values   # (3,)
    vox_ids = ((means_canon - vox_min) / voxel_size).long()  # (G, 3)

    # Hash voxel ID to a single integer: hash = x + D*y + D^2*z  (D large prime)
    D = 100003
    vox_hash = vox_ids[:, 0] + D * vox_ids[:, 1] + D * D * vox_ids[:, 2]  # (G,)

    # Per voxel: discard if only high-uncertainty Gaussians; sample 1 low-u Gaussian
    unique_vox, vox_inv = torch.unique(vox_hash, return_inverse=True)  # N_vox
    candidates = []
    for vid in range(unique_vox.shape[0]):
        in_vox = (vox_inv == vid).nonzero(as_tuple=True)[0]  # Gaussian indices in voxel
        # Check if any low-uncertainty Gaussian exists in this voxel (at any frame)
        low_u_in_vox = low_u_mask[in_vox].any(dim=1)  # (n_in_vox,) True if ever low-u
        good = in_vox[low_u_in_vox]
        if good.numel() == 0:
            continue   # all high-uncertainty → skip
        
        # Pick the most reliable Gaussian in this voxel 
        # Metric: highest number of reliable frames; tie-break by lowest mean uncertainty
        good_sig_period = low_u_mask[good].sum(dim=1)      # (N_good,)
        good_mean_u = u_scalar[good].mean(dim=1)           # (N_good,)
        # score = -sig_period + mean_u * 1e-8 (lower is better)
        score = -good_sig_period.float() + good_mean_u * 1e-8
        best_idx = score.argmin()
        pick = good[best_idx].unsqueeze(0)
        candidates.append(pick)

    if len(candidates) == 0:
        raise RuntimeError("No key node candidates found; try relaxing u_tau_percentile.")
    candidates = torch.cat(candidates)  # (N_cand,)
    

    # Stage 2: Filter by significant period >= spt_threshold
    # Significant period = number of frames where u < tau
    sig_period = low_u_mask[candidates].sum(dim=1)  # (N_cand,)
    valid = sig_period >= spt_threshold
    candidates_before_spt = candidates.clone()
    candidates = candidates[valid]
    
    
    if candidates.numel() == 0:
        raise RuntimeError(
            f"No key nodes survive SPT={spt_threshold} filter. "
            "Try reducing spt_threshold."
        )

    # Keep at most G * key_ratio key nodes (top-confidence)
    max_key = min(max(int(G * key_ratio), 1), max(int(max_key_nodes), 1))
    if candidates.numel() > max_key:
        # Rank primarily by reliable frame count (descending = most stable anchors first)
        # Tie-break using lowest mean uncertainty across all frames
        c_sig_period = low_u_mask[candidates].sum(dim=1)   # (N_cand,)
        c_mean_u = u_scalar[candidates].mean(dim=1)        # (N_cand,)
        score = -c_sig_period.float() + c_mean_u * 1e-8
        top = score.topk(max_key, largest=False).indices
        candidates = candidates[top]
        
    key_idx = candidates   # (N_k,)

    # Non-key = all Gaussians not in key set
    is_key = torch.zeros(G, dtype=torch.bool, device=device)
    is_key[key_idx] = True
    nonkey_idx = (~is_key).nonzero(as_tuple=True)[0]  # (N_n,)

    # ---- 2. Key-key edges: UA-kNN (Eq. 9) ----
    # For each key node i, select K neighbors among other key nodes,
    # evaluated at best frame t_hat_i = argmin_t u_{i,t},
    # using Mahalanobis distance (weighted by U_i + U_j).
    N_k = key_idx.shape[0]
    k_eff = min(knn_k, N_k - 1)   # can't have more neighbors than other key nodes

    key_means_all = means_t[key_idx]    # (N_k, T, 3)
    key_u_all     = u_scalar[key_idx]   # (N_k, T)

    # Best frame per key node
    best_t = key_u_all.argmin(dim=1)   # (N_k,)
    key_pos_at_best = key_means_all[torch.arange(N_k, device=device), best_t]  # (N_k, 3)
    key_u_at_best   = key_u_all[torch.arange(N_k, device=device), best_t]       # (N_k,)

    # UA-kNN: distance(i, j) = ||p_i - p_j|| / sqrt(u_i + u_j)  (Eq. 9 — Mahalanobis)
    # We use isotropic approximation (scalar uncertainty) for efficiency.
    # Eq. 9 uses the full matrix U_i + U_j, here approximated as scalar u_i + u_j.
    diff = key_pos_at_best.unsqueeze(1) - key_pos_at_best.unsqueeze(0)  # (N_k, N_k, 3)
    dist_sq = (diff ** 2).sum(dim=-1)   # (N_k, N_k)
    u_sum = key_u_at_best.unsqueeze(1) + key_u_at_best.unsqueeze(0) + 1e-8  # (N_k, N_k)
    mahala_dist = dist_sq / u_sum       # (N_k, N_k)  UA-kNN distance

    # Exclude self (set diagonal to infinity)
    mahala_dist.fill_diagonal_(float("inf"))

    if k_eff > 0:
        _, key_nbr_local = mahala_dist.topk(k_eff, dim=-1, largest=False)  # (N_k, K)
    else:
        key_nbr_local = torch.zeros(N_k, 1, dtype=torch.long, device=device)
        k_eff = 1

    # Edge weights: inverse UA distance (softmax-normalized per node)
    nbr_dists = mahala_dist.gather(1, key_nbr_local)  # (N_k, K)
    nbr_dists_clamped = nbr_dists.clamp(min=1e-8)
    key_nbr_weights = F.softmax(-nbr_dists_clamped, dim=-1)  # (N_k, K)

    # ---- 3. Non-key → key assignment (Eq. 10) ----
    # Assign each non-key node i to key node j that minimizes
    # sum_t ||p_{i,t} - p_{j,t}||_{U_i^t + U_j^t} (scalar Mahalanobis approx).
    N_n = nonkey_idx.shape[0]
    nk_means_all = means_t[nonkey_idx]   # (N_n, T, 3)
    nk_u_all     = u_scalar[nonkey_idx]  # (N_n, T)

    # Compute aggregated distance in nested non-key/key chunks to bound peak memory.
    # This is mathematically identical to the full distance matrix argmin:
    #   argmin_j sum_t ||p_i,t - p_j,t||^2 / (u_i,t + u_j,t)
    CHUNK = max(int(assignment_chunk_size), 1)
    KEY_CHUNK = max(int(key_assignment_chunk_size), 1)
    nonkey_key_local = torch.empty(N_n, dtype=torch.long, device=device)

    for start in range(0, N_n, CHUNK):
        end = min(start + CHUNK, N_n)
        best_dist = torch.full((end - start,), float("inf"), device=device)
        best_key = torch.zeros((end - start,), dtype=torch.long, device=device)

        nk_means_chunk = nk_means_all[start:end]
        nk_u_chunk = nk_u_all[start:end]

        for key_start in range(0, N_k, KEY_CHUNK):
            key_end = min(key_start + KEY_CHUNK, N_k)

            d = (
                nk_means_chunk[:, None, :, :]
                - key_means_all[None, key_start:key_end, :, :]
            )
            d_sq = (d * d).sum(dim=-1)
            u_s = (
                nk_u_chunk[:, None, :]
                + key_u_all[None, key_start:key_end, :]
                + 1e-8
            )
            dist_chunk = (d_sq / u_s).sum(dim=-1)
            local_dist, local_key = dist_chunk.min(dim=-1)

            update = local_dist < best_dist
            best_dist[update] = local_dist[update]
            best_key[update] = local_key[update] + key_start

        nonkey_key_local[start:end] = best_key

    # Non-key inherits assigned key node's edges + the key node itself (ε_i = ε_j ∪ {j})
    assigned_key_nbrs = key_nbr_local[nonkey_key_local]  # (N_n, K) neighbors of assigned key
    # Append the assigned key node itself as an extra neighbor
    extra = nonkey_key_local.unsqueeze(-1)               # (N_n, 1)
    nonkey_nbrs = torch.cat([assigned_key_nbrs, extra], dim=-1)  # (N_n, K+1)

    # DQB blending weights: inverse Mahalanobis distance to each neighbor
    # at the best frame of the non-key node
    nk_best_t = nk_u_all.argmin(dim=1)  # (N_n,)
    nk_pos_best = nk_means_all[torch.arange(N_n, device=device), nk_best_t]  # (N_n, 3)
    nk_u_best   = nk_u_all[torch.arange(N_n, device=device), nk_best_t]      # (N_n,)

    # Neighbor key positions at non-key node's best frame
    # nonkey_nbrs: (N_n, K+1) indices into key_idx (local)
    nbr_key_global = key_idx[nonkey_nbrs.reshape(-1)].reshape(N_n, k_eff + 1)  # (N_n, K+1) global idx
    nbr_t = nk_best_t.unsqueeze(-1).expand(-1, k_eff + 1)                      # (N_n, K+1)
    nbr_pos = means_t[nbr_key_global.reshape(-1), nbr_t.reshape(-1)].reshape(N_n, k_eff + 1, 3)  # (N_n, K+1, 3)
    nbr_u   = u_scalar[nbr_key_global.reshape(-1), nbr_t.reshape(-1)].reshape(N_n, k_eff + 1)    # (N_n, K+1)

    diff_nk = nk_pos_best.unsqueeze(1) - nbr_pos           # (N_n, K+1, 3)
    dist_sq_nk = (diff_nk ** 2).sum(dim=-1)                # (N_n, K+1)
    u_sum_nk = nk_u_best.unsqueeze(1) + nbr_u + 1e-8       # (N_n, K+1)
    mahala_nk = dist_sq_nk / u_sum_nk                      # (N_n, K+1)
    nonkey_nbr_weights = F.softmax(-mahala_nk.clamp(min=1e-8), dim=-1)  # (N_n, K+1)
    
    return USplat4DGraph(
        key_idx=key_idx,
        nonkey_idx=nonkey_idx,
        key_nbrs=key_nbr_local,           # (N_k, K) indices into key_idx
        key_nbr_weights=key_nbr_weights,  # (N_k, K)
        nonkey_key_idx=nonkey_key_local,  # (N_n,)   index into key_idx
        nonkey_nbrs=nonkey_nbrs,          # (N_n, K+1) indices into key_idx
        nonkey_nbr_weights=nonkey_nbr_weights,  # (N_n, K+1)
    )