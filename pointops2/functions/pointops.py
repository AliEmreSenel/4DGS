from __future__ import annotations

import torch
from torch.autograd import Function

import pointops2_cuda as pointops_cuda


class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, offset: torch.Tensor, new_offset: torch.Tensor) -> torch.Tensor:
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i - 1], n_max)

        idx = torch.zeros((int(new_offset[b - 1].item()),), device=xyz.device, dtype=torch.int32)
        tmp = torch.full((n,), 1e10, device=xyz.device, dtype=torch.float32)
        pointops_cuda.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
        del tmp
        return idx


furthestsampling = FurthestSampling.apply


class KNNQuery(Function):
    @staticmethod
    def forward(
        ctx,
        nsample: int,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor | None,
        offset: torch.Tensor,
        new_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist: (m, nsample)
        """
        if new_xyz is None:
            new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()

        m = new_xyz.shape[0]
        idx = torch.zeros((m, nsample), device=xyz.device, dtype=torch.int32)
        dist2 = torch.zeros((m, nsample), device=xyz.device, dtype=torch.float32)
        pointops_cuda.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
        return idx, torch.sqrt(dist2)


knnquery = KNNQuery.apply


__all__ = ["furthestsampling", "knnquery"]
