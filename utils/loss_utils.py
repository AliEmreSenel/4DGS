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
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    MultiScaleStructuralSimilarityIndexMeasure,
)

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def _ensure_nchw(image):
    if image.dim() == 3:
        return image.unsqueeze(0), True
    if image.dim() != 4:
        raise ValueError("Expected CHW or NCHW tensor")
    return image, False

def ssim(img1, img2, window_size=11, size_average=True):
    img1, squeezed_1 = _ensure_nchw(img1)
    img2, squeezed_2 = _ensure_nchw(img2)
    if squeezed_1 != squeezed_2 or img1.shape != img2.shape:
        raise ValueError("img1 and img2 must have matching shapes")

    reduction = "elementwise_mean" if size_average else "none"
    return structural_similarity_index_measure(
        img1,
        img2,
        gaussian_kernel=True,
        sigma=1.5,
        kernel_size=window_size,
        reduction=reduction,
        data_range=1.0,
    )
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
lpips_metrics = {}

def msssim(rgb, gts):
    # assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    # assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return ms_ssim(rgb, gts).item()

def lpips(rgb, gts, net_type="alex"):
    key = (net_type, str(rgb.device))
    if key not in lpips_metrics:
        metric = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=True)
        lpips_metrics[key] = metric.to(rgb.device).eval()

    with torch.no_grad():
        return lpips_metrics[key](rgb, gts).item()