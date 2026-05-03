from typing import NamedTuple

import torch
import torch.nn as nn

from externals.diff_gaussian_rasterization_ms_nosorting import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple
    ]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    theta,
    phi,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    compute_scores=False,
    compute_score_squares=False,
    score_error_map=None,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        theta,
        phi,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        compute_scores,
        compute_score_squares,
        score_error_map,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        theta,
        phi,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        compute_scores,
        compute_score_squares,
        score_error_map,
    ):
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            theta,
            phi,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            compute_scores,
            compute_score_squares,
            score_error_map,
            raster_settings.debug,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                (
                    num_rendered,
                    color,
                    accum_weights_ptr,
                    accum_weights_count,
                    accum_max_count,
                    radii,
                    kernel_time,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                    w_fg,
                    transmittance,
                    gaussian_scores,
                    gaussian_score_max_error,
                ) = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_ms_nosort_fw.dump")
                raise ex
        else:
            (
                num_rendered,
                color,
                accum_weights_ptr,
                accum_weights_count,
                accum_max_count,
                radii,
                kernel_time,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                w_fg,
                transmittance,
                gaussian_scores,
                gaussian_score_max_error,
            ) = _C.rasterize_gaussians(*args)

        del accum_weights_ptr, accum_weights_count, accum_max_count

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            color,
            colors_precomp,
            means3D,
            scales,
            rotations,
            theta,
            phi,
            w_fg,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            gaussian_scores,
        )
        return color, radii, kernel_time, transmittance, gaussian_scores, gaussian_score_max_error

    @staticmethod
    def backward(
        ctx,
        grad_out_color,
        grad_radii,
        grad_kernel_time,
        grad_transmittance,
        grad_gaussian_scores,
        grad_gaussian_score_max_error,
    ):
        del grad_radii, grad_kernel_time, grad_transmittance, grad_gaussian_scores, grad_gaussian_score_max_error

        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            color,
            colors_precomp,
            means3D,
            scales,
            rotations,
            theta,
            phi,
            w_fg,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            gaussian_scores,
        ) = ctx.saved_tensors

        args = (
            raster_settings.bg,
            means3D,
            radii,
            color,
            colors_precomp,
            scales,
            rotations,
            theta,
            phi,
            w_fg,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_theta,
                    grad_phi,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_ms_nosort_bw.dump")
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_theta,
                grad_phi,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
            ) = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_theta,
            grad_phi,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,  # raster_settings
            None,  # compute_scores
            None,  # compute_score_squares
            None,  # score_error_map
        )
        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
            )
        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        theta,
        phi,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        compute_scores=False,
        compute_score_squares=False,
        score_error_map=None,
    ):
        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception("Please provide exactly one of either SHs or precomputed colors")

        if (scales is None or rotations is None) and cov3D_precomp is None:
            raise Exception(
                "Please provide scale/rotation pair, or precomputed 3D covariance"
            )

        empty = means3D.new_empty((0,))
        if shs is None:
            shs = empty
        if colors_precomp is None:
            colors_precomp = empty
        if scales is None:
            scales = empty
        if rotations is None:
            rotations = empty
        if cov3D_precomp is None:
            cov3D_precomp = empty
        if score_error_map is None:
            score_error_map = empty

        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            theta,
            phi,
            scales,
            rotations,
            cov3D_precomp,
            self.raster_settings,
            compute_scores,
            compute_score_squares,
            score_error_map,
        )
