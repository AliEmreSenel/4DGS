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
import numpy as np
from utils.general_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    build_rotation_4d,
    build_scaling_rotation_4d,
)
from torch import nn
import torch.nn.functional as F
import os
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from externals.simple_knn import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import sh_channels_4d
from utils.gpcc_utils import compress_gpcc, decompress_gpcc, calculate_morton_order
from utils.compress_utils import huffman_encode, huffman_decode


class MobileOpacityPhiNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        mid_dim = hidden_dim // 2
        self.phi_head = nn.Linear(mid_dim, 1)
        self.opacity_head = nn.Linear(mid_dim, 1)
        self._init_heads()

    def _init_heads(self):
        nn.init.constant_(self.phi_head.weight, 0.0)
        nn.init.constant_(self.phi_head.bias, 1.0)

        nn.init.constant_(self.opacity_head.weight, 0.0)
        nn.init.constant_(
            self.opacity_head.bias, inverse_sigmoid(torch.tensor(0.1)).item()
        )

    def forward(self, shs, scales, viewdirs, rotations, time_features):
        shs = shs.view(shs.shape[0], -1)
        shs = F.normalize(shs, dim=1)
        scales = F.normalize(scales, dim=1)
        rotations = F.normalize(rotations, dim=1)

        feat = torch.cat([shs, viewdirs, scales, rotations, time_features], dim=1)
        feat = self.backbone(feat)

        phi = F.relu(self.phi_head(feat))
        opacity = torch.sigmoid(self.opacity_head(feat))
        return phi, opacity


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L.transpose(1, 2) @ L
            symm = strip_symmetric(actual_covariance)
            return symm

        def build_covariance_from_scaling_rotation_4d(
            scaling, scaling_modifier, rotation_l, rotation_r, dt=0.0
        ):
            L = build_scaling_rotation_4d(
                scaling_modifier * scaling, rotation_l, rotation_r
            )
            actual_covariance = L @ L.transpose(1, 2)
            cov_11 = actual_covariance[:, :3, :3]
            cov_12 = actual_covariance[:, 0:3, 3:4]
            cov_t = actual_covariance[:, 3:4, 3:4]
            current_covariance = cov_11 - cov_12 @ cov_12.transpose(1, 2) / cov_t
            symm = strip_symmetric(current_covariance)
            if dt.shape[1] > 1:
                mean_offset = (cov_12.squeeze(-1) / cov_t.squeeze(-1))[:, None, :] * dt[
                    ..., None
                ]
                mean_offset = mean_offset[..., None]  # [num_pts, num_time, 3, 1]
            else:
                mean_offset = cov_12.squeeze(-1) / cov_t.squeeze(-1) * dt
            return symm, mean_offset.squeeze(-1)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        if not self.rot_4d:
            self.covariance_activation = build_covariance_from_scaling_rotation
        else:
            self.covariance_activation = build_covariance_from_scaling_rotation_4d

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(
        self,
        sh_degree: int,
        gaussian_dim: int = 3,
        time_duration: list = [-0.5, 0.5],
        rot_4d: bool = False,
        force_sh_3d: bool = False,
        sh_degree_t: int = 0,
        prefilter_var: float = -1.0,
        isotropic_gaussians: bool = False,
    ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.gaussian_dim = gaussian_dim
        self._t = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self.time_duration = time_duration
        self.rot_4d = rot_4d
        self.isotropic_gaussians = isotropic_gaussians
        self._rotation_r = torch.empty(0)
        self.force_sh_3d = force_sh_3d
        self.t_gradient_accum = torch.empty(0)
        if self.rot_4d or self.force_sh_3d:
            assert self.gaussian_dim == 4
        self.env_map = torch.empty(0)

        self.active_sh_degree_t = 0
        self.max_sh_degree_t = sh_degree_t

        self.prefilter_var = prefilter_var
        self.mobilegs_opacity_phi_nn = None
        self.mobilegs_opacity_phi_optimizer = None

        self.setup_functions()

    def _build_mobilegs_opacity_phi_nn(self):
        input_dim = 3 * self.get_max_sh_channels + 3 + 3 + 4 + 3
        return MobileOpacityPhiNN(input_dim).cuda()

    def get_mobilegs_opacity_phi(self, shs, scales, viewdirs, rotations, time_features):
        if self.mobilegs_opacity_phi_nn is None:
            self.mobilegs_opacity_phi_nn = self._build_mobilegs_opacity_phi_nn()
        return self.mobilegs_opacity_phi_nn(
            shs, scales, viewdirs, rotations, time_features
        )

    def capture(self):
        mobilegs_state = None
        if self.mobilegs_opacity_phi_nn is not None:
            mobilegs_state = {
                "model": self.mobilegs_opacity_phi_nn.state_dict(),
                "optimizer": (
                    self.mobilegs_opacity_phi_optimizer.state_dict()
                    if self.mobilegs_opacity_phi_optimizer is not None
                    else None
                ),
            }

        if self.gaussian_dim == 3:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                mobilegs_state,
            )
        elif self.gaussian_dim == 4:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.t_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self._t,
                self._scaling_t,
                self._rotation_r,
                self.rot_4d,
                self.env_map,
                self.active_sh_degree_t,
                mobilegs_state,
            )

    def restore(self, model_args, training_args):
        t_gradient_accum = None
        mobilegs_state = None
        if self.gaussian_dim == 3:
            if len(model_args) >= 13:
                (
                    self.active_sh_degree,
                    self._xyz,
                    self._features_dc,
                    self._features_rest,
                    self._scaling,
                    self._rotation,
                    self._opacity,
                    self.max_radii2D,
                    xyz_gradient_accum,
                    denom,
                    opt_dict,
                    self.spatial_lr_scale,
                    mobilegs_state,
                ) = model_args[:13]
            else:
                (
                    self.active_sh_degree,
                    self._xyz,
                    self._features_dc,
                    self._features_rest,
                    self._scaling,
                    self._rotation,
                    self._opacity,
                    self.max_radii2D,
                    xyz_gradient_accum,
                    denom,
                    opt_dict,
                    self.spatial_lr_scale,
                ) = model_args
        elif self.gaussian_dim == 4:
            if len(model_args) >= 20:
                (
                    self.active_sh_degree,
                    self._xyz,
                    self._features_dc,
                    self._features_rest,
                    self._scaling,
                    self._rotation,
                    self._opacity,
                    self.max_radii2D,
                    xyz_gradient_accum,
                    t_gradient_accum,
                    denom,
                    opt_dict,
                    self.spatial_lr_scale,
                    self._t,
                    self._scaling_t,
                    self._rotation_r,
                    self.rot_4d,
                    self.env_map,
                    self.active_sh_degree_t,
                    mobilegs_state,
                ) = model_args[:20]
            else:
                (
                    self.active_sh_degree,
                    self._xyz,
                    self._features_dc,
                    self._features_rest,
                    self._scaling,
                    self._rotation,
                    self._opacity,
                    self.max_radii2D,
                    xyz_gradient_accum,
                    t_gradient_accum,
                    denom,
                    opt_dict,
                    self.spatial_lr_scale,
                    self._t,
                    self._scaling_t,
                    self._rotation_r,
                    self.rot_4d,
                    self.env_map,
                    self.active_sh_degree_t,
                ) = model_args

        if mobilegs_state is not None and mobilegs_state.get("model") is not None:
            if self.mobilegs_opacity_phi_nn is None:
                self.mobilegs_opacity_phi_nn = self._build_mobilegs_opacity_phi_nn()
            self.mobilegs_opacity_phi_nn.load_state_dict(mobilegs_state["model"])

        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            if self.gaussian_dim == 4:
                self.t_gradient_accum = t_gradient_accum
            self.denom = denom
            try:
                self.optimizer.load_state_dict(opt_dict)
            except ValueError as exc:
                print(
                    f"Warning: optimizer state is incompatible with current Gaussian parameterization, continuing with fresh optimizer state. Details: {exc}"
                )

            if (
                mobilegs_state is not None
                and mobilegs_state.get("optimizer") is not None
                and self.mobilegs_opacity_phi_optimizer is not None
            ):
                try:
                    self.mobilegs_opacity_phi_optimizer.load_state_dict(
                        mobilegs_state["optimizer"]
                    )
                except ValueError as exc:
                    print(
                        f"Warning: MobileGS MLP optimizer state is incompatible, continuing with fresh state. Details: {exc}"
                    )

    @property
    def get_scaling(self):
        scaling = self.scaling_activation(self._scaling)
        if self.isotropic_gaussians and scaling.shape[0] > 0 and scaling.shape[1] == 1:
            return scaling.repeat(1, 3)
        return scaling

    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)

    @property
    def get_scaling_xyzt(self):
        return torch.cat([self.get_scaling, self.get_scaling_t], dim=1)

    @property
    def get_rotation(self):
        if self.isotropic_gaussians:
            return self._identity_rotation(
                self.get_xyz.shape[0], self.get_xyz.device, self.get_xyz.dtype
            )
        return self.rotation_activation(self._rotation)

    @property
    def get_rotation_r(self):
        if self.isotropic_gaussians:
            return self._identity_rotation(
                self.get_xyz.shape[0], self.get_xyz.device, self.get_xyz.dtype
            )
        return self.rotation_activation(self._rotation_r)

    def _identity_rotation(self, n_pts, device, dtype):
        rots = torch.zeros((n_pts, 4), device=device, dtype=dtype)
        rots[:, 0] = 1
        return rots

    def _to_isotropic_scaling(self, scaling_param):
        if scaling_param.numel() == 0:
            return scaling_param
        if scaling_param.shape[1] == 1:
            return scaling_param
        return torch.log(torch.exp(scaling_param).mean(dim=1, keepdim=True))

    def _apply_isotropic_parameterization(self):
        if not self.isotropic_gaussians:
            return
        self._scaling = nn.Parameter(
            self._to_isotropic_scaling(self._scaling).requires_grad_(True)
        )
        self._rotation = torch.empty(
            (0, 4), device=self._scaling.device, dtype=self._scaling.dtype
        )
        if self.gaussian_dim == 4 and self.rot_4d:
            self._rotation_r = torch.empty(
                (0, 4), device=self._scaling.device, dtype=self._scaling.dtype
            )

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_t(self):
        return self._t

    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim=1)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_max_sh_channels(self):
        if self.gaussian_dim == 3 or self.force_sh_3d:
            return (self.max_sh_degree + 1) ** 2
        elif self.gaussian_dim == 4 and self.max_sh_degree_t == 0:
            return sh_channels_4d[self.max_sh_degree]
        elif self.gaussian_dim == 4 and self.max_sh_degree_t > 0:
            return (self.max_sh_degree + 1) ** 2 * (self.max_sh_degree_t + 1)

    def get_cov_t(self, scaling_modifier=1):
        if self.rot_4d:
            L = build_scaling_rotation_4d(
                scaling_modifier * self.get_scaling_xyzt,
                self.get_rotation,
                self.get_rotation_r,
            )
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance[:, 3, 3].unsqueeze(1)
        else:
            return self.get_scaling_t * scaling_modifier

    def get_marginal_t(self, timestamp, scaling_modifier=1):  # Standard
        sigma = self.get_cov_t(scaling_modifier)
        if self.prefilter_var > 0.0:
            sigma += self.prefilter_var
        return torch.exp(
            -0.5 * (self.get_t - timestamp) ** 2 / sigma
        )  # / torch.sqrt(2*torch.pi*sigma)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.get_rotation
        )

    def get_current_covariance_and_mean_offset(self, scaling_modifier=1, timestamp=0.0):
        return self.covariance_activation(
            self.get_scaling_xyzt,
            scaling_modifier,
            self.get_rotation,
            self.get_rotation_r,
            dt=timestamp - self.get_t,
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        elif self.max_sh_degree_t and self.active_sh_degree_t < self.max_sh_degree_t:
            self.active_sh_degree_t += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        if self.gaussian_dim == 4:
            if pcd.time is None:
                fused_times = (
                    torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1
                ) * (
                    self.time_duration[1] - self.time_duration[0]
                ) + self.time_duration[
                    0
                ]
            else:
                fused_times = torch.from_numpy(pcd.time).cuda().float()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic_gaussians:
            scales = scales.repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
            dist_t = (
                torch.zeros_like(fused_times, device="cuda")
                + (self.time_duration[1] - self.time_duration[0]) / 5
            )
            scales_t = torch.log(torch.sqrt(dist_t))
            if self.rot_4d:
                rots_r = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots_r[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        if self.isotropic_gaussians:
            self._rotation = torch.empty((0, 4), device="cuda")
        else:
            self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                if self.isotropic_gaussians:
                    self._rotation_r = torch.empty((0, 4), device="cuda")
                else:
                    self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    def create_from_pth(self, path, spatial_lr_scale):
        assert self.gaussian_dim == 4 and self.rot_4d
        self.spatial_lr_scale = spatial_lr_scale
        init_4d_gaussian = torch.load(path)
        fused_point_cloud = init_4d_gaussian["xyz"].cuda()
        features_dc = init_4d_gaussian["features_dc"].cuda()
        features_rest = init_4d_gaussian["features_rest"].cuda()
        fused_times = init_4d_gaussian["t"].cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = init_4d_gaussian["scaling"].cuda()
        if self.isotropic_gaussians:
            scales = self._to_isotropic_scaling(scales)
        rots = init_4d_gaussian["rotation"].cuda()
        scales_t = init_4d_gaussian["scaling_t"].cuda()
        rots_r = init_4d_gaussian["rotation_r"].cuda()

        opacities = init_4d_gaussian["opacity"].cuda()

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features_dc.transpose(1, 2).requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features_rest.transpose(1, 2).requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        if self.isotropic_gaussians:
            self._rotation = torch.empty((0, 4), device="cuda")
        else:
            self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._t = nn.Parameter(fused_times.requires_grad_(True))
        self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
        if self.isotropic_gaussians:
            self._rotation_r = torch.empty((0, 4), device="cuda")
        else:
            self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
        ]
        if not self.isotropic_gaussians:
            l.append(
                {
                    "params": [self._rotation],
                    "lr": training_args.rotation_lr,
                    "name": "rotation",
                }
            )
        if self.gaussian_dim == 4:  # TODO: tune time_lr_scale
            if training_args.position_t_lr_init < 0:
                training_args.position_t_lr_init = training_args.position_lr_init
            self.t_gradient_accum = torch.zeros(
                (self.get_xyz.shape[0], 1), device="cuda"
            )
            l.append(
                {
                    "params": [self._t],
                    "lr": training_args.position_t_lr_init * self.spatial_lr_scale,
                    "name": "t",
                }
            )
            l.append(
                {
                    "params": [self._scaling_t],
                    "lr": training_args.scaling_lr,
                    "name": "scaling_t",
                }
            )
            if self.rot_4d and not self.isotropic_gaussians:
                l.append(
                    {
                        "params": [self._rotation_r],
                        "lr": training_args.rotation_lr,
                        "name": "rotation_r",
                    }
                )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        mobilegs_lr = getattr(training_args, "mobilegs_opacity_phi_lr", 0.0)
        if mobilegs_lr > 0.0:
            if self.mobilegs_opacity_phi_nn is None:
                self.mobilegs_opacity_phi_nn = self._build_mobilegs_opacity_phi_nn()
            self.mobilegs_opacity_phi_nn.train()
            self.mobilegs_opacity_phi_optimizer = torch.optim.Adam(
                self.mobilegs_opacity_phi_nn.parameters(),
                lr=mobilegs_lr,
                eps=1e-15,
            )
        else:
            self.mobilegs_opacity_phi_optimizer = None

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
            # if param_group["name"] == "t" and self.gaussian_dim == 4:
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        if not self.isotropic_gaussians:
            self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.gaussian_dim == 4:
            self._t = optimizable_tensors["t"]
            self._scaling_t = optimizable_tensors["scaling_t"]
            if self.rot_4d and not self.isotropic_gaussians:
                self._rotation_r = optimizable_tensors["rotation_r"]
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

    @torch.no_grad()
    def compute_spatio_temporal_variation_score(
        self, scene, pipe, background, render_fn
    ):
        if self.gaussian_dim != 4:
            raise ValueError("Spatio-temporal pruning is only defined for 4D Gaussians")

        training_cameras = scene.getTrainCameras()
        if len(training_cameras) == 0:
            return torch.zeros((self.get_xyz.shape[0],), device=self.get_xyz.device)

        spatial_scores = torch.zeros(
            (self.get_xyz.shape[0],), device=self.get_xyz.device
        )
        for img, viewpoint_camera in training_cameras:
            render_pkg = render_fn(
                viewpoint_camera, self, pipe, background, return_gaussian_scores=True
            )
            gaussian_scores = render_pkg.get("gaussian_scores")
            if gaussian_scores is None or gaussian_scores.numel() == 0:
                continue
            spatial_scores = spatial_scores + gaussian_scores.to(spatial_scores.device)

        timestamps = torch.tensor(
            [
                float(viewpoint_camera.timestamp)
                for img, viewpoint_camera in training_cameras
            ],
            device=self.get_xyz.device,
            dtype=self.get_xyz.dtype,
        )
        unique_timestamps = torch.unique(timestamps)

        mu_t = self.get_t.squeeze(-1)
        sigma_t = self.get_cov_t(scaling_modifier=1.0).squeeze(-1).clamp_min(1e-6)
        temporal_variation = torch.zeros_like(mu_t)
        for timestamp in unique_timestamps:
            diff = timestamp - mu_t
            p_i_t = torch.exp(-0.5 * (diff / sigma_t) ** 2)
            second_derivative = (((diff**2) / (sigma_t**2)) - (1.0 / sigma_t)) * p_i_t
            temporal_variation = temporal_variation + 1.0 / (
                0.5 * torch.tanh(torch.abs(second_derivative)) + 0.5
            )

        volume_4d = torch.prod(self.get_scaling, dim=1) * self.get_scaling_t.squeeze(-1)
        volume_4d = volume_4d.clamp_min(1e-8)
        volume_normalized = volume_4d / (torch.norm(volume_4d, p=2) + 1e-8)

        temporal_score = temporal_variation * volume_normalized
        return spatial_scores * temporal_score

    @torch.no_grad()
    def prune_with_spatio_temporal_score(
        self, scores, pruning_ratio, reset_optimizer_state=True
    ):
        if scores.numel() == 0:
            return

        num_points = scores.shape[0]
        num_to_prune = int(num_points * pruning_ratio)
        if num_to_prune <= 0:
            return

        num_to_keep = max(num_points - num_to_prune, 1)
        keep_scores, keep_indices = torch.topk(scores, k=num_to_keep, largest=True)
        del keep_scores
        prune_mask = torch.ones(num_points, dtype=torch.bool, device=scores.device)
        prune_mask[keep_indices] = False
        self.prune_points(prune_mask)

        if reset_optimizer_state and self.optimizer is not None:
            for group in self.optimizer.param_groups:
                state = self.optimizer.state.get(group["params"][0], None)
                if state is not None:
                    if "exp_avg" in state:
                        state["exp_avg"].zero_()
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].zero_()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_t,
        new_scaling_t,
        new_rotation_r,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
        }
        if not self.isotropic_gaussians:
            d["rotation"] = new_rotation
        if self.gaussian_dim == 4:
            d["t"] = new_t
            d["scaling_t"] = new_scaling_t
            if self.rot_4d and not self.isotropic_gaussians:
                d["rotation_r"] = new_rotation_r

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        if not self.isotropic_gaussians:
            self._rotation = optimizable_tensors["rotation"]
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors["t"]
            self._scaling_t = optimizable_tensors["scaling_t"]
            if self.rot_4d and not self.isotropic_gaussians:
                self._rotation_r = optimizable_tensors["rotation_r"]
            self.t_gradient_accum = torch.zeros(
                (self.get_xyz.shape[0], 1), device="cuda"
            )

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(
        self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, N=2
    ):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )
        # print(f"num_to_densify_pos: {torch.where(padded_grad >= grad_threshold, True, False).sum()}, num_to_split_pos: {selected_pts_mask.sum()}")

        new_scaling = self.scaling_inverse_activation(
            self.scaling_activation(self._scaling[selected_pts_mask]).repeat(N, 1)
            / (0.8 * N)
        )
        new_rotation = (
            None
            if self.isotropic_gaussians
            else self._rotation[selected_pts_mask].repeat(N, 1)
        )
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        if not self.rot_4d:
            stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            if self.isotropic_gaussians:
                new_xyz = samples + self.get_xyz[selected_pts_mask].repeat(N, 1)
            else:
                rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
                new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(
                    -1
                ) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_t = None
            new_scaling_t = None
            new_rotation_r = None
            if self.gaussian_dim == 4:
                stds_t = self.get_scaling_t[selected_pts_mask].repeat(N, 1)
                means_t = torch.zeros((stds_t.size(0), 1), device="cuda")
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
                new_scaling_t = self.scaling_inverse_activation(
                    self.get_scaling_t[selected_pts_mask].repeat(N, 1) / (0.8 * N)
                )
        else:
            stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 4), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            if self.isotropic_gaussians:
                new_xyzt = samples + self.get_xyzt[selected_pts_mask].repeat(N, 1)
            else:
                rots = build_rotation_4d(
                    self._rotation[selected_pts_mask],
                    self._rotation_r[selected_pts_mask],
                ).repeat(N, 1, 1)
                new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(
                    -1
                ) + self.get_xyzt[selected_pts_mask].repeat(N, 1)
            new_xyz = new_xyzt[..., 0:3]
            new_t = new_xyzt[..., 3:4]
            new_scaling_t = self.scaling_inverse_activation(
                self.get_scaling_t[selected_pts_mask].repeat(N, 1) / (0.8 * N)
            )
            new_rotation_r = (
                None
                if self.isotropic_gaussians
                else self._rotation_r[selected_pts_mask].repeat(N, 1)
            )

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_t,
            new_scaling_t,
            new_rotation_r,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(
        self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold
    ):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )
        # print(f"num_to_densify_pos: {torch.where(grads >= grad_threshold, True, False).sum()}, num_to_clone_pos: {selected_pts_mask.sum()}")

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = (
            None if self.isotropic_gaussians else self._rotation[selected_pts_mask]
        )
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.rot_4d and not self.isotropic_gaussians:
                new_rotation_r = self._rotation_r[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_t,
            new_scaling_t,
            new_rotation_r,
        )

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        max_grad_t=None,
        prune_only=False,
    ):
        if not prune_only:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            if self.gaussian_dim == 4:
                grads_t = self.t_gradient_accum / self.denom
                grads_t[grads_t.isnan()] = 0.0
            else:
                grads_t = None

            self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t)
            self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(
        self, viewspace_point_tensor, update_filter, avg_t_grad=None
    ):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4 and avg_t_grad is not None:
            tgrad = avg_t_grad[update_filter]
            if tgrad.ndim == 1:
                tgrad = tgrad.unsqueeze(-1)
            self.t_gradient_accum[update_filter] += tgrad

    def add_densification_stats_grad(
        self, viewspace_point_grad, update_filter, avg_t_grad=None
    ):
        grad = viewspace_point_grad[update_filter]
        if grad.ndim == 2 and grad.shape[1] >= 2:
            grad = torch.norm(grad[:, :2], dim=-1, keepdim=True)
        elif grad.ndim == 1:
            grad = grad.unsqueeze(-1)

        self.xyz_gradient_accum[update_filter] += grad
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4 and avg_t_grad is not None:
            tgrad = avg_t_grad[update_filter]
            if tgrad.ndim == 1:
                tgrad = tgrad.unsqueeze(-1)
            self.t_gradient_accum[update_filter] += tgrad

    def _compressible_tensors(self):
        out = {
            "features_dc": self._features_dc.detach(),
            "features_rest": self._features_rest.detach(),
            "scaling": self._scaling.detach(),
            "opacity": self._opacity.detach(),
        }
        if not self.isotropic_gaussians:
            out["rotation"] = self._rotation.detach()
        if self.gaussian_dim == 4:
            out["t"] = self._t.detach()
            out["scaling_t"] = self._scaling_t.detach()
            if self.rot_4d and not self.isotropic_gaussians:
                out["rotation_r"] = self._rotation_r.detach()
        return out

    def _quantize_xyz_u16(self, xyz: torch.Tensor):
        xyz = xyz.detach().float()
        xyz_min = xyz.amin(dim=0)
        xyz_max = xyz.amax(dim=0)
        xyz_extent = (xyz_max - xyz_min).clamp_min(1e-8)
        xyz_q = torch.round((xyz - xyz_min) * (65535.0 / xyz_extent)).to(torch.int32)
        meta = {
            "min": xyz_min.cpu(),
            "max": xyz_max.cpu(),
        }
        return xyz_q, meta

    def _dequantize_xyz_u16(self, xyz_q: torch.Tensor, meta: dict):
        xyz_min = meta["min"].to(xyz_q.device).float()
        xyz_max = meta["max"].to(xyz_q.device).float()
        xyz_extent = (xyz_max - xyz_min).clamp_min(1e-8)
        xyz = xyz_q.float() * (xyz_extent / 65535.0) + xyz_min
        return xyz

    def _encode_scalar_q_huffman(self, tensor: torch.Tensor, num_bits: int = 8):
        x = tensor.detach().float().reshape(tensor.shape[0], -1).contiguous()

        x_min = x.amin(dim=0, keepdim=True)
        x_max = x.amax(dim=0, keepdim=True)
        qmax = float((1 << num_bits) - 1)
        step = ((x_max - x_min) / qmax).clamp_min(1e-8)

        q = torch.round((x - x_min) / step).to(torch.int32).cpu().numpy().reshape(-1)
        huf_idx, huf_tab = huffman_encode(q)

        return {
            "shape": list(tensor.shape),
            "num_bits": int(num_bits),
            "min": x_min.cpu().numpy(),
            "step": step.cpu().numpy(),
            "index": huf_idx,
            "htable": huf_tab,
        }

    def _decode_scalar_q_huffman(self, pack: dict, device="cuda"):
        shape = tuple(pack["shape"])
        flat = np.asarray(huffman_decode(pack["index"], pack["htable"]), dtype=np.int32)
        flat = flat.reshape(shape[0], -1)

        x_min = torch.from_numpy(np.asarray(pack["min"])).to(
            device=device, dtype=torch.float32
        )
        step = torch.from_numpy(np.asarray(pack["step"])).to(
            device=device, dtype=torch.float32
        )

        x = torch.from_numpy(flat).to(device=device, dtype=torch.float32) * step + x_min
        return x.reshape(shape)

    def capture_compressed(self, attr_bits: int = 8):
        xyz_q, xyz_meta = self._quantize_xyz_u16(self.get_xyz)

        sort_idx = calculate_morton_order(xyz_q.int())
        xyz_q_sorted = xyz_q[sort_idx]

        save_dict = {
            "format": "gaussian-compressed-v1",
            "meta": {
                "gaussian_dim": self.gaussian_dim,
                "rot_4d": self.rot_4d,
                "isotropic_gaussians": self.isotropic_gaussians,
                "force_sh_3d": self.force_sh_3d,
                "max_sh_degree": self.max_sh_degree,
                "active_sh_degree": self.active_sh_degree,
                "max_sh_degree_t": self.max_sh_degree_t,
                "active_sh_degree_t": self.active_sh_degree_t,
                "time_duration": list(self.time_duration),
                "prefilter_var": self.prefilter_var,
                "attr_bits": int(attr_bits),
                "xyz_quant": xyz_meta,
                "spatial_lr_scale": self.spatial_lr_scale,
                "num_points": int(self.get_xyz.shape[0]),
            },
            "xyz": compress_gpcc(xyz_q_sorted.cpu()),
            "attr": {},
        }

        if (
            self.gaussian_dim == 4
            and self.env_map is not None
            and self.env_map.numel() > 0
        ):
            save_dict["env_map"] = self.env_map.detach().cpu()

        for name, tensor in self._compressible_tensors().items():
            save_dict["attr"][name] = self._encode_scalar_q_huffman(
                tensor[sort_idx], num_bits=attr_bits
            )

        return save_dict

    def restore_compressed(self, save_dict, training_args=None, device: str = "cuda"):
        assert save_dict["format"] == "gaussian-compressed-v1"
        meta = save_dict["meta"]

        assert self.gaussian_dim == meta["gaussian_dim"]
        assert self.rot_4d == meta["rot_4d"]
        assert self.isotropic_gaussians == meta["isotropic_gaussians"]
        assert self.force_sh_3d == meta["force_sh_3d"]
        assert self.max_sh_degree == meta["max_sh_degree"]
        assert self.max_sh_degree_t == meta["max_sh_degree_t"]

        self.active_sh_degree = meta["active_sh_degree"]
        self.active_sh_degree_t = meta.get("active_sh_degree_t", 0)
        self.time_duration = meta.get("time_duration", self.time_duration)
        self.prefilter_var = meta.get("prefilter_var", self.prefilter_var)
        self.spatial_lr_scale = meta.get("spatial_lr_scale", self.spatial_lr_scale)

        xyz_q = decompress_gpcc(save_dict["xyz"]).to(device=device, dtype=torch.int32)
        sort_idx = calculate_morton_order(xyz_q.int())
        xyz_q = xyz_q[sort_idx]
        xyz = self._dequantize_xyz_u16(xyz_q, meta["xyz_quant"])

        attrs = {}
        for name, pack in save_dict["attr"].items():
            attrs[name] = self._decode_scalar_q_huffman(pack, device=device)

        n_pts = xyz.shape[0]
        assert (
            n_pts == meta["num_points"]
        ), f"Decoded xyz count mismatch: {n_pts} vs {meta['num_points']}"
        for name, t in attrs.items():
            assert (
                t.shape[0] == n_pts
            ), f"Attribute {name} has {t.shape[0]} rows, expected {n_pts}"

        self._xyz = nn.Parameter(xyz.contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(
            attrs["features_dc"].contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            attrs["features_rest"].contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(attrs["scaling"].contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(attrs["opacity"].contiguous().requires_grad_(True))

        if self.isotropic_gaussians:
            self._rotation = torch.empty((0, 4), device=device, dtype=self._xyz.dtype)
        else:
            self._rotation = nn.Parameter(
                attrs["rotation"].contiguous().requires_grad_(True)
            )

        if self.gaussian_dim == 4:
            self._t = nn.Parameter(attrs["t"].contiguous().requires_grad_(True))
            self._scaling_t = nn.Parameter(
                attrs["scaling_t"].contiguous().requires_grad_(True)
            )
            if self.rot_4d:
                if self.isotropic_gaussians:
                    self._rotation_r = torch.empty(
                        (0, 4), device=device, dtype=self._xyz.dtype
                    )
                else:
                    self._rotation_r = nn.Parameter(
                        attrs["rotation_r"].contiguous().requires_grad_(True)
                    )

        env_map = save_dict.get("env_map")
        if env_map is None:
            self.env_map = torch.empty(0, device=device)
        else:
            self.env_map = env_map.to(device)

        self.max_radii2D = torch.zeros((n_pts), device=device)
        self.xyz_gradient_accum = torch.zeros((n_pts, 1), device=device)
        self.denom = torch.zeros((n_pts, 1), device=device)
        if self.gaussian_dim == 4:
            self.t_gradient_accum = torch.zeros((n_pts, 1), device=device)

        self.optimizer = None
        if training_args is not None:
            self.training_setup(training_args)

    def save_compressed(self, path, attr_bits: int = 8):
        parent = os.path.dirname(path)
        if parent:
            mkdir_p(parent)
        torch.save(self.capture_compressed(attr_bits=attr_bits), path)

    def load_compressed(self, path, training_args=None, device: str = "cuda"):
        save_dict = torch.load(path, map_location="cpu")
        self.restore_compressed(save_dict, training_args=training_args, device=device)
