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
import random
import torch
from torch import nn
from utils.loss_utils import l1_loss, ssim, msssim, lpips as lpips_metric
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, knn
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
from utils.uncertainty import compute_uncertainty_all_frames
from utils.graph import build_graph, USplat4DGraph
from utils.usplat_losses import key_node_loss, non_key_node_loss

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def save_gaussian_args(model_path, gaussian_kwargs):
    with open(os.path.join(model_path, "gaussian_args"), "w") as f:
        f.write(str(Namespace(**gaussian_kwargs)))


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
        for t_idx, (_, cam) in enumerate(training_dataset):
            cam = cam.cuda(non_blocking=True, copy=False)
            _, delta = gaussians.get_current_covariance_and_mean_offset(
                1.0, cam.timestamp
            )
            means_t_all[:, t_idx] = gaussians.get_xyz + delta

        graph = build_graph(
            means_t=means_t_all,
            u_scalar=u_scalar,
            key_ratio=0.02,
            spt_threshold=5,
            knn_k=8,
            device=background.device,
        )
        p_pretrained = gaussians.get_xyz.detach().clone()

    print(
        f"[USplat4D] Graph rebuilt: {graph.num_key} key nodes, "
        f"{graph.num_nonkey} non-key nodes"
    )
    return u_scalar, graph, p_pretrained


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size, isotropic_gaussians,
             use_usplat=False, usplat_start_iter=10000):
    
    if dataset.frame_ratio > 1:
        time_duration = [
            time_duration[0] / dataset.frame_ratio,
            time_duration[1] / dataset.frame_ratio,
        ]

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussian_kwargs = {
        "sh_degree": dataset.sh_degree,
        "gaussian_dim": gaussian_dim,
        "time_duration": time_duration,
        "rot_4d": rot_4d,
        "force_sh_3d": force_sh_3d,
        "sh_degree_t": 2 if pipe.eval_shfs_4d else 0,
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

    gaussians.training_setup(opt)
    
    # USplat4D setup
    graph = None
    p_pretrained = None
    u_scalar = None
    usplat_state_dirty = False
    
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

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

    if opt.final_prune_from_iter >= 0 and final_prune_ratio > 0.0 and gaussian_dim != 4:
        raise ValueError("Final pruning is ST-pruning and requires gaussian_dim == 4")

    num_workers = 12 if dataset.dataloader else 0
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": lambda x: x,
        "drop_last": True,
        "pin_memory": dataset.dataloader,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2
    training_dataloader = DataLoader(training_dataset, **dataloader_kwargs)

    iteration = first_iter
    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
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
                u_scalar, graph, p_pretrained = rebuild_usplat_state(
                    gaussians=gaussians,
                    training_dataset=training_dataset,
                    pipe=pipe,
                    background=background,
                    opt=opt,
                    iteration=iteration,
                )
                usplat_state_dirty = False

            batch_Ll1 = torch.tensor(0.0, device="cuda")
            batch_Lssim = torch.tensor(0.0, device="cuda")
            batch_loss = torch.tensor(0.0, device="cuda")
            batch_Ldepth = torch.tensor(0.0, device="cuda")
            batch_Lopa_mask = torch.tensor(0.0, device="cuda")
            batch_Lrigid = torch.tensor(0.0, device="cuda")
            batch_Lmotion = torch.tensor(0.0, device="cuda")
            batch_psnr = torch.tensor(0.0, device="cuda")
            reg_loss = torch.tensor(0.0, device="cuda")
            Ldepth = torch.tensor(0.0, device="cuda")
            Lopa_mask = torch.tensor(0.0, device="cuda")
            Lrigid = torch.tensor(0.0, device="cuda")
            Lmotion = torch.tensor(0.0, device="cuda")
            
            Lkey = torch.tensor(0.0, device=background.device)
            Lnon_key = torch.tensor(0.0, device=background.device)

            for batch_idx in range(batch_size):
                gt_image, viewpoint_cam = batch_data[batch_idx]
                gt_image = gt_image.to(background.device, non_blocking=True)
                viewpoint_cam = viewpoint_cam.cuda(non_blocking=True, copy=False)

                render_pkg = render(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    background,
                    apply_random_dropout=True,
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
                
                # --- Compute USplat losses every iteration after graph state is ready ---
                if use_usplat and graph is not None and iteration >= usplat_start_iter:

                    # Density control schedule
                    total_usplat = opt.iterations - usplat_start_iter
                    in_first_10 = iteration < usplat_start_iter + int(0.1 * total_usplat)
                    in_last_20  = iteration > opt.iterations - int(0.2 * total_usplat)
                    if in_first_10 or in_last_20:
                        should_densify = False

                    with torch.no_grad():
                        _, delta = gaussians.get_current_covariance_and_mean_offset(
                            1.0, viewpoint_cam.timestamp
                        )
                        means_current = gaussians.get_xyz + delta  # (G, 3)

                    N_k = graph.num_key
                    N_n = graph.num_nonkey

                    # Current positions — shape (N, B, 3) with B=1
                    pos_key_t = means_current[graph.key_idx].unsqueeze(1)      # (N_k, 1, 3)
                    pos_nk_t  = means_current[graph.nonkey_idx].unsqueeze(1)   # (N_n, 1, 3)

                    # Pretrained positions — same shape
                    pos_key_pre = p_pretrained[graph.key_idx].unsqueeze(1)     # (N_k, 1, 3)
                    pos_nk_pre  = p_pretrained[graph.nonkey_idx].unsqueeze(1)  # (N_n, 1, 3)

                    # Canonical positions (no time dim)
                    pos_o_key = p_pretrained[graph.key_idx]                    # (N_k, 3)
                    pos_o_nk  = p_pretrained[graph.nonkey_idx]                 # (N_n, 3)

                    # Identity quaternions for isotropic Gaussians
                    quats_key = torch.zeros(N_k, 1, 4, device=background.device)
                    quats_key[..., 0] = 1.0
                    quats_nk  = torch.zeros(N_n, 1, 4, device=background.device)
                    quats_nk[..., 0] = 1.0

                    # SE(3) transforms: identity rotation + current position as translation
                    def make_transforms(pos):
                        # pos: (N, 1, 3) → transforms: (N, 1, 3, 4)
                        N = pos.shape[0]
                        T_mat = torch.zeros(N, 1, 3, 4, device=background.device)
                        T_mat[:, :, 0, 0] = 1.0
                        T_mat[:, :, 1, 1] = 1.0
                        T_mat[:, :, 2, 2] = 1.0
                        T_mat[:, :, :, 3] = pos[:, :, :]
                        return T_mat

                    transforms_key = make_transforms(pos_key_t)   # (N_k, 1, 3, 4)
                    transforms_nk  = make_transforms(pos_nk_t)    # (N_n, 1, 3, 4)

                    # Uncertainty — mean over time, shape (N, 1)
                    u_key = u_scalar[graph.key_idx].mean(dim=1, keepdim=True)    # (N_k, 1)
                    u_nk  = u_scalar[graph.nonkey_idx].mean(dim=1, keepdim=True) # (N_n, 1)

                    # Camera rotation — shape (B, 3, 3) with B=1
                    w2c = viewpoint_cam.world_view_transform        # (4, 4)
                    R_wc_t = w2c[:3, :3].T.unsqueeze(0)            # (1, 3, 3)

                    # Key rotations and translations for DQB — shape (N_k, B, 3, 3) and (N_k, B, 3)
                    R_key_t = transforms_key[:, :, :, :3]          # (N_k, 1, 3, 3)
                    t_key_t = transforms_key[:, :, :, 3]           # (N_k, 1, 3)

                    # --- Key node loss ---
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

                    # --- Non-key node loss ---
                    if N_n > 0:
                        Lnon_key = non_key_node_loss(
                            pos_nk_t=pos_nk_t,
                            quats_nk_t=quats_nk,
                            transforms_nk_t=transforms_nk,
                            pos_nk_pretrained=pos_nk_pre,
                            u_nk=u_nk,
                            R_wc_t=R_wc_t,
                            pos_o_nk=pos_o_nk,
                            R_key_t=R_key_t,
                            t_key_t=t_key_t,
                            pos_key_t=pos_key_t,
                            quats_key_t=quats_key,
                            transforms_key_t=transforms_key,
                            pos_o_key=pos_o_key,
                            nonkey_nbrs_local=graph.nonkey_nbrs,
                            nonkey_nbr_weights=graph.nonkey_nbr_weights,
                            nonkey_nbrs_global=graph.nonkey_nbrs,
                        )
                    else:
                        Lnon_key = torch.tensor(0.0, device=background.device)

                    loss = loss + opt.lambda_key * Lkey + opt.lambda_non_key * Lnon_key
                
                batch_Ll1 += Ll1_i.detach()
                batch_Lssim += Lssim_i.detach()
                batch_psnr += psnr(image, gt_image).mean().detach()

                loss = loss / batch_size
                batch_loss += loss.detach()
                loss.backward()
                if should_densify:
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
                batch_loss += reg_loss.detach()
                if reg_loss.requires_grad:
                    reg_loss.backward()

            Ll1 = batch_Ll1 / batch_size
            Lssim = batch_Lssim / batch_size
            loss = batch_loss
            if opt.lambda_depth > 0:
                Ldepth = batch_Ldepth / batch_size
            if opt.lambda_opa_mask > 0:
                Lopa_mask = batch_Lopa_mask / batch_size
            if opt.lambda_rigid > 0:
                Lrigid = batch_Lrigid
            if opt.lambda_motion > 0:
                Lmotion = batch_Lmotion
            if should_densify:
                if batch_size > 1:
                    visibility_count = torch.stack(batch_visibility_filter, 1).sum(1)
                    visibility_filter = visibility_count > 0
                    radii = torch.stack(batch_radii, 1).max(1)[0]

                    batch_viewspace_point_grad = torch.stack(batch_point_grad, 1).sum(1)
                    batch_viewspace_point_grad[visibility_filter] = (
                        batch_viewspace_point_grad[visibility_filter]
                        * batch_size
                        / visibility_count[visibility_filter]
                    )
                    batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)

                    if gaussians.gaussian_dim == 4:
                        batch_t_grad = gaussians._t.grad.clone()[:, 0].detach()
                        batch_t_grad[visibility_filter] = (
                            batch_t_grad[visibility_filter]
                            * batch_size
                            / visibility_count[visibility_filter]
                        )
                        batch_t_grad = batch_t_grad.unsqueeze(1)
                else:
                    if gaussians.gaussian_dim == 4:
                        batch_t_grad = gaussians._t.grad.clone().detach()

            loss_dict = {"Ll1": Ll1, "Lssim": Lssim}
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
                    psnr_for_log = (batch_psnr / batch_size).item()
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
                        torch.save(
                            (gaussians.capture(), iteration),
                            scene.model_path + "/chkpnt_best.pth",
                        )

                if iteration in saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if should_densify:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    if batch_size == 1:
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
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            opt.thresh_opa_prune,
                            scene.cameras_extent,
                            size_threshold,
                            opt.densify_grad_t_threshold,
                        )
                        if use_usplat and iteration >= usplat_start_iter:
                            graph = None
                            p_pretrained = None
                            u_scalar = None
                            usplat_state_dirty = True

                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
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
                    print(
                        f"[ITER {iteration}] Pruning Gaussians with ratio {opt.spatio_temporal_pruning_ratio}"
                    )
                    gaussians.prune_with_spatio_temporal_score(
                        spatio_temporal_scores,
                        opt.spatio_temporal_pruning_ratio,
                        reset_optimizer_state=True,
                        probabilistic=getattr(
                            opt, "spatio_temporal_pruning_random", False
                        ),
                    )
                    if use_usplat and iteration >= usplat_start_iter:
                        graph = None
                        p_pretrained = None
                        u_scalar = None
                        usplat_state_dirty = True

                if should_final_st_prune:
                    print(
                        f"\n[ITER {iteration}] Computing final spatio-temporal pruning scores"
                    )
                    final_spatio_temporal_scores = (
                        gaussians.compute_spatio_temporal_variation_score(
                            scene, pipe, background, render
                        )
                    )
                    print(
                        f"[ITER {iteration}] Final ST-pruning Gaussians with ratio {final_prune_ratio}"
                    )
                    gaussians.prune_with_spatio_temporal_score(
                        final_spatio_temporal_scores,
                        final_prune_ratio,
                        reset_optimizer_state=True,
                        probabilistic=getattr(
                            opt, "spatio_temporal_pruning_random", False
                        ),
                    )
                    if use_usplat and iteration >= usplat_start_iter:
                        graph = None
                        p_pretrained = None
                        u_scalar = None
                        usplat_state_dirty = True
                    scene.save(iteration)

                if use_usplat and iteration >= usplat_start_iter and usplat_state_dirty:
                    u_scalar, graph, p_pretrained = rebuild_usplat_state(
                        gaussians=gaussians,
                        training_dataset=training_dataset,
                        pipe=pipe,
                        background=background,
                        opt=opt,
                        iteration=iteration,
                    )
                    usplat_state_dirty = False


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
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
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
    parser.add_argument("--gaussian_dim", type=int, default=3)
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

    # Load config and use it as parser defaults
    if pre_args.config is not None:
        cfg = OmegaConf.load(pre_args.config)
        cfg_defaults = flatten_cfg(cfg)

        for k in cfg_defaults:
            if not hasattr(pre_args, k):
                raise ValueError(f"Unknown config key: {k}")

        parser.set_defaults(**cfg_defaults)

    # Second pass: real parse, CLI now overrides config
    args = parser.parse_args(sys.argv[1:])
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
             args.use_usplat, args.usplat_start_iter)

    print("\nTraining complete.")
