# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn).
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def get_heat_value(pose_coord, heatmap):
    _, h, w = heatmap.shape
    heatmap_nocenter = heatmap[:-1].flatten(1,2).transpose(0,1)

    y_b = torch.clamp(torch.floor(pose_coord[:,:,1]), 0, h-1).long()
    x_l = torch.clamp(torch.floor(pose_coord[:,:,0]), 0, w-1).long()
    heatval = torch.gather(heatmap_nocenter, 0, y_b * w + x_l).unsqueeze(-1)
    return heatval


def cal_area_2_torch(v):
    w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
    h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
    return w * w + h * h


def nms_core(cfg, pose_coord, heat_score):
    num_people, num_joints, _ = pose_coord.shape
    pose_area = cal_area_2_torch(pose_coord)[:,None].repeat(1,num_people*num_joints)
    pose_area = pose_area.reshape(num_people,num_people,num_joints)
    
    pose_diff = pose_coord[:, None, :, :] - pose_coord
    pose_diff.pow_(2)
    pose_dist = pose_diff.sum(3)
    pose_dist.sqrt_()
    pose_thre = cfg.TEST.NMS_THRE * torch.sqrt(pose_area)
    pose_dist = (pose_dist < pose_thre).sum(2)
    nms_pose = pose_dist > cfg.TEST.NMS_NUM_THRE

    ignored_pose_inds = []
    keep_pose_inds = []
    for i in range(nms_pose.shape[0]):
        if i in ignored_pose_inds:
            continue
        keep_inds = nms_pose[i].nonzero().cpu().numpy()
        keep_inds = [list(kind)[0] for kind in keep_inds]
        keep_scores = heat_score[keep_inds]
        ind = torch.argmax(keep_scores)
        keep_ind = keep_inds[ind]
        if keep_ind in ignored_pose_inds:
            continue
        keep_pose_inds += [keep_ind]
        ignored_pose_inds += list(set(keep_inds)-set(ignored_pose_inds))

    return keep_pose_inds


def pose_nms(cfg, heatmap_avg, poses):
    """
    NMS for the regressed poses results.

    Args:
        heatmap_avg (Tensor): Avg of the heatmaps at all scales (1, 1+num_joints, w, h)
        poses (List[Tensor]): pose proposals per scale [(num_people, num_joints, 3)]
    """
    # pick the scale "1.0" as reference
    scale1_index = sorted(cfg.TEST.SCALE_FACTOR, reverse=True).index(1.0)
    pose_norm = poses[scale1_index]
    max_score = pose_norm[:, :, 2].max() if pose_norm.shape[0] else 1

    # normalize scores across scales
    for i, pose in enumerate(poses):
        if i != scale1_index:
            max_score_scale = pose[:, :, 2].max() if pose.shape[0] else 1
            pose[:, :, 2] = pose[:, :, 2] / (max_score_scale if max_score_scale > 0 else 1) \
                            * max_score * cfg.TEST.DECREASE

    # concat coords/scores across scales (still on device for heat operations)
    pose_score = torch.cat([pose[:, :, 2:] for pose in poses], dim=0)   # (N, J, 1)
    pose_coord = torch.cat([pose[:, :, :2] for pose in poses], dim=0)   # (N, J, 2)

    if pose_coord.shape[0] == 0:
        return [], []

    num_people, num_joints, _ = pose_coord.shape

    # heat value per joint/person for scoring
    heatval = get_heat_value(pose_coord, heatmap_avg[0])                # (N, J, 1), same device as heatmap
    heat_score = (torch.sum(heatval, dim=1) / num_joints)[:, 0]         # (N,)

    # we will keep a CPU copy of poses for JSON/NumPy later, but keep heat_score on its device
    poses_cpu = torch.cat([pose_coord.detach().cpu(),
                           (pose_score * heatval).detach().cpu()], dim=2)  # (N, J, 3) on CPU

    # NMS on the device of pose_coord/heat_score
    keep_pose_inds = nms_core(cfg, pose_coord, heat_score)  # list[int]
    if len(keep_pose_inds) == 0:
        return [], []

    keep_pose_inds_t = torch.as_tensor(keep_pose_inds, dtype=torch.long, device=heat_score.device)
    # filter heat_score on its device, and poses on CPU with CPU indices
    heat_score = heat_score.index_select(0, keep_pose_inds_t)
    poses_cpu = poses_cpu.index_select(0, keep_pose_inds_t.detach().cpu())

    # limit max people with top-k (be careful with devices!)
    if heat_score.numel() > cfg.DATASET.MAX_NUM_PEOPLE:
        heat_score, topk_inds = torch.topk(heat_score, cfg.DATASET.MAX_NUM_PEOPLE)
        poses_cpu = poses_cpu.index_select(0, topk_inds.detach().cpu())

    poses_list = [poses_cpu.numpy()]
    scores = [p[:, 2].mean() for p in poses_list[0]]
    return poses_list, scores
