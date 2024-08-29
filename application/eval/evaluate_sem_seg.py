import json
import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import open3d as o3d
import open_clip
import plyfile
from scipy.spatial import cKDTree
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree
import torch
import torchmetrics as tm

from hovsg.labels.label_constants import SCANNET_COLOR_MAP_20, SCANNET_LABELS_20
from hovsg.utils.eval_utils import (
    load_feature_map,
    knn_interpolation,
    read_ply_and_assign_colors,
    read_semantic_classes,
    sim_2_label,
    read_semantic_classes_replica,
    text_prompt,
    read_ply_and_assign_colors_replica
)
from hovsg.utils.metric import (
    frequency_weighted_iou,
    mean_iou,
    mean_accuracy,
    pixel_accuracy,
    per_class_iou,
)

@hydra.main(version_base=None, config_path="../../config", config_name="eval_sem_seg")
def main(params: DictConfig):

    # load CLIP model
    if params.models.clip.type == "ViT-L/14@336px":
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained=str(params.models.clip.checkpoint),
            device=params.main.device,
        )
        clip_feat_dim = 768
    elif params.models.clip.type == "ViT-H-14":
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained=str(params.models.clip.checkpoint),
            device=params.main.device,
        )
        clip_feat_dim = 1024
    clip_model.eval()

    # Load Feature Map
    masked_pcd, mask_feats = load_feature_map(params.main.feature_map_path)
    # read semantic classes
    scene_name = params.main.scene_name
    if params.main.dataset == "scannet":
        SCANNET_LABELS_20_list = list(SCANNET_LABELS_20)
        labels = SCANNET_LABELS_20_list
        labels_id = list(SCANNET_COLOR_MAP_20.keys())
    elif params.main.dataset == "replica":
        semantic_info_path = os.path.join(
            params.main.replica_dataset_gt_path, scene_name, "habitat", "info_semantic.json"
        )
        class_id_name = read_semantic_classes_replica(semantic_info_path, crete_color_map=True)
        # add background class with id len(class_id_name)+1
        class_id_name[0] = "background"
        labels = list(class_id_name.values())
        labels_id = list(class_id_name.keys())

    sim = text_prompt(clip_model, clip_feat_dim, mask_feats, labels, templates=True)
    labels = sim_2_label(sim, labels_id)
    labels = np.array(labels)

    # create a new pcd from the labeld pcd masks
    pcd = o3d.geometry.PointCloud()
    if params.main.dataset == "scannet":
        colors = np.array([SCANNET_COLOR_MAP_20[i] for i in labels]) / 255.0
    elif params.main.dataset == "replica":
        # assign color based on labels id
        colors_map = {}
        with open(params.main.replica_color_map, "r") as f:
            colors_map = json.load(f)
        colors_map = {int(k): v for k, v in colors_map.items()}
        # create colors for labels based on colors_map
        colors = np.zeros((len(labels), 3))
        for i, label in enumerate(labels):
            colors[i] = colors_map[label]

    ## FOR MASK BASED SEGMENTATION ##
    for i in range(len(masked_pcd)):
        pcd += masked_pcd[i].paint_uniform_color(colors[i])
    # o3d.io.write_point_cloud(os.path.join(save_dir, "pred_pcd.ply"), pcd)

    # load ground truth pcd
    if params.main.dataset == "scannet":
        pcd_gt = o3d.io.read_point_cloud(
            os.path.join(params.main.scannet_dataset_gt_path, scene_name, f"{scene_name}_vh_clean_2.labels.ply")
        )
    elif params.main.dataset == "replica":
        ply_path = os.path.join(params.main.replica_dataset_gt_path, scene_name, "habitat", "mesh_semantic.ply")
        gt_pcd, gt_labels, gt_instance_pcd, gt_instance_id = read_ply_and_assign_colors_replica(
            ply_path, semantic_info_path
        )
        # save the gt pcd using same colors as predicted pcd
        gt_pcd.points = o3d.utility.Vector3dVector(np.asarray(gt_pcd.points))
        # assing colors to gt pcd based on labels
        colors = np.zeros((len(gt_labels), 3))
        for i, label in enumerate(gt_labels):
            colors[i] = colors_map[label]
        gt_pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.io.write_point_cloud(os.path.join(save_dir, "gt_pcd.ply"), gt_pcd)

    if params.main.dataset == "scannet":
        # create labels_pred
        label_pre = np.zeros((len(pcd.points), 1))
        for i in range(len(pcd.points)):
            # find the color of the point
            color = np.asarray(pcd.colors[i]) * 255.0
            # find the index of the color in the color map
            color_map_array = np.array(list(SCANNET_COLOR_MAP_20.values()))
            color_diff = np.sum(np.abs(color_map_array - color), axis=1)
            min_diff_index = np.argmin(color_diff)
            # find the label of the point
            label = np.array(list(SCANNET_COLOR_MAP_20.keys()))[min_diff_index]
            label_pre[i] = label
        label_pred = torch.tensor(label_pre)
        # create labels_gt
        labels_gt = np.zeros((len(pcd_gt.points), 1))
        for i in range(len(pcd_gt.points)):
            # find the color of the point
            color = np.asarray(pcd_gt.colors[i]) * 255.0
            # find the index of the color in the color map
            color_map_array = np.array(list(SCANNET_COLOR_MAP_20.values()))
            color_diff = np.sum(np.abs(color_map_array - color), axis=1)
            min_diff_index = np.argmin(color_diff)
            # find the label of the point
            label = np.array(list(SCANNET_COLOR_MAP_20.keys()))[min_diff_index]
            labels_gt[i] = label
        labels_gt = torch.tensor(labels_gt)
        # concat coords and labels for predicied pcd
        coords_labels = np.zeros((len(pcd.points), 4))
        coords_labels[:, :3] = np.asarray(pcd.points)
        coords_labels[:, -1] = label_pred[:, 0]
        coords_gt = np.zeros((len(pcd_gt.points), 4))
        coords_gt[:, :3] = np.asarray(pcd_gt.points)
        coords_gt[:, -1] = labels_gt[:, 0]
        match_pc = knn_interpolation(coords_labels, coords_gt, k=5)
        label_pred = match_pc[:, -1].reshape(-1, 1)
        labels_gt = labels_gt.numpy()

    elif params.main.dataset == "replica":
        pred_labels = []
        for i in range(len(masked_pcd)):
            pred_labels.append(np.repeat(labels[i], len(masked_pcd[i].points)))
        pred_labels = np.hstack(pred_labels)

        pred_labels = pred_labels.reshape(-1, 1)
        gt_labels = gt_labels.reshape(-1, 1)

        # concat coords and labels for predicied pcd
        coords_labels = np.zeros((len(pcd.points), 4))
        coords_labels[:, :3] = np.asarray(pcd.points)
        coords_labels[:, -1] = pred_labels[:, 0]
        # concat coords and labels for gt pcd
        coords_gt = np.zeros((len(gt_pcd.points), 4))
        coords_gt[:, :3] = np.asarray(gt_pcd.points)
        coords_gt[:, -1] = gt_labels[:, 0]
        # knn interpolation
        match_pc = knn_interpolation(coords_labels, coords_gt, k=5)
        pred_labels = match_pc[:, -1].reshape(-1, 1)
        ## MATCHING ##
        labels_gt = gt_labels
        label_pred = pred_labels
        assert len(labels_gt) == len(pred_labels)

    # show predicted pcd
    # o3d.io.write_point_cloud(os.path.join(save_dir, "predicted_pcd.ply"), pcd)

    # print number of unique labels in the ground truth and predicted pointclouds
    print("Number of unique labels in the GT pcd: ", len(np.unique(labels_gt)))
    print("Number of unique labels in the pred pcd ", len(np.unique(label_pred)))

    # openscene evaluation
    if params.main.dataset == "scannet":
        ignore = [0, 1, 7, 8, 20]
    elif params.main.dataset == "replica":
        ignore = [-1, 0]
        for id, name in class_id_name.items():
            if (
                "wall" in name
                or "floor" in name
                or "ceiling" in name
                or "door" in name
                or "window" in name
                or "background" in name
            ):
                ignore.append(id)
    print("################ {} ################".format(scene_name))
    ious = per_class_iou(label_pred, labels_gt, ignore=ignore)
    print("per class iou: ", ious)
    miou = mean_iou(label_pred, labels_gt, ignore=ignore)
    print("miou: ", miou)
    fmiou = frequency_weighted_iou(label_pred, labels_gt, ignore=ignore)
    print("fmiou: ", fmiou)
    macc = mean_accuracy(label_pred, labels_gt, ignore=ignore)
    print("macc: ", macc)
    pacc = pixel_accuracy(label_pred, labels_gt, ignore=ignore)
    print("pacc: ", pacc)
    print("#######################################")

if __name__ == "__main__":
    main()
