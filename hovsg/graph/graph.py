"""
    Class to represent the HOV-SG graph
"""

import os
import copy
from typing import Any, Dict, List, Set, Tuple, Union
from pathlib import Path

import cv2
from omegaconf import DictConfig
import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from scipy.ndimage import binary_erosion, median_filter
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d

from sklearn.cluster import DBSCAN
from tqdm import tqdm
import networkx as nx

import torch
import open_clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from hovsg.graph.object import Object
from hovsg.graph.room import Room
from hovsg.graph.floor import Floor

from hovsg.dataloader.hm3dsem import HM3DSemDataset
from hovsg.dataloader.scannet import ScannetDataset
from hovsg.dataloader.replica import ReplicaDataset

from hovsg.utils.clip_utils import get_img_feats, get_text_feats_multiple_templates
from hovsg.models.sam_clip_feats_extractor import extract_feats_per_pixel
from hovsg.utils.graph_utils import (
    seq_merge,
    pcd_denoise_dbscan,
    feats_denoise_dbscan,
    distance_transform,
    map_grid_to_point_cloud,
    compute_room_embeddings,
    find_intersection_share,
    hierarchical_merge,
)

from hovsg.graph.navigation_graph import NavigationGraph
from hovsg.utils.label_feats import get_label_feats
from hovsg.utils.constants import MATTERPORT_GT_LABELS, CLIP_DIM
from hovsg.utils.llm_utils import (
    parse_floor_room_object_gpt35,
    parse_hier_query,
    infer_floor_id_from_query,
)

# pylint: disable=all


class Graph:
    """
    Class to represent the HOV-SG graph
    :param cfg: Config file
    :param inf_params: Inference parameters
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.full_pcd = o3d.geometry.PointCloud()
        self.mask_feats = []
        self.mask_feats_d = []
        self.mask_pcds = []
        self.mask_weights = []
        self.objects = []
        self.rooms = []
        self.floors = []
        self.full_feats_array = []
        self.graph = nx.Graph()
        self.graph.add_node(0, name="building", type="building")
        self.room_masks = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load CLIP model
        if self.cfg.models.clip.type == "ViT-L/14@336px":
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14",
                pretrained=str(self.cfg.models.clip.checkpoint),
                device=self.device,
            )
            self.clip_feat_dim = CLIP_DIM["ViT-L-14"]
            # self.clip_feat_dim = constants.clip_feat_dim[self.cfg.models.clip.type]
        elif self.cfg.models.clip.type == "ViT-H-14":
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14",
                pretrained=str(self.cfg.models.clip.checkpoint),
                device=self.device,
            )
            self.clip_feat_dim = CLIP_DIM["ViT-H-14"]
        self.clip_model.eval()
        if not hasattr(self.cfg, "pipeline"):
            print("-- entering querying and evaluation mode")
            return

        self.graph_tmp_folder = os.path.join(cfg.main.save_path, "tmp")
        if not os.path.exists(self.graph_tmp_folder):
            os.makedirs(self.graph_tmp_folder)

        # load the SAM model
        model_type = self.cfg.models.sam.type
        self.sam = sam_model_registry[model_type](
            checkpoint=str(self.cfg.models.sam.checkpoint)
        )
        self.sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=self.cfg.models.sam.points_per_side,
            pred_iou_thresh=self.cfg.models.sam.pred_iou_thresh,
            points_per_batch=self.cfg.models.sam.points_per_batch,
            stability_score_thresh=self.cfg.models.sam.stability_score_thresh,
            crop_n_layers=self.cfg.models.sam.crop_n_layers,
            min_mask_region_area=self.cfg.models.sam.min_mask_region_area,
        )
        self.sam.eval()
        # load the dataset
        dataset_cfg = {"root_dir": self.cfg.main.dataset_path, "transforms": None}
        if self.cfg.main.dataset == "hm3dsem":
            self.dataset = HM3DSemDataset(dataset_cfg)
        elif self.cfg.main.dataset == "scannet":
            self.dataset = ScannetDataset(dataset_cfg)
        elif self.cfg.main.dataset == "replica":
            self.dataset = ReplicaDataset(dataset_cfg)
        else:
            print("Dataset not supported")
            return

    def create_feature_map(self, save_path=None):
        """
        Create the feature map of the HOV-SG (full point cloud + feature map point level + feature map mask level)
        :param save_path : str, optional, The path to save the feature map
        """

        if self.dataset is None:
            print("No dataset loaded")
            return

        # create the RGB-D point cloud
        for i in tqdm(range(0, len(self.dataset), self.cfg.pipeline.skip_frames), desc="Creating RGB-D point cloud"):
            rgb_image, depth_image, pose, _, depth_intrinsics = self.dataset[i]
            self.full_pcd += self.dataset.create_pcd(rgb_image, depth_image, pose)

        # filter point cloud
        self.full_pcd = self.full_pcd.voxel_down_sample(
            voxel_size=self.cfg.pipeline.voxel_size
        )
        self.full_pcd = pcd_denoise_dbscan(self.full_pcd, eps=0.01, min_points=100)

        # create tree from full point cloud
        locs_in = np.array(self.full_pcd.points)
        tree_pcd = cKDTree(locs_in)
        n_points = locs_in.shape[0]
        counter = torch.zeros((n_points, 1), device="cpu")
        sum_features = torch.zeros((n_points, self.clip_feat_dim), device="cpu")

        # extract features for each frame
        frames_pcd = []
        frames_feats = []
        for i in tqdm(range(0, len(self.dataset), self.cfg.pipeline.skip_frames), desc="Extracting features"):
            rgb_image, depth_image, pose, _, _ = self.dataset[i]
            if rgb_image.size != depth_image.size:
                rgb_image = rgb_image.resize(depth_image.size)
            F_2D, F_masks, masks, F_g = extract_feats_per_pixel(
                np.array(rgb_image),
                self.mask_generator,
                self.clip_model,
                self.preprocess,
                clip_feat_dim=self.clip_feat_dim,
                bbox_margin=self.cfg.pipeline.clip_bbox_margin,
                maskedd_weight=self.cfg.pipeline.clip_masked_weight,
            )
            F_2D = F_2D.cpu()
            pcd = self.dataset.create_pcd(rgb_image, depth_image, pose)
            masks_3d = self.dataset.create_3d_masks(
                masks,
                depth_image,
                self.full_pcd,
                tree_pcd,
                pose,
                down_size=self.cfg.pipeline.voxel_size,
                filter_distance=self.cfg.pipeline.max_mask_distance,
            )
            frames_pcd.append(masks_3d)
            frames_feats.append(F_masks)
            # fuse features for each point in the full pcd
            mask = np.array(depth_image) > 0
            mask = torch.from_numpy(mask)
            F_2D = F_2D[mask]
            # using cKdtree to find the closest point in the full pcd for each point in frame pcd
            dis, idx = tree_pcd.query(np.asarray(pcd.points), k=1, workers=-1)
            sum_features[idx] += F_2D
            counter[idx] += 1
        # compute the average features
        counter[counter == 0] = 1e-5
        sum_features = sum_features / counter
        self.full_feats_array = sum_features.cpu().numpy()
        self.full_feats_array: np.ndarray

        # merging the masks
        if self.cfg.pipeline.merge_type == "hierarchical":
            tqdm.write("Merging 3d masks hierarchically")
            self.mask_pcds = hierarchical_merge(
                frames_pcd, 
                self.cfg.pipeline.init_overlap_thresh, 
                self.cfg.pipeline.overlap_thresh_factor, 
                self.cfg.pipeline.voxel_size, 
                self.cfg.pipeline.iou_thresh,
            )
        elif self.cfg.pipeline.merge_type == "sequential":
            tqdm.write("Merging 3d masks sequentially") 
            self.mask_pcds = seq_merge(
                frames_pcd, 
                self.cfg.pipeline.init_overlap_thresh, 
                self.cfg.pipeline.voxel_size, 
                self.cfg.pipeline.iou_thresh
            )

        # remove any small pcds
        for i, pcd in enumerate(self.mask_pcds):
            if pcd.is_empty() or len(pcd.points) < 100:
                self.mask_pcds.pop(i)
        # fuse point features in every 3d mask
        masks_feats = []
        for i, mask_3d in tqdm(enumerate(self.mask_pcds), desc="Fusing features"):
            # find the points in the mask
            mask_3d = mask_3d.voxel_down_sample(self.cfg.pipeline.voxel_size * 2)
            points = np.asarray(mask_3d.points)
            dist, idx = tree_pcd.query(points, k=1, workers=-1)
            feats = self.full_feats_array[idx]
            feats = np.nan_to_num(feats)
            # filter feats with dbscan
            if feats.shape[0] == 0:
                masks_feats.append(
                    np.zeros((1, self.clip_feat_dim), dtype=self.full_feats_array.dtype)
                )
                continue
            feats = feats_denoise_dbscan(feats, eps=0.01, min_points=100)
            masks_feats.append(feats)
        self.mask_feats = masks_feats
        print("number of masks: ", len(self.mask_feats))
        print("number of pcds in hovsg: ", len(self.mask_pcds))
        assert len(self.mask_pcds) == len(self.mask_feats)

    def segment_floors(self, path, flip_zy=False):
        """
        Segment the floors from the full point cloud
        :param path: str, The path to save the intermediate results
        """
        # downsample the point cloud
        downpcd = self.full_pcd.voxel_down_sample(voxel_size=0.05)
        # flip the z and y axis
        if flip_zy:
            downpcd.points = o3d.utility.Vector3dVector(
                np.array(downpcd.points)[:, [0, 2, 1]]
            )
            downpcd.transform(np.eye(4) * np.array([1, 1, -1, 1]))
        # rotate the point cloud to align floor with the y axis
        T1 = np.eye(4)
        T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        downpcd = np.asarray(downpcd.points)
        print("downpcd", downpcd.shape)

        # divide z axis range into 0.01m bin
        reselotion = 0.01
        bins = np.abs(np.max(downpcd[:, 1]) - np.min(downpcd[:, 1])) / reselotion
        print("bins", bins)
        z_hist = np.histogram(downpcd[:, 1], bins=int(bins))
        # smooth the histogram
        z_hist_smooth = gaussian_filter1d(z_hist[0], sigma=2)
        # Find the peaks in this histogram.
        distance = 0.2 / reselotion
        print("distance", distance)
        # set the min peak height based on the histogram
        print(np.mean(z_hist_smooth))
        min_peak_height = np.percentile(z_hist_smooth, 90)
        print("min_peak_height", min_peak_height)
        peaks, _ = find_peaks(z_hist_smooth, distance=distance, height=min_peak_height)

        # plot the histogram
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(
                min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r"
            )
            plt.savefig(os.path.join(self.graph_tmp_folder, "floor_histogram.png"))

        # cluster the peaks using DBSCAN
        peaks_locations = z_hist[1][peaks]
        clustering = DBSCAN(eps=1, min_samples=1).fit(peaks_locations.reshape(-1, 1))
        labels = clustering.labels_

        # plot the histogram
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(
                min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r"
            )
            # plot the clusters
            for i in range(len(np.unique(labels))):
                plt.plot(
                    z_hist[1][peaks[labels == i]],
                    z_hist_smooth[peaks[labels == i]],
                    "o",
                )
            plt.savefig(
                os.path.join(self.graph_tmp_folder, "floor_histogram_cluster.png")
            )

        # for each cluster find the top 2 peaks
        clustred_peaks = []
        for i in range(len(np.unique(labels))):
            # for first and last cluster, find the top 1 peak
            if i == 0 or i == len(np.unique(labels)) - 1:
                p = peaks[labels == i]
                top_p = p[np.argsort(z_hist_smooth[p])[-1:]].tolist()
                top_p = [z_hist[1][p] for p in top_p]
                clustred_peaks.append(top_p)
                continue
            p = peaks[labels == i]
            top_p = p[np.argsort(z_hist_smooth[p])[-2:]].tolist()
            top_p = [z_hist[1][p] for p in top_p]
            clustred_peaks.append(top_p)
        clustred_peaks = [item for sublist in clustred_peaks for item in sublist]
        clustred_peaks = np.sort(clustred_peaks)
        print("clustred_peaks", clustred_peaks)

        floors = []
        # for every two consecutive peaks with 2m distance, assign floor level
        for i in range(0, len(clustred_peaks) - 1, 2):
            floors.append([clustred_peaks[i], clustred_peaks[i + 1]])
        print("floors", floors)
        # for the first floor extend the floor to the ground
        floors[0][0] = (floors[0][0] + np.min(downpcd[:, 1])) / 2
        # for the last floor extend the floor to the ceiling
        floors[-1][1] = (floors[-1][1] + np.max(downpcd[:, 1])) / 2
        print("number of floors: ", len(floors))

        floors_pcd = []
        for i, floor in enumerate(floors):
            floor_obj = Floor(str(i), name="floor_" + str(i))
            floor_pcd = self.full_pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(-np.inf, floor[0], -np.inf),
                    max_bound=(np.inf, floor[1], np.inf),
                )
            )
            bbox = floor_pcd.get_axis_aligned_bounding_box()
            floor_obj.vertices = np.asarray(bbox.get_box_points())
            floor_obj.pcd = floor_pcd
            floor_obj.floor_zero_level = np.min(np.array(floor_pcd.points)[:, 1])
            floor_obj.floor_height = floor[1] - floor_obj.floor_zero_level
            self.floors.append(floor_obj)
            floors_pcd.append(floor_pcd)
        return floors

    def segment_rooms(self, floor: Floor, path):
        """
        Segment the rooms from the floor point cloud
        :param floor: Floor, The floor object
        :param path: str, The path to save the intermediate results
        """

        tmp_floor_path = os.path.join(self.graph_tmp_folder, floor.floor_id)
        if not os.path.exists(tmp_floor_path):
            os.makedirs(tmp_floor_path, exist_ok=True)

        floor_pcd = floor.pcd
        xyz = np.asarray(floor_pcd.points)
        xyz_full = xyz.copy()
        floor_zero_level = floor.floor_zero_level
        floor_height = floor.floor_height
        ## Slice below the ceiling ##
        xyz = xyz[xyz[:, 1] < floor_zero_level + floor_height - 0.3]
        xyz = xyz[xyz[:, 1] >= floor_zero_level + 1.5]
        xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + floor_height - 0.2]
        ## Slice above the floor and below the ceiling ##
        # xyz = xyz[xyz[:, 1] < floor_zero_level + 1.8]
        # xyz = xyz[xyz[:, 1] > floor_zero_level + 0.8]
        # xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + 1.8]

        # project the point cloud to 2d
        pcd_2d = xyz[:, [0, 2]]
        xyz_full = xyz_full[:, [0, 2]]

        # define the grid size and resolution based on the 2d point cloud
        grid_size = (
            int(np.max(pcd_2d[:, 0]) - np.min(pcd_2d[:, 0])),
            int(np.max(pcd_2d[:, 1]) - np.min(pcd_2d[:, 1])),
        )
        grid_size = (grid_size[0] + 1, grid_size[1] + 1)
        resolution = self.cfg.pipeline.grid_resolution
        print("grid_size: ", resolution)

        # calc 2d histogram of the floor using the xyz point cloud to extract the walls skeleton
        num_bins = (int(grid_size[0] // resolution), int(grid_size[1] // resolution))
        num_bins = (num_bins[1] + 1, num_bins[0] + 1)
        hist, _, _ = np.histogram2d(pcd_2d[:, 1], pcd_2d[:, 0], bins=num_bins)
        if self.cfg.pipeline.save_intermediate_results:
            # plot the histogram
            plt.figure()
            plt.imshow(hist, interpolation="nearest", cmap="jet", origin="lower")
            plt.colorbar()
            plt.savefig(os.path.join(tmp_floor_path, "2D_histogram.png"))

        # applythresholding
        hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist = cv2.GaussianBlur(hist, (5, 5), 1)
        hist_threshold = 0.25 * np.max(hist)
        _, walls_skeleton = cv2.threshold(hist, hist_threshold, 255, cv2.THRESH_BINARY)

        # create a bigger image to avoid losing the walls
        walls_skeleton = cv2.copyMakeBorder(
            walls_skeleton, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )

        # apply closing to the walls skeleton
        kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        walls_skeleton = cv2.morphologyEx(
            walls_skeleton, cv2.MORPH_CLOSE, kernal, iterations=1
        )

        # extract outside boundary from histogram of xyz_full
        hist_full, _, _ = np.histogram2d(xyz_full[:, 1], xyz_full[:, 0], bins=num_bins)
        hist_full = cv2.normalize(hist_full, hist_full, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        hist_full = cv2.GaussianBlur(hist_full, (21, 21), 2)
        _, outside_boundary = cv2.threshold(hist_full, 0, 255, cv2.THRESH_BINARY)

        # create a bigger image to avoid losing the walls
        outside_boundary = cv2.copyMakeBorder(
            outside_boundary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )

        # apply closing to the outside boundary
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        outside_boundary = cv2.morphologyEx(
            outside_boundary, cv2.MORPH_CLOSE, kernal, iterations=3
        )

        # extract the outside contour from the outside boundary
        contours, _ = cv2.findContours(
            outside_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outside_boundary = np.zeros_like(outside_boundary)
        cv2.drawContours(outside_boundary, contours, -1, (255, 255, 255), -1)
        outside_boundary = outside_boundary.astype(np.uint8)

        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.imshow(walls_skeleton, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "walls_skeleton.png"))

            plt.figure()
            plt.imshow(outside_boundary, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "outside_boundary.png"))

        # combine the walls skelton and outside boundary
        full_map = cv2.bitwise_or(walls_skeleton, cv2.bitwise_not(outside_boundary))

        # apply closing to the full map
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        full_map = cv2.morphologyEx(full_map, cv2.MORPH_CLOSE, kernal, iterations=2)

        if self.cfg.pipeline.save_intermediate_results:
            # plot the full map
            plt.figure()
            plt.imshow(full_map, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "full_map.png"))
        # apply distance transform to the full map
        room_vertices = distance_transform(full_map, resolution, tmp_floor_path)

        # using the 2D room vertices, map the room back to the original point cloud using KDTree
        room_pcds = []
        room_masks = []
        room_2d_points = []
        floor_tree = cKDTree(np.array(floor_pcd.points))
        for i in tqdm(range(len(room_vertices)), desc="Assign floor points to rooms"):
            room = np.zeros_like(full_map)
            room[room_vertices[i][0], room_vertices[i][1]] = 255
            room_masks.append(room)
            room_m = map_grid_to_point_cloud(room, resolution, pcd_2d)
            room_2d_points.append(room_m)
            # extrude the 2D room to 3D room by adding z value from floor zero level to floor zero level + floor height, step by 0.1m
            z_levels = np.arange(
                floor_zero_level, floor_zero_level + floor_height, 0.05
            )
            z_levels = z_levels.reshape(-1, 1)
            z_levels *= -1
            room_m3dd = []
            for z in z_levels:
                room_m3d = np.hstack((room_m, np.ones((room_m.shape[0], 1)) * z))
                room_m3dd.append(room_m3d)
            room_m3d = np.concatenate(room_m3dd, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(room_m3d)
            # rotate floor pcd to align with the original point cloud
            T1 = np.eye(4)
            T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
            pcd.transform(T1)
            # find the nearest point in the original point cloud
            _, idx = floor_tree.query(np.array(pcd.points), workers=-1)
            pcd = floor_pcd.select_by_index(idx)
            room_pcds.append(pcd)
        self.room_masks[floor.floor_id] = room_masks

        # compute the features of room: input a list of poses and images, output a list of embeddings list
        rgb_list = []
        pose_list = []
        F_g_list = []

        all_global_clip_feats = dict()
        for i, img_id in tqdm(enumerate(range(0, len(self.dataset), self.cfg.pipeline.skip_frames)), desc="Computing room features"):
            rgb_image, _, pose, _, _ = self.dataset[img_id]
            F_g = get_img_feats(np.array(rgb_image), self.preprocess, self.clip_model)
            all_global_clip_feats[str(img_id)] = F_g
            rgb_list.append(rgb_image)
            pose_list.append(pose)
            F_g_list.append(F_g)
        np.savez(
            os.path.join(self.graph_tmp_folder, "room_views.npz"),
            **all_global_clip_feats,
        )

        pcd_min = np.min(np.array(floor_pcd.points), axis=0)
        pcd_max = np.max(np.array(floor_pcd.points), axis=0)
        assert pcd_min.shape[0] == 3

        repr_embs_list, repr_img_ids_list = compute_room_embeddings(
            room_pcds, pose_list, F_g_list, pcd_min, pcd_max, 10, tmp_floor_path
        )
        assert len(repr_embs_list) == len(room_2d_points)
        assert len(repr_img_ids_list) == len(room_2d_points)

        room_index = 0
        for i in range(len(room_2d_points)):
            room = Room(
                str(floor.floor_id) + "_" + str(room_index),
                floor.floor_id,
                name="room_" + str(room_index),
            )
            room.pcd = room_pcds[i]
            room.vertices = room_2d_points[i]
            self.floors[int(floor.floor_id)].add_room(room)
            room.room_height = floor_height
            room.room_zero_level = floor.floor_zero_level
            room.embeddings = repr_embs_list[i]
            room.represent_images = [int(k * self.cfg.pipeline.skip_frames) for k in repr_img_ids_list[i]]
            self.rooms.append(room)
            room_index += 1
        print(
            "number of rooms in floor {} is {}".format(
                floor.floor_id, len(self.floors[int(floor.floor_id)].rooms)
            )
        )

    def identify_object(self, object_feat, text_feats, classes):
        """
        Identify the object class by computing the similarity between the object feature and the text features
        we use COCO-Stuff dataset classes as the text features (183) class.
        :param object_feat: np.ndarray, The object feature
        :param text_feats: np.ndarray, The text features
        :param classes: List, The list of classes
        :return: str, The object class
        """
        similarity = np.dot(object_feat, text_feats.T)
        # find the class with the highest similarity
        return classes[np.argmax(similarity)]

    def segment_objects(self, save_dir: str = None):
        """
        Per floor, assign each object to the room with the highest overlap.
        :param save_dir: str, optional, The path to save the intermediate results
        """
        for i, pcd in enumerate(self.mask_pcds):
            self.mask_pcds[i] = pcd_denoise_dbscan(pcd, eps=0.05, min_points=10)
        text_feats, classes = get_label_feats(
            self.clip_model,
            self.clip_feat_dim,
            self.cfg.pipeline.obj_labels,
            self.cfg.main.save_path,
        )

        pbar = tqdm(enumerate(self.floors), total=len(self.floors), desc="Floor: ")
        margin = 0.2
        for f_idx, floor in pbar:
            pbar.set_description(f"Floor: {f_idx}")
            floor_pcd = floor.pcd
            objects_inside_floor = list()
            # assign objects to rooms
            for i, pcd in enumerate(self.mask_pcds):
                min_z = np.min(np.asarray(pcd.points)[:, 1])
                max_z = np.max(np.asarray(pcd.points)[:, 1])
                if min_z > floor.floor_zero_level - margin and max_z < (
                    floor.floor_zero_level + floor.floor_height + margin
                ):
                    objects_inside_floor.append(i)

            # show the second layer of pbar with tqdm
            obj_pbar = tqdm(
                enumerate(objects_inside_floor),
                total=len(objects_inside_floor),
                desc="Object: ",
                leave=False,
            )
            for obj_floor_idx, mask_idx in obj_pbar:
                room_assoc = list()
                for r_idx, room in enumerate(floor.rooms):
                    room_assoc.append(
                        find_intersection_share(
                            room.vertices,
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]],
                            0.2,
                        )
                    )
                # for outlier objects, utilize Euclidean distance between room centers and mask centers
                if np.sum(room_assoc) == 0:
                    for r_idx, room in enumerate(floor.rooms):
                        # use negative distance to align with the similarity metric
                        room_assoc[r_idx] = -1 * np.linalg.norm(
                            np.mean(room.vertices, axis=0)
                            - np.mean(
                                np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]],
                                axis=0,
                            )
                        )
                    if self.cfg.pipeline.save_intermediate_results:
                        plt.clf()
                        fig, ax = plt.subplots()
                        for r_idx, room in enumerate(floor.rooms):
                            if np.argmax(room_assoc) == r_idx:
                                plt.scatter(
                                    room.vertices[:, 0],
                                    room.vertices[:, 1],
                                    color="red",
                                )
                            else:
                                continue
                                # plt.scatter(room.vertices[:, 0], room.vertices[:, 1])
                        plt.scatter(
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]][:, 0],
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]][:, 1],
                            s=0.05,
                            alpha=0.5,
                            color="green",
                        )
                        ax.set_aspect("equal")

                        debug_objects_dir = os.path.join(
                            self.graph_tmp_folder, "objects"
                        )
                        os.makedirs(debug_objects_dir, exist_ok=True)
                        plt.savefig(
                            os.path.join(
                                debug_objects_dir,
                                f"{floor.rooms[np.argmax(room_assoc)].room_id}_{floor.rooms[np.argmax(room_assoc)].object_counter}.png",
                            )
                        )

                closest_room_idx = np.argmax(room_assoc)

                name = self.identify_object(
                    self.mask_feats[mask_idx], text_feats, classes
                )
                # if [i for i in ["wall", "floor", "ceiling", "window", "door", "roof", "railing"] if i in name]:
                #     continue
                parent_room = floor.rooms[closest_room_idx]
                object = Object(
                    parent_room.room_id + "_" + str(parent_room.object_counter),
                    parent_room.room_id,
                )
                parent_room.object_counter += 1
                object.name = name
                obj_pbar.set_description(
                    f"object name: {object.name}, {object.object_id}"
                )
                object.pcd = self.mask_pcds[mask_idx]
                object.vertices = np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]]
                object.embedding = self.mask_feats[mask_idx]
                floor.rooms[closest_room_idx].add_object(object)
                self.objects.append(object)

    def create_graph(self):
        """
        Create the full HOV-SG graph as a networkx graph
        """
        # add nodes to the graph
        for floor in self.floors:
            self.graph.add_node(floor, name="floor", type="floor")
            self.graph.add_edge(0, floor)
            for room in floor.rooms:
                self.graph.add_node(room, name="room", type="room")
                self.graph.add_edge(floor, room)
                for object in room.objects:
                    self.graph.add_node(object, name=object.name, type="object")
                    self.graph.add_edge(room, object)

    def save_graph(self, path):
        """
        Save the HOV-SG graph
        :param path: str, The path to save the graph
        """
        # create a folder for the graph
        if not os.path.exists(path):
            os.makedirs(path)
        # create a folder for floors, rooms and objects
        if not os.path.exists(os.path.join(path, "floors")):
            os.makedirs(os.path.join(path, "floors"))
        if not os.path.exists(os.path.join(path, "rooms")):
            os.makedirs(os.path.join(path, "rooms"))
        if not os.path.exists(os.path.join(path, "objects")):
            os.makedirs(os.path.join(path, "objects"))
        # save the graph
        for i, node in enumerate(self.graph.nodes(data=True)):
            topo_obj, node_dict = node
            if type(topo_obj) == Floor:
                topo_obj.save(os.path.join(path, "floors"))
            elif type(topo_obj) == Room:
                topo_obj.save(os.path.join(path, "rooms"))
            elif type(topo_obj) == Object:
                topo_obj.save(os.path.join(path, "objects"))

    def load_graph(self, path):
        """
        Load the HOV-SG graph
        :param path: str, The path to load the graph
        """
        print(".. loading predicted graph")
        # load floors
        floor_files = os.listdir(os.path.join(path, "floors"))
        floor_files.sort()
        floor_files = sorted([f for f in floor_files if f.endswith(".ply")])
        for floor_file in floor_files:
            floor_file = floor_file.split(".")[0]
            floor = Floor(str(floor_file), name="floor_" + str(floor_file))
            floor.load(os.path.join(path, "floors"))
            self.floors.append(floor)
            self.graph.add_node(floor, name="floor_" + str(floor_file), type="floor")
            self.graph.add_edge(0, floor)
        print("# pred floors: ", len(self.floors))
        # load rooms
        room_files = os.listdir(os.path.join(path, "rooms"))
        room_files.sort()
        room_files = [f for f in room_files if f.endswith(".ply")]
        for room_file in room_files:
            room_file = room_file.split(".")[0]
            room = Room(str(room_file), room_file.split("_")[0])
            room.load(os.path.join(path, "rooms"))
            self.rooms.append(room)
            self.graph.add_node(room, name="room_" + str(room_file), type="room")
            self.graph.add_edge(self.floors[int(room_file.split("_")[0])], room)
            if isinstance(self.floors[int(room.floor_id)].rooms[0], str):
                self.floors[int(room.floor_id)].rooms = []
            self.floors[int(room.floor_id)].rooms.append(room)
        print("# pred rooms: ", len(self.rooms))
        # load objects
        object_files = os.listdir(os.path.join(path, "objects"))
        object_files.sort()
        object_files = [f for f in object_files if f.endswith(".ply")]
        for object_file in object_files:
            object_file = object_file.split(".")[0]
            room_id = "_".join(object_file.split("_")[:2])
            parent_room = None
            for room in self.rooms:
                if room.room_id == room_id:
                    parent_room = room
                    break
            assert (
                parent_room is not None
            ), f"Couldn't find the room with room id {room_id}"
            objectt = Object(
                str(object_file), room_id, name="object_" + str(object_file)
            )
            objectt.load(os.path.join(path, "objects"))
            objectt.room_id = room_id  # object_file.split("_")[1]
            self.objects.append(objectt)
            self.graph.add_node(
                objectt, name="object_" + str(object_file), type="object"
            )
            self.graph.add_edge(parent_room, objectt)
            # add object to the room
            parent_room.add_object(objectt)
        print("# pred objects: ", len(self.objects))
        print("-------------------")

    def build_graph(self, save_path=None):
        """
        Build the HOV-SG, by segmenting the floors, rooms, and objects and creating the graph.
        :param save_path: str, The path to save the intermediate results
        """
        print("segmenting floors...")
        self.segment_floors(save_path)

        print("segmenting rooms...")
        for floor in self.floors:
            self.segment_rooms(floor, save_path)

        print("segmenting/identifying objects...")
        self.segment_objects(save_path)

        print("number of objects: ", len(self.objects))
        if self.cfg.pipeline.merge_objects_graph:
            # merge objects that close to each other with same name
            for room in tqdm(self.rooms):
                print("room: ", room.room_id)
                print(" number of objects before merging: ", len(room.objects))
                room.merge_objects()
                print(" number of objects after merging: ", len(room.objects))

        print("creating graph...")
        self.create_graph()

        # create navigation graph for each floor
        self.create_nav_graph()

        # save the graph
        self.save_graph(os.path.join(save_path, "graph"))

        print("# floors: ", len(self.floors))
        print("# rooms: ", len(self.rooms))
        print("# objects: ", len(self.objects))
        print("--> HOV-SG representation successfully built")

    def create_nav_graph(self):
        """
        Create the navigation graph for each floor and connect the floors together
        """
        last_nav_graph = None
        global_voronoi = None

        # create a folder for the resulting navigation graph
        nav_dir = os.path.join(self.cfg.main.save_path, "graph", "nav_graph")
        os.makedirs(nav_dir, exist_ok=True)

        # get pose list
        poses_list = []
        for i in range(0, len(self.dataset), self.cfg.pipeline.skip_frames):
            _, _, pose, _, _ = self.dataset[i]
            poses_list.append(pose)

        for floor_id, floor in enumerate(self.floors):
            nav_graph = NavigationGraph(floor.pcd, cell_size=0.03)
            upperbound = None
            if floor_id + 1 < len(self.floors):
                upperbound = self.floors[floor_id + 1].floor_zero_level
            floor_poses_list = nav_graph.get_floor_poses(floor, poses_list, upperbound)

            sparse_stairs_voronoi = nav_graph.get_stairs_graph_with_poses_v2(
                floor, floor_id, poses_list, nav_dir
            )
            sparse_floor_voronoi = nav_graph.get_floor_graph(
                floor, floor_poses_list, nav_dir
            )
            if sparse_stairs_voronoi is not None:
                print(f"connecting stairs and floor {floor_id}")
                sparse_floor_voronoi = nav_graph.connect_stairs_and_floor_graphs(
                    sparse_stairs_voronoi, sparse_floor_voronoi, nav_dir
                )
            NavigationGraph.save_voronoi_graph(
                sparse_floor_voronoi, nav_dir, "sparse_voronoi"
            )

            if last_nav_graph is not None and last_nav_graph.has_stairs:
                print(f"connecting two floors {floor_id}")
                global_voronoi = nav_graph.connect_voronoi_graphs(
                    last_nav_graph.sparse_floor_voronoi, nav_graph.sparse_floor_voronoi
                )
            last_nav_graph = nav_graph

        if global_voronoi is None:
            global_voronoi = last_nav_graph.sparse_floor_voronoi

        NavigationGraph.save_voronoi_graph(global_voronoi, nav_dir, "global_nav_graph")

    def generate_room_names(
        self,
        generate_method: str = "label",
        default_room_types: List[str] = None,
    ):
        """Generate a name for each room node based on children nodes' embedding

        Args:
            generate_method (str): "label" or "obj_embedding" or "view_embedding"
            default_room_types (List[str]): optionally provide a list of default room types so that the
                                            room names can only be one of the provided options. When the
                                            generate_method is set to "embedding", this list is mandatory.
            clip_model (Any): when the generate_method is set to "embedding", a clip model needs to be
                              provided to the method.
            clip_feat_dim (int): when the generate_method is set to "embedding", the clip features dimension
                                 needs to be provided to this method
        """
        for i in range(len(self.rooms)):
            if generate_method in ["obj_embedding", "view_embedding"]:
                print(default_room_types)
                assert (
                    default_room_types is not None
                ), "You should provide a list of default room types"
                assert self.clip_model is not None, "You should provide a clip model"
                assert (
                    self.clip_feat_dim is not None
                ), "You should provide the clip features dimension"
            self.rooms: List[Room]
            if generate_method in ["obj_embedding", "label"]:
                self.rooms[i].infer_room_type_from_objects(
                    infer_method=generate_method,
                    default_room_types=default_room_types,
                    clip_model=self.clip_model,
                    clip_feat_dim=self.clip_feat_dim,
                )
            elif generate_method in ["view_embedding"]:
                self.rooms[i].infer_room_type_from_view_embedding(
                    default_room_types, self.clip_model, self.clip_feat_dim
                )
            else:
                return NotImplementedError

    def query_graph(self, query):
        """
        search in graph of the openmap with a text query
        """
        text_feats = get_text_feats_multiple_templates(
            [query], self.clip_model, self.clip_feat_dim
        )
        # compute similarity between the text query and the objects embeddings in the graph
        similarity = np.dot(text_feats, np.array([o.embedding for o in self.objects]).T)
        # similarity = compute_similarity(text_feats, np.array([o.embedding for o in self.objects]))
        # find top 5 similar objects
        top_index = np.argsort(similarity[0])[::-1][:5]
        # print the top 5 similar objects
        for i in top_index:
            print(self.objects[i].name, similarity[0])
            print("room: ", self.objects[i].room_id)
            obj_pcd = self.objects[i].pcd.paint_uniform_color([1, 0, 0])
            # find the room with a room id that matches the object's room id
            for room in self.rooms:
                if room.room_id == self.objects[i].room_id:
                    room_pcd = room.pcd
                    break
            o3d.visualization.draw_geometries([room_pcd, obj_pcd])

        # return the object with the highest similarity
        return self.objects[top_index[0]]

    def query_floor(self, query: str, query_method: str = "clip") -> int:
        """search a floor based on the number of the text query

        Args:
            query (str): a number in text format
            query_method (str): "clip" match the clip embeddings of the query text and the text description of all floors.
                                "gpt" provide the floor ids in the graph and the text query to a gpt agent, and ask for the
                                matching floor id.

        Returns:
            int: The target floor id in self.floors
        """
        # TODO: assume that the self.floors are ordered according to the floor level in ascending order. Check again.
        zero_levels_list = [x.floor_zero_level for x in self.floors]
        print("zero_levels_list: ", zero_levels_list)
        zero_level_order_ids = np.argsort(zero_levels_list)

        # check whether the query is a number that is an integer
        try:
            return zero_level_order_ids[int(query) - 1]
        except:
            if query_method == "clip":
                text_feats = get_text_feats_multiple_templates(
                    [query], self.clip_model, self.clip_feat_dim
                )
                floor_names = ["floor " + str(i) for i in range(len(self.floors))]
                floor_embs = get_text_feats_multiple_templates(
                    floor_names, self.clip_model, self.clip_feat_dim
                )
                sim_mat = np.dot(text_feats, floor_embs.T)
                # sim_mat = compute_similarity(text_feats, floor_embs)
                print(sim_mat)
                top_index = np.argsort(sim_mat[0])[::-1][0]
                return zero_level_order_ids[top_index]

            elif query_method == "gpt":
                floor_ids_list = [i + 1 for i in range(len(self.floors))]
                floor_id = infer_floor_id_from_query(floor_ids_list, query)
                return zero_level_order_ids[floor_id - 1]

    def query_room(
        self, query: str, floor_id: int = -1, query_method: str = "view_embedding"
    ) -> List[int]:
        """search a room node with a text query

        Args:
            query (str): a text describing the room
            floor_id (int): -1 means global search. 0-(max_floor - 1) means searching the target room on a specific floor.
            query_method (str): "label" use pre-defined label stored in the room node. "view_embedding" use the room embedding
                                stored in the room node. "children_embedding" use all children objects' embedding and find
                                the most representative one for the room.
        Returns:
            (Room): the target room ids in self.rooms which matches th equery the best
        """
        query_text_feats = get_text_feats_multiple_templates(
            [query], self.clip_model, self.clip_feat_dim
        )

        rooms_list = self.rooms
        if floor_id != -1:
            rooms_list = self.floors[floor_id].rooms
            # for room_id in self.floors[floor_id].rooms:
            #     for room in self.rooms:
            #         if room.room_id == room_id:
            #             rooms_list.append(room)
        rooms_list: List[Room]
        if query_method == "label":
            for room in rooms_list:
                assert (
                    room.name is not None
                ), "The name attribute for the room has not been generated"
            room_names_list = [room.name for room in rooms_list]
            print(room_names_list)
            room_embs = get_text_feats_multiple_templates(
                room_names_list, self.clip_model, self.clip_feat_dim
            )
            similarity = np.dot(query_text_feats, room_embs.T)
            # similarity = compute_similarity(query_text_feats, room_embs)
            top_index = np.argsort(similarity[0])[::-1]
            # print the top 5 matching rooms
            for i in top_index:
                print(rooms_list[i].name, similarity[0][i])
                print("room: ", rooms_list[i].room_id)

            same_sim_indices = []
            tar_sim = similarity[0, top_index[0]]
            same_sim_indices.append(top_index[0])
            for i in top_index[1:]:
                if np.abs(similarity[0, i] - tar_sim) < 1e-3:
                    same_sim_indices.append(i)

            target_rooms = [rooms_list[i] for i in same_sim_indices]
            target_room_ids = [target_room.room_id for target_room in target_rooms]
            target_ids = [
                i for i, x in enumerate(self.rooms) if x.room_id in target_room_ids
            ]
            return target_ids
        elif query_method == "view_embedding":
            room2query_sim = dict()
            for room in self.rooms:
                room_query_sim_median = np.max(
                    np.dot(query_text_feats, np.stack(room.embeddings).T)
                )
                # room_query_sim_median = np.max(compute_similarity(query_text_feats, np.stack(room.embeddings)))
                room2query_sim[room.room_id] = room_query_sim_median
            room2query_sim_sorted = {
                int(k.split("_")[-1]): v
                for k, v in sorted(
                    room2query_sim.items(), key=lambda item: item[1], reverse=True
                )
            }
            return list(room2query_sim_sorted.keys())[
                0 : min(len(room2query_sim_sorted), 3)
            ]  # return three highest-ranking rooms
        elif query_method == "children_embedding":
            return NotImplementedError

    def query_object(
        self,
        query: str,
        room_ids: List[int] = [],
        query_method: str = "clip",
        top_k: int = 1,
        negative_prompt: List[str] = [],
    ) -> Tuple[List[int], List[int]]:
        """search an object (from a room) with a text query

        Args:
            query (str): a description of the object
            room_ids (List[int], optional): The room ids. Defaults to [], which means search from all rooms.
            query_method (str, optional): "clip" means using clip features of the objects and the query to search.
                                          Defaults to "clip".
            top_k (int, optional): The number of top results to return. Default to 1.
            negative_prompt (List[str], optional): A list of categories used as negative prompt.


        Returns:
            Tuple[int, int]: The target object id in self.objects and the corresponding room id in self.rooms.
        """

        if query in negative_prompt:
            query_id = negative_prompt.index(query)
        else:
            query_id = None

        if query_id is None:
            query = [query, *negative_prompt]
            query_id = 0
        else:
            query = negative_prompt

        print(f"query_id: {query_id}")
        print(f"categories list: {query}")

        query_text_feats = get_text_feats_multiple_templates(
            query, self.clip_model, self.clip_feat_dim
        )  # (len(categories), feat_dim)
        print(f"text_feats.shape: {query_text_feats.shape}")
        print(query_text_feats[:, :10])

        room_ids_list = []
        for obj in self.objects:
            for i, room in enumerate(self.rooms):
                if obj.room_id == room.room_id:
                    room_ids_list.append(i)
                    break

        if len(room_ids) != 0:
            objects_list = []
            room_ids_list = []
            for i in room_ids:
                objects_list.extend(self.rooms[i].objects)
                room_ids_list.extend([i] * len(self.rooms[i].objects))
        objects_list: List[Object]
        if query_method == "clip":
            object_embs = np.array([obj.embedding for obj in objects_list])
            sim_mat = np.dot(query_text_feats, object_embs.T)
            # sim_mat = compute_similarity(query_text_feats, object_embs)  # (len(categories), len(objects_list))
            top_index = np.argsort(sim_mat[query_id])[::-1][:10]
            for i in top_index:
                print("object name, score: ", objects_list[i].name, sim_mat[0][i])
                print("object id: ", objects_list[i].object_id)

            # plt.hist(sim_mat.flatten(), bins=100)
            # plt.show()

            top_index = np.argsort(sim_mat[query_id])[::-1][:top_k]
            if len(negative_prompt) > 0:
                cls_ids = np.argmax(sim_mat, axis=0)  # category id for each object
                print(f"cls_ids: {cls_ids}")
                max_scores = np.max(sim_mat, axis=0)  # max scores for each object
                obj_ids = np.where(cls_ids == query_id)[
                    0
                ]  # find the obj ids that assign max score to the target category
                if len(obj_ids) > 0:
                    obj_scores = max_scores[obj_ids]
                    resort_ids = np.argsort(
                        -obj_scores
                    )  # sort the obj ids based on max score (descending)
                    top_index = obj_ids[resort_ids]  # get the top index
                    top_index = top_index[:top_k]

            target_object_id = [objects_list[i].object_id for i in top_index]
            target_room_id = [room_ids_list[i] for i in top_index]
            target_id = []
            for ti in target_object_id:
                target_id.append(
                    [i for i, x in enumerate(self.objects) if x.object_id == ti][0]
                )

            return target_id, target_room_id
        return NotImplementedError

    def query_hierarchy(
        self, query: str, top_k: int = 1
    ) -> Tuple[Floor, Room, List[Object]]:
        """return the target floor, room, and the list of top k objects

        Args:
            query (str): the long query like "object X in room Y on floor Z"
            top_k (int, optional): The number of top results to return. Default to 1.

        Returns:
            Tuple[Floor, List[Room], List[Object]]: return a floor object, a room object and a list of object objects
        """

        negative_labels = ["background"]

        floor_query, room_query, object_query = parse_hier_query(self.cfg, query)
        # log these in a txt file
        # with open("room_obj_query_log.txt", "a") as f:
        #     f.write(f"query: {query} -- {floor_query}, {room_query}, {object_query}\n")

        floor_id = self.query_floor(floor_query) if floor_query is not None else -1
        print(f"floor id: {floor_id}")
        room_ids = (
            self.query_room(room_query, floor_id=floor_id)
            if room_query is not None
            else []
        )
        print(f"room ids: {room_ids}")

        object_ids, room_ids = (
            self.query_object(
                object_query,
                room_ids=room_ids,
                top_k=top_k,
                negative_prompt=negative_labels,
            )
            if object_query is not None
            else ([], [])
        )
        print(f"object id: {object_ids}")

        return (
            self.floors[floor_id] if floor_id != -1 else None,
            [self.rooms[k] for k in room_ids],
            [self.objects[i] for i in object_ids],
        )

    def save_full_pcd(self, path):
        """
        Save the full pcd to disk
        :param path: str, The path to save the full pcd
        """
        if not os.path.exists(path):
            os.makedirs(path)
        o3d.io.write_point_cloud(os.path.join(path, "full_pcd.ply"), self.full_pcd)
        print("full pcd saved to disk in {}".format(path))
        return None

    def load_full_pcd(self, path):
        """
        Load the full pcd from disk
        :param path: str, The path to load the full pcd
        """
        if not os.path.exists(path):
            print("full pcd not found in {}".format(path))
            return None
        self.full_pcd = o3d.io.read_point_cloud(os.path.join(path, "full_pcd.ply"))
        print(
            "full pcd loaded from disk with shape {}".format(
                np.asarray(self.full_pcd.points).shape
            )
        )
        return self.full_pcd

    def save_full_pcd_feats(self, path):
        """
        Save the full pcd with feats to disk
        :param path: str, The path to save the full pcd feats
        """
        if not os.path.exists(path):
            os.makedirs(path)
        # check if the full pcd feats is empty list
        if len(self.mask_feats) != 0:
            self.mask_feats = np.array(self.mask_feats)
            torch.save(
                torch.from_numpy(self.mask_feats), os.path.join(path, "mask_feats.pt")
            )
        if len(self.full_feats_array) != 0:
            torch.save(
                torch.from_numpy(self.full_feats_array),
                os.path.join(path, "full_feats.pt"),
            )
        print("full pcd feats saved to disk in {}".format(path))
        return None

    def load_full_pcd_feats(self, path, full_feats=False, normalize=True):
        """
        Load the full pcd with feats from disk
        :param path: str, The path to load the full pcd feats
        :param full_feats: bool, Whether to load the full feats or the mask feats
        :param normalize: bool, Whether to normalize the feats
        """
        if not os.path.exists(path):
            print("full pcd feats not found in {}".format(path))
            return None
        if full_feats:
            self.full_feats_array = torch.load(
                os.path.join(path, "full_feats.pt")
            ).float()
            if normalize:
                self.full_feats_array = (
                    torch.nn.functional.normalize(self.full_feats_array, p=2, dim=-1)
                    .cpu()
                    .numpy()
                )
            else:
                self.full_feats_array = self.full_feats_array.cpu().numpy()
            print(
                "full pcd feats loaded from disk with shape {}".format(
                    self.full_feats_array.shape
                )
            )
            return self.full_feats_array
        else:
            self.mask_feats = torch.load(os.path.join(path, "mask_feats.pt")).float()
            if normalize:
                self.mask_feats = (
                    torch.nn.functional.normalize(self.mask_feats, p=2, dim=-1)
                    .cpu()
                    .numpy()
                )
            else:
                self.mask_feats = self.mask_feats.cpu().numpy()
            print(
                "full pcd feats loaded from disk with shape {}".format(
                    self.mask_feats.shape
                )
            )
            return self.mask_feats

    def print_details(self):
        """
        Print the details of the graph
        """
        print("number of floors: ", len(self.floors))
        print("number of rooms: ", len(self.rooms))
        print("number of objects: ", len(self.objects))
        return None

    def save_masked_pcds(self, path, state="both"):
        """
        Save the masked pcds to disk
        :params state: str 'both' or 'objects' or 'full' to save the full masked pcds or only the objects.
        """
        # # remove any small pcds
        tqdm.write("-- removing small and empty masks --")
        for i, pcd in enumerate(self.mask_pcds):
            if len(pcd.points) < 50:
                self.mask_pcds.pop(i)
                self.mask_feats.pop(i)

        for i, pcd in enumerate(self.mask_pcds):
            if pcd.is_empty():
                self.mask_pcds.pop(i)
                self.mask_feats.pop(i)

        if state == "both":
            if not os.path.exists(path):
                os.makedirs(path)
            objects_path = os.path.join(path, "objects")
            if not os.path.exists(objects_path):
                os.makedirs(objects_path)
            print("number of masked pcds: ", len(self.mask_pcds))
            print("number of mask_feats: ", len(self.mask_feats))
            for i, pcd in enumerate(self.mask_pcds):
                o3d.io.write_point_cloud(
                    os.path.join(objects_path, "pcd_{}.ply".format(i)), pcd
                )

            masked_pcd = o3d.geometry.PointCloud()
            for pcd in self.mask_pcds:
                pcd.paint_uniform_color(np.random.rand(3))
                masked_pcd += pcd
            o3d.io.write_point_cloud(os.path.join(path, "masked_pcd.ply"), masked_pcd)
            print("masked pcds saved to disk in {}".format(path))

        elif state == "objects":
            if not os.path.exists(path):
                os.makedirs(path)
            for i, pcd in enumerate(self.mask_pcds):
                o3d.io.write_point_cloud(
                    os.path.join(objects_path, "pcd_{}.ply".format(i)), pcd
                )
            print("masked pcds saved to disk in {}".format(path))

        elif state == "full":
            if not os.path.exists(path):
                os.makedirs(path)
            masked_pcd = o3d.geometry.PointCloud()
            for pcd in self.mask_pcds:
                pcd.paint_uniform_color(np.random.rand(3))
                masked_pcd += pcd
            o3d.io.write_point_cloud(os.path.join(path, "masked_pcd.ply"), masked_pcd)
            print("masked pcds saved to disk in {}".format(path))

    def load_masked_pcds(self, path):
        """
        Load the masked pcds from disk
        """
        # make sure that self.mask_feats is already loaded
        if len(self.mask_feats) == 0:
            print("load full pcd feats first")
            return None
        if os.path.exists(os.path.join(path, "objects")):
            self.mask_pcds = []
            number_of_pcds = len(os.listdir(os.path.join(path, "objects")))
            not_found = []
            for i in range(number_of_pcds):
                if os.path.exists(
                    os.path.join(path, "objects", "pcd_{}.ply".format(i))
                ):
                    self.mask_pcds.append(
                        o3d.io.read_point_cloud(
                            os.path.join(path, "objects", "pcd_{}.ply".format(i))
                        )
                    )
                else:
                    print("masked pcd {} not found in {}".format(i, path))
                    not_found.append(i)
            print(
                "number of masked pcds loaded from disk {}".format(len(self.mask_pcds))
            )
            # remove masks_feats that are not found
            self.mask_feats = np.delete(self.mask_feats, not_found, axis=0)
            print(
                "number of mask_feats loaded from disk {}".format(len(self.mask_feats))
            )
            return self.mask_pcds
        else:
            print("masked pcds for objects not found in {}".format(path))
            return None

    def transform(self, transform):
        """
        Transform the openmap full pcd and masked pcds
        :param transform: np.ndarray, The transformation matrix
        """
        self.full_pcd.transform(transform)
        for i, pcd in enumerate(self.mask_pcds):
            self.mask_pcds[i].transform(transform)
        return None

    def visualize_instances(self):
        """
        visualize the instance of obejcts in the graph
        """
        all_objects_pcd = o3d.geometry.PointCloud()
        number_of_objects = 0
        for i, node in enumerate(self.graph.nodes):
            if type(node) == Object:
                print("object name: ", node.name, node.object_id)
                print("number of points: ", len(node.pcd.points))
                all_objects_pcd += node.pcd
                number_of_objects += 1
        print("number of objects: ", number_of_objects)
        o3d.visualization.draw_geometries([all_objects_pcd])
        return None
