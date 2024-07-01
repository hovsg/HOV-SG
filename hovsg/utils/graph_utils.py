from collections import Counter, defaultdict
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import cv2

import faiss
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import open3d as o3d
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree, distance
from sklearn.cluster import DBSCAN, KMeans
import torch
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
from tqdm import tqdm

import open3d as o3d


def find_intersection_share(map_points, obj_points, radius=0.05):
    """
    Calculate the percentage of overlapping points normalized by the query objects size.

    Parameters:
    base_points (numpy.ndarray): shape (n1, 3).
    query_points (numpy.ndarray): shape (n2, 3).
    radius (float): Radius for KD-Tree query (adjust based on point density).

    Returns:
    float: Overlapping ratio between 0 and 1.
    """
    obj_tree_points = cKDTree(obj_points)

    # Query all points in pcd1 for nearby points in pcd2
    _, indices = obj_tree_points.query(map_points, k=1, distance_upper_bound=radius, p=2, workers=-1)
    # Remove indices that are out of range
    indices = indices[indices != obj_points.shape[0]]

    # Calculate the overlapping ratio, handle the case where one of the point clouds is empty
    if map_points.shape[0] == 0 or obj_points.shape[0] == 0:
        overlapping_ratio = 0
    else:
        overlapping_ratio = indices.shape[0] / obj_points.shape[0]

    del indices
    return overlapping_ratio


def compute_room_embeddings(
    room_pcds: List[o3d.geometry.PointCloud],
    pose_list: List[np.ndarray],
    emb_list: List[np.ndarray],
    pcd_min: np.ndarray,
    pcd_max: np.ndarray,
    num_views: int = 5,
    save_path: Union[str, Path] = None,
) -> Tuple[List[List[np.ndarray]], List[List[int]]]:
    """assign all images to their corresponding room regions respectively, apply k-mean clustering to the CLIP
       embeddings of images in each room, and select 5 representative embeddings to represent the room.

    Args:
        room_masks (List[np.ndarray]): a list of 2D array representing the mask of each room
        pose_list (List[np.ndarray]): a list of pose of the images
        emb_list (List[np.ndarray]): a list of CLIP embeddings of the images
        pcd_min (np.ndarray): the minimum X, Y, Z of the 3D point cloud of the floor
        pcd_max (np.ndarray): the maximum X, Y, Z of the 3D point cloud of the floor
        resolution (int): the maximum height of current floor
        num_views (int): the number of views considered in each room
        save_path (Union[str, Path]): a path to save debug info

    Returns:
        repr_embs_list (List[List[np.ndarray]]): a list of CLIP embeddings list, each of the room has a
                                                 list of num_views CLIP embeddings
        repr_img_ids_list (List[List[int]]]): a list of image ids list, each of the room has a list of
                                              num_views image indices

    """
    # save_path = Path(save_path) / "debug"
    # if save_path is not None:
    #     os.makedirs(save_path, exist_ok=True)

    img2room_id = []
    room_id2img_id = defaultdict(list)

    flattened_room_points = list()
    plt.figure()
    # colormap over all rooms
    cmap = cm.get_cmap("tab20")
    for room_idx, room_pcd in enumerate(room_pcds):
        room_2d_points = np.stack([np.asarray(room_pcd.points)[:, 0], np.asarray(room_pcd.points)[:, 2]], axis=1)
        plt.scatter(room_2d_points[:, 0], room_2d_points[:, 1], s=0.1, c=cmap(room_idx))
        flattened_room_points.append(room_2d_points)

    pbar = tqdm(enumerate(pose_list), total=len(pose_list), desc="assign camera to room")
    pose_cmap = cm.get_cmap("Set1")
    for i, pose in pbar:
        pos = pose[0, 3], pose[2, 3]
        z = pose[1, 3]
        # Check if camera pose is inside the floor bounds
        if z < pcd_min[1] or z > pcd_max[1]:
            img2room_id.append(-1)
            continue
        # Find the closest room given the camera pose
        room_dists = []
        for room_points in flattened_room_points:
            room_dists.append(np.min(distance.cdist(np.array([pos]), np.array(room_points), metric="euclidean")))
        closest_room_idx = np.argmin(room_dists)
        plt.scatter(pos[0], pos[1], s=3.0, c=pose_cmap(closest_room_idx))

        img2room_id.append(closest_room_idx)
        room_id2img_id[closest_room_idx].append(i)

    # check whether one of the rooms has not been assigned any image
    for room_id in range(len(flattened_room_points)):
        if room_id not in room_id2img_id:
            # (double)-assign closest image to the room
            closest_cam_pose = list()
            pbar = tqdm(
                enumerate(pose_list), total=len(pose_list), desc="find closest camera pose to room w/o assigned image"
            )
            for i, pose in pbar:
                pos = pose[0, 3], pose[2, 3]
                z = pose[1, 3]
                # Check if camera pose is inside the floor bounds
                if z < pcd_min[1] or z > pcd_max[1]:
                    closest_cam_pose.append(
                        np.min(
                            distance.cdist(
                                np.array([pos]), np.array(flattened_room_points[room_id]), metric="euclidean"
                            )
                        )
                    )
                else:
                    closest_cam_pose.append(np.inf)
            assert len(closest_cam_pose) == len(pose_list)
            closest_cam_pose_idx = np.argmin(np.array(closest_cam_pose))
            room_id2img_id[room_id].append(closest_cam_pose_idx)

    plt.savefig(os.path.join(save_path, "pcd_camera_pose.png"))

    repr_img_ids_list = []
    repr_embs_list = []
    for room_id in range(len(flattened_room_points)):
        img_ids = room_id2img_id[room_id]
        if len(img_ids) == 0:
            repr_img_ids_list.append([])
            repr_embs_list.append([])
            continue

        repr_img_ids = []
        repr_embs = []
        room_clip_embeddings = [emb_list[i] for i in img_ids]
        room_clip_embeddings = np.squeeze(np.array(room_clip_embeddings), axis=1)
        if len(img_ids) < num_views:
            repr_img_ids_list.append(img_ids)
            repr_embs_list.append([emb for emb in room_clip_embeddings])
            continue
        # To tune the parameter, follow the guideline here:
        # https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#clustering-sparse-data-with-k-means
        kmeans = KMeans(n_clusters=num_views, max_iter=100, n_init=5, random_state=0).fit(room_clip_embeddings)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        unique_labels = np.unique(labels)
        print(unique_labels)
        for unique_label in unique_labels:
            ids = np.where(labels == unique_label)[0]
            cluster = room_clip_embeddings[ids]
            # mean_feats = np.mean(cluster, axis=0)
            mean_feats = centers[unique_label]
            similarity = np.dot(cluster, mean_feats)
            max_idx = np.argmax(similarity)
            feats = cluster[max_idx]
            # img_ids -> ids -> max_idx
            repr_img_ids.append(img_ids[ids[max_idx]])
            repr_embs.append(feats)
        repr_img_ids_list.append(repr_img_ids)
        repr_embs_list.append(repr_embs)
    return repr_embs_list, repr_img_ids_list

def map_grid_to_point_cloud(occupancy_grid_map, resolution, point_cloud):
    """
    Map the occupancy grid back to the original coordinates in the point cloud.

    Parameters:
        occupancy_grid_map (numpy.array): Occupancy grid map as a 2D numpy array, where each cell is marked as either 0 (unoccupied) or 1 (occupied).
        grid_size (tuple): A tuple (width, height) representing the size of the occupancy grid map in meters.
        resolution (float): The resolution of each cell in the grid map in meters.
        point_cloud (numpy.array): 2D numpy array of shape (N, 2), where N is the number of points and each row represents a point (x, y).

    Returns:
        numpy.array: A subset of the original point cloud containing points that correspond to occupied cells in the occupancy grid.
    """

    # make sure image is binary
    occupancy_grid_map = (occupancy_grid_map > 0).astype(np.uint8)

    # Get the occupied cell indices
    y_cells, x_cells = np.where(occupancy_grid_map == 1)

    # Compute the corresponding point coordinates for occupied cells
    # NOTE: The coordinates are shifted by 10.5 cells to account for the padding added to the grid map
    mapped_x_coords = (x_cells - 10.5) * resolution + np.min(point_cloud[:, 0])
    mapped_y_coords = (y_cells - 10.5) * resolution + np.min(point_cloud[:, 1])

    # Stack the mapped x and y coordinates to form the mapped point cloud
    mapped_point_cloud = np.column_stack((mapped_x_coords, mapped_y_coords))

    return mapped_point_cloud

def distance_transform(occupancy_map, reselotion, tmp_path):
    """
        Perform distance transform on the occupancy map to find the distance of each cell to the nearest occupied cell.
        :param occupancy_map: 2D numpy array representing the occupancy map.
        :param reselotion: The resolution of each cell in the grid map in meters.
        :param path: The path to save the distance transform image.
        :return: The distance transform of the occupancy map.
    """

    print("occupancy_map shape: ", occupancy_map.shape)
    bw = occupancy_map.copy()
    full_map = occupancy_map.copy()

    # invert the image
    bw = cv2.bitwise_not(bw)

    # Perform the distance transform algorithm
    bw = np.uint8(bw)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    print("range of dist: ", np.min(dist), np.max(dist))
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
    plt.figure()
    plt.imshow(dist, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist.png"))

    dist = np.uint8(dist)
    # apply Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(dist, (11, 1), 10)
    plt.figure()
    plt.imshow(blur, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist_blur.png"))
    _, dist = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.figure()
    plt.imshow(dist, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist_thresh.png"))

    # Create the CV_8U version of the distance image
    # It is needed for findContours()
    dist_8u = dist.astype("uint8")
    # Find total markers
    contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("number of seeds, aka rooms: ", len(contours))

    # print the area of each seed
    for i in range(len(contours)):
        print("area of seed {}: ".format(i), cv2.contourArea(contours[i]))

    # remove small seed contours
    min_area_m = 0.5
    min_area = (min_area_m / reselotion) ** 2
    print("min_area: ", min_area)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    print("number of contours after remove small seeds: ", len(contours))

    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i + 1), -1)
    # Draw the background marker
    circle_radius = 1  # in pixels
    cv2.circle(markers, (3, 3), circle_radius, len(contours) + 1, -1)

    # Perform the watershed algorithm
    full_map = cv2.cvtColor(full_map, cv2.COLOR_GRAY2BGR)
    cv2.watershed(full_map, markers)

    plt.figure()
    plt.imshow(markers, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "markers.png"))

    # find the vertices of each room
    room_vertices = []
    for i in range(len(contours)):
        room_vertices.append(np.where(markers == i + 1))
    room_vertices = np.array(room_vertices, dtype=object).squeeze()
    print("room_vertices shape: ", room_vertices.shape)

    return room_vertices


def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    Taken from ConceptGraphs
    Compute IoU between two sets of axis-aligned 3D bounding boxes.
    
    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Compute min and max for each box
    bbox1_min, _ = bbox1.min(dim=1) # Shape: (M, 3)
    bbox1_max, _ = bbox1.max(dim=1) # Shape: (M, 3)
    bbox2_min, _ = bbox2.min(dim=1) # Shape: (N, 3)
    bbox2_max, _ = bbox2.max(dim=1) # Shape: (N, 3)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)

    # Compute volume of intersection box
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

    return iou


def find_overlapping_ratio_faiss(pcd1, pcd2, radius=0.02):
    """
    Calculate the percentage of overlapping points between two point clouds using FAISS.

    Parameters:
    pcd1 (numpy.ndarray): Point cloud 1, shape (n1, 3).
    pcd2 (numpy.ndarray): Point cloud 2, shape (n2, 3).
    radius (float): Radius for KD-Tree query (adjust based on point density).

    Returns:
    float: Overlapping ratio between 0 and 1.
    """
    if type(pcd1) == o3d.geometry.PointCloud and type(pcd2) == o3d.geometry.PointCloud:
        pcd1 = np.asarray(pcd1.points)
        pcd2 = np.asarray(pcd2.points)

    if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
        return 0

    # Create the FAISS index for each point cloud
    index1 = faiss.IndexFlatL2(pcd1.shape[1])
    index2 = faiss.IndexFlatL2(pcd2.shape[1])
    index1.add(pcd1.astype(np.float32))
    index2.add(pcd2.astype(np.float32))

    # Query all points in pcd1 for nearby points in pcd2
    D1, I1 = index2.search(pcd1.astype(np.float32), k=1)
    D2, I2 = index1.search(pcd2.astype(np.float32), k=1)

    number_of_points_overlapping1 = np.sum(D1 < radius**2)
    number_of_points_overlapping2 = np.sum(D2 < radius**2)

    overlapping_ratio = np.max(
        [number_of_points_overlapping1 / pcd1.shape[0], number_of_points_overlapping2 / pcd2.shape[0]]
    )

    return overlapping_ratio

def merge_point_clouds_list(pcd_list, voxel_size=0.02):
    """
        Merge a list of point clouds into a single point cloud.
        :param pcd_list: List of point clouds to merge.
        :param voxel_size: Voxel size for downsampling.
        :return: Merged point cloud.
    """
    merged_pcd = pcd_list[0]
    for pcd in pcd_list[1:]:
        merged_pcd += pcd
    merged_pcd = pcd_denoise_dbscan(merged_pcd, eps=0.1, min_points=10)
    return merged_pcd

def feats_denoise_dbscan(feats, eps=0.02, min_points=2):
    """
        Denoise the features using DBSCAN
        :param feats: Features to denoise.
        :param eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other.
        :param min_points: The number of samples in a neighborhood for a point to be considered as a core point.
        :return: Denoised features.
    """
    # Convert to numpy arrays
    feats = np.array(feats)
    # Create DBSCAN object
    clustering = DBSCAN(eps=eps, min_samples=min_points, metric="cosine").fit(feats)

    # Get the labels
    labels = clustering.labels_

    # Count all labels in the cluster
    counter = Counter(labels)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        # Create mask for points in the largest cluster
        largest_mask = labels == most_common_label
        # Apply mask
        largest_cluster_feats = feats[largest_mask]
        feats = largest_cluster_feats
        # take the feature with the highest similarity to the mean of the cluster
        if len(feats) > 1:
            mean_feats = np.mean(largest_cluster_feats, axis=0)
            # similarity = np.dot(largest_cluster_feats, mean_feats)
            # max_idx = np.argmax(similarity)
            # feats = feats[max_idx]
            feats = mean_feats
    else:
        feats = np.mean(feats, axis=0)
    return feats


def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10):
    """
    Denoise the point cloud using DBSCAN.
    :param pcd: Point cloud to denoise.
    :param eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_points: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: Denoised point cloud.
    """
    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )

    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]

        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]

        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)

        pcd = largest_cluster_pcd

    return pcd


def compute_3d_bbox_iou(bbox1, bbox2, padding=0):
    """
    Compute 3D Intersection over Union (IoU) between two point clouds.
    :param pcd1 (open3d.geometry.PointCloud): Point cloud 1.
    :param pcd2 (open3d.geometry.PointCloud): Point cloud 2.
    :param padding (float): Padding to add to the bounding box.
    :return: 3D IoU between 0 and 1.
    """
    # Get the coordinates of the first bounding box
    bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
    bbox1_max = np.asarray(bbox1.get_max_bound()) + padding

    # Get the coordinates of the second bounding box
    bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
    bbox2_max = np.asarray(bbox2.get_max_bound()) + padding

    # Compute the overlap between the two bounding boxes
    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_volume = np.prod(overlap_size)
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)

    obj_1_overlap = overlap_volume / bbox1_volume
    obj_2_overlap = overlap_volume / bbox2_volume
    max_overlap = max(obj_1_overlap, obj_2_overlap)

    iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume)

    return iou


def merge_3d_masks(mask_list, overlap_threshold=0.5, radius=0.02, iou_thresh=0.05):
    """
    merge the overlapped 3D masks in the list of masks using matrix
    :param pcd_list (list): list of point clouds
    :param overlap_threshold (float): threshold for overlapping ratio
    :param radius (float): radius for faiss search
    :param iou_thresh (float): threshold for iou
    :return: merged point clouds and features
    """
    
    aa_bb = [pcd.get_axis_aligned_bounding_box() for pcd in mask_list]
    overlap_matrix = np.zeros((len(mask_list), len(mask_list)))

    # create matrix of overlapping ratios
    for i in range(len(mask_list)):
        for j in range(i + 1, len(mask_list)):
            if compute_3d_bbox_iou(aa_bb[i], aa_bb[j]) > iou_thresh:
                overlap_matrix[i, j] = find_overlapping_ratio_faiss(mask_list[i], mask_list[j], radius=1.5 * radius)

    # check if overlap_matrix is zero size
    if overlap_matrix.shape[0] == 0:
        return mask_list
    graph = overlap_matrix > overlap_threshold
    n_components, component_labels = connected_components(graph)
    component_indices = [np.where(component_labels == i)[0] for i in range(n_components)]
    # merge the masks in each component
    pcd_list_merged = []
    for indices in component_indices:
        pcd_list_merged.append(merge_point_clouds_list([mask_list[i] for i in indices], voxel_size=0.5 * radius))

    return pcd_list_merged


def merge_adjacent_frames(frames_pcd, th, down_size, proxy_th):
    """
        Merge adjacent frames in the list of frames
        :param frames_pcd (list): list of point clouds
        :param th (float): threshold for overlapping ratio
        :param down_size (float): radius for downsampling
        :param proxy_th (float): threshold for iou
        :return: merged point clouds and features
    """
    new_frames_pcd = []
    for i in tqdm(range(0, len(frames_pcd), 2)):
        # if the number of frames is odd, the last frame is appended without merging.
        if i == len(frames_pcd) - 1:
            new_frames_pcd.append(frames_pcd[i])
            break
        pcd_list = frames_pcd[i] + frames_pcd[i + 1]

        pcd_list = merge_3d_masks(
            pcd_list,
            overlap_threshold=th,
            radius=down_size,
            iou_thresh=proxy_th,
        )
        new_frames_pcd.append(pcd_list)

    return new_frames_pcd


def hierarchical_merge(frames_pcd, th, th_factor, down_size, proxy_th):
    """
        Hierarchical merge the frames in the list of frames
        :param frames_pcd (list): list of point clouds
        :param th (float): threshold for overlapping ratio
        :param th_factor (float): factor for decreasing the threshold
        :param down_size (float): radius for downsampling
        :param proxy_th (float): threshold for iou
        :return: merged point clouds and features
    """
    while len(frames_pcd) > 1:
        frames_pcd = merge_adjacent_frames(frames_pcd, th, down_size, proxy_th)
        if len(frames_pcd) > 1:
            th -= th_factor * (len(frames_pcd) - 2) / max(1, len(frames_pcd) - 1)
            print("th: ", th)
    # apply one more merge
    frames_pcd = frames_pcd[0]
    frames_pcd = merge_3d_masks(
        frames_pcd, overlap_threshold=0.75, radius=down_size, iou_thresh=proxy_th
    )
    return frames_pcd


def seq_merge(frames_pcd, th, down_size, proxy_th):
    """
        Merge the frames in the list of frames sequentially
        :param frames_pcd (list): list of point clouds
        :param th (float): threshold for overlapping ratio
        :param down_size (float): radius for downsampling
        :param proxy_th (float): threshold for iou
        :return: merged point clouds and features
    """

    global_masks = frames_pcd[0]
    for i in tqdm(range(1, len(frames_pcd))):
        mask_list = global_masks + frames_pcd[i]
        merged_mask_list = merge_3d_masks(
            mask_list,
            overlap_threshold=th,
            radius=down_size,
            iou_thresh=proxy_th,
        )
        global_masks = mask_list
    
    # apply one more merge
    global_masks = merge_3d_masks(
        global_masks, overlap_threshold=th, radius=down_size, iou_thresh=proxy_th
    )
    return global_masks

