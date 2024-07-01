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
from hovsg.utils.clip_utils import get_text_feats, get_text_feats_multiple_templates



# Function to read PLY file and assign colors based on object_id for replica dataset
def read_ply_and_assign_colors_replica(file_path, semantic_info_path):
    """
    Read PLY file and assign colors based on object_id for replica dataset
    :param file_path: path to PLY file
    :param semantic_info_path: path to semantic info JSON file
    :return: point cloud, class ids, point cloud instance, object ids
    """
    # Read PLY file
    plydata = plyfile.PlyData.read(file_path)
    # Read semantic info
    with open(semantic_info_path) as f:
        semantic_info = json.load(f)

    object_class_mapping = {obj["id"]: obj["class_id"] for obj in semantic_info["objects"]}
    unique_class_ids = np.unique(list(object_class_mapping.values()))

    # Extract vertex data
    vertices = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    # Extract object_id and normalize it to use as color
    face_vertices = plydata["face"]["vertex_indices"]
    object_ids = plydata["face"]["object_id"]
    vertices1 = []
    object_ids1 = []
    for i, face in enumerate(face_vertices):
        vertices1.append(vertices[face])
        object_ids1.append(np.repeat(object_ids[i], len(face)))
    vertices1 = np.vstack(vertices1)
    object_ids1 = np.hstack(object_ids1)

    # set random color for every unique object_id/instance id
    unique_object_ids = np.unique(object_ids)
    instance_colors = np.zeros((len(object_ids1), 3))
    unique_colors = np.random.rand(len(unique_object_ids), 3)
    for i, object_id in enumerate(unique_object_ids):
        instance_colors[object_ids1 == object_id] = unique_colors[i]

    # semantic colors
    class_ids = []
    for object_id in object_ids1:
        if object_id in object_class_mapping.keys():
            class_ids.append(object_class_mapping[object_id])
        else:
            class_ids.append(0)
    class_ids = np.array(class_ids)
    print("class_ids: ", class_ids.shape)
    class_colors = np.zeros((len(object_ids1), 3))
    unique_class_colors = np.random.rand(len(unique_class_ids), 3)
    for i, class_id in enumerate(unique_class_ids):
        class_colors[class_ids == class_id] = unique_class_colors[i]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices1)
    pcd.colors = o3d.utility.Vector3dVector(class_colors)
    pcd_instance = o3d.geometry.PointCloud()
    pcd_instance.points = o3d.utility.Vector3dVector(vertices1)
    pcd_instance.colors = o3d.utility.Vector3dVector(instance_colors)
    return pcd, class_ids, pcd_instance, object_ids1

def text_prompt(clip_model, clip_feat_dim, mask_feats, text, templates=True):
    """
    Compute similarity between text and mask_feats
    :param clip_model: CLIP model
    :param clip_feat_dim: CLIP feature dimension
    :param mask_feats: mask features
    :param text: text
    :param templates: whether to use templates
    :return: similarity
    """
    text_list = text
    if templates:
        text_feats = get_text_feats_multiple_templates(
            text_list, clip_model, clip_feat_dim
        )
    else:
        text_feats = get_text_feats(text_list, clip_model, clip_feat_dim)
    similarity = torch.nn.functional.cosine_similarity(
        torch.from_numpy(mask_feats).unsqueeze(1), torch.from_numpy(text_feats).unsqueeze(0), dim=2
    )
    similarity = similarity.cpu().numpy()
    return similarity


def read_semantic_classes_replica(semantic_info_path, crete_color_map=False):
    """
    Read semantic classes for replica dataset
    :param semantic_info_path: path to semantic info JSON file
    :param crete_color_map: whether to create color map
    :return: class id names
    """
    with open(semantic_info_path) as f:
        semantic_info = json.load(f)
    class_id_names = {obj["id"]: obj["name"] for obj in semantic_info["classes"]}
    if crete_color_map:
        unique_class_ids = np.unique(list(class_id_names.keys()))
        unique_colors = np.random.rand(len(unique_class_ids), 3)
        class_id_colors = {class_id: unique_colors[i] for i, class_id in enumerate(unique_class_ids)}
        # convert to string
        class_id_colors = {str(k): v.tolist() for k, v in class_id_colors.items()}
        # save class_id_colors to json file to use later
        with open("class_id_colors.json", "w") as f:
            json.dump(class_id_colors, f)
    return class_id_names


def load_feture_map(path, normalize=True):
    """
    Load features map from disk, mask_feats.pt and objects/pcd_i.ply
    :param path: path to feature map
    :param normalize: whether to normalize features
    :return: mask_pcds, mask_feats
    """
    if not os.path.exists(path):
        raise FileNotFoundError("Feature map not found in {}".format(path))
    # load mask_feats
    mask_feats = torch.load(os.path.join(path, "mask_feats.pt")).float()
    if normalize:
        mask_feats = torch.nn.functional.normalize(mask_feats, p=2, dim=-1).cpu().numpy()
    else:
        mask_feats = mask_feats.cpu().numpy()
    print("full pcd feats loaded from disk with shape {}".format(mask_feats.shape))
    # load masked pcds
    if os.path.exists(os.path.join(path, "objects")):
        mask_pcds = []
        number_of_pcds = len(os.listdir(os.path.join(path, "objects")))
        not_found = []
        for i in range(number_of_pcds):
            if os.path.exists(os.path.join(path, "objects", "pcd_{}.ply".format(i))):
                mask_pcds.append(
                    o3d.io.read_point_cloud(os.path.join(path, "objects", "pcd_{}.ply".format(i)))
                )
            else:
                print("masked pcd {} not found in {}".format(i, path))
                not_found.append(i)
        print("number of masked pcds loaded from disk {}".format(len(mask_pcds)))
        # remove masks_feats that are not found
        mask_feats = np.delete(mask_feats, not_found, axis=0)
        print("number of mask_feats loaded from disk {}".format(len(mask_feats)))
        return mask_pcds, mask_feats
    else:
        raise FileNotFoundError("objects directory not found in {}".format(path))
        
def compute_3d_iou(pcd1, pcd2, padding=0):
    """
    Compute 3D Intersection over Union (IoU) between two point clouds.
    :param pcd1 (open3d.geometry.PointCloud): Point cloud 1.
    :param pcd2 (open3d.geometry.PointCloud): Point cloud 2.
    :param padding (float): Padding to add to the bounding box.
    :return: 3D IoU between 0 and 1.
    """
    bbox1 = pcd1.get_axis_aligned_bounding_box()
    bbox2 = pcd2.get_axis_aligned_bounding_box()

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


def find_overlapping_ratio(pcd1, pcd2, radius=0.02):
    """
    Calculate the percentage of overlapping points between two point clouds using KD-Trees.

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
    tree_pcd2 = cKDTree(pcd2)
    tree_pcd1 = cKDTree(pcd1)

    # Query all points in pcd1 for nearby points in pcd2
    _, indices1 = tree_pcd2.query(pcd1, k=1, distance_upper_bound=radius, p=2, workers=-1)
    _, indices2 = tree_pcd1.query(pcd2, k=1, distance_upper_bound=radius, p=2, workers=-1)

    # Remove indices that are out of range
    indices1 = indices1[indices1 != pcd2.shape[0]]
    indices2 = indices2[indices2 != pcd1.shape[0]]

    # Calculate the overlapping ratio, handle the case where one of the point clouds is empty
    if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
        overlapping_ratio = 0
    else:
        overlapping_ratio = (len(indices1) + len(indices2)) / (pcd1.shape[0] + pcd2.shape[0])

    return overlapping_ratio


def sim_2_label(similarity, labels_id):
    """
    Convert similarity matrix to labels
    :param similarity: similarity matrix
    :param labels_id: labels id
    :return: labels
    """
    # find the label index with the highest similarity
    label_indices = similarity.argmax(axis=1)
    print("label_indices: ", label_indices)
    # convert label indices to label names
    labels = np.array([labels_id[i] for i in label_indices])
    return labels


def Tree_interpolation(pred_pc: np.ndarray, gt_pc: np.ndarray):
    """
    using cKdtree assign create new pred_label based on gt_label
    for each point in pred_pc, find the nearest point in gt_pc and assign its label
    """
    pred_label = pred_pc[:, -1]
    gt_label = gt_pc[:, -1]
    pred_label_new = np.zeros((gt_pc.shape[0], 1))
    tree = cKDTree(pred_pc[:, :3])
    pred_label_new = pred_label[tree.query(gt_pc[:, :3])[1]]
    print("pred_label_new: ", pred_label_new.shape)
    return pred_label_new


def knn_interpolation(cumulated_pc: np.ndarray, full_sized_data: np.ndarray, k):
    """
    Using k-nn interpolation to find labels of points of the full sized pointcloud
    :param cumulated_pc: cumulated pointcloud results after running the network
    :param full_sized_data: full sized point cloud
    :param k: k for k nearest neighbor interpolation
    :return: pointcloud with predicted labels in last column and ground truth labels in last but one column
    """

    labeled = cumulated_pc[cumulated_pc[:, -1] != -1]
    to_be_predicted = full_sized_data.copy()

    ball_tree = BallTree(labeled[:, :3], metric="minkowski")

    knn_classes = labeled[ball_tree.query(to_be_predicted[:, :3], k=k)[1]][:, :, -1].astype(int)
    print("knn_classes: ", knn_classes.shape)

    interpolated = np.zeros(knn_classes.shape[0])

    for i in range(knn_classes.shape[0]):
        interpolated[i] = np.bincount(knn_classes[i]).argmax()

    output = np.zeros((to_be_predicted.shape[0], to_be_predicted.shape[1] + 1))
    output[:, :-1] = to_be_predicted

    output[:, -1] = interpolated

    assert output.shape[0] == full_sized_data.shape[0]

    return output


# Function to read PLY file and assign colors based on object_id
def read_ply_and_assign_colors(file_path, semantic_info_path):
    """
    Read PLY file and assign colors based on object_id
    :param file_path: path to PLY file
    :param semantic_info_path: path to semantic info JSON file
    :return: point cloud, class ids, point cloud instance, object ids
    """
    # Read PLY file
    plydata = plyfile.PlyData.read(file_path)
    # Read semantic info
    with open(semantic_info_path) as f:
        semantic_info = json.load(f)

    object_class_mapping = {obj["id"]: obj["class_id"] for obj in semantic_info["objects"]}
    unique_class_ids = np.unique(list(object_class_mapping.values()))

    # Extract vertex data
    vertices = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    # Extract object_id and normalize it to use as color
    face_vertices = plydata["face"]["vertex_indices"]
    object_ids = plydata["face"]["object_id"]
    vertices1 = []
    object_ids1 = []
    for i, face in enumerate(face_vertices):
        vertices1.append(vertices[face])
        object_ids1.append(np.repeat(object_ids[i], len(face)))
    vertices1 = np.vstack(vertices1)
    object_ids1 = np.hstack(object_ids1)

    # set random color for every unique object_id/instance id
    unique_object_ids = np.unique(object_ids)
    instance_colors = np.zeros((len(object_ids1), 3))
    unique_colors = np.random.rand(len(unique_object_ids), 3)
    for i, object_id in enumerate(unique_object_ids):
        instance_colors[object_ids1 == object_id] = unique_colors[i]

    # semantic colors
    class_ids = []
    for object_id in object_ids1:
        if object_id in object_class_mapping.keys():
            class_ids.append(object_class_mapping[object_id])
        else:
            class_ids.append(0)
    class_ids = np.array(class_ids)
    print("class_ids: ", class_ids.shape)
    class_colors = np.zeros((len(object_ids1), 3))
    unique_class_colors = np.random.rand(len(unique_class_ids), 3)
    for i, class_id in enumerate(unique_class_ids):
        class_colors[class_ids == class_id] = unique_class_colors[i]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices1)
    pcd.colors = o3d.utility.Vector3dVector(class_colors)
    pcd_instance = o3d.geometry.PointCloud()
    pcd_instance.points = o3d.utility.Vector3dVector(vertices1)
    pcd_instance.colors = o3d.utility.Vector3dVector(instance_colors)
    return pcd, class_ids, pcd_instance, object_ids1


def read_semantic_classes(semantic_info_path):
    with open(semantic_info_path) as f:
        semantic_info = json.load(f)
    class_id_names = {obj["id"]: obj["name"] for obj in semantic_info["classes"]}
    return class_id_names
