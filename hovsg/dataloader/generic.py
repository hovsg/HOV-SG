
import os
from abc import ABC, abstractmethod
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import cv2

from torch.utils.data import Dataset, DataLoader

class RGBDDataset(Dataset, ABC):
    """
    Abstract class for RGBD datasets.

    This class provides a base structure for loading RGBD datasets. Subclasses
    need to implement the abstract methods to handle specific dataset formats.
    """

    def __init__(self, cfg):
        """
        Args:
            root_dir: Path to the root directory containing the dataset.
            mode: "train", "val", or "test" depending on the data split.
            transforms: Optional transformations to apply to the data.
        """
        self.root_dir = cfg["root_dir"]
        self.transforms = cfg["transforms"]
        self.rgb_intrinsics = None
        self.depth_intrinsics = None
        self.scale = None
        self.data_list = self._get_data_list()

    @abstractmethod
    def _get_data_list(self):
        """
        This method should be implemented by subclasses to define how to 
        get a list of data samples (RGB and depth image paths) based on the 
        dataset format and mode (train, val, test).
        """
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
            Get a data sample based on the given index.
        """
        pass

    def _load_image(self, path):
        """
        This method should be implemented by subclasses to load the RGB image 
        based on the dataset format (e.g., OpenCV, PIL).
        """
        pass

    def _load_depth(self, path):
        """
        This method should be implemented by subclasses to load the depth image 
        based on the dataset format (e.g., OpenCV, PIL).
        """
        pass
    
    def _load_pose(self, path):
        """
        This method should be implemented by subclasses to load the camera pose 
        based on the dataset format.
        """
        pass
    
    def _load_rgb_intrinsics(self, path):
        """
        This method should be implemented by subclasses to load the RGB camera 
        intrinsics based on the dataset format.
        """
        pass
    
    def _load_depth_intrinsics(self, path):
        """
        This method should be implemented by subclasses to load the depth camera 
        intrinsics based on the dataset format.
        """
        pass
    
    def create_pcd(self, rgb, depth, camera_pose=None, mask_img=False, filter_distance=np.inf):
        """
        Create Open3D point cloud from RGB and depth images, and camera pose. filter_distance is used to filter out
        points that are further than a certain distance.
        :param rgb (pil image): RGB image
        :param depth (pil image): Depth image
        :param camera_pose (np.array): Camera pose
        :param mask_img (bool): Mask image
        :param filter_distance (float): Filter distance
        :return: Open3D point cloud
        """
        # convert rgb and depth images to numpy arrays
        rgb = np.array(rgb).astype(np.uint8)
        depth = np.array(depth)
        # resize rgb image to match depth image size if needed
        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        # load depth camera intrinsics
        H = rgb.shape[0]
        W = rgb.shape[1]
        camera_matrix = self.depth_intrinsics
        scale = self.scale
        # create point cloud
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        depth = depth.astype(np.float32) / scale
        if mask_img:
            depth = depth * rgb
        mask = depth > 0
        x = x[mask]
        y = y[mask]
        depth = depth[mask]
        # convert to 3D
        X = (x - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
        Y = (y - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
        Z = depth
        if Z.mean() > filter_distance:
            return o3d.geometry.PointCloud()
        # convert to open3d point cloud
        points = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if not mask_img:
            colors = rgb[mask]
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd.transform(camera_pose)
        return pcd
    
    def create_3d_masks(self, masks, depth, full_pcd, full_pcd_tree, camera_pose, down_size=0.02, filter_distance=None):
        """
        create 3d masks from 2D masks
        Args:
            masks: list of 2D masks
            depth: depth image
            full_pcd: full point cloud
            full_pcd_tree: KD-Tree of full point cloud
            camera_pose: camera pose
            down_size: voxel size for downsampling
        Returns:
            list of 3D masks as Open3D point clouds
        """
        camera_matrix = self.depth_intrinsics
        depth_scale = self.scale
        pcd_list = []
        pcd = np.asarray(full_pcd.points)
        depth = np.array(depth)
        for i in range(len(masks)):
            # get the mask
            mask = masks[i]["segmentation"]
            mask = np.array(mask)
            # create pcd from mask
            pcd_masked = self.create_pcd(mask, depth, camera_pose, mask_img=True, filter_distance=filter_distance)
            # using KD-Tree to find the nearest points in the point cloud
            pcd_masked = np.asarray(pcd_masked.points)
            dist, indices = full_pcd_tree.query(pcd_masked, k=1, workers=-1)
            pcd_masked = pcd[indices]
            pcd_mask = o3d.geometry.PointCloud()
            pcd_mask.points = o3d.utility.Vector3dVector(pcd_masked)
            colors = np.asarray(full_pcd.colors)
            colors = colors[indices]
            pcd_mask.colors = o3d.utility.Vector3dVector(colors)
            pcd_mask = pcd_mask.voxel_down_sample(voxel_size=down_size)
            pcd_list.append(pcd_mask)
        return pcd_list