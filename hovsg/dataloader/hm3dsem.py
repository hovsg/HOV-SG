"""
    Habitat Matterport 3D Semantics dataset loader.
"""

import sys
import os
import math
import numpy as np
from PIL import Image
import torchvision
import open3d as o3d

from hovsg.dataloader.generic import RGBDDataset

# pylint: disable=all

class HM3DSemDataset(RGBDDataset):
    """
    Dataset class for the Habitat Matterport3D Semantic dataset.

    This class provides an interface to load RGB-D data samples from the ScanNet
    dataset. The dataset format is assumed to follow the ScanNet v2 dataset format.
    """    
    def __init__(self, cfg):
        """
        Args:
            root_dir: Path to the root directory containing the dataset.
            transforms: Optional transformations to apply to the data.
        """
        super(HM3DSemDataset, self).__init__(cfg)
        self.root_dir = cfg["root_dir"]
        self.transforms = cfg["transforms"]
        self.data_list = self._get_data_list()
        self.rgb_H = self._load_image(self.data_list[0][0]).size[1]
        self.rgb_W = self._load_image(self.data_list[0][0]).size[0]
        self.depth_intrinsics = self._load_depth_intrinsics(self.rgb_H, self.rgb_W)
        self.scale = 1000.0
    
    def __getitem__(self, idx):
        """
        Get a data sample based on the given index.

        Args:
            idx: Index of the data sample.

        Returns:
            RGB image and depth image as numpy arrays.
        """
        rgb_path, depth_path, pose_path = self.data_list[idx]
        rgb_image = self._load_image(rgb_path)
        depth_image = self._load_depth(depth_path)
        pose = self._load_pose(pose_path)
        depth_intrinsics = self._load_depth_intrinsics(self.rgb_H, self.rgb_W)
        if self.transforms is not None:
            rgb_image = self.transforms(rgb_image)
            depth_image = self.transforms(depth_image)   
        return rgb_image, depth_image, pose, list(), depth_intrinsics
    
    def _get_data_list(self):
        """
        Get a list of RGB-D data samples based on the dataset format and mode.

        Returns:
            List of RGB-D data samples (RGB image path, depth image path).
        """
        rgb_data_list = []
        depth_data_list = []
        pose_data_list = []
        rgb_data_list = os.listdir(self.root_dir + "/rgb")
        rgb_data_list = [self.root_dir + "/rgb/" + x for x in rgb_data_list]
        depth_data_list = os.listdir(self.root_dir + "/depth")
        depth_data_list = [self.root_dir + "/depth/" + x for x in depth_data_list]
        pose_data_list = os.listdir(self.root_dir + "/pose")
        pose_data_list = [self.root_dir + "/pose/" + x for x in pose_data_list]
        # sort the data list
        rgb_data_list.sort()
        depth_data_list.sort()
        pose_data_list.sort()
        return list(zip(rgb_data_list, depth_data_list, pose_data_list))
        
    def _load_image(self, path):
        """
        Load the RGB image from the given path.

        Args:
            path: Path to the RGB image file.

        Returns:
            RGB image as a numpy array.
        """
        # Load the RGB image using PIL
        rgb_image = Image.open(path)
        return rgb_image

    def _load_depth(self, path):
        """
        Load the depth image from the given path.

        Args:
            path: Path to the depth image file.

        Returns:
            Depth image as a numpy array.
        """
        # Load the depth image using OpenCV
        depth_image = Image.open(path)
        return depth_image
    
    def _load_pose(self, path):
        """
        Load the camera pose from the given path.

        Args:
            path: Path to the camera pose file.

        Returns:
            Camera pose as a numpy array (4x4 matrix).
        """
        with open(path, "r") as file:
            line = file.readline().strip()
            values = line.split()
            values = [float(val) for val in values]
            transformation_matrix = np.array(values).reshape((4, 4))
            C = np.eye(4)
            C[1, 1] = -1
            C[2, 2] = -1
            transformation_matrix = np.matmul(transformation_matrix, C)
        return transformation_matrix
    
    def _load_depth_intrinsics(self, H, W):
        """
        Load the depth camera intrinsics.

        Returns:
            Depth camera intrinsics as a numpy array (3x3 matrix).
        """        
        hfov = 90 * np.pi / 180
        vfov = 2 * math.atan(np.tan(hfov / 2) * H / W)
        fx = W / (2.0 * np.tan(hfov / 2.0))
        fy = H / (2.0 * np.tan(vfov / 2.0))
        cx = W / 2
        cy = H / 2
        depth_camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return depth_camera_matrix

    def create__pcd(self, rgb, depth, camera_pose=None):
        """
        Create a point cloud from RGB-D images.

        Args:
            rgb: RGB image as a numpy array.
            depth: Depth image as a numpy array.
            camera_pose: Camera pose as a numpy array (4x4 matrix).

        Returns:
            Point cloud as an Open3D object.
        """
        # convert rgb and depth images to numpy arrays
        rgb = np.array(rgb)
        depth = np.array(depth)
        # load depth camera intrinsics
        H = rgb.shape[0]
        W = rgb.shape[1]
        camera_matrix = self._load_depth_intrinsics(H, W)
        # create point cloud
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        depth = depth.astype(np.float32) / 1000.0
        mask = depth > 0
        x = x[mask]
        y = y[mask]
        depth = depth[mask]
        # convert to 3D
        X = (x - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
        Y = (y - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
        Z = depth
        # convert to open3d point cloud
        points = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = rgb[mask]
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd.transform(camera_pose)
        return pcd
    
