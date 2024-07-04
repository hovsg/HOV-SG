
import os 
import numpy as np
from PIL import Image
import json
import open3d as o3d

from hovsg.dataloader.generic import RGBDDataset

class ReplicaDataset(RGBDDataset):
    """
    Dataset class for the ScanNet dataset.

    This class provides an interface to load RGB-D data samples from the ScanNet
    dataset. The dataset format is assumed to follow the ScanNet v2 dataset format.
    """
    
    def __init__(self, cfg):
        """
        Args:
            root_dir: Path to the root directory containing the dataset.
            mode: "train", "val", or "test" depending on the data split.
            transforms: Optional transformations to apply to the data.
        """
        super(ReplicaDataset, self).__init__(cfg)
        self.root_dir = cfg["root_dir"]
        self.transforms = cfg["transforms"]
        self.depth_intrinsics, self.scale = self._load_depth_intrinsics(os.path.split(self.root_dir)[0] + "/cam_params.json")
        self.data_list = self._get_data_list()
        
    def __getitem__(self, idx):
        """
        Get a data sample based on the given index.

        Args:
            idx: Index of the data sample.

        Returns:
            RGB image and depth image as numpy arrays.
        """
        rgb_path, depth_path = self.data_list[idx]
        rgb_image = self._load_image(rgb_path)
        depth_image = self._load_depth(depth_path)
        pose = self._load_pose(self.root_dir, idx)
        if self.transforms is not None:
            # convert to Tensor
            rgb_image = self.transforms(rgb_image)
            depth_image = self.transforms(depth_image)
        return rgb_image, depth_image, pose, list(), self.depth_intrinsics
    
    def _get_data_list(self):
        """
        Get a list of RGB-D data samples based on the dataset format and mode.

        Returns:
            List of RGB-D data samples (RGB image path, depth image path).
        """
        rgb_data_list = []
        depth_data_list = []
        pose_data_list = []
        # lis all files in the root directory + results, that start wuth rgb, depth
        for root, dirs, files in os.walk(os.path.join(self.root_dir, "results")):
            for file in files:
                if file.startswith("frame"):
                    rgb_data_list.append(os.path.join(root, file))
                elif file.startswith("depth"):
                    depth_data_list.append(os.path.join(root, file))
        # sort the data list
        rgb_data_list.sort()
        depth_data_list.sort()
        return list(zip(rgb_data_list, depth_data_list))
        
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
    
    def _load_pose(self, path, idx):
        """
        Load the camera pose from the given path.

        Args:
            path: Path to the camera pose file.

        Returns:
            Camera pose as a numpy array (4x4 matrix).
        """
        path = os.path.join(path, "traj.txt")
        with open(path, "r") as file:
            lines = file.readlines()
            if 0 <= idx < len(lines):
                line = lines[idx]
                values = [float(val) for val in line.split()]
                # Reshape the 16 values into a 4x4 matrix
                transformation_matrix = np.array(values).reshape((4, 4))
                return transformation_matrix
    
    def _load_depth_intrinsics(self, path):
        """
        Load the depth camera intrinsics from the given path.

        Args:
            path: Path to the depth camera intrinsics file.

        Returns:
            RGB camera intrinsics as a numpy array (3x3 matrix)
            and the depth scale factor.
        """
        with open(path, "r") as file:
            data = json.load(file)
            camera_params = data.get("camera")
            if camera_params:
                w = camera_params.get("w")
                h = camera_params.get("h")
                fx = camera_params.get("fx")
                fy = camera_params.get("fy")
                cx = camera_params.get("cx")
                cy = camera_params.get("cy")
                scale = camera_params.get("scale")
                # Creating the camera matrix K
                K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                K = np.array(K)
                return K, scale
    
    def create__pcd(self, rgb, depth, camera_pose=None):
        """
        This method should be implemented by subclasses to create a point cloud 
        from RGB-D images.
        """
        rgb = np.array(rgb)
        depth = np.array(depth)
        scale = self.scale
        camera_matrix = self.depth_intrinsics
        x, y = np.meshgrid(np.arange(rgb.shape[1]), np.arange(rgb.shape[0]))
        # neglect points with depth = 0
        depth = depth.astype(np.float32) / scale
        mask = depth > 0
        x = x[mask]
        y = y[mask]
        depth = depth[mask]

        # convert to 3D
        X = (x - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
        Y = (y - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
        Z = depth

        # convert to camera coordinate
        points = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = rgb[mask]
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd.transform(camera_pose)
        return pcd
