
import os 
import numpy as np
from PIL import Image
import torchvision
import open3d as o3d

from hovsg.dataloader.generic import RGBDDataset


class ScannetDataset(RGBDDataset):
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
        super(ScannetDataset, self).__init__(cfg)
        self.root_dir = cfg["root_dir"]
        self.transforms = cfg["transforms"]
        self.rgb_intrinsics = self._load_rgb_intrinsics(self.root_dir + "intrinsic/intrinsic_color.txt")
        self.depth_intrinsics = self._load_depth_intrinsics(self.root_dir + "intrinsic/intrinsic_depth.txt")
        self.scale = 1000.0
        self.data_list = self._get_data_list()
        
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

        if self.transforms is not None:
            # convert to Tensor
            rgb_image = self.transforms(rgb_image)
            depth_image = self.transforms(depth_image)
            
        return rgb_image, depth_image, pose, self.rgb_intrinsics, self.depth_intrinsics
    
    def _get_data_list(self):
        """
        Get a list of RGB-D data samples based on the dataset format and mode.

        Returns:
            List of RGB-D data samples (RGB image path, depth image path).
        """
        rgb_data_list = []
        depth_data_list = []
        pose_data_list = []
        rgb_data_list = os.listdir(self.root_dir + "color")
        rgb_data_list = [self.root_dir + "color/" + x for x in rgb_data_list]
        depth_data_list = os.listdir(self.root_dir + "depth")
        depth_data_list = [self.root_dir + "depth/" + x for x in depth_data_list]
        pose_data_list = os.listdir(self.root_dir + "pose")
        pose_data_list = [self.root_dir + "pose/" + x for x in pose_data_list]
        # sort the data list
        rgb_data_list.sort()
        depth_data_list.sort()
        pose_data_list.sort()
        return list(zip(rgb_data_list, depth_data_list, pose_data_list))
        
    
    def _get_test_data_list(self):
        """
        Get a list of test data samples.

        Returns:
            List of test data samples (RGB image path, depth image path).
        """
        test_data_list = []
        # Load test data samples from the dataset
        # ...
        return test_data_list

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
        with open(path, "r") as f:
            pose = []
            for line in f:
                pose.append([float(x) for x in line.split()])
        pose = np.array(pose)
        return pose
    
    def _load_rgb_intrinsics(self, path):
        """
        Load the RGB camera intrinsics from the given path.

        Args:
            path: Path to the RGB camera intrinsics file.

        Returns:
            RGB camera intrinsics as a numpy array (3x3 matrix).
        """
        with open(path, "r") as f:
            intrinsics = []
            for line in f:
                intrinsics.append([float(x) for x in line.split()])
        intrinsics = np.array(intrinsics)
        return intrinsics
    
    def _load_depth_intrinsics(self, path):
        """
        Load the depth camera intrinsics from the given path.

        Args:
            path: Path to the depth camera intrinsics file.

        Returns:
            Depth camera intrinsics as a numpy array (3x3 matrix).
        """
        with open(path, "r") as f:
            intrinsics = []
            for line in f:
                intrinsics.append([float(x) for x in line.split()])
        intrinsics = np.array(intrinsics)
        return intrinsics

    def create__pcd(self, rgb, depth, camera_pose=None):
        """
        This method should be implemented by subclasses to create a point cloud 
        from RGB-D images.
        """
        rgb = np.array(rgb)
        depth = np.array(depth)
        rgb = np.array(Image.fromarray(rgb).resize((depth.shape[1], depth.shape[0])))
        depth_scale = 1000.0
        camera_matrix = self.depth_intrinsics
        depth_img = depth.astype(np.float32) / depth_scale
        x, y = np.meshgrid(np.arange(depth_img.shape[1]), np.arange(depth_img.shape[0]))
        mask = depth_img > 0 
        x = x[mask]
        y = y[mask]
        depth_img = depth_img[mask]
        X = (x - camera_matrix[0, 2]) * depth_img / camera_matrix[0, 0]
        Y = (y - camera_matrix[1, 2]) * depth_img / camera_matrix[1, 1]
        Z = depth_img
        pcd = np.hstack(([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1), np.ones((X.shape[0], 1))]))

        # apply projection matrix
        if camera_pose is not None:
            pcd = np.dot(camera_pose, pcd.T).T
            pcd = pcd[:, :3] / pcd[:, 3:]
        colors = rgb.reshape(-1, 3) / 255
        colors = colors[mask.reshape(-1)]
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcd)
        pcd1.colors = o3d.utility.Vector3dVector(colors)
        return pcd1

    
    