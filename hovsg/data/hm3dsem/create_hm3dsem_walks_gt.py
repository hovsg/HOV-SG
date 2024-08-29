from collections import defaultdict
import json
import math
import os

import cv2
import habitat_sim
import hydra
import numpy as np
import open3d as o3d
import pandas as pd

from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageColor
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from hovsg.data.hm3dsem.habitat_utils import make_cfg


class PanopticObject:
    def __init__(self, line):
        # semantics object info
        self.id = int(line[0])
        self.hex = line[1]
        self.category = eval(line[2])
        self.region_id = int(line[3])
        print(self.category)
        self.floor_id = None
        self.rgb = np.array(ImageColor.getcolor("#" + self.hex, "RGB"))
        self.type = "object"
        self.mapped = False

        # habitat object info
        self.aabb_center = None
        self.aabb_dims = None
        self.obb_center = None
        self.obb_dims = None
        self.obb_rotation = None
        self.obb_local_to_world = None
        self.obb_world_to_local = None
        self.obb_volume = None
        self.obb_half_extents = None

        # point cloud data
        self.points = None
        self.colors = None

    def __print__(self):
        print("id:", self.id, "category:", self.category, "hex_color:", self.hex, "region_id:", self.region_id)

    def hex2id(self, hex_color):
        return self.id

    def __str__(self) -> str:
        return f"{self.floor_id}_{self.region_id}_{self.id}"


def rgb2hex(color_array):
    color_array = color_array * 255
    return "#%02x%02x%02x" % (int(color_array[0]), int(color_array[1]), int(color_array[2]))


class PanopticRegion:
    def __init__(self, region_id):
        self.id = region_id
        self.graph_id = None
        self.floor_id = None
        self.objects = []
        self.type = "room"
        self.voted_category = None
        self.category = None

        self.min_height = None
        self.max_height = None
        self.mean_height = None
        self.region_points = None
        self.bev_region_points = None

    def project_regions(self):
        # Aggregate region point cloud
        print("Projecting region ", self.id, " w/ ", len(self.objects), " objects")
        self.region_points = np.concatenate([obj.points for obj in self.objects if obj.mapped], axis=0)
        self.region_colors = np.concatenate([obj.colors for obj in self.objects if obj.mapped], axis=0)
        self.min_height = np.min(self.region_points[:, 1])
        self.max_height = np.max(self.region_points[:, 1])
        self.mean_height = np.mean(self.region_points[:, 1])

        # project region point cloud to xz plane and use min height as region height
        self.region_point_cloud = o3d.geometry.PointCloud()
        self.region_point_cloud.points = o3d.utility.Vector3dVector(self.region_points)
        self.region_point_cloud.colors = o3d.utility.Vector3dVector(self.region_colors)
        self.bev_region_point_cloud = o3d.geometry.PointCloud()
        self.bev_region_point_cloud.points = o3d.utility.Vector3dVector(
            np.stack(
                [
                    self.region_points[:, 0],
                    np.repeat(self.min_height, self.region_points.shape[0]),
                    self.region_points[:, 2],
                ],
                axis=1,
            )
        )
        self.bev_region_point_cloud = self.bev_region_point_cloud.voxel_down_sample(0.05)

    def __print__(self) -> str:
        return f"{self.floor_id}_{self.id}"


class PanopticLevel:
    def __init__(self, level_id, lower, upper):
        self.id = level_id
        self.lower = lower
        self.upper = upper
        self.type = "floor"

        self.regions = []
        self.objects = []

    def __print__(self) -> str:
        return f"{self.id}"


class PanopticScene:
    def __init__(self, scene_dir_name, habitat_scene, panoptic_object_list):
        self.scene = habitat_scene
        self.scene_dir_name = scene_dir_name
        self.objects = panoptic_object_list
        self.regions = defaultdict(PanopticRegion)
        self.floors = defaultdict(PanopticLevel)

        self.scene_info = {"levels": [], "regions": [], "objects": []}

        self.id2obj_idx = {}
        for i, obj in enumerate(self.objects):
            self.id2obj_idx[obj.id] = i

        self.hex2obj_idx = {}
        for i, obj in enumerate(self.objects):
            self.hex2obj_idx[obj.hex] = i

        self.hex2id = {}
        for i, obj in enumerate(self.objects):
            self.hex2id[obj.hex] = obj.id
        self.id2hex = {v: k for k, v in self.hex2id.items()}
        self.id2rgb = {id: ImageColor.getcolor("#" + hex, "RGB") for id, hex in self.id2hex.items()}

        self.append_habitat_infos()

    def append_habitat_infos(self):
        for obj in self.scene.objects:
            obj_id = int(obj.id.split("_")[1])

            if obj_id in self.id2obj_idx:
                self.objects[self.id2obj_idx[obj_id]].aabb_center = obj.aabb.center.tolist()
                self.objects[self.id2obj_idx[obj_id]].aabb_dims = obj.aabb.sizes.tolist()
                self.objects[self.id2obj_idx[obj_id]].obb_center = obj.obb.center.tolist()
                self.objects[self.id2obj_idx[obj_id]].obb_dims = obj.obb.sizes.tolist()
                self.objects[self.id2obj_idx[obj_id]].obb_rotation = obj.obb.rotation.tolist()
                self.objects[self.id2obj_idx[obj_id]].obb_local_to_world = obj.obb.local_to_world.tolist()
                self.objects[self.id2obj_idx[obj_id]].obb_world_to_local = obj.obb.world_to_local.tolist()
                self.objects[self.id2obj_idx[obj_id]].obb_volume = obj.obb.volume
                self.objects[self.id2obj_idx[obj_id]].obb_half_extents = obj.obb.half_extents.tolist()

    def get_object(self, key):
        if isinstance(key, str):
            return self.objects[self.hex2obj_idx[key]]
        elif isinstance(key, int):
            return self.objects[self.id2obj_idx[key]]
        else:
            raise NotImplementedError

    def construct_regions(self):
        for obj in self.objects:
            if obj.mapped:
                if (obj.region_id) not in self.regions:
                    self.regions[int(obj.region_id)] = PanopticRegion(int(obj.region_id))
                self.regions[int(obj.region_id)].objects.append(obj)

        for region_id in self.regions.keys():
            self.regions[region_id].project_regions()

    def label_mapped_objects(self):
        for obj in self.objects:
            if obj.points is not None:
                obj.mapped = True

    def label_regions(self, region_votes_file_path, region_labels_file_path):
        # load region labels
        region_votes = pd.read_csv(
            region_votes_file_path, header=0, usecols=["Scene Name", "Region #", "Weighted Room Proposal"], sep=","
        )
        print("Labeling regions based on provided category votes")
        for ind in region_votes.index:
            if region_votes["Scene Name"][ind] == self.scene_dir_name:
                if int(region_votes["Region #"][ind]) in self.regions:
                    self.regions[int(region_votes["Region #"][ind])].voted_category = (
                        region_votes["Weighted Room Proposal"][ind].strip().lower()
                    )
                else:
                    print(
                        "Region {} not contained in the mapped scene, thus not labeling it".format(
                            region_votes["Region #"][ind]
                        )
                    )

        region_labels = pd.read_csv(
            region_labels_file_path, header=0, usecols=["Scene Name", "Region #", "Region Category"], sep=","
        )
        print("Labeling regions based on own manual category labels")
        for ind in region_labels.index:
            if region_labels["Scene Name"][ind] == self.scene_dir_name:
                if int(region_labels["Region #"][ind]) in self.regions:
                    self.regions[int(region_labels["Region #"][ind])].category = (
                        region_labels["Region Category"][ind].strip().lower()
                    )
                else:
                    print(
                        "Region {} not contained in the mapped scene, thus not labeling it".format(
                            region_labels["Region #"][ind]
                        )
                    )

    def get_region_objects(self, region_id):
        if isinstance(region_id, str):
            region_id = int(region_id)
        return self.regions[region_id]
    
    def obtain_floor_separation_heights(self, floor_labels_file_path):

        floor_height_labels = pd.read_csv(floor_labels_file_path, 
                                          header=0, 
                                          usecols=["Scene Name", "Separation Heights"], 
                                          sep=",")

        self.floor_coords = []
        for ind in floor_height_labels.index:
            if floor_height_labels["Scene Name"][ind] == self.scene_dir_name:
                self.floor_coords = eval(floor_height_labels["Separation Heights"][ind])

        assert len(self.floor_coords) > 0

        self.num_floors = len(self.floor_coords) - 1
        print("number of floors:", self.num_floors)
        print("floor_coords:", self.floor_coords)
        self.assign_regions_to_floors()

    def assign_regions_to_floors(self):
        # go through all regions and assign floor id
        for floor_idx in range(self.num_floors):
            lower = self.floor_coords[floor_idx]
            upper = self.floor_coords[floor_idx + 1]
            self.floors[floor_idx] = PanopticLevel(floor_idx, lower, upper)
        for region in list(self.regions.values()):
            for floor in list(self.floors.values()):
                if region.mean_height > floor.lower and region.mean_height < floor.upper:
                    print("region {} assigned to floor {}".format(region.id, floor.id))
                    region.floor_id = floor.id
                    floor.regions.append(region)
                    floor.objects.extend(region.objects)
                    # Assign floor id to objects
                    for region_obj in region.objects:
                        self.objects[self.id2obj_idx[region_obj.id]].floor_id = floor.id
            assert region.floor_id is not None
            # based on region categorization into floors, take min/max heights to override floor heights
        for floor_idx in range(self.num_floors):
            self.floors[floor_idx].lower = np.mean([region.min_height for region in self.floors[floor_idx].regions])
            self.floors[floor_idx].upper = np.mean([region.max_height for region in self.floors[floor_idx].regions])
            print("updated means:", self.floors[floor_idx].lower, self.floors[floor_idx].upper)

    def write_metadata(self, save_dir):
        # write level information
        for floor_idx, floor_obj in self.floors.items():
            floor_item = {"id": floor_idx, "lower": floor_obj.lower, "upper": floor_obj.upper}
            floor_item["regions"] = [region.id for region in floor_obj.regions]
            floor_item["objects"] = [obj.id for obj in floor_obj.objects]
            self.scene_info["levels"].append(floor_item)

        # write region information
        for region_id, region_obj in self.regions.items():
            region_item = {
                "id": region_id,
                "floor_id": region_obj.floor_id,
                "voted_category": region_obj.voted_category,
                "category": region_obj.category,
                "min_height": region_obj.min_height,
                "max_height": region_obj.max_height,
                "mean_height": region_obj.mean_height,
                "bev_region_points": np.array(region_obj.bev_region_point_cloud.points).tolist(),
            }
            region_item["objects"] = [obj.id for obj in region_obj.objects if obj.mapped]
            self.scene_info["regions"].append(region_item)

        for obj in self.objects:
            if obj.mapped:
                object_item = {
                    "id": obj.id,
                    "category": obj.category,
                    "hex": obj.hex,
                    "region_id": obj.region_id,
                    "floor_id": obj.floor_id,
                    "aabb_center": obj.aabb_center,
                    "aabb_dims": obj.aabb_dims,
                    "obb_center": obj.obb_center,
                    "obb_dims": obj.obb_dims,
                    "obb_rotation": obj.obb_rotation,
                    "obb_local_to_world": obj.obb_local_to_world,
                    "obb_world_to_local": obj.obb_world_to_local,
                    "obb_volume": obj.obb_volume,
                    "obb_half_extents": obj.obb_half_extents,
                    # "points": obj.points.tolist(),
                    # "colors": object.colors.tolist(),
                }
                self.scene_info["objects"].append(object_item)

        # save scene info as JSON
        with open(os.path.join(save_dir, "scene_info.json"), "w") as file:
            json.dump(self.scene_info, file)


def read_camera_pose_hmp3d(file_path):
    """
    for habitat mp3d dataset, read first line of camera pose file
    16 separate by space, reshape to 4x4 matrix
    """
    with open(file_path, "r") as file:
        line = file.readline().strip()
        values = line.split()
        values = [float(val) for val in values]
        transformation_matrix = np.array(values).reshape((4, 4))
        C = np.eye(4)
        C[1, 1] = -1
        C[2, 2] = -1
        transformation_matrix = np.matmul(transformation_matrix, C)
    return transformation_matrix


def create_pcd_hmp3d(rgb, depth, camera_pose=None):
    """
    for habitat mp3d dataset, create point cloud from RGBD images
    params:
        rgb_img: numpy array of shape (H, W, 3)
        depth_img: numpy array of shape (H, W)
        camera_matrix: numpy array of shape (3, 3)
        depth_scale: depth scale factor
    return:
        pcd: Open3D point cloud
    """
    H = rgb.shape[0]
    W = rgb.shape[1]

    hfov = 90 * np.pi / 180
    vfov = 2 * math.atan(np.tan(hfov / 2) * H / W)
    fx = W / (2.0 * np.tan(hfov / 2.0))
    fy = H / (2.0 * np.tan(vfov / 2.0))
    cx = W / 2
    cy = H / 2
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

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


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


def parse_semantics(scene_dir, scene_mesh, txt_path, raw_scene_dir, dataset_dir, scene_name):

    sim_settings = {
        "scene": scene_mesh,
        "default_agent": 0,
        "sensor_height": 1.5,
        "color_sensor": True,
        "depth_sensor": True,
        "semantic_sensor": True,
        "lidar_sensor": False,
        "move_forward": 0.2,
        "move_backward": 0.2,
        "turn_left": 5,
        "turn_right": 5,
        "look_up": 5,
        "look_down": 5,
        "look_left": 5,
        "look_right": 5,
        "width": 1080,
        "height": 720,
        "enable_physics": False,
        "seed": 42,
        "lidar_fov": 360,
        "depth_img_for_lidar_n": 20,
        # "img_save_dir": save_dir,
        "raw_data_dir": raw_scene_dir,
        "dataset_dir": dataset_dir,
        "scene_name": scene_name,
    }

    sim_cfg = make_cfg(sim_settings, 
                       dataset_dir, 
                       raw_scene_dir,
                       scene_name)
    sim = habitat_sim.Simulator(sim_cfg)
    scene = sim.semantic_scene

    panoptic_object_list = []

    # load txt file
    with open(txt_path, "r") as file:
        lines = file.readlines()[1:]

    for i, line in enumerate(lines):
        object_desc = line.strip().split(",")
        panoptic_object_list.append(PanopticObject(object_desc))

    return PanopticScene(scene_dir, scene, panoptic_object_list)


@hydra.main(version_base=None, config_path="../../../config", config_name="create_graph")
def main(params: DictConfig):
    
    dataset_dir = params.main.raw_data_path # raw HM3D dataset
    walks_path = params.main.dataset_path # processed hm3dsem_walks dataset

    split = params.main.split
    scene_dir = params.main.scene_id
    scene_name = scene_dir.split("-")[-1]

    raw_scene_dir = "{}/{}/{}/".format(dataset_dir, split, scene_dir)
    scene_mesh = os.path.join(raw_scene_dir, scene_name + ".glb")
    panoptics_labels_file = os.path.join(raw_scene_dir, scene_name + ".semantic.txt")
    panoptic_scene = parse_semantics(
        scene_dir, scene_mesh, panoptics_labels_file, raw_scene_dir, dataset_dir, scene_name
    )

    rgb_image_path = os.path.join(walks_path, split, scene_dir, "rgb")
    panoptic_image_path = os.path.join(walks_path, split, scene_dir, "semantic")
    depth_image_path = os.path.join(walks_path, split, scene_dir, "depth")
    camera_pose_file_path = os.path.join(walks_path, split, scene_dir, "pose")

    floor_labels_file_path = os.path.join(params.main.package_path, "data/hm3dsem/metadata/Per_Scene_Floor_Sep.csv")

    region_votes_file_path = os.path.join(params.main.package_path, "data/hm3dsem/metadata/Per_Scene_Region_Weighted_Votes.csv")
    region_labels_file_path = os.path.join(params.main.package_path, "data/hm3dsem/metadata/Per_Scene_Region_Labels.csv")

    # read rgb, depth, camera pose successively
    all_rgb_image_files = os.listdir(rgb_image_path)
    all_depth_image_files = os.listdir(depth_image_path)
    all_pose_files = os.listdir(camera_pose_file_path)

    all_rgb_image_files.sort()
    all_depth_image_files.sort()
    all_pose_files.sort()

    rgb_pcd = o3d.geometry.PointCloud()
    panoptic_pcd = o3d.geometry.PointCloud()
    poses = []

    num_point_clouds = 0
    for i in tqdm(range(0, len(all_rgb_image_files), params.dataset.hm3dsem.gt_skip_frames), desc="Processing frames"):
        file_name = all_rgb_image_files[i]
        tqdm.write(file_name)
        rgb = cv2.imread(os.path.join(rgb_image_path, file_name))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        panoptic_ids = np.load(os.path.join(panoptic_image_path, file_name.replace("png", "npy")))
        panoptic_rgb = np.zeros((panoptic_ids.shape[0], panoptic_ids.shape[1], 3), dtype=np.uint8)
        for id, id_rgb in panoptic_scene.id2rgb.items():
            panoptic_rgb[panoptic_ids == id, :] = id_rgb
        # panoptic_image = Image.fromarray(panoptic_rgb) # as sanity check
        # panoptic_image.save(os.path.join(walks_path, split, scene_dir, "panoptic", file_name))
        depth = cv2.imread(os.path.join(depth_image_path, file_name), cv2.IMREAD_ANYDEPTH)
        camera_pose = read_camera_pose_hmp3d(os.path.join(camera_pose_file_path, file_name.replace("png", "txt")))
        poses.append(camera_pose[:3, 3])
        frame_color_pcd = create_pcd_hmp3d(rgb, depth, camera_pose)
        frame_panoptic_pcd = create_pcd_hmp3d(panoptic_rgb, depth, camera_pose)
        rgb_pcd += frame_color_pcd
        panoptic_pcd += frame_panoptic_pcd
        num_point_clouds += 1

        if num_point_clouds % 500 == 0:
            # downsample point cloud
            tqdm.write("--> downsampling point cloud")
            rgb_pcd = rgb_pcd.voxel_down_sample(0.02)
            panoptic_pcd = panoptic_pcd.voxel_down_sample(0.02)

    # # plot camera trajectory
    # pose_min_coord = np.min(np.array(poses))
    # pose_max_coord = np.max(np.array(poses))
    # plt.figure()
    # plt.xlim(pose_min_coord, pose_max_coord)
    # plt.ylim(pose_min_coord, pose_max_coord)
    # plt.gca().invert_yaxis()
    # plt.plot(np.array(poses)[:, 0], np.array(poses)[:, 2])
    # plt.grid()
    # plt.savefig(os.path.join(walks_path, split, scene_dir, "cam_trajectory.png"))

    print("full_pcd:", len(rgb_pcd.points))
    rgb_pcd = rgb_pcd.voxel_down_sample(0.02)
    print("full_pcd after voxelization:", len(rgb_pcd.points))
    # save point cloud and mesh
    o3d.io.write_point_cloud(os.path.join(walks_path, split, scene_dir, "scene_rgb.ply"), rgb_pcd)

    print("panoptic_pcd:", len(panoptic_pcd.points))
    panoptic_pcd = panoptic_pcd.voxel_down_sample(0.02)
    o3d.io.write_point_cloud(os.path.join(walks_path, split, scene_dir, "scene_panoptic.ply"), panoptic_pcd)
    print("panoptic_pcd after voxelization:", len(panoptic_pcd.points))

    # Go through panoptic point cloud and extract all points per object instance
    # and save to separate point cloud file
    if not os.path.exists(os.path.join(walks_path, split, scene_dir, "objects")):
        os.makedirs(os.path.join(walks_path, split, scene_dir, "objects"))
    if not os.path.exists(os.path.join(walks_path, split, scene_dir, "regions")):
        os.makedirs(os.path.join(walks_path, split, scene_dir, "regions"))

    pan_colors = np.asarray(panoptic_pcd.colors)  # * 255).astype(np.uint8)
    for obj in panoptic_scene.objects:
        id_filter = np.all(np.isclose(pan_colors - obj.rgb / 255.0, 0.0), axis=1)
        print("Object ID: ", obj.id, "w/", np.sum(id_filter), "points")
        obj_points = np.asarray(rgb_pcd.points)[id_filter, :]
        obj_colors = np.asarray(rgb_pcd.colors)[id_filter, :]

        if len(obj_points) > 0:
            obj.mapped = True
            # Add obj point cloud to objects and regions
            panoptic_scene.objects[panoptic_scene.id2obj_idx[obj.id]].points = obj_points
            panoptic_scene.objects[panoptic_scene.id2obj_idx[obj.id]].colors = obj_colors

            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
            obj_pcd.colors = o3d.utility.Vector3dVector(obj_colors)
            o3d.io.write_point_cloud(
                os.path.join(walks_path, split, scene_dir, "objects", "{}.ply".format(obj.id)), obj_pcd
            )
    panoptic_scene.construct_regions()
    panoptic_scene.label_regions(region_votes_file_path, region_labels_file_path)
    panoptic_scene.obtain_floor_separation_heights(floor_labels_file_path)

    # save region point clouds
    for region in panoptic_scene.regions.values():
        o3d.io.write_point_cloud(
            os.path.join(walks_path, split, scene_dir, "regions", "{}.ply".format(region.id)),
            region.region_point_cloud,
        )

    for region in list(panoptic_scene.regions.values()):
        print(
            "floor",
            region.floor_id,
            "region",
            region.id,
            "# obj",
            len([obj for obj in region.objects if obj.mapped]),
            "min/max/mean r-height",
            region.min_height,
            region.max_height,
            region.mean_height,
        )

    # write meta data infos to JSON file
    panoptic_scene.write_metadata(os.path.join(walks_path, split, scene_dir))


if __name__ == "__main__":
    main()