from collections import defaultdict
import json
import math
import os

import cv2
import habitat_sim
from matplotlib import pyplot as plt
import numpy as np

# function to display the topdown map
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Union, Dict


def make_cfg(
    settings: Dict, root_dataset_dir: str, raw_data_dir: str, scene_name: str
) -> habitat_sim.Configuration:
    """Create a configuration for the simulator based on the input parameter dictionary.
       This function is for Habitat3DSemantic dataset.

    Args:
        settings (Dict): Necessary parameters for configuring the simulator
        root_dataset_dir (str): The root directory where all scenes' data is stored.
        raw_data_dir (str): The directory where the raw data is stored
        scene_name (str): The name of the scene (without the extension)

    Returns:
        sim_cfg(habitat_sim.Configuration): The configuration for the simulator
    """
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = os.path.join(raw_data_dir, scene_name + ".basis.glb")
    backend_cfg.scene_dataset_config_file = os.path.join(
        root_dataset_dir, "hm3d_annotated_basis.scene_dataset_config.json"
    )

    print("-------------")
    print(backend_cfg.scene_id)
    print(backend_cfg.scene_dataset_config_file)
    print("-------------")

    sensor_spec = []
    back_rgb_sensor_spec = make_sensor_spec(
        "back_color_sensor",
        habitat_sim.SensorType.COLOR,
        settings["height"],
        settings["width"],
        [0.0, settings["sensor_height"], 1.3],
        orientation=[-math.pi / 8, 0, 0],
    )
    sensor_spec.append(back_rgb_sensor_spec)

    if settings["color_sensor"]:
        rgb_sensor_spec = make_sensor_spec(
            "color_sensor",
            habitat_sim.SensorType.COLOR,
            settings["height"],
            settings["width"],
            [0.0, settings["sensor_height"], 0.0],
        )
        sensor_spec.append(rgb_sensor_spec)

    if settings["depth_sensor"]:
        depth_sensor_spec = make_sensor_spec(
            "depth_sensor",
            habitat_sim.SensorType.DEPTH,
            settings["height"],
            settings["width"],
            [0.0, settings["sensor_height"], 0.0],
        )
        sensor_spec.append(depth_sensor_spec)

    if settings["semantic_sensor"]:
        semantic_sensor_spec = make_sensor_spec(
            "semantic",
            habitat_sim.SensorType.SEMANTIC,
            settings["height"],
            settings["width"],
            [0.0, settings["sensor_height"], 0.0],
        )
        sensor_spec.append(semantic_sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_spec
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=settings["move_forward"]),
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward",
            habitat_sim.agent.ActuationSpec(amount=settings["move_backward"]),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=settings["turn_right"])
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=settings["turn_right"])
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=settings["look_up"])
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=settings["look_down"])
        ),
        "look_left": habitat_sim.agent.ActionSpec(
            "look_left", habitat_sim.agent.ActuationSpec(amount=settings["look_left"])
        ),
        "look_right": habitat_sim.agent.ActionSpec(
            "look_right", habitat_sim.agent.ActuationSpec(amount=settings["look_right"])
        ),
    }

    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    # sim_cfg.create_renderer = True

    return sim_cfg


def make_sensor_spec(
    uuid: str,
    sensor_type: habitat_sim.SensorType,
    h: int,
    w: int,
    position: Union[List, np.ndarray],
    orientation: Union[List, np.ndarray] = None,
) -> habitat_sim.CameraSensorSpec:
    """Create the sensor configuration for the habitat-sim simulator.

    Args:
        uuid (str): sensor unique string identifier (unique name of the sensor).
        sensor_type (habitat_sim.SensorType): type of the sensor. For example, habitat_sim.SensorType.COLOR, habitat_sim.SensorType.DEPTH, habitat_sim.SensorType.SEMANTIC.
        h (int): height of the image.
        w (int): width of the image.
        position (Union[List, np.ndarray]): a 3D array of the camera position relative to the robot base. Forward is -z, right is x, up is y.
        orientation (Union[List, np.ndarray], optional): a 3D array of the camera orientation. Defaults to None.

    Returns:
        sensor_spec(habitat_sim.CameraSensorSpec): A sensor configuration for the habitat-sim simulator
    """

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = uuid
    sensor_spec.sensor_type = sensor_type
    sensor_spec.resolution = [h, w]
    sensor_spec.position = position
    if orientation is not None:
        sensor_spec.orientation = np.array(orientation)

    sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    return sensor_spec


# new version one file for all poses, line by line
def load_poses_from_file(pose_file_path: str) -> List[np.ndarray]:
    """Load camera poses from a file. The file store a sequence of 4x4 transformation matrices of
       the camera poses. Each row in the file contains the flatten 4x4 matrix of one camera pose.

    Args:
        pose_file_path (str): The path to the camera pose file (.txt).

    Returns:
        poses_list(List[np.ndarray]): A list of (4,4) np.ndarray representing the camera poses.
    """

    poses_list = []
    with open(pose_file_path, "r") as file:
        for line in file:
            pose_data = np.fromstring(line.strip(), sep="\t")
            pose = pose_data.reshape((4, 4))
            rot = R.from_matrix(pose[:3, :3])
            quat = rot.as_quat()
            pose_vec = np.array(
                [pose[0, 3], pose[1, 3], pose[2, 3], quat[0], quat[1], quat[2], quat[3]]
            )
            poses_list.append(pose_vec)
    return poses_list


def save_obs(
    root_save_dir: str, sim_setting: Dict, observations: Dict, pose: str, save_count: int
) -> None:
    """Save the observation dictionary (rgb, depth, semantic, etc.) as images or numpy arrays in separate folders

    Args:
        root_save_dir (str): The directory where the observation data will be saved.
        sim_setting (Dict): A dictionary specifying the parameters used to initialize the habitat simulator.
        observations (Dict): A dictionary containing the observations from the habitat simulator. Keys include 'color_sensor', 'depth_sensor', 'semantic_sensor', etc.
        save_count (int): The index of the current observation which will be left-zero-filled and used as the name of the saved file.
    """
    if sim_setting["lidar_sensor"]:
        lidar_depths = []
        for i in range(sim_setting["depth_img_for_lidar_n"]):
            lidar_depths.append(observations[f"depth_sensor_ground_{i}"])

    # save rgb
    save_name = (
        sim_setting["scene"].split("/")[-1].split(".")[0] + f"_{save_count:06}.png"
    )
    save_dir = os.path.join(root_save_dir, "rgb")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    obs = observations["color_sensor"][:, :, [2, 1, 0]] / 255
    cv2.imwrite(save_path, observations["color_sensor"][:, :, [2, 1, 0]])

    # save depth
    if sim_setting["depth_sensor"]:
        save_name = (
            sim_setting["scene"].split("/")[-1].split(".")[0] + f"_{save_count:06}.png"
        )
        save_dir = os.path.join(root_save_dir, "depth")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        obs = observations["depth_sensor"]
        obs = (obs * 1000).astype(np.uint16)
        cv2.imwrite(save_path, obs)

    # save semantic
    if sim_setting["semantic_sensor"]:
        save_name = (
            sim_setting["scene"].split("/")[-1].split(".")[0] + f"_{save_count:06}.npy"
        )
        save_dir = os.path.join(root_save_dir, "semantic")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        obs = observations["semantic"]
        with open(save_path, "wb") as f:
            np.save(f, obs)

    # save pose
    save_name = (
        sim_setting["scene"].split("/")[-1].split(".")[0] + f"_{save_count:06}.txt"
    )
    save_dir = os.path.join(root_save_dir, "pose")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    # convert quaternion to rotation matrix
    rot = R.from_quat(pose[3:])
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = rot.as_matrix()
    pose_mat[:3, 3] = pose[:3]
    with open(save_path, "w") as f:
        f.write("\t".join([str(x) for x in pose_mat.flatten()]))
        f.write("\n")



def print_scene_recur(
    scene: habitat_sim.scene.SemanticScene, limit_output: int = 10
) -> None:
    """Print the scene information recursively

    Args:
        scene (habitat_sim.scene.SemanticScene): The habitat_sim.scene.SemanticScene object containing the scene information.
        limit_output (int, optional): The number of object infos to be shown. Defaults to 10.
    """

    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
    )
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return None

