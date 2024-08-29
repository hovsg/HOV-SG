import math
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import argparse

import habitat_sim
import numpy as np

from typing import Dict, List, Tuple

from habitat_utils import save_obs, make_cfg, load_poses_from_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="the directory where hm3dsem dataset is stored. This directory should contain val and train subfolders",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="the directory where the generated data will be saved. This directory should contain val or train subfolders.",
    )
    parser.add_argument(
        "--pose_dir",
        type=str,
        default="data/hm3dsem_poses",
        help="the file containing the poses to be used for generating the data",
    )
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()
    split = args.split
    all_scene_names = [
        "00824-Dd4bFSTQ8gi",
        "00829-QaLdnwvtxbs",
        "00843-DYehNKdT76V",
        "00847-bCPU9suPUw9",
        "00849-a8BtkwhxdRV",
        "00861-GLAQ4DNUx5U",
        "00862-LT9Jq6dN3Ea",
        "00873-bxsVRursffK",
        "00877-4ok3usBNeis",
        "00890-6s7QHgap2fW",
    ]
    root_dataset_dir = args.dataset_dir
    root_save_dir = args.save_dir
    for scene_dir in all_scene_names:
        scene_name = scene_dir.split("-")[-1]
        split_dir = f"{root_dataset_dir}/{split}/"
        scene_data_dir = f"{root_dataset_dir}/{split}/{scene_dir}/"
        save_dir = f"{root_save_dir}/{split}/{scene_dir}"

        scene_mesh = os.path.join(scene_data_dir, scene_name + ".glb")
        print("scene:", scene_mesh)
        os.makedirs(save_dir, exist_ok=True)

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
            "img_save_dir": save_dir,
        }
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

        sim_cfg = make_cfg(sim_settings, root_dataset_dir, scene_data_dir, scene_name)
        sim = habitat_sim.Simulator(sim_cfg)
        scene = sim.semantic_scene
        print(scene.semantic_index_map)

        # # initialize the agent
        agent = sim.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        random_pt = sim.pathfinder.get_random_navigable_point()
        agent_state.position = random_pt
        agent.set_state(agent_state)

        agent_state = agent.get_state()
        print(
            "agent_state: position",
            agent_state.position,
            "rotation",
            agent_state.rotation,
        )

        init_agent_state = agent_state
        actions_list = []

        agent_height = agent_state.position[1]
        obs = sim.get_sensor_observations(0)
        last_action = None
        release_count = 0

        pose_file = os.path.join(args.pose_dir, scene_dir + ".txt")
        poses_list = load_poses_from_file(pose_file)

        pbar = tqdm(poses_list, total=len(poses_list), desc="saving frames")
        steps = 0
        for pose in pbar:
            pbar.set_description(f"saving frame {steps}/{len(poses_list) + 1}")
            agent = sim.get_agent(0)
            agent_state = agent.get_state()
            agent_state.sensor_states["color_sensor"].position = pose[:3]
            agent_state.sensor_states["color_sensor"].rotation = pose[3:]
            agent_state.sensor_states["depth_sensor"].position = pose[:3]
            agent_state.sensor_states["depth_sensor"].rotation = pose[3:]
            agent_state.sensor_states["semantic"].position = pose[:3]
            agent_state.sensor_states["semantic"].rotation = pose[3:]
            agent.set_state(agent_state, reset_sensors=True, infer_sensor_states=False)
            obs = sim.get_sensor_observations(0)
            rgb = obs["color_sensor"]
            depth = obs["depth_sensor"]
            depth = ((depth / 10) * 255).astype(np.uint8)
            semantic = obs["semantic"]
            # cv2.imshow("rgb", rgb)
            # cv2.imshow("depth", depth)
            # cv2.waitKey()
            save_obs(save_dir, sim_settings, obs, pose, steps)
            steps += 1


if __name__ == "__main__":
    main()
