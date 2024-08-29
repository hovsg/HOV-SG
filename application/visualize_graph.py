import os
import json
import glob
import numpy as np
from collections import defaultdict

import hydra
import open3d as o3d
import matplotlib.pyplot as plt
import pyvista as pv
from tqdm import tqdm
from omegaconf import DictConfig



def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)

@hydra.main(version_base=None, config_path="../config", config_name="visualize_graph")
def main(params: DictConfig):
    # Initialize the PyVista plotter
    p = pv.Plotter()

    # Load paths to floor PLY files and corresponding JSON metadata
    floors_ply_paths = sorted(glob.glob(os.path.join(params.graph_path, "floors", "*.ply")))
    floors_info_paths = sorted(glob.glob(os.path.join(params.graph_path, "floors", "*.json")))

    # Initialize data structures for storing point clouds and metadata
    floor_pcds = {}
    floor_infos = {}
    hier_topo = defaultdict(dict)
    init_offset = np.array([7.0, 2.5, 4.0])  # Initial offset for visualization

    # Process each floor
    for counter, (ply_path, info_path) in enumerate(zip(floors_ply_paths, floors_info_paths)):
        with open(info_path, "r") as fp:
            floor_info = json.load(fp)
        # Store relevant floor metadata
        floor_infos[floor_info["floor_id"]] = {
            k: v for k, v in floor_info.items() if k in ["floor_id", "name", "rooms", "floor_height", "floor_zero_level", "vertices"]
        }
        # Apply visualization offset to each floor
        floor_infos[floor_info["floor_id"]]["viz_offset"] = init_offset * counter
        for r_id in floor_info["rooms"]:
            hier_topo[floor_info["floor_id"]][r_id] = []

        # Load the floor point cloud
        floor_pcds[floor_info["floor_id"]] = o3d.io.read_point_cloud(ply_path)

    # Load paths to room PLY files and corresponding JSON metadata
    rooms_ply_paths = sorted(glob.glob(os.path.join(params.graph_path, "rooms", "*.ply")))
    rooms_info_paths = sorted(glob.glob(os.path.join(params.graph_path, "rooms", "*.json")))

    # Initialize data structures for storing room point clouds and metadata
    room_pcds = {}
    room_infos = {}

    # Process each room
    for ply_path, info_path in zip(rooms_ply_paths, rooms_info_paths):
        with open(info_path, "r") as fp:
            room_info = json.load(fp)
        # Store relevant room metadata
        room_infos[room_info["room_id"]] = {
            k: v for k, v in room_info.items() if k in ["room_id", "name", "floor_id", "room_height", "room_zero_level", "vertices"]
        }
        for o_id in room_info["objects"]:
            hier_topo[room_info["floor_id"]][room_info["room_id"]].append(o_id)

        # Load the room point cloud and apply filtering
        orig_cloud = o3d.io.read_point_cloud(ply_path)
        orig_cloud_xyz = np.asarray(orig_cloud.points)
        below_ceiling_filter = (
            orig_cloud_xyz[:, 1]
            < room_infos[room_info["room_id"]]["room_zero_level"]
            + room_infos[room_info["room_id"]]["room_height"]
            - 0.4
        )
        room_pcds[room_info["room_id"]] = orig_cloud.select_by_index(np.where(below_ceiling_filter)[0])
        cloud_xyz = np.asarray(room_pcds[room_info["room_id"]].points)
        cloud_xyz += floor_infos[room_info["floor_id"]]["viz_offset"]
        cloud = pv.PolyData(cloud_xyz)
        room_pcds[room_info["room_id"]].colors = o3d.utility.Vector3dVector(
            np.clip(np.array(room_pcds[room_info["room_id"]].colors) * 1.2, 0.0, 1.0)
        )
        # p.add_mesh(
        #     cloud,
        #     scalars=np.asarray(room_pcds[room_info["room_id"]].colors),
        #     rgb=True,
        #     point_size=5,
        #     opacity=0.8,
        #     show_vertices=True,
        # )

    # Load paths to object PLY files and corresponding JSON metadata
    objects_ply_paths = sorted(glob.glob(os.path.join(params.graph_path, "objects", "*.ply")))
    objects_info_paths = sorted(glob.glob(os.path.join(params.graph_path, "objects", "*.json")))

    # Initialize data structures for storing object point clouds, metadata, and features
    object_pcds = {}
    object_infos = {}
    object_feats = {}

    # Process each object
    for ply_path, info_path in zip(objects_ply_paths, objects_info_paths):
        with open(info_path, "r") as fp:
            object_info = json.load(fp)
        # Store relevant object metadata
        object_infos[object_info["object_id"]] = {
            k: v for k, v in object_info.items() if k in ["object_id", "name", "room_id", "object_height", "object_zero_level"]
        }
        object_feats[object_info["object_id"]] = np.asarray(object_info["embedding"])
        hier_topo[room_infos[object_info["room_id"]]["floor_id"]][room_infos[object_info["room_id"]]["room_id"]].append(
            object_info["object_id"]
        )

        # Load the object point cloud and apply visualization offset
        object_pcds[object_info["object_id"]] = o3d.io.read_point_cloud(ply_path)
        cloud_xyz = np.asarray(object_pcds[object_info["object_id"]].points)
        cloud_xyz += floor_infos[room_infos[object_info["room_id"]]["floor_id"]]["viz_offset"]

    # Calculate centroids for floors
    max_floor_id = list(hier_topo.keys())[-1]
    max_floor_centroid = np.mean(np.asarray(floor_pcds[max_floor_id].points), axis=0)
    floor_centroids = {floor_id: np.mean(np.asarray(floor_pcds[floor_id].points), axis=0) for floor_id in hier_topo.keys()}
    floor_centroids_viz = {floor_id: floor_centroids[floor_id] + floor_infos[floor_id]["viz_offset"] + [0.0, 4.0, 0.0]
                           for floor_id in hier_topo.keys()}

    # Calculate the root node centroid for visualization
    root_offset = [
        np.mean(np.stack(list(floor_centroids_viz.values())).T, axis=1)[0],
        6.0,
        np.mean(np.stack(list(floor_centroids_viz.values())).T, axis=1)[2],
    ]
    root_node_centroid_viz = max_floor_centroid + floor_infos[max_floor_id]["viz_offset"] + root_offset

    # Visualize the centroids of floors
    for floor_id, floor_centroid_viz in floor_centroids_viz.items():
        p.add_mesh(pv.Sphere(center=tuple(floor_centroid_viz), radius=0.5), color="orange")

    # Calculate and visualize the centroids of rooms
    room_centroids = {room_id: np.mean(np.asarray(room_pcds[room_id].points), axis=0) for room_id in room_infos.keys()}
    room_centroids_viz = {room_id: room_centroids[room_id] + [0.0, 3.5, 0] for room_id in room_infos.keys()}
    for room_id, room_centroid_viz in room_centroids_viz.items():
        p.add_mesh(pv.Sphere(center=tuple(room_centroid_viz), radius=0.25), color="blue")
        p.add_mesh(
            pv.Line(tuple(floor_centroids_viz[room_infos[room_id]["floor_id"]]), tuple(room_centroid_viz)),
            line_width=4,
        )

    # Calculate and visualize the centroids of objects
    obj_centroids = {obj_id: np.mean(np.asarray(object_pcds[obj_id].points), axis=0) for obj_id in object_infos.keys()}
    obj_centroids_viz = {obj_id: obj_centroids[obj_id] for obj_id in object_infos.keys()}
    for obj_id, obj_info in object_infos.items():
        if (
            not any(
                substring in obj_info["name"].lower()
                for substring in ["wall", "floor", "ceiling", "paneling", "banner", "overhang"]
            )
            and len(object_pcds[obj_id].points) > 100
        ):
            p.add_mesh(
                pv.Line(tuple(room_centroids_viz[obj_info["room_id"]]), tuple(obj_centroids_viz[obj_id])),
                line_width=1.5,
                opacity=0.5,
            )
            print("included object of category:", obj_info["name"])
            # add object point cloud
            object_pcds[obj_id].paint_uniform_color(np.random.rand(3))
            cloud_xyz = np.asarray(object_pcds[obj_id].points)
            cloud = pv.PolyData(cloud_xyz)
            p.add_mesh(
                cloud,
                scalars=np.asarray(object_pcds[obj_id].colors),
                rgb=True,
                point_size=5,
                show_vertices=True,
            )


    # Show the visualization
    p.show()

if __name__ == "__main__":
    main()
