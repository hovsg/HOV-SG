from collections import defaultdict
import copy
import glob
import json
from multiprocessing import current_process, Process, Queue
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Union, Any

import cv2
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, median_filter
from scipy.signal import find_peaks
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
import skfmm
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from hovsg.graph.floor import Floor
from hovsg.graph.object import Object
from hovsg.utils.constants import MATTERPORT_LABELS_40
from hovsg.utils.clip_utils import get_text_feats_62_templates

# change matplotlib backend to a gui one
plt.switch_backend("TkAgg")


def compute_sdf(boundary_mask, distance_scale=1):
    dx = 1  # downsample factor
    f = distance_scale / dx  # distance function scale

    # cv2.resize(boundary_map, None, fx=1/dx, fy=1/dx, interpolation=cv2.INTER_NEAREST)
    sdf = skfmm.distance(1 - boundary_mask)
    sdf[sdf > f] = f
    sdf = sdf / f
    sdf = 1 - sdf
    return sdf


class NavigationGraph:
    def __init__(self, floor_pcd: o3d.geometry.PointCloud, cell_size: int):
        """Initialization of the NavigationGraph class.

        Args:
            floor_pcd (o3d.geometry.PointCloud): The point cloud of the floor.
            cell_size (int): the resolution of the cell (m/cell)
        """
        self.pcd_min = np.min(np.array(floor_pcd.points), axis=0)
        self.pcd_max = np.max(np.array(floor_pcd.points), axis=0)
        self.grid_size = np.ceil((self.pcd_max - self.pcd_min) / cell_size + 1).astype(
            np.int32
        )
        self.grid_size = self.grid_size[[0, 2]]
        self.cell_size = cell_size
        self.has_stairs = False
        # self.midpoint = grid_size // 2
        # assert self.grid_size % 2 == 0

        # self.reset()

    def get_floor_poses(
        self,
        floor: Floor,
        poses_list: List[np.ndarray],
        upper_floor_min_height: float = None,
        camera_height: float = 1.5,
    ) -> List[np.ndarray]:
        """Return the list of poses that are within the height range of the floor.

        Args:
            floor (Floor): The Floor object.
            poses_list (List[np.ndarray]): The list of camera poses for the exploration of the whole scene (multi-floor).
            upper_floor_min_height (float, optional): The uppebound of the height range that is used to determined the poses
                belonging to the floor. If None, then use the upper bound of the floor point cloud. Defaults to None.
            camera_height (float, optional): The height of the camera. Defaults to 1.5.

        Returns:
            floor_poses_list (List[np.ndarray]): A list of (4, 4) numpy arrays representing the poses that are within the
                height range of the floor.
        """
        floor_min_height = np.min(np.asarray(floor.pcd.points)[:, 1])
        floor_max_height = np.max(np.asarray(floor.pcd.points)[:, 1])
        floor_poses_list = []
        upperbound = (
            upper_floor_min_height
            if upper_floor_min_height is not None
            else floor_max_height
        )
        for pose in poses_list:

            if (
                pose[1, 3] - camera_height >= floor_min_height
                and pose[1, 3] - camera_height < upperbound
            ):
                floor_poses_list.append(pose)
        return floor_poses_list

    def reset(self):
        """Reset the map size to double the current grid size."""

        self.map = np.zeros([self.grid_size] * 2, dtype=np.float32)

    def to_grid(self, world_coords: np.ndarray) -> np.ndarray:
        """Convert the world coordinates to grid coordinates (integer).

        Args:
            world_coords (np.ndarray): The point in the world coordinate frame.

        Returns:
            np.ndarray: The point in the grid coordinate frame.
        """
        return np.int32((world_coords - self.pcd_min) / self.cell_size)

    def to_world(self, grid_coords: np.ndarray) -> np.ndarray:
        """Convert the grid coordinates to world coordinates.

        Args:
            grid_coords (np.ndarray): The point in the grid coordinate frame.

        Returns:
            np.ndarray: The point in the world coordinate frame.
        """
        return grid_coords * self.cell_size + self.pcd_min

    def obstacles_vertices(
        self,
        floor_pcd: o3d.geometry.PointCloud,
        floor_info: Dict,
        height_region: List[float] = [0.2, 1.5],
    ) -> np.ndarray:
        """Get the point cloud within a certain height range from the floor as the obstacles.

        Args:
            floor_pcd (o3d.geometry.PointCloud): The point cloud of the floor.
            floor_info (Dict): The floor information that contains the minimum height of the floor.
            height_region (List[float], optional): The height range relative to the minimum of the floor in which the
                point cloud is considered as obstacles. Defaults to [0.2, 1.5].

        Returns:
            floor_pcd (np.ndarray): The obstacle point cloud of the floor.
        """
        floor_pcd = np.array(floor_pcd.points)
        floor_zero_level = floor_info["floor_zero_level"]
        floor_height = floor_info["floor_height"]
        # remove points above 2 meters from the floor
        floor_pcd = floor_pcd[floor_pcd[:, 1] < floor_zero_level + height_region[1]]
        # remove ground points (z < 0.1)
        floor_pcd = floor_pcd[floor_pcd[:, 1] > floor_zero_level + height_region[0]]
        # project to 2D
        floor_pcd = floor_pcd[:, [0, 2]]
        # plot for debugging
        # plt.scatter(floor_pcd[:, 0], floor_pcd[:, 1], s=1)
        # plt.show()
        return floor_pcd

    def floor_region_vertices(
        self,
        floor_pcd: o3d.geometry.PointCloud,
        floor_info: Dict,
        height_max: float = 1.5,
    ):
        """Get the point cloud within a certain height range from the floor.

        Args:
            floor_pcd (o3d.geometry.PointCloud): The point cloud of the floor.
            floor_info (Dict): The floor information that contains the minimum height of the floor.
            height_max (float, optional): The maximum relative height considered as within the floor. Defaults to 1.5.

        Returns:
            floor_pcd (np.ndarray): The obstacle point cloud of the floor.
        """
        floor_pcd = np.array(floor_pcd.points)
        floor_pcd = floor_pcd[
            floor_pcd[:, 1] < floor_info["floor_zero_level"] + height_max
        ]
        floor_pcd = floor_pcd[:, [0, 2]]
        return floor_pcd

    def create_occupancy_grid(
        self, point_cloud: np.ndarray, dilation_radius: int = 5, filter_size: int = 3
    ) -> np.ndarray:
        """Create the occupancy grid map from the point cloud and apply dilation and smoothing.

        Args:
            point_cloud (np.ndarray): The point cloud to be converted to the occupancy grid map.
            dilation_radius (int, optional): The steps of dilation (embolding the boundary). Defaults to 5.
            filter_size (int, optional): The filter size for gaussian blurring for smoothing the grid map.
                Defaults to 3.

        Returns:
            occupancy_grid_map (np.ndarray): The occupancy grid map.
        """
        # Initialize an empty grid map with all cells marked as unoccupied (0)
        occupancy_grid_map = np.zeros(self.grid_size[::-1], dtype=int)

        # handle point cloud with negative values
        point_cloud = point_cloud - self.pcd_min[[0, 2]]

        # Iterate through the point cloud and mark the corresponding cells as occupied (1)
        x_cells = np.floor(point_cloud[:, 0] / self.cell_size).astype(int)
        y_cells = np.floor(point_cloud[:, 1] / self.cell_size).astype(int)
        # increment the cell value by 1
        occupancy_grid_map[y_cells, x_cells] = 1

        if dilation_radius > 0:
            occupancy_grid_map = cv2.dilate(
                occupancy_grid_map.astype(np.uint8),
                np.ones((dilation_radius, dilation_radius)),
                iterations=1,
            )
        # apply guassian filter to smooth the occupancy grid map
        if filter_size > 0:
            occupancy_grid_map = cv2.GaussianBlur(
                occupancy_grid_map.astype(np.float32), (filter_size, filter_size), 0
            )

        return occupancy_grid_map

    def get_height_map(
        self,
        point_cloud: np.ndarray,
        floor_dir: str = None,
        height_region: np.ndarray = None,
        knn: int = 10,
    ):
        """Get the grid map where each cell stores the maximum height of the cell.

        Args:
            point_cloud (np.ndarray): The point cloud of the grid map.
            floor_dir (str, optional): The directory where the intermediate results are stored. Defaults to None.
            height_region (np.ndarray, optional): A binary mask specifying where to compute the height map. Defaults to None.
            knn (int, optional): When the cell has no points projected to it, apply knn to get the nearest height.
                Defaults to 10.

        Returns:
            _type_: _description_
        """
        # Initialize an empty grid map with all cells marked as unoccupied (0)
        height_map = np.zeros(self.grid_size[::-1], dtype=float)

        # handle point cloud with negative values
        vertices = point_cloud[:, [0, 2]] - self.pcd_min[[0, 2]]

        # Iterate through the point cloud and mark the corresponding cells as occupied (1)
        x_cells = np.floor(vertices[:, 0] / self.cell_size).astype(int)
        y_cells = np.floor(vertices[:, 1] / self.cell_size).astype(int)
        # increment the cell value by 1
        height_map[y_cells, x_cells] = point_cloud[:, 1]
        if height_region is not None:
            rows, cols = np.where(height_map > 0)
            interpolate_region = np.logical_and(height_map == 0, height_region)
            interpolate_rows, interpolate_cols = np.where(interpolate_region)
            inter_pos = np.array([interpolate_rows, interpolate_cols]).T
            height_pos = np.array([rows, cols]).T
            dist_mat = cdist(inter_pos, height_pos)
            height_ids = np.argsort(dist_mat, axis=1)
            for i, (int_row, int_col) in enumerate(
                zip(interpolate_rows, interpolate_cols)
            ):
                knn = min(knn, len(height_ids[0]))
                row_col = height_pos[height_ids[i][:knn]]
                height_list = height_map[row_col[:, 0], row_col[:, 1]]
                height_map[int_row, int_col] = np.mean(height_list)

        height_map = median_filter(height_map, size=3)

        save_height_map = (height_map.copy() / np.max(height_map) * 255).astype(
            np.uint8
        )
        save_height_map = cv2.applyColorMap(save_height_map, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(floor_dir, "height_map.png"), save_height_map)

        return height_map

    def get_largest_region(self, binary_map: np.ndarray) -> np.ndarray:
        """Get the largest disconnected island region in the binary map.

        Args:
            binary_map (np.ndarray): The binary map.

        Returns:
            np.ndarray: the largest region in the binary map.
        """
        # Threshold it so it becomes binary
        # ret, thresh = cv2.threshold(binary_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        input = (binary_map > 0).astype(np.uint8)
        output = cv2.connectedComponentsWithStats(input, 8, cv2.CV_8UC1)
        areas = output[2][:, -1]
        # TODO: the top region is 0 region, so we need to sort the areas and get the second largest
        # but I am not sure if the largest region is always the background
        id = np.argsort(areas)[::-1][1]
        return output[1] == id

    def get_poses_region(
        self,
        poses_list: List[np.ndarray],
        floor_dir: str = None,
        radius: float = 0.5,
        save: bool = True,
        cluster: bool = True,
    ) -> np.ndarray:
        """Get the region grid map where the pose positions and their circular surrounding are. Each location is marked with a circle with a certain radius.

        Args:
            poses_list (List[np.ndarray]): The list of poses.
            floor_dir (str, optional): The directory where intermediate results are stored. Defaults to None.
            radius (float, optional): The radius of the circle around the pose location. Defaults to 0.5.
            save (bool, optional): The boolean controlling whether to store the pose region grid map as a file.
                Defaults to True.
            cluster (bool, optional): The boolean controlling whether to cluster the poses' height. Defaults to True.

        Returns:
            pose_map (np.ndarray): The resulting pose region grid map. 1 is the pose region, 0 is the non-pose region.
        """
        pose_heights = np.array([pose[1, 3] for pose in poses_list])
        clusters = DBSCAN(eps=0.1).fit(pose_heights.reshape(-1, 1))
        labels, counts = np.unique(clusters.labels_, return_counts=True)
        id = np.argmax(counts)
        mask = clusters.labels_ == labels[id]
        major_height = np.mean(pose_heights[mask])
        if cluster:
            poses_list = [
                pose for pose in poses_list if np.abs(pose[1, 3] - major_height) < 0.1
            ]

            poses_min = np.min(np.array(poses_list)[:, 1, 3])
            poses_list = [pose for pose in poses_list if pose[1, 3] < poses_min + 0.1]

        poses_list = np.array(poses_list)
        poses_list = poses_list[:, [0, 2], 3]
        poses_list = (poses_list - self.pcd_min[[0, 2]]) / self.cell_size
        poses_list = np.int32(poses_list)
        poses_map = np.zeros(self.grid_size[::-1], dtype=np.uint8)
        for pose in poses_list:
            cv2.circle(poses_map, tuple(pose), int(radius / self.cell_size), 1, -1)
        if save:
            cv2.imwrite(
                os.path.join(floor_dir, f"poses_region_map.png"), poses_map * 255
            )
        return poses_map

    def get_main_free_map(
        self,
        floor_pcd: o3d.geometry.PointCloud,
        floor_info: Dict,
        floor_dir: str = None,
        floor_poses_list: List[np.ndarray] = None,
        save: bool = True,
    ) -> np.ndarray:
        """Get the free region of the floor as a grid map.

        Args:
            floor_pcd (o3d.geometry.PointCloud): The floor point cloud.
            floor_info (Dict): The information dictionary of the floor.
            floor_dir (str, optional): Directory where the intermediate results will be stored. Defaults to None.
            floor_poses_list (List[np.ndarray], optional): When provided, also use the pose region as the free space. Defaults to None.
            save (bool, optional): Boolean to control whether to save the intermediate results. Defaults to True.

        Returns:
            main_free_map (np.ndarray): The free region grid map. 1 is free, 0 is occupied.
        """
        obstaces_vertices = self.obstacles_vertices(floor_pcd, floor_info)
        floor_region_vertices = self.floor_region_vertices(floor_pcd, floor_info)
        floor_occupancy_map = self.create_occupancy_grid(floor_region_vertices)
        if save:
            os.makedirs(floor_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(floor_dir, "floor_region_no_closing.png"),
                floor_occupancy_map.astype(np.uint8) * 255,
            )
        # floor_occupancy_map = binary_closing(floor_occupancy_map, iterations=5)
        poses_map = self.get_poses_region(
            floor_poses_list, floor_dir=floor_dir, radius=0.5, save=save
        )
        np.logical_or(floor_occupancy_map, poses_map, out=floor_occupancy_map)
        self.map = self.create_occupancy_grid(obstaces_vertices)
        floor_free_map = floor_occupancy_map - self.map
        floor_free_map = np.where(floor_free_map < 0, 0, floor_free_map).astype(
            np.uint8
        )
        main_free_map = self.get_largest_region(floor_free_map)
        if save:
            cv2.imwrite(
                os.path.join(floor_dir, "floor_obstacles.png"),
                self.map.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "floor_region.png"),
                floor_occupancy_map.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "floor_free.png"),
                floor_free_map.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "floor_free_main.png"),
                main_free_map.astype(np.uint8) * 255,
            )
        return main_free_map

    def get_top_down_rgb_map(
        self, floor_pcd: o3d.geometry.PointCloud, floor_info: Dict, floor_dir: str
    ) -> np.ndarray:
        """Generate the top-down RGB map of the floor.

        Args:
            floor_pcd (o3d.geometry.PointCloud): Floor point cloud.
            floor_info (Dict): Floor information dictionary.
            floor_dir (str): Directory where the intermediate results will be stored.

        Returns:
            top_down_bgr (np.ndarray): (H, W, 3) The top-down BGR map of the floor.
        """
        floor_point_cloud = np.asarray(floor_pcd.points)
        floor_point_color = np.asarray(floor_pcd.colors)
        zero_level = floor_info["floor_zero_level"]
        floor_height = floor_info["floor_height"]
        mask = floor_point_cloud[:, 1] < zero_level + 1.5  # floor_height - 0.5
        floor_point_cloud = floor_point_cloud[mask]
        floor_point_color = floor_point_color[mask]
        floor_grid_vertices = np.int32(
            (floor_point_cloud - self.pcd_min) / self.cell_size
        )
        top_down = np.zeros([self.grid_size[1], self.grid_size[0], 3], dtype=np.uint8)
        top_down_height = -1000 * np.ones(
            [self.grid_size[1], self.grid_size[0]], dtype=np.float32
        )
        for p_i, floor_p in enumerate(floor_grid_vertices):
            col, height, row = floor_p
            if height > top_down_height[row, col]:
                top_down[row, col] = (floor_point_color[p_i] * 255).astype(np.uint8)
                top_down_height[row, col] = height
        top_down = median_filter(top_down, size=3)

        top_down_bgr = cv2.cvtColor(top_down, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(floor_dir, "top_down_rgb.png"), top_down_bgr)
        # cv2.imwrite(os.path.join(floor_dir, "top_down_rgb.png"), top_down)
        return top_down_bgr

    def get_voronoi_graph(
        self,
        main_free_map: np.ndarray,
        map_rgb: np.ndarray,
        floor_dir: str,
        floor_id: str,
        name: str = "",
        height_map: np.ndarray = None,
    ) -> nx.Graph:
        """Generate the Voronoi Graph of the floor based on the free space map substracting obstacle map.

        Args:
            main_free_map (np.ndarray): Free space map.
            map_rgb (np.ndarray): RGB map of the floor.
            floor_dir (str): Directory where the intermediate results will be stored.
            floor_id (str): The floor id. For example, "0" for the first floor.
            name (str, optional): The name is used for saving the intermediate results. Defaults to "".
            height_map (np.ndarray, optional): Height map used for getting the 3D positional attributes of
                the Voronoi node. Defaults to None.

        Returns:
            nx.Graph: Resulting Voronoi graph.
        """
        boundary_map = binary_erosion(main_free_map, iterations=1).astype(np.uint8)
        boundary_map = main_free_map - boundary_map
        cv2.imwrite(
            os.path.join(floor_dir, f"boundary_{name}.png"),
            boundary_map.astype(np.uint8) * 255,
        )
        rows, cols = np.where(boundary_map == 1)
        boundaries = np.array(list(zip(rows, cols)))
        voronoi = Voronoi(boundaries)

        fig_free = main_free_map.copy().astype(np.uint8) * 255
        fig_free = cv2.cvtColor(fig_free, cv2.COLOR_GRAY2BGR)
        map_bgr = cv2.cvtColor(map_rgb, cv2.COLOR_RGB2BGR)
        fig = map_bgr.copy()
        vertices = []
        if height_map is None:
            height_map = np.ones_like(boundary_map) * self.pcd_min[1]
        voronoi_graph = nx.Graph()
        for simplex in voronoi.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.any(simplex < 0):
                continue
            src, tar = voronoi.vertices[simplex]
            if (
                src[0] < 0
                or src[0] >= fig.shape[0]
                or src[1] < 0
                or src[1] >= fig.shape[1]
                or tar[0] < 0
                or tar[0] >= fig.shape[0]
                or tar[1] < 0
                or tar[1] >= fig.shape[1]
            ):
                continue
            if (
                main_free_map[int(src[0]), int(src[1])] == 0
                or main_free_map[int(tar[0]), int(tar[1])] == 0
            ):
                continue
            cv2.line(
                fig,
                tuple(np.int32(src[::-1])),
                tuple(np.int32(tar[::-1])),
                (0, 0, 255),
                1,
            )
            cv2.line(
                fig_free,
                tuple(np.int32(src[::-1])),
                tuple(np.int32(tar[::-1])),
                (0, 0, 255),
                1,
            )
            cv2.circle(fig, tuple(np.int32(src[::-1])), 2, (255, 0, 0), -1)
            cv2.circle(fig_free, tuple(np.int32(src[::-1])), 2, (255, 0, 0), -1)

            # check if src and tar already exist in the graph
            if (src[0], src[1], floor_id) not in voronoi_graph.nodes:
                height = height_map[int(src[0]), int(src[1])]
                voronoi_graph.add_node(
                    (src[0], src[1], floor_id),
                    pos=(
                        src[1] * self.cell_size + self.pcd_min[0],
                        height,
                        src[0] * self.cell_size + self.pcd_min[2],
                    ),
                    floor_id=floor_id,
                )
            if (tar[0], tar[1], floor_id) not in voronoi_graph.nodes:
                height = height_map[int(tar[0]), int(tar[1])]
                voronoi_graph.add_node(
                    (tar[0], tar[1], floor_id),
                    pos=(
                        tar[1] * self.cell_size + self.pcd_min[0],
                        height,
                        tar[0] * self.cell_size + self.pcd_min[2],
                    ),
                    floor_id=floor_id,
                )
            # check if the edge already exists
            if (src[0], src[1], floor_id) not in voronoi_graph[
                (tar[0], tar[1], floor_id)
            ]:
                voronoi_graph.add_edge(
                    (src[0], src[1], floor_id),
                    (tar[0], tar[1], floor_id),
                    dist=np.linalg.norm(src - tar),
                )

        cv2.imwrite(os.path.join(floor_dir, f"vor_{name}.png"), fig)
        cv2.imwrite(os.path.join(floor_dir, f"vor_free_{name}.png"), fig_free)
        # vertices = np.array(vertices)
        return voronoi_graph

    def sparsify_graph(self, floor_graph: nx.Graph, resampling_dist: float = 0.4):
        """
        Sparsify a topology graph by removing nodes with degree 2.
        This algorithm first starts at degree-one nodes (dead ends) and
        removes all degree-two nodes until confluence nodes are found.
        Next, we find close pairs of higher-order degree nodes and
        delete all nodes if the shortest path between two nodes consists
        only of degree-two nodes.
        Args:
            floor_graph (nx.Graph): graph to sparsify
        Returns:
            nx.Graph: sparsified graph
        """
        graph = copy.deepcopy(floor_graph)

        if len(graph.nodes) < 10:
            return graph
        # all nodes with degree 1 or 3+
        new_node_candidates = [
            node for node in list(graph.nodes) if (graph.degree(node) != 2)
        ]

        new_graph = nx.Graph()
        for i, node in enumerate(new_node_candidates):
            new_graph.add_node(
                node,
                pos=graph.nodes[node]["pos"],
                floor_id=graph.nodes[node]["floor_id"],
            )
        new_nodes = set(new_graph.nodes)
        new_nodes_list = list(new_graph.nodes)

        print(
            f"Getting paths between all nodes. Node number: {len(new_node_candidates)}/{len(graph.nodes)}"
        )

        st = time.time()
        all_path_dense_graph = dict(nx.all_pairs_dijkstra_path(graph, weight="dist"))
        ed = time.time()
        print("time for computing all pairs shortest path: ", ed - st, " seconds")
        sampled_edges_to_add = list()
        pbar = tqdm(range(len(new_graph.nodes)), desc="Sparsifying graph")
        for i in pbar:
            inner_pbar = tqdm(
                range(len(new_graph.nodes)), desc="Sparsifying graph", leave=False
            )
            for j in inner_pbar:
                if i >= j:
                    continue
                # Go through all edges along path and extract dist
                node1 = new_nodes_list[i]
                node2 = new_nodes_list[j]
                try:
                    path = all_path_dense_graph[node1][node2]
                    for node in path[1:-1]:
                        if graph.degree(node) > 2:
                            break
                    else:
                        sampled_edges_to_add.append(
                            (
                                path[0],
                                path[-1],
                                np.linalg.norm(np.array(path[0]) - np.array(path[-1])),
                            )
                        )
                        dist = [
                            graph.edges[path[k], path[k + 1]]["dist"]
                            for k in range(len(path) - 1)
                        ]
                        mov_agg_dist = 0
                        predecessor = path[0]
                        # connect the nodes if there is a path between them that does not go through any other of the new nodes
                        if (
                            len(path)
                            and len(set(path[1:-1]).intersection(new_nodes)) == 0
                        ):
                            for cand_idx, cand_node in enumerate(path[1:-1]):
                                mov_agg_dist += dist[cand_idx]
                                if mov_agg_dist * self.cell_size > resampling_dist:
                                    sampled_edges_to_add.append(
                                        (
                                            predecessor,
                                            cand_node,
                                            np.linalg.norm(
                                                np.array(predecessor)
                                                - np.array(cand_node)
                                            ),
                                        )
                                    )
                                    predecessor = cand_node
                                    mov_agg_dist = 0
                                else:
                                    continue
                            sampled_edges_to_add.append(
                                (
                                    predecessor,
                                    path[-1],
                                    np.linalg.norm(
                                        np.array(predecessor) - np.array(path[-1])
                                    ),
                                )
                            )
                except:
                    continue

        for edge_param in sampled_edges_to_add:
            k, l, dist = edge_param
            if k not in new_graph.nodes:
                new_graph.add_node(
                    k, pos=graph.nodes[k]["pos"], floor_id=graph.nodes[k]["floor_id"]
                )
            if l not in new_graph.nodes:
                new_graph.add_node(
                    l, pos=graph.nodes[l]["pos"], floor_id=graph.nodes[l]["floor_id"]
                )
            new_graph.add_edge(k, l, dist=dist)

        self.floor_graph = new_graph
        return new_graph

    def trim_graph(self, graph: nx.Graph, trim_deg: int = 1):
        """
        Trim a graph by removing all nodes with degree 1.
        Args:
            graph (nx.Graph): graph to trim
            trim_deg (int): degree of nodes to remove
        Returns:
            nx.Graph: trimmed graph
        """
        graph = copy.deepcopy(graph)
        while True:
            degree_one_nodes = [
                node for node in list(graph.nodes) if (graph.degree(node) == trim_deg)
            ]
            if len(degree_one_nodes) == 0:
                break
            graph.remove_nodes_from(degree_one_nodes)
        return graph

    def draw_graph_on_map(
        self, map: np.ndarray, graph: nx.Graph, floor_dir: str, name: str
    ) -> np.ndarray:
        """Draw graph on the grid map

        Args:
            map (np.ndarray): The grid map where the graph will be drawn on.
            graph (nx.Graph): The drawn graph.
            floor_dir (str): The directory where the intermediate results are stored.
            name (str): The saved name of the intermediate results.

        Returns:
            fig (np.ndarray): The map where the graph is drawn.
        """
        fig = map.copy().astype(np.uint8)
        if np.max(map) <= 1:
            fig = (map.copy() * 255).astype(np.uint8)
        if len(fig.shape) == 2:
            fig = cv2.cvtColor(fig, cv2.COLOR_GRAY2BGR)
        for edge in graph.edges:
            v1, v2 = edge
            v1 = np.int32(v1)[::-1][1:]
            v2 = np.int32(v2)[::-1][1:]
            cv2.line(fig, tuple(v1), tuple(v2), (0, 0, 255), 1)
            cv2.circle(fig, tuple(v1), 2, (255, 0, 0), -1)
            cv2.circle(fig, tuple(v2), 2, (255, 0, 0), -1)
        cv2.imwrite(os.path.join(floor_dir, f"{name}.png"), fig)
        # cv2.imshow(f"graph on map {name}", fig)
        # cv2.waitKey()
        return fig

    def get_stairs_objects(
        self, objects_list: List[Object], clip_model: Any, clip_feat_dim: int
    ) -> List[Object]:
        """Get the objects that are stairs or staircase.

        Args:
            objects_list (List[Object]): The list of Object instances.
            clip_model (Any): The clip model.
            clip_feat_dim (int): The clip model dimension.

        Returns:
            stairs_list (List[Object]): The list of stairs objects.
        """
        text_feats = get_text_feats_62_templates(
            MATTERPORT_LABELS_40, clip_model, clip_feat_dim
        )
        stair_id = MATTERPORT_LABELS_40.index("stairs")
        print(stair_id)
        stairs_list = []
        for obj in objects_list:
            emb = obj.embedding
            sim_mat = emb @ text_feats.T
            label_id = np.argmax(sim_mat)
            if label_id == stair_id:  # staircase
                print(obj.object_id)
                stairs_list.append(obj)
            elif obj.name == "staircase":
                stairs_list.append(obj)

        return stairs_list

    def get_stairs_graph_with_poses_v2(
        self, floor: Floor, floor_id: str, poses_list: List[np.ndarray], floor_dir: str
    ) -> nx.Graph:
        """A way of getting stairs graph based on the poses in between floors.

        Args:
            floor (Floor): Floor instance.
            floor_id (str): Floor id.
            poses_list (List[np.ndarray]): The list of poses in the whole scene.
            floor_dir (str): The directory where the intermediate results are stored.

        Returns:
            spares_voronoi_graph (nx.Graph): The stairs graph.
        """
        floor_pcd = floor.pcd
        floor_info = {
            "floor_zero_level": floor.floor_zero_level,
            "floor_height": floor.floor_height,
        }
        top_down = self.get_top_down_rgb_map(floor_pcd, floor_info, floor_dir)
        pose_heights = np.array([pose[1, 3] - 1.5 for pose in poses_list])
        hist = np.histogram(pose_heights, bins=100)
        min_peak_height = 0.3 * np.max(hist[0])
        input_hist = np.concatenate([[np.min(hist[0])], hist[0], [np.min(hist[0])]])
        peaks, _ = find_peaks(input_hist, height=min_peak_height)
        peaks = peaks - 1
        plt.clf()
        plt.plot(hist[1][:-1], hist[0])
        plt.plot(hist[1][peaks], hist[0][peaks], "x")
        plt.savefig(os.path.join(floor_dir, "hist_peaks.png"))
        if floor_id >= len(peaks) - 1:
            self.has_stairs = False
            return nx.Graph()

        height_min = hist[1][peaks[int(floor_id)] + 1]
        height_max = hist[1][peaks[int(floor_id) + 1]]
        if pose_heights[0] > pose_heights[-1]:
            print("reverse pose sequence")
            pose_heights = pose_heights[::-1]
            poses_list = poses_list[::-1]
        min_st_id = 0
        if pose_heights[0] > height_min:
            for i, h in enumerate(pose_heights):
                if h <= height_min:
                    min_st_id = i
                    break

        st_id = None
        ed_id = None
        for hi, h in enumerate(pose_heights):
            if h > height_min and hi > min_st_id:
                st_id = hi
                break
        for hi, h in enumerate(pose_heights):
            if h > height_max:
                ed_id = hi
                break
        for hi in range(st_id, ed_id):
            if pose_heights[hi] < height_min:
                st_id = hi

        hist = np.histogram(pose_heights[st_id:ed_id], bins=100)
        plt.clf()
        plt.plot(hist[1][:-1], hist[0])
        plt.savefig(os.path.join(floor_dir, "hist_stairs.png"))
        buffer_down = 5
        buffer_up = 20
        st_id = np.max([st_id - buffer_down, 0])
        ed_id = ed_id + buffer_up
        stairs_poses = poses_list[st_id:ed_id]
        tar_poses = []
        for pose in stairs_poses:
            pos = pose[:3, 3]
            tar_poses.append(pos)
            pos_2d = np.round((pos - self.pcd_min) / self.cell_size).astype(np.int32)
            cv2.circle(top_down, tuple(np.int32(pos_2d[[0, 2]])), 2, (255, 0, 0), -1)
            cv2.imwrite(os.path.join(floor_dir, "top_down_rgb_poses.png"), top_down)

        min_height = np.min([pos[1] for pos in tar_poses])
        # cv2.imshow("stairs poses", top_down)
        # cv2.waitKey()
        voronoi_graph = nx.Graph()
        last_pos = None
        for pos in tar_poses:
            pos_2d = (pos - self.pcd_min) / self.cell_size
            voronoi_graph.add_node(
                (pos_2d[2], pos_2d[0], floor_id),
                pos=(
                    pos[0],
                    pos[1] - min_height + floor_info["floor_zero_level"],
                    pos[2],
                ),
                floor_id=floor_id,
            )
            if last_pos is not None:
                voronoi_graph.add_edge(
                    (pos_2d[2], pos_2d[0], floor_id),
                    (last_pos[2], last_pos[0], floor_id),
                    dist=np.linalg.norm(pos_2d - last_pos),
                )
                last_pos = pos_2d
            else:
                last_pos = pos_2d

        # sparse_voronoi_graph = self.sparsify_graph(voronoi_graph, resampling_dist=0.4)
        sparse_voronoi_graph = voronoi_graph
        self.has_stairs = True
        self.draw_graph_on_map(
            top_down, sparse_voronoi_graph, floor_dir, "sparse_vor_stairs"
        )
        return sparse_voronoi_graph

    def get_stairs_graph_with_poses(
        self, floor: Floor, floor_id: str, poses_list: List[np.ndarray], floor_dir: str
    ) -> nx.Graph:
        """A way of getting stairs graph based on the poses in between floors.

        Args:
            floor (Floor): Floor instance.
            floor_id (str): Floor id.
            poses_list (List[np.ndarray]): The list of poses in the whole scene.
            floor_dir (str): The directory where the intermediate results are stored.

        Returns:
            spares_voronoi_graph (nx.Graph): The stairs graph.
        """
        floor_pcd = floor.pcd
        floor_info = {
            "floor_zero_level": floor.floor_zero_level,
            "floor_height": floor.floor_height,
        }
        top_down = self.get_top_down_rgb_map(floor_pcd, floor_info, floor_dir)
        pose_heights = np.array([pose[1, 3] for pose in poses_list])
        min_height = np.min(pose_heights)
        max_height = np.max(pose_heights)
        if max_height - min_height < 0.5:
            self.has_stairs = False
            return nx.Graph()

        pose_heights = np.array([pose[1, 3] for pose in poses_list])
        clusters = DBSCAN(eps=0.1).fit(pose_heights.reshape(-1, 1))
        labels, counts = np.unique(clusters.labels_, return_counts=True)
        id = np.argmax(counts)
        mask = clusters.labels_ == labels[id]
        major_height = np.mean(pose_heights[mask])
        major_height_poses = [pose[:3, 3] for pose, m in zip(poses_list, mask) if m]
        for l in labels:
            print(np.mean(pose_heights[clusters.labels_ == l]))
        non_min_height_poses = [
            pose[:3, 3] for pose in poses_list if pose[1, 3] > major_height + 0.1
        ]
        # clusters = DBSCAN(eps=0.5).fit(np.array(non_min_height_poses))
        # labels, counts = np.unique(clusters.labels_, return_counts=True)
        # id = np.argmax(counts)
        # mask = clusters.labels_ == labels[id]
        # non_min_height_poses = [pose for pose, m in zip(non_min_height_poses, mask) if m]
        # find the closest pose in major_height_poses to the non_min_height_poses
        dist_mat = cdist(np.array(non_min_height_poses), np.array(major_height_poses))
        row, col = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
        non_min_height_poses.append(major_height_poses[col])
        poses_sorted_by_height = sorted(non_min_height_poses, key=lambda x: x[1])
        if len(poses_sorted_by_height) > 0:
            self.has_stairs = True

        all_poses = np.array([pose[:3, 3] for pose in poses_list])
        top_down_copy = top_down.copy()
        top_down_copy_1 = top_down.copy()

        for pose in non_min_height_poses:
            pos_2d = np.round((pose - self.pcd_min) / self.cell_size).astype(np.int32)
            cv2.circle(top_down, tuple(np.int32(pos_2d[[0, 2]])), 2, (255, 0, 0), -1)
            cv2.imwrite(os.path.join(floor_dir, "top_down_rgb_poses.png"), top_down)

        for pose in major_height_poses:
            pos_2d = np.round((pose - self.pcd_min) / self.cell_size).astype(np.int32)
            cv2.circle(
                top_down_copy, tuple(np.int32(pos_2d[[0, 2]])), 2, (255, 0, 0), -1
            )
            cv2.imwrite(
                os.path.join(floor_dir, "top_down_rgb_major_poses.png"), top_down_copy
            )

        rest_poses = []
        for pos in [*major_height_poses, *non_min_height_poses]:
            cdist_mat = cdist(np.array([pos]), np.array(all_poses))
            min_dist = np.min(cdist_mat) if len(cdist_mat) > 0 else np.inf
            if min_dist > 0.1:
                rest_poses.append(pos)
        print(rest_poses)

        for pose in all_poses:
            pos_2d = np.round((pose - self.pcd_min) / self.cell_size).astype(np.int32)
            cv2.circle(
                top_down_copy_1, tuple(np.int32(pos_2d[[0, 2]])), 2, (255, 0, 0), -1
            )
            cv2.imwrite(
                os.path.join(floor_dir, "top_down_rgb_all_poses.png"), top_down_copy_1
            )

        tar_poses = []
        last_pose = np.ones(3) * -np.inf
        for pose in poses_sorted_by_height:
            if pose[1] > last_pose[1]:
                last_pose = pose
                tar_poses.append(last_pose)
                continue
            elif pose[1] < last_pose[1]:
                continue
            if np.linalg.norm(pose[:2] - last_pose[:2]) > 0.1:
                last_pose = pose
                tar_poses.append(last_pose)

        voronoi_graph = nx.Graph()
        last_pos = None
        for pos in tar_poses:
            pos_2d = (pos - self.pcd_min) / self.cell_size
            print(
                pos_2d[2],
                pos_2d[0],
                pos[0],
                pos[1] - min_height + floor_info["floor_zero_level"],
                pos[2],
            )
            voronoi_graph.add_node(
                (pos_2d[2], pos_2d[0], floor_id),
                pos=(
                    pos[0],
                    pos[1] - min_height + floor_info["floor_zero_level"],
                    pos[2],
                ),
                floor_id=floor_id,
            )
            if last_pos is not None:
                voronoi_graph.add_edge(
                    (pos_2d[2], pos_2d[0], floor_id),
                    (last_pos[2], last_pos[0], floor_id),
                    dist=np.linalg.norm(pos_2d - last_pos),
                )
                last_pos = pos_2d
            else:
                last_pos = pos_2d

        # sparse_voronoi_graph = self.sparsify_graph(voronoi_graph, resampling_dist=0.4)
        sparse_voronoi_graph = voronoi_graph
        self.has_stairs = True
        self.draw_graph_on_map(
            top_down, sparse_voronoi_graph, floor_dir, "sparse_vor_stairs"
        )
        return sparse_voronoi_graph

    def get_stairs_graph(
        self,
        floor_id: str,
        objects_list: List[Object],
        floor_dir: str,
        clip_model: Any = None,
        clip_feat_dim: int = None,
    ) -> nx.Graph:
        """Get the objects that are stairs or staircase, generate a region of stairs and compute the stairs graph.

        Args:
            floor_id (str): Floor id.
            objects_list (List[Object]): The list of Object instances.
            floor_dir (str): The directory where the intermediate results are stored.
            clip_model (Any): The clip model.
            clip_feat_dim (int): The clip model dimension.

        Returns:
            spares_voronoi_graph (nx.Graph): The stairs graph.
        """
        stairs_list = self.get_stairs_objects(objects_list, clip_model, clip_feat_dim)

        floor_stairs_list = []
        for obj in stairs_list:
            obj_min = np.min(np.array(obj.pcd.points), axis=0)
            obj_max = np.max(np.array(obj.pcd.points), axis=0)
            if obj_min[1] > self.pcd_min[1] and obj_max[1] < self.pcd_max[1]:
                floor_stairs_list.append(obj)

        # compute the stairs navigation graph (voronoi graph)
        if len(floor_stairs_list) == 0:
            self.has_stairs = False
            return None
        self.has_stairs = True

        # merge all points in floor_stairs_list
        floor_stairs_pcd = np.concatenate(
            [obj.pcd.points for obj in floor_stairs_list], axis=0
        )

        # save floor_stairs_pcd to ply
        floor_stairs_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(floor_stairs_pcd)
        )
        save_path = os.path.join(floor_dir, f"floor_{floor_id}_stairs.ply")
        o3d.io.write_point_cloud(save_path, floor_stairs_pcd)

        stairs_map_no_close = self.create_occupancy_grid(
            np.asarray(floor_stairs_pcd.points)[:, [0, 2]]
        ).astype(np.uint8)
        cv2.imwrite(
            os.path.join(floor_dir, "stairs_map_no_closing.png"),
            stairs_map_no_close.astype(np.uint8) * 255,
        )
        stairs_map_no_erosion = binary_closing(
            stairs_map_no_close, iterations=50, structure=np.ones((3, 3))
        ).astype(np.uint8)
        cv2.imwrite(
            os.path.join(floor_dir, "stairs_map_no_erosion.png"),
            stairs_map_no_erosion.astype(np.uint8) * 255,
        )
        stairs_map = binary_erosion(stairs_map_no_erosion, iterations=5).astype(
            np.uint8
        )
        cv2.imwrite(
            os.path.join(floor_dir, "stairs_map.png"), stairs_map.astype(np.uint8) * 255
        )
        stairs_height = self.get_height_map(
            np.asarray(floor_stairs_pcd.points), floor_dir, stairs_map_no_erosion, 10
        )
        stairs_voronoi = self.get_voronoi_graph(
            stairs_map,
            stairs_map,
            floor_dir,
            floor_id,
            name="stairs",
            height_map=stairs_height,
        )
        # sparse_stairs_voronoi = self.sparsify_graph(stairs_voronoi, resampling_dist=0.4)
        sparse_stairs_voronoi = stairs_voronoi
        sparse_stairs_voronoi = self.trim_graph(sparse_stairs_voronoi, trim_deg=1)
        self.draw_graph_on_map(
            stairs_map, sparse_stairs_voronoi, floor_dir, "sparse_vor_stairs"
        )
        return sparse_stairs_voronoi

    def get_floor_graph(
        self, floor: Floor, floor_poses_list: List[np.ndarray], floor_dir: str
    ) -> nx.Graph:
        """Get the floor voronoi graph.

        Args:
            floor (Floor): Floor instance.
            floor_poses_list (List[np.ndarray]): The list of poses on the floor.
            floor_dir (str): The directory where the intermediate results are stored.

        Returns:
            spares_voronoi_graph (nx.Graph): The floor graph.
        """
        # compute the floor navigation graph (voronoi graph)
        floor_info = {
            "floor_zero_level": floor.floor_zero_level,
            "floor_height": floor.floor_height,
        }
        main_free_map = self.get_main_free_map(
            floor.pcd, floor_info, floor_dir, floor_poses_list
        ).astype(np.uint8)
        top_down = self.get_top_down_rgb_map(floor.pcd, floor_info, floor_dir)
        voronoi_graph = self.get_voronoi_graph(
            main_free_map, top_down, floor_dir, int(floor.floor_id)
        )
        sparse_floor_voronoi = self.sparsify_graph(voronoi_graph, resampling_dist=0.4)
        # sparse_floor_voronoi = voronoi_graph

        self.draw_graph_on_map(
            top_down, sparse_floor_voronoi, floor_dir, "sparse_vor_rgb"
        )
        self.draw_graph_on_map(
            main_free_map, sparse_floor_voronoi, floor_dir, "sparse_vor"
        )

        # save necessary data
        self.top_down = top_down
        self.sparse_floor_voronoi = sparse_floor_voronoi
        return sparse_floor_voronoi

    def connect_voronoi_graphs(
        self, src_graph: nx.Graph, tar_graph: nx.Graph, floor_dir: str = None
    ) -> nx.Graph:
        """Connect two graphs by finding the closest node in the source graph to the target graph.

        Args:
            src_graph (nx.Graph): The source graph.
            tar_graph (nx.Graph): The target graph.
            floor_dir (str, optional): The directory where intermediate results will be saved. Defaults to None.

        Returns:
            tar_graph (nx.Graph): The resulting graph.
        """
        # find closest node from src_graph to tar_graph
        floor_nodes = [
            (i, tar_graph.nodes[node]["pos"])
            for i, node in enumerate(tar_graph.nodes)
            if tar_graph.degree(node) > 1
        ]
        floor_node_ids = [node[0] for node in floor_nodes]
        floor_nodes = np.array([node[1] for node in floor_nodes])
        stairs_nodes = np.array(
            [
                src_graph.nodes[node]["pos"] for node in src_graph.nodes
            ]  # if src_graph.degree(node) == 1]
        )
        print(np.unique(stairs_nodes[:, 1], return_counts=True))
        print(np.unique(floor_nodes[:, 1], return_counts=True))

        dist_mat = cdist(stairs_nodes, floor_nodes)
        row, col = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
        col = floor_node_ids[col]
        stair_node = list(src_graph.nodes)[row]
        floor_node = list(tar_graph.nodes)[col]
        print("stairs, floor")
        print(stair_node, floor_node)
        print(src_graph.nodes[stair_node]["pos"], tar_graph.nodes[floor_node]["pos"])
        print(
            src_graph.nodes[stair_node]["floor_id"],
            tar_graph.nodes[floor_node]["floor_id"],
        )

        tar_graph = nx.compose(tar_graph, src_graph)
        tar_graph.add_edge(
            stair_node,
            floor_node,
            dist=np.linalg.norm(np.array(stair_node[:2]) - np.array(floor_node[:2])),
        )
        if floor_dir is not None:
            fig = self.draw_graph_on_map(
                self.top_down, tar_graph, floor_dir, "vor_rgb_combined_highlighted"
            )
            src_pos = np.int32([stair_node[1], stair_node[0]])
            tar_pos = np.int32([floor_node[1], floor_node[0]])
            cv2.circle(fig, tuple(src_pos), 3, (0, 255, 0), -1)
            cv2.circle(fig, tuple(tar_pos), 3, (0, 255, 0), -1)
        return tar_graph

    def connect_stairs_and_floor_graphs(
        self,
        sparse_stairs_voronoi: nx.Graph,
        sparse_floor_voronoi: nx.Graph,
        floor_dir: str,
    ):
        """Connect stairs graph to the floor graph by finding the closest node in the source graph to the target graph.

        Args:
            sparse_stairs_voronoi (nx.Graph): The stairs graph.
            sparse_floor_voronoi (nx.Graph): The floor graph.
            floor_dir (str, optional): The directory where intermediate results will be saved. Defaults to None.

        Returns:
            sparse_floor_voronoi (nx.Graph): The resulting graph.
        """
        # connect sparse_floor_voronoi and sparse_stairs_voronoi
        if not self.has_stairs:
            return sparse_floor_voronoi
        sparse_floor_voronoi = self.connect_voronoi_graphs(
            sparse_stairs_voronoi, sparse_floor_voronoi, floor_dir
        )
        self.draw_graph_on_map(
            self.top_down, sparse_floor_voronoi, floor_dir, "vor_rgb_combined"
        )
        self.sparse_floor_voronoi = sparse_floor_voronoi
        return sparse_floor_voronoi

    @staticmethod
    def save_voronoi_graph(graph: nx.Graph, floor_dir: str, name: str) -> None:
        """Save the Voronoi graph to a json file.

        Args:
            graph (nx.Graph): The Voronoi graph.
            floor_dir (str): The directory where the intermediate results are stored.
            name (str): The name of the file.
        """
        graph_path = os.path.join(floor_dir, f"{name}_graph.json")
        graph_json = nx.node_link_data(graph)
        with open(graph_path, "w") as f:
            json.dump(graph_json, f, indent=4)
