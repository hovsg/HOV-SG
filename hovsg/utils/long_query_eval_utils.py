from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from omegaconf import DictConfig, OmegaConf
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.evaluate_graph_hm3dsem import HM3DSemanticEvaluator


def parse_json_gt_graph_hm3dsem(json_path: Union[str, Path]) -> Dict:
    with open(json_path, "r") as f:
        data = json.load(f)

    tree = nx.DiGraph()

    tree.add_node("root", **{"id": "root", "category": "building"})
    levels = ["levels", "regions", "objects"]

    for level in data["levels"]:
        tree.add_node(level["id"], **level)
        tree.add_edge("root", level["id"])

    ignored_room_ids = set()
    for region in data["regions"]:
        level_id, _ = region["id"].split("_")
        if region["category"] in ["living room", "kitchen", "bedroom", "bathroom"]:
            tree.add_node(region["id"], **region)
            tree.add_edge(level_id, region["id"])
        else:
            ignored_room_ids.add(region["id"])
            continue

    for object in data["objects"]:
        if object["category"] in [
            "wall",
            "floor",
            "ceiling",
            "door",
            "window",
            "stairs",
            "railing",
            "objects",
            "misc",
            "void",
            "",
        ]:
            continue
        level_id, region_sub_id, _ = object["id"].split("_")
        region_id = "_".join([level_id, region_sub_id])
        if region_id in ignored_room_ids:
            continue

        tree.add_node(object["id"], **object)
        tree.add_edge(region_id, object["id"])

    return tree


def generate_long_queries(tree: nx.DiGraph) -> Tuple[List[str], List[Tuple]]:
    """Generate long hierarchical queries from the given tree. The queries are in the form of <object X in region Y on floor Z>.

    Args:
        tree (nx.DiGraph): The hierarchical graph.

    Returns:
        queries (List[str]): The list of long queries.
        gt_nodes_tuples (List[Tuple]): The list of GT node tuples [(floor_id, room_id, object_id), ...]
    """
    queries = []
    gt_nodes_tuples = []
    nodes = [
        node
        for node, out_degree in tree.out_degree()
        if out_degree == 0 and len(node.split("_")) == 3
    ]
    for node in nodes:
        floor_id, room_sub_id, object_sub_id = node.split("_")
        room_id = "_".join([floor_id, room_sub_id])
        room_cat = tree.nodes[room_id]["category"]
        object_cat = tree.nodes[node]["category"]
        query = object_cat + " in region " + room_cat + " on floor " + floor_id
        queries.append(query)
        nodes_tuple = (floor_id, room_id, node)
        gt_nodes_tuples.append(nodes_tuple)

    return queries, gt_nodes_tuples


def generate_gt_object_nodes(
    tree: nx.DiGraph, gt_nodes_tuples: Tuple[str, str, str]
) -> List[List[str]]:
    """The GT object nodes ids. Given a hierarchical query "X object in region Y on floor Z", there
       might be multiple correct target objects. For example, there are multiple "region Y" on the floor,
       and "object X" exists in several "region Y". This function computes all valid object instances given
       a certain query.

    Args:
        tree (nx.DiGraph): The hierarchical graph.
        gt_nodes_tuples (Tuple[str, str, str]): a list of tuple (floor_id, room_id, object_id). We check
                                                the category of room_id and object_id on floor_id and
                                                find same category combination of room and object on the
                                                same floor.

    Returns:
        List[List[str]]: a list of object node ids lists.
    """
    gt_node_ids = []
    room_nodes = [
        node for node, out_degree in tree.out_degree() if len(node.split("_")) == 2
    ]
    for i, gt_nodes_tuple in enumerate(gt_nodes_tuples):
        floor_id, room_id, object_id = gt_nodes_tuple
        room_category = tree.nodes[room_id]["category"]
        object_category = tree.nodes[object_id]["category"]

        # select all room of the same category at the same level
        candidate_room_nodes = [
            node for node in room_nodes if tree.nodes[node]["category"] == room_category
        ]

        # select all object of the same category in candidate rooms
        object_nodes = []
        for room_node in candidate_room_nodes:
            successors = tree.successors(room_node)
            for obj_node in successors:
                if tree.nodes[obj_node]["category"] == object_category:
                    object_nodes.append(obj_node)
        gt_node_ids.append(object_nodes)
    return gt_node_ids


def filter_duplicates_long_queries(
    queries: List[str], gt_nodes_tuples: List[Tuple[str, str, str]]
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """Filter out repeated long queries

    Args:
        queries (List[str]): a list of long queries.
        gt_nodes_tuples (List[Tuple[str, str, str]]): a list of tuple (floor_id, room_id, object_id).

    Returns:
        Tuple[List[str], List[Tuple[str, str, str]]]: filtered queries and gt_nodes_tuples
    """
    filtered_queries = []
    filtered_gt_nodes_tuples = []
    queries_set = set()
    for query, gt_nodes_tuple in zip(queries, gt_nodes_tuples):
        if query in queries_set:
            continue
        queries_set.add(query)
        filtered_queries.append(query)
        filtered_gt_nodes_tuples.append(gt_nodes_tuple)

    return filtered_queries, filtered_gt_nodes_tuples


def aggregate_duplicates_long_queries(
    queries: List[str], gt_nodes_tuples: List[Tuple[str, str, str]]
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """Filter out repeated long queries

    Args:
        queries (List[str]): a list of long queries.
        gt_nodes_tuples (List[Tuple[str, str, str]]): a list of tuple (floor_id, room_id, object_id).

    Returns:
        Tuple[List[str], List[Tuple[str, str, str]]]: filtered queries and gt_nodes_tuples
    """
    aggregated_queries = dict()
    for query, gt_nodes_tuple in zip(queries, gt_nodes_tuples):
        if query in aggregated_queries.keys():
            aggregated_queries[query].append(gt_nodes_tuple)
        else:
            aggregated_queries[query] = [gt_nodes_tuple]

    return aggregated_queries


def generate_long_query_dataset_hm3dsem(
    params, scene_dirs: List[Union[Path, str]]
) -> Dict[str, Tuple[List[str], List[nx.DiGraph]]]:
    """generate long query dataset

    Args:
        scene_dirs (List[Union[Path, str]]): dirs to collected habitat scenes where scene_info.json is included

    Returns:
        Dict[str, Tuple[List[str], List[nx.DiGraph], List[List[str]]]]: a dictionary map from the scene name to a tuple.
                                                       The tuple contains a list of long queries, a list of GT node
                                                       name tuples, the graph of the tree, and the GT object node ids.
    """
    query_spec = params.eval.hm3dsem.long_query.spec
    dataset = {}
    for scene_dir in scene_dirs:
        scene_dir = Path(scene_dir)
        print(f"Generating long queries for scene {scene_dir.as_posix()}")
        params.main.scene = str(scene_dir).split("/")[-1]
        # get last trailing folder name
        evaluator = HM3DSemanticEvaluator(params)
        evaluator.load_gt_graph_from_json(scene_dir / "scene_info.json")

        # Generate GT text queries and accompanying text queries (floor_id, room_id, object_id)
        queries = []
        gt_node_tuples = []
        for gt_obj_node in evaluator.gt_objects:
            query = ""
            query += gt_obj_node.category if "obj" in query_spec else ""
            if "room" in query_spec:
                query += " in the "
                query += evaluator.gt_rooms[gt_obj_node.region_id].category
            if "floor" in query_spec:
                query += " on floor "
                query += str(gt_obj_node.floor_id)

            # Depending on the query spec, define the GT tuple that is asked for given the query
            if "obj" in query_spec and "room" in query_spec and "floor" in query_spec:
                nodes_tuple = (
                    gt_obj_node.floor_id,
                    gt_obj_node.region_id,
                    gt_obj_node.id,
                )
            elif "obj" in query_spec and "room" in query_spec:
                nodes_tuple = (None, gt_obj_node.region_id, gt_obj_node.id)
            elif "obj" in query_spec and "floor" in query_spec:
                nodes_tuple = (gt_obj_node.floor_id, None, gt_obj_node.id)
            elif "room" in query_spec and "floor" in query_spec:
                nodes_tuple = (gt_obj_node.floor_id, gt_obj_node.region_id, None)
            elif "obj" in query_spec:
                nodes_tuple = (None, None, gt_obj_node.id)

            queries.append(query)
            gt_node_tuples.append(nodes_tuple)
        aggregated_queries = aggregate_duplicates_long_queries(queries, gt_node_tuples)

        dataset[scene_dir.name] = (evaluator, aggregated_queries)

    return dataset


def generate_long_query_dataset_hm3dsem(
    params, scene_dirs: List[Union[Path, str]]
) -> Dict[str, Tuple[List[str], List[nx.DiGraph]]]:
    """generate long query dataset

    Args:
        scene_dirs (List[Union[Path, str]]): dirs to collected habitat scenes where scene_info.json is included

    Returns:
        Dict[str, Tuple[List[str], List[nx.DiGraph], List[List[str]]]]: a dictionary map from the scene name to a tuple.
                                                       The tuple contains a list of long queries, a list of GT node
                                                       name tuples, the graph of the tree, and the GT object node ids.
    """
    dataset = {}
    for scene_dir in scene_dirs:
        scene_dir = Path(scene_dir)
        print(f"Generating long queries for scene {scene_dir.as_posix()}")
        json_path = scene_dir / "scene_info.json"
        tree = parse_json_gt_graph_hm3dsem(json_path)
        queries, gt_nodes_tuples = generate_long_queries(tree)
        queries, gt_nodes_tuples = filter_duplicates_long_queries(
            queries, gt_nodes_tuples
        )
        gt_object_nodes_ids = generate_gt_object_nodes(tree, gt_nodes_tuples)
        dataset[scene_dir.name] = (queries, gt_nodes_tuples, tree, gt_object_nodes_ids)

    return dataset


def create_3d_bounding_box(center: List[float], dimensions: List[float]) -> np.ndarray:
    """Create a 3D bounding box given the center and dimensions.

    Args:
        center (List[float]): 3D Center
        dimensions (List[float]): 3D Dimention

    Returns:
        np.ndarray: (8, 3) The list of points at the bounding box vertices.
    """
    center = np.array(center)
    dimensions = np.array(dimensions)
    # Calculate half of each dimension
    half_dims = dimensions / 2.0

    # Define the vertices of the bounding box
    vertices = [
        (-1, -1, -1),  # Vertex 0
        (-1, -1, 1),  # Vertex 1
        (-1, 1, -1),  # Vertex 2
        (-1, 1, 1),  # Vertex 3
        (1, -1, -1),  # Vertex 4
        (1, -1, 1),  # Vertex 5
        (1, 1, -1),  # Vertex 6
        (1, 1, 1),  # Vertex 7
    ]

    # Scale vertices by half_dims and shift by center
    vertices = np.array(vertices) * half_dims + center

    return vertices


def crop_pc_with_aabb(
    pcd: o3d.geometry.PointCloud, center: List[float], dim: List[float]
) -> o3d.geometry.PointCloud:
    """Crop a point cloud with an axis-aligned bounding box.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud.
        center (List[float]): 3D Center.
        dim (List[float]): 3D Dimension.

    Returns:
        o3d.geometry.PointCloud: Cropped point cloud.
    """
    bbox_pts = create_3d_bounding_box(center, dim)
    min_b = np.min(bbox_pts, axis=0).reshape((3, 1))
    max_b = np.max(bbox_pts, axis=0).reshape((3, 1))
    if not np.array_equal(min_b, max_b):
        return pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)
        )
    return None


def save_pcd(save_path: str, pcd: Union[np.ndarray, o3d.geometry.PointCloud]):
    """Save the point cloud

    Args:
        save_path (str): Save path.
        pcd (Union[np.ndarray, o3d.geometry.PointCloud]): Point cloud.
    """
    if isinstance(pcd, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(save_path, pcd)


def find_similar_category_id(class_name: str, classes_list: List[str]):
    """
    Return the id of the most similar name to class_name in classes_list
    """
    if class_name in classes_list:
        print(f"found {class_name} in the classes_list: {classes_list}")
        return classes_list.index(class_name)
    return None
    # import openai

    # openai_key = os.environ["OPENAI_KEY"]
    # openai.api_key = openai_key
    # classes_list_str = ",".join(classes_list)
    # question = f"""
    # Q: What is television most relevant to among tv_monitor,plant,chair. A:tv_monitor\n
    # Q: What is drawer most relevant to among tv_monitor,chest_of_drawers,chair. A:chest_of_drawers\n
    # Q: What is {class_name} most relevant to among {classes_list_str}. A:"""
    # response = openai.Completion.create(
    #     engine="text-davinci-002",
    #     prompt=question,
    #     max_tokens=64,
    #     temperature=0.0,
    #     stop="\n",
    # )
    # result = response["choices"][0]["text"].strip()
    # print(f"Similar category of {class_name} is {result}")
    # return classes_list.index(result)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="long_query_eval_config_hm3dsem",
)
def main(params):
    scene_names = params.dataset[params.main.dataset][params.main.split].scene_names
    scene_dirs = [
        Path(params.paths.data) / "hm3dsem_walks" / params.main.split / scene_name
        for scene_name in scene_names
    ]

    dataset = generate_long_query_dataset_hm3dsem(params, scene_dirs)
    print(dataset)


if __name__ == "__main__":
    main()
