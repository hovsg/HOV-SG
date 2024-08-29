import os
import sys
import json
import torch

import numpy as np
import open3d as o3d
import networkx as nx

from collections import defaultdict
from PIL import ImageColor
from scipy.optimize import linear_sum_assignment
from torchmetrics.functional import pairwise_cosine_similarity

from hovsg.data.hm3dsem.create_hm3dsem_walks_gt import PanopticLevel, PanopticRegion, PanopticObject
from hovsg.utils.eval_utils import find_box_center_and_dims, get_3d_iou
from hovsg.utils.graph_utils import find_overlapping_ratio_faiss, find_intersection_share
from hovsg.graph.floor import Floor
from hovsg.graph.room import Room
from hovsg.graph.object import Object


class PanopticBuildingEval:
    def __init__(self, building_id):
        self.id = building_id
        self.type = "building"
        self.name = "building"

    def __str__(self):
        return f"{self.id}"


class PanopticLevelEval(PanopticLevel):
    def __str__(self):
        return f"{self.id}"


class PanopticRegionEval(PanopticRegion):
    def __init__(self, region_id, floor_id, category, voted_category, min_height, max_height, mean_height):
        self.id = region_id
        self.floor_id = floor_id
        self.voted_category = voted_category
        self.category = category
        self.hier_id = f"{self.floor_id}_{self.id}"
        self.objects = []
        self.type = "room"

        self.min_height = min_height
        self.max_height = max_height
        self.mean_height = mean_height
        self.region_points = None
        self.bev_region_points = None

    def __str__(self) -> str:
        return f"{self.floor_id}_{self.id}"


class PanopticObjectEval(PanopticObject):
    def __init__(self, object_id, region_id, floor_id, category, hex):
        # semantics object info
        self.id = object_id
        self.hex = hex
        self.category = category
        self.region_id = region_id
        self.floor_id = floor_id
        self.rgb = np.array(ImageColor.getcolor("#" + self.hex, "RGB"))
        self.type = "object"
        self.hier_id = f"{self.floor_id}_{self.region_id}_{self.id}"

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

    def __str__(self) -> str:
        return f"{self.floor_id}_{self.region_id}_{self.id}"


class HM3DSemanticEvaluator:
    def __init__(self, params):
        self.params = params
        self.gt_graph = nx.DiGraph()

        self.gt_floors = dict()
        self.gt_rooms = dict()
        self.gt_objects = []

        self.metrics = defaultdict(dict)

    def load_gt_graph_from_json(self, path):
        self.gt_scene_infos_path = path
        print("Loading GT graph from: ", self.gt_scene_infos_path)
        with open(self.gt_scene_infos_path, "r") as file:
            scene_info = json.load(file)

        building = PanopticBuildingEval(-1)
        self.gt_graph.add_node(building, name="building", type="building")

        for level_info in scene_info["levels"]:
            level_id = level_info["id"]
            floor = PanopticLevelEval(level_id, level_info["lower"], level_info["upper"])
            floor.regions = level_info["regions"]
            floor.objects = level_info["objects"]

            self.gt_graph.add_node(floor, name=f"floor_{level_id}", type="floor")
            self.gt_graph.add_edge(building, floor)
            self.gt_floors[floor.id] = floor

        for region_info in scene_info["regions"]:
            room = PanopticRegionEval(
                region_info["id"],
                region_info["floor_id"],
                region_info["category"],
                region_info["voted_category"],
                region_info["min_height"],
                region_info["max_height"],
                region_info["mean_height"],
            )
            room.graph_id = f"{room.floor_id}_{room.id}"
            room.bev_region_points = np.array(region_info["bev_region_points"])
            room.bev_pcd = o3d.geometry.PointCloud()
            room.bev_pcd.points = o3d.utility.Vector3dVector(room.bev_region_points)
            room.objects = region_info["objects"]

            print(room)

            self.gt_graph.add_node(room, name=f"room_{room.id}", type="room")
            self.gt_graph.add_edge(self.gt_floors[int(room.floor_id)], room)
            self.gt_rooms[room.id] = room

        for obj_info in scene_info["objects"]:
            obj = PanopticObjectEval(
                obj_info["id"], obj_info["region_id"], obj_info["floor_id"], obj_info["category"], obj_info["hex"]
            )
            obj.aabb_center, obj.aabb_dims = obj_info["aabb_center"], obj_info["aabb_dims"]
            obj.obb_center, obj.obb_dims = obj_info["obb_center"], obj_info["obb_dims"]
            obj.obb_rotation = obj_info["obb_rotation"]
            obj.obb_local_to_world = obj_info["obb_local_to_world"]
            obj.obb_world_to_local = obj_info["obb_world_to_local"]
            obj.obb_volume = obj_info["obb_volume"]
            obj.obb_half_extents = obj_info["obb_half_extents"]

            # load points from object pcd under self.gt_scene_infos_path + "/objects"
            obj_pcd_path = os.path.join(
                os.path.dirname(self.gt_scene_infos_path), "objects", str(obj_info["id"]) + ".ply"
            )
            obj.pcd = o3d.io.read_point_cloud(obj_pcd_path)
            obj.points = np.asarray(obj.pcd.points)

            self.gt_graph.add_node(obj, name=obj.category, type="object")
            self.gt_graph.add_edge(self.gt_rooms[int(obj.region_id)], obj)
            self.gt_objects.append(obj)

        print("----------------------------")
        print("GT graph loaded:")
        print("Number of GT floors: ", len([node for node in self.gt_graph.nodes if node.type == "floor"]))
        print("Number of GT rooms: ", len([node for node in self.gt_graph.nodes if node.type == "room"]))
        print("Number of GT objects: ", len([node for node in self.gt_graph.nodes if node.type == "object"]))
        print("----------------------------")

    def get_results(self):
        return self.metrics

    def evaluate_floors(self, pred_graph):
        """
        Evaluate the floor prediction by comparing low an upper bounds of the predicted floor with the ground truth floor
        """
        gt_floors = [node for node in self.gt_graph.nodes if node.type == "floor"]
        pred_floors = [node for node in pred_graph.nodes if type(node) == Floor]
        
        gt_floors_bounds = []
        for gt_floor in gt_floors:
            gt_floors_bounds.append([gt_floor.lower, gt_floor.upper])
        gt_floors_bounds = [y for x in gt_floors_bounds for y in x]
        gt_floors_bounds.sort()
        
        gt_floors_bounds_ = [
            (gt_floors_bounds[i] + gt_floors_bounds[i + 1]) / 2 for i in range(1, len(gt_floors_bounds) - 1, 2)
        ]
        gt_floors_bounds = [gt_floors_bounds[0]] + gt_floors_bounds_ + [gt_floors_bounds[-1]]

        pred_floors_bounds = []
        for pred_floor in pred_floors:
            pred_floor_center, pred_floor_dims = find_box_center_and_dims(pred_floor.vertices)
            pred_floor_y_bounds = [
                pred_floor_center[1] - pred_floor_dims[1] / 2,
                pred_floor_center[1] + pred_floor_dims[1] / 2,
            ]
            pred_floors_bounds.append(pred_floor_y_bounds)
        pred_floors_bounds = [y for x in pred_floors_bounds for y in x]
        pred_floors_bounds.sort()
        pred_floors_bounds_ = [
            (pred_floors_bounds[i] + pred_floors_bounds[i + 1]) / 2 for i in range(1, len(pred_floors_bounds) - 1, 2)
        ]
        pred_floors_bounds = [pred_floors_bounds[0]] + pred_floors_bounds_ + [pred_floors_bounds[-1]]
        
        # calc distance between each gt floor and each openmap floor
        dist = np.abs(np.array(gt_floors_bounds) - np.array(pred_floors_bounds))

        TP, TN, FP, FN = 0, 0, 0, 0
        floor_dist_threshold = 0.5
        for i in range(len(dist)):
            if dist[i] < floor_dist_threshold:
                TP += 1
        FP = len(pred_floors_bounds) - TP
        FN = len(gt_floors_bounds) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        print("----- Floor Evaluation -----")
        floor_metrics = {"tp": TP, "tn": TN, "fp": FP, "fn": FN, "acc": accuracy, "prec": precision, "recall": recall}
        self.metrics["floors"] = floor_metrics
        for k, v in floor_metrics.items():
            print("{}: {}".format(k, v))
        print("----------------------------")

    def evaluate_rooms(self, pred_graph):
        """
        Evaluate the room prediction by comparing 3D IoU of the predicted room with the ground truth room
        """
        gt_floors = [node for node in self.gt_graph.nodes if node.type == "floor"]
        gt_rooms = [node for node in self.gt_graph.nodes if node.type == "room"]
        pred_rooms = [node for node in pred_graph.nodes if isinstance(node, Room)]

        # find openmap room corresponding to each gt room based on distance between centers
        room_association_matrix = np.zeros((len(pred_rooms), len(gt_rooms)))
        for gt_floor in gt_floors:
            for gt_room in gt_rooms:
                if gt_room.mean_height > gt_floor.lower and gt_room.mean_height < gt_floor.upper:
                    for pred_room in pred_rooms:
                        pred_room_points = np.asarray(pred_room.pcd.points)
                        pred_mean_height = pred_room.room_zero_level + pred_room.room_height / 2
                        if pred_mean_height > gt_room.min_height and pred_mean_height < gt_room.max_height:
                            pred_room.bev_pcd = o3d.geometry.PointCloud()
                            pred_room_points[:, 1] = gt_room.min_height  # overwrite this to get planes on same height
                            pred_room.bev_pcd.points = o3d.utility.Vector3dVector(pred_room_points)
                            gt_room.bev_pcd = gt_room.bev_pcd.voxel_down_sample(voxel_size=0.05)
                            pred_room.bev_pcd = pred_room.bev_pcd.voxel_down_sample(voxel_size=0.05)
                            overlap = find_overlapping_ratio_faiss(pred_room.bev_pcd, gt_room.bev_pcd, 0.05)

                            room_association_matrix[pred_rooms.index(pred_room)][gt_rooms.index(gt_room)] = overlap

        hydra_room_overlap_over_pred = np.zeros((len(pred_rooms), len(gt_rooms)))
        hydra_room_overlap_over_gt = np.zeros((len(pred_rooms), len(gt_rooms)))
        for gt_floor in gt_floors:
            for gt_room in gt_rooms:
                if gt_room.mean_height > gt_floor.lower and gt_room.mean_height < gt_floor.upper:
                    for pred_room in pred_rooms:
                        pred_room_points = np.asarray(pred_room.pcd.points)
                        pred_mean_height = pred_room.room_zero_level + pred_room.room_height / 2
                        if pred_mean_height > gt_room.min_height and pred_mean_height < gt_room.max_height:
                            pred_room.bev_pcd = o3d.geometry.PointCloud()
                            pred_room_points[:, 1] = gt_room.min_height  # overwrite this to get planes on same height
                            pred_room.bev_pcd.points = o3d.utility.Vector3dVector(pred_room_points)
                            pred_room.bev_pcd.colors = o3d.utility.Vector3dVector(
                                np.array([[0, 0, 1] for i in range(len(pred_room.bev_pcd.points))])
                            )
                            gt_room.bev_pcd.colors = o3d.utility.Vector3dVector(
                                np.array([[1, 0, 0] for i in range(len(gt_room.bev_pcd.points))])
                            )
                            gt_room.bev_pcd = gt_room.bev_pcd.voxel_down_sample(voxel_size=0.05)
                            pred_room.bev_pcd = pred_room.bev_pcd.voxel_down_sample(voxel_size=0.05)
                            overlap_over_pred = min(
                                find_intersection_share(
                                    np.asarray(gt_room.bev_pcd.points), np.asarray(pred_room.bev_pcd.points), 0.05
                                ),
                                1.0,
                            )
                            overlap_over_gt = min(
                                find_intersection_share(
                                    np.asarray(pred_room.bev_pcd.points), np.asarray(gt_room.bev_pcd.points), 0.05
                                ),
                                1.0,
                            )
                            hydra_room_overlap_over_pred[pred_rooms.index(pred_room)][
                                gt_rooms.index(gt_room)
                            ] = overlap_over_pred
                            hydra_room_overlap_over_gt[pred_rooms.index(pred_room)][
                                gt_rooms.index(gt_room)
                            ] = overlap_over_gt

        hydra_precision = 0.0
        hydra_recall = 0.0
        for i in range(len(pred_rooms)):
            # get max overlap for each pred room
            prec_max_overlap = np.max(hydra_room_overlap_over_pred[i, :])
            hydra_precision += prec_max_overlap

        for j in range(len(gt_rooms)):
            # get max overlap for each gt room
            rec_max_overlap = np.max(hydra_room_overlap_over_gt[:, j])
            hydra_recall += rec_max_overlap

        hydra_precision = hydra_precision / len(pred_rooms)
        hydra_recall = hydra_recall / len(gt_rooms)

        # calculate TP, FP, FN for rooms
        row_ind, col_ind = linear_sum_assignment(room_association_matrix, maximize=True)
        acc_values = list()
        prec_values = list()
        rec_values = list()
        for eval_idx, thresh in enumerate(np.linspace(0.0, 1.0, 11, endpoint=True)):
            TP, TN, FP, FN = 0, 0, 0, 0
            iou_threshold = thresh
            TP = np.sum(room_association_matrix[row_ind, col_ind] > iou_threshold)
            FP = len(pred_rooms) - TP
            FN = len(gt_rooms) - TP

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

            acc_values.append(accuracy)
            prec_values.append(precision)
            rec_values.append(recall)

        rec_values.sort()
        avg_prec = np.trapz(prec_values, rec_values)

        print("----- Room Evaluation ------")
        room_metrics = {
            "acc@IoU=0.5": acc_values[6],
            "ap": avg_prec,
            "prec@IoU=0.5": prec_values[6],
            "recall@IoU=0.5": rec_values[6],
            "gt": len(gt_rooms),
            "pred": len(pred_rooms),
            "hydra_prec": hydra_precision,
            "hydra_recall": hydra_recall,
        }
        self.metrics["rooms"] = room_metrics
        for k, v in room_metrics.items():
            print("{}: {}".format(k, v))
        print("----------------------------")

    def evaluate_objects(self, eval_metric, pred_graph, gt_classes, gt_text_feats):
        """
        Evaluate the object prediction by comparing 3D IoU of the predicted object with the ground truth object
        """
        object_metrics = {"instances": {}, "instances_iou50": {}, "ins_semantics": {}}
        self.metrics["objects"] = object_metrics
        print("----- Object Evaluation ------")

        gt_objects = [node for node, n_data in self.gt_graph.nodes(data=True) if (node.type == "object")]
        pred_objects = [node for node in pred_graph.nodes if (type(node) == Object)]

        # Evaluation of class-agnostic instance segmentation on point clouds
        # find openmap object corresponding to each gt object based on distance between centers
        obj_iou_assoc_matrix = np.zeros((len(pred_objects), len(gt_objects)))
        obj_overlap_assoc_matrix = np.zeros((len(pred_objects), len(gt_objects)))

        for gt_obj in gt_objects:
            gt_obj_center = gt_obj.obb_center
            gt_obj_dims = gt_obj.obb_dims

            gt_obj_bbox = np.array(gt_obj.pcd.get_axis_aligned_bounding_box().get_box_points())
            for pred_obj in pred_objects:
                pred_obj_bbox = np.array(pred_obj.pcd.get_axis_aligned_bounding_box().get_box_points())
                pred_obj_center, pred_obj_dims = find_box_center_and_dims(pred_obj_bbox)

                iou = get_3d_iou(gt_obj_center, gt_obj_dims, pred_obj_center, pred_obj_dims)
                obj_iou_assoc_matrix[pred_objects.index(pred_obj)][gt_objects.index(gt_obj)] = iou

                overlap = 0.0
                if iou > 0.0:
                    overlap = find_overlapping_ratio_faiss(pred_obj.pcd, gt_obj.pcd, 0.02)
                    obj_overlap_assoc_matrix[pred_objects.index(pred_obj)][gt_objects.index(gt_obj)] = overlap

        if eval_metric == "iou":
            row_ind, col_ind = linear_sum_assignment(obj_iou_assoc_matrix, maximize=True)
        elif eval_metric == "overlap":
            row_ind, col_ind = linear_sum_assignment(obj_overlap_assoc_matrix, maximize=True)

        acc_values = list()
        prec_values = list()
        rec_values = list()
        for eval_idx, thresh in enumerate(np.linspace(0.0, 1.0, 11, endpoint=True)):
            TP, TN, FP, FN = 0, 0, 0, 0
            TP = np.sum(obj_overlap_assoc_matrix[row_ind, col_ind] > thresh)
            FP = len(pred_objects) - TP
            FN = len(gt_objects) - TP
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            acc_values.append(accuracy)
            prec_values.append(precision)
            rec_values.append(recall)
        rec_values.sort()
        avg_prec = np.trapz(prec_values, rec_values)

        obj_instance_metrics = {"ap": avg_prec, "gt": len(gt_objects), "pred": len(pred_objects)}
        self.metrics["objects"]["instances"] = obj_instance_metrics
        print("- - - - - - - - - - - - - - - ")
        print("Object Instance Evaluation:")
        for k, v in obj_instance_metrics.items():
            print("{}: {}".format(k, v))
        print("- - - - - - - - - - - - - - - ")

        TP, TN, FP, FN = 0, 0, 0, 0
        threshold = 0.5
        TP = np.sum(obj_overlap_assoc_matrix[row_ind, col_ind] > threshold)
        FP = len(pred_objects) - TP
        FN = len(gt_objects) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        obj_instance_iou50_metrics = {
            "acc": accuracy,
            "prec": precision,
            "recall": recall,
            "tp": int(TP),
            "fp": int(FP),
            "fn": int(FN),
            "gt": len(gt_objects),
            "pred": len(pred_objects),
        }
        self.metrics["objects"]["instances_iou50"] = obj_instance_iou50_metrics
        print("Object Instance Evaluation @ IoU={}:".format(threshold))
        for k, v in obj_instance_iou50_metrics.items():
            print("{}: {}".format(k, v))
        print("- - - - - - - - - - - - - - - ")


        print("Top-k semantics evaluation")
        # eval top-k accuracy at specific thresholds
        top_k_acc_representative, _ = self.object_semantics_eval_tp_auc(
            self.params.eval.hm3dsem.top_k_object_semantic_eval,
            row_ind,
            col_ind,
            pred_objects,
            gt_objects,
            gt_text_feats,
            gt_classes,
        )

        # eval top-k auc score based on a range of thresholds
        _, tp_multi_top_k_acc_auc = self.object_semantics_eval_tp_auc(
            [k for k in range(0, len(gt_classes), 10)],
            row_ind,
            col_ind,
            pred_objects,
            gt_objects,
            gt_text_feats,
            gt_classes,
        )

        obj_instance_semantics_topk_class_metrics = dict()
        obj_instance_semantics_topk_class_metrics["tp_top_k_acc"] = top_k_acc_representative
        obj_instance_semantics_topk_class_metrics["top_k_auc"] = tp_multi_top_k_acc_auc

        self.metrics["objects"]["ins_semantics"] = obj_instance_semantics_topk_class_metrics
        print("TP Instance Semantics Evaluation (top-k)")
        for k, v in obj_instance_semantics_topk_class_metrics.items():
            print("{}: {}".format(k, v))
        print("- - - - - - - - - - - - - - - ")

    def object_semantics_eval_tp_auc(
        self, top_k_spec, row_ind, col_ind, pred_objects, gt_objects, gt_text_feats, gt_classes
    ):
        success_k = {k: list() for k in top_k_spec}
        for pred_idx, gt_idx in zip(row_ind, col_ind):
            # dot_sim = np.dot(pred_objects[pred_idx].embedding, gt_text_feats.T)
            dot_sim = (
                pairwise_cosine_similarity(
                    torch.from_numpy(pred_objects[pred_idx].embedding.reshape(1, -1)).float(),
                    torch.from_numpy(gt_text_feats).float(),
                )
                .squeeze(0)
                .numpy()
            )
            # sort the dot similarity scores in descending order
            sorted_dot_similarity = np.sort(dot_sim)[::-1]
            for k in top_k_spec:
                top_k_idx = np.argsort(dot_sim)[::-1][:k]
                # get names of top k classes
                top_k_classes = [gt_classes[idx] for idx in top_k_idx]
                if gt_objects[gt_idx].category in top_k_classes:
                    success_k[k].append((pred_idx))
        top_k_acc = {k: len(v) / len(col_ind) for k, v in success_k.items()}

        norm_top_k = [k / len(gt_classes) for k in top_k_spec]
        tp_top_k_auc = np.trapz(list(top_k_acc.values()), norm_top_k)
        return top_k_acc, tp_top_k_auc
