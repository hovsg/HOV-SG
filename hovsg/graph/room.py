"""
Room class to represent a room in a HOV-SGraph.
"""

import json
import os
from collections import defaultdict
from typing import Any, List

import numpy as np
import open3d as o3d

from hovsg.utils.clip_utils import get_img_feats, get_text_feats_multiple_templates




class Room:
    """
    Class to represent a room in a building.
    :param room_id: Unique identifier for the room
    :param floor_id: Identifier of the floor this room belongs to
    :param name: Name of the room (e.g., "Living Room", "Bedroom")
    """
    def __init__(self, room_id, floor_id, name=None):
        self.room_id = room_id  # Unique identifier for the room
        self.name = name  # Name of the room (e.g., "Living Room", "Bedroom")
        self.category = None  # placeholder for a GT category
        self.floor_id = floor_id  # Identifier of the floor this room belongs to
        self.objects = []  # List of objects inside the room
        self.vertices = []  # indices of the room in the point cloud 8 vertices
        self.embeddings = []  # List of tensors of embeddings of the room
        self.pcd = None  # Point cloud of the room
        self.room_height = None  # Height of the room
        self.room_zero_level = None  # Zero level of the room
        self.represent_images = []  # 5 images that represent the appearance of the room
        self.object_counter = 0

    def add_object(self, objectt):
        """
        Method to add objects to the room
        :param objectt: Object object to be added to the room
        """
        self.objects.append(objectt)  # Method to add objects to the room

    def set_txt_embeddings(self, text):
        self.embeddings.append(get_text_feats_multiple_templates(text))

    def merge_objects(self, overlap_threshold=0.01, radius=0.1):
        """
        Merge objects that are close to each other and have the same name
        """
        # for every object in the room with the same name, calculate the overlap between them
        # if the overlap is more than the threshold, merge them
        # repeat until no more merging is possible

        overlap_scores = np.zeros((len(self.objects), len(self.objects)))
        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects):
                if i >= j:
                    continue
                if obj1.name == obj2.name:
                    overlap = find_overlapping_ratio(obj1.pcd, obj2.pcd, radius)
                    if overlap > overlap_threshold:
                        overlap_scores[i, j] = overlap
                        overlap_scores[j, i] = overlap

        new_room_objects = defaultdict(list)
        merging_idcs = list()
        i_idcs, j_idcs = np.where(overlap_scores > 0)
        for i, j in zip(i_idcs, j_idcs):
            merging_idcs.extend([i, j])
            if i not in list(new_room_objects.keys()) and j not in list(new_room_objects.keys()):
                new_room_objects[i].append(j)
            else:
                if i in list(new_room_objects.keys()):
                    new_room_objects[i].append(j)
                elif j in list(new_room_objects.keys()):
                    new_room_objects[j].append(i)
        # Add all remaning objects not involved in merging
        merging_idcs = list(set(merging_idcs))
        for idx in range(len(self.objects)):
            if idx not in merging_idcs:
                new_room_objects[idx].append(idx)

        # actual merging and re-indexing
        object_index = 0
        self.objects_new = []
        for i, j in new_room_objects.items():
            j = list(set(j))
            print(i, j)
            if len(j) == 1:
                jj = j[0]
                if i == jj:
                    obj = self.objects[i]
                    obj.object_id = self.room_id + "_" + str(object_index)
                    self.objects_new.append(obj)
                    object_index += 1
                if i != jj:
                    obj1 = self.objects[i]
                    obj2 = self.objects[jj]
                    obj1 += obj2
                    obj1.object_id = self.room_id + "_" + str(object_index)
                    self.objects_new.append(obj1)
                    object_index += 1
            elif len(j) > 1:
                obj1 = self.objects[i]
                for jj in j:
                    obj_jj = self.objects[jj]
                    obj1 += obj_jj
                    obj1.object_id = self.room_id + "_" + str(object_index)
                self.objects_new.append(obj1)
                object_index += 1
        self.objects = self.objects_new

    def infer_room_type_from_view_embedding(
        self,
        default_room_types: List[str],
        clip_model: Any,
        clip_feat_dim: int,
    ) -> str:
        """Use the embeddings stored inside the room to infer room type. We should already
           save k views CLIP embeddings for each room. We match the k embeddings with room
           types' textual CLIP embeddings to get a room label for each of the k views. Then
           we count which room type has the most votes and return that.

        Args:
            default_room_types (List[str]): the output room type should only be a room type from the list.
            clip_model (Any): when the generate_method is set to "embedding", a clip model needs to be
                              provided to the method.
            clip_feat_dim (int): when the generate_method is set to "embedding", the clip features dimension
                                 needs to be provided to this method

        Returns:
            str: a room type from the default_room_types list
        """
        if len(self.embeddings) == 0:
            print("empty embeddings")
            return "unknown room type"
        text_feats = get_text_feats_multiple_templates(default_room_types, clip_model, clip_feat_dim)
        embeddings = np.array(self.embeddings)
        sim_mat = np.dot(embeddings, text_feats.T)
        # sim_mat = compute_similarity(embeddings, text_feats)
        print(sim_mat)
        col_ids = np.argmax(sim_mat, axis=1)
        votes = [default_room_types[i] for i in col_ids]
        print(f"the votes are: {votes}")
        unique, counts = np.unique(col_ids, return_counts=True)
        unique_id = np.argmax(counts)
        type_id = unique[unique_id]
        self.name = default_room_types[type_id]
        print(f"The room view ids are {self.represent_images}")
        print(f"The room type is {default_room_types[type_id]}")
        return default_room_types[type_id]

    def infer_room_type_from_objects(
        self,
        infer_method: str = "label",
        default_room_types: List[str] = None,
        clip_model: Any = None,
        clip_feat_dim: int = None,
    ) -> str:
        """Use the objects contained in the room to infer a room type. We want to ask GPT what kind of room it is from
        the names for the objects contained in the room.

        Args:
            infer_method (str): "label" if we want to directly use the pre-computed object names in the children nodes.
                                "obj_embedding" if we want to use the embedding of the room node's children to infer the
                                room type. default_room_types can not be None if infer_method is "embedding".
            default_room_types (List[str] = None): the output room type should only be a room type from the list.
            clip_model (Any): when the generate_method is set to "embedding", a clip model needs to be
                              provided to the method.
            clip_feat_dim (int): when the generate_method is set to "embedding", the clip features dimension
                                 needs to be provided to this method

        Returns:
            str: room type name
        """
        from llm.llm_utils import infer_room_type_from_object_list_chat

        if infer_method == "label":
            objects_list = []
            for obj_i, obj in enumerate(self.objects):
                obj: Object
                if not any(
                    substring in obj.name.lower()
                    for substring in [
                        "wall",
                        "floor",
                        "ceiling",
                        "railing",
                        "roof",
                        "void",
                        "unlabeled",
                        "misc",
                    ]
                ):
                    objects_list.append(obj.name)
            room_type = infer_room_type_from_object_list_chat(objects_list, default_room_type=default_room_types)
            self.name = room_type
        if infer_method == "obj_embedding":
            assert default_room_types, "default_room_types can not be None if infer_method is 'embedding'"
            object_embs = []
            for obj_i, obj in enumerate(self.objects):
                obj: Object
                object_embs.append(obj.embedding)

            represent_feat = feats_denoise_dbscan(object_embs).reshape((1, -1))
            text_feats = get_text_feats_multiple_templates(default_room_types, clip_model, clip_feat_dim)
            sim_mat = compute_similarity(represent_feat, text_feats)
            col_id = np.argmax(sim_mat)
            self.name = default_room_types[col_id]
        print("room_id, name: ", self.room_id, self.name)

    def save(self, path):
        """
        Save the room in folder as ply for the point cloud
        and json for the metadata
        """
        # save the point cloud
        o3d.io.write_point_cloud(os.path.join(path, str(self.room_id) + ".ply"), self.pcd)
        # save the metadata
        metadata = {
            "room_id": self.room_id,
            "name": self.name,
            "floor_id": self.floor_id,
            "objects": [obj.object_id for obj in self.objects],
            "vertices": self.vertices.tolist(),
            "room_height": self.room_height,
            "room_zero_level": self.room_zero_level,
            "embeddings": [i.tolist() for i in self.embeddings],
            "represent_images": self.represent_images,
        }
        with open(os.path.join(path, str(self.room_id) + ".json"), "w") as outfile:
            json.dump(metadata, outfile)

    def load(self, path):
        """
        Load the room from folder as ply for the point cloud
        and json for the metadata
        """
        # load the point cloud
        self.pcd = o3d.io.read_point_cloud(os.path.join(path, str(self.room_id) + ".ply"))
        # load the metadata
        with open(path + "/" + str(self.room_id) + ".json") as json_file:
            metadata = json.load(json_file)
            self.name = metadata["name"]
            self.floor_id = metadata["floor_id"]
            self.vertices = np.asarray(metadata["vertices"])
            self.room_height = metadata["room_height"]
            self.room_zero_level = metadata["room_zero_level"]
            self.embeddings = [np.asarray(i) for i in metadata["embeddings"]]
            self.represent_images = metadata["represent_images"]

    def __str__(self):
        return f"Room ID: {self.room_id}, Name: {self.name}, Floor ID: {self.floor_id}, Objects: {len(self.objects)}"
