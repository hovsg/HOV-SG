"""
    Class to represent a floor in a HOV-Sgraph.
"""

import json
import os

import numpy as np
import open3d as o3d




class Floor:
    """
    Class to represent a floor in a building.
    :param floor_id: Unique identifier for the floor
    :param name: Name of the floor (e.g., "First", "Second")
    """
    def __init__(self, floor_id, name=None):
        self.floor_id = floor_id  # Unique identifier for the floor
        self.name = name  # Name of the floor (e.g., "First", "Second")
        self.rooms = []  # List of rooms in the floor
        self.txt_embeddings = []  # List of tensors of text embeddings of the floor
        self.pcd = None  # Point cloud of the floor
        self.vertices = []  # indices of the floor in the point cloud 8 vertices
        self.floor_height = None  # Height of the floor
        self.floor_zero_level = None  # Zero level of the floor

    def add_room(self, room):
        """
        Method to add rooms to the floor
        :param room: Room object to be added to the floor
        """
        self.rooms.append(room)  # Method to add rooms to the floor

    def save(self, path):
        """
        Save the floor in folder as ply for the point cloud
        and json for the metadata
        """
        # save the point cloud
        o3d.io.write_point_cloud(os.path.join(path, str(self.floor_id) + ".ply"), self.pcd)
        # save the metadata
        metadata = {
            "floor_id": self.floor_id,
            "name": self.name,
            "rooms": [room.room_id for room in self.rooms],
            "vertices": self.vertices.tolist(),
            "floor_height": self.floor_height,
            "floor_zero_level": self.floor_zero_level,
        }
        with open(os.path.join(path, str(self.floor_id) + ".json"), "w") as outfile:
            json.dump(metadata, outfile)

    def load(self, path):
        """
        Load the floor from folder as ply for the point cloud
        and json for the metadata
        """
        # load the point cloud
        self.pcd = o3d.io.read_point_cloud(path + "/" + str(self.floor_id) + ".ply")
        # load the metadata
        with open(path + "/" + str(self.floor_id) + ".json") as json_file:
            metadata = json.load(json_file)
            self.name = metadata["name"]
            self.rooms = metadata["rooms"]
            self.vertices = np.asarray(metadata["vertices"])
            self.floor_height = metadata["floor_height"]
            self.floor_zero_level = metadata["floor_zero_level"]

    def __str__(self):
        return f"Floor ID: {self.floor_id}, Name: {self.name}, Rooms: {len(self.rooms)}"
