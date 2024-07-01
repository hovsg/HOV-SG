from copy import deepcopy

from hovsg.graph.graph import Graph
import hydra
import open3d as o3d
from omegaconf import DictConfig

# pylint: disable=all


@hydra.main(version_base=None, config_path="../config", config_name="visualize_query_graph_config")
def main(params: DictConfig):
    # Load graph
    hovsg = Graph(params)
    hovsg.load_graph(params.main.graph_path)
    # generate room names
    hovsg.generate_room_names(
            generate_method="view_embedding",
            default_room_types=[
                "office",
                "kitchen",
                "bathroom",
                "seminar room",
                "meeting room",
                "dinning room",
                "corridor",
            ])
    
    
    # loop forever and ask for query, until user click 'q'
    while True:
        query = input("Enter query: ")
        if query == "q":
            break
        floor, room, obj = hovsg.query_hierarchy(query, top_k=5)
        # visualize the query
        print(floor.floor_id, [(r.room_id, r.name) for r in room], [o.object_id for o in obj])
        # use open3d to visualize room.pcd and color the points where obj.pcd is
        for i in range(len(obj)):
            obj_pcd = obj[i].pcd.paint_uniform_color([0, 1, 0])
            room_pcd = room[i].pcd
            obj_pcd = deepcopy(obj[i].pcd)
            room_pcd = deepcopy(room[i].pcd)
            print(obj_pcd.get_center())
            o3d.visualization.draw_geometries([room_pcd, obj_pcd])


if __name__ == "__main__":
    main()
