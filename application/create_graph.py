import os
import hydra
from omegaconf import DictConfig
from hovsg.graph.graph import Graph

# pylint: disable=all


@hydra.main(version_base=None, config_path="../config", config_name="create_graph_config")
def main(params: DictConfig):
    # create logging directory
    save_dir = os.path.join(params.main.save_path, params.main.dataset, params.main.scene_id)
    params.main.save_path = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # create graph
    hovsg = Graph(params)
    hovsg.create_feature_map() # create feature map

    # save full point cloud, features, and masked point clouds (pcd for all objects)
    hovsg.save_masked_pcds(path=save_dir, state="both")
    hovsg.save_full_pcd(path=save_dir)
    hovsg.save_full_pcd_feats(path=save_dir)

    # for debugging: load preconstructed map as follows
    # hovsg.load_full_pcd(path=save_dir)
    # hovsg.load_full_pcd_feats(path=save_dir)
    # hovsg.load_masked_pcds(path=save_dir)

    # create graph, only if dataset is not Replia or ScanNet
    print(params.main.dataset)
    hovsg.build_graph(save_path=save_dir)


if __name__ == "__main__":
    main()
