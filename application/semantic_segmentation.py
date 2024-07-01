import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hovsg.graph.graph import Graph

# pylint: disable=all


@hydra.main(version_base=None, config_path="../config", config_name="semantic_segmentation_config")
def main(params: DictConfig):
    # Create save directory
    save_dir = os.path.join(params.main.save_path, params.main.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Create graph
    hovsg = Graph(params)
    # Create feature map
    hovsg.create_feature_map()
    # Save full point cloud, features, and masked point clouds (pcd for all objects)
    hovsg.save_masked_pcds(path=save_dir, state="both")
    hovsg.save_full_pcd(path=save_dir)
    hovsg.save_full_pcd_feats(path=save_dir)

if __name__ == "__main__":
    main()