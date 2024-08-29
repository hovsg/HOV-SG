import os
import sys
import hydra
import json

import open_clip
import matplotlib.pyplot as plt
import networkx as nx

from omegaconf import DictConfig, OmegaConf
from networkx.drawing.nx_agraph import graphviz_layout

from hovsg.graph.graph import Graph
from hovsg.graph.object import Object
from hovsg.eval.hm3dsem_evaluator import PanopticBuildingEval, PanopticLevelEval, PanopticRegionEval, PanopticObjectEval, HM3DSemanticEvaluator
from hovsg.utils.label_feats import get_label_feats


def visualize_graph(graph, save_path=None):
    """
    Visualize the graph of the openmap
    """
    # print number of nodes and edges in the graph
    print("number of nodes in the graph: ", len(graph.nodes))
    print("number of edges in the graph: ", len(graph.edges))
    # draw the graph
    plt.clf()
    plt.figure(figsize=(20, 20))
    pos = graphviz_layout(graph, prog="dot")
    nx.draw(graph, pos, with_labels=True, font_weight="bold", font_size=6)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


@hydra.main(version_base=None, config_path="../../config", config_name="eval_graph")
def main(params: DictConfig):

    # load CLIP model
    if params.models.clip.type == "ViT-L/14@336px":
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained=str(params.models.clip.checkpoint),
            device=params.main.device,
        )
        clip_feat_dim = 768
    elif params.models.clip.type == "ViT-H-14":
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained=str(params.models.clip.checkpoint),
            device=params.main.device,
        )
        clip_feat_dim = 1024
    clip_model.eval()


    # overwrite it with the path to the actual scene
    dataset_path = os.path.join(params.main.dataset_path, params.main.split, params.main.scene_id)
    params.main.dataset_path = dataset_path

    # load gt graph
    evaluator = HM3DSemanticEvaluator(params)
    evaluator.load_gt_graph_from_json(os.path.join(dataset_path, "scene_info.json"))

    save_dir = os.path.join(params.main.save_path, params.main.dataset, params.main.split, params.main.scene_id)

    hovsg = Graph(params)

    hovsg.load_graph(path=os.path.join(save_dir, "graph/")) # load the predicted graph
    hovsg.graph.remove_node(0)

    # relabel objects based on the hm3dsem labels
    text_feats, classes = get_label_feats(clip_model, 
                                          clip_feat_dim, 
                                          params.eval.obj_labels, 
                                          params.main.save_path)
    
    for node in hovsg.graph.nodes:
        if type(node) == Object:
            name = hovsg.identify_object(node.embedding, text_feats, classes)
            node.name = name
    evaluator.evaluate_floors(hovsg.graph)
    evaluator.evaluate_rooms(hovsg.graph)
    evaluator.evaluate_objects(
        eval_metric=params.eval.association_metric,
        pred_graph=hovsg.graph,
        gt_classes=classes,
        gt_text_feats=text_feats,
    )

    print(evaluator.metrics)


if __name__ == "__main__":
    main()