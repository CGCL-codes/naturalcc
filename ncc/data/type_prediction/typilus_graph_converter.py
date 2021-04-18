import os
import numpy as np
from typing import Dict, Generic, List, NamedTuple, Tuple, TypeVar
from ncc.data.type_prediction.utils import enforce_not_None

TNodeData = TypeVar("TNodeData")

IGNORED_TYPES = {
    "typing.Any",
    "Any",
    "",
    "typing.NoReturn",
    "NoReturn",
    "nothing",
    "None",
    "T",
    "_T",
    "_T0",
    "_T1",
    "_T2",
    "_T3",
    "_T4",
    "_T5",
    "_T6",
    "_T7",
}


class GraphData(Generic[TNodeData]):
    __slots__ = ("node_information", "edges", "reference_nodes")

    def __init__(
        self,
        node_information: List[TNodeData],
        edges: Dict[str, List[Tuple[int, int]]],
        reference_nodes: Dict[str, List[int]],
    ):
        self.node_information = node_information
        self.edges = edges
        self.reference_nodes = reference_nodes


def convert(typilus_graph):
    __tensorize_samples_with_no_annotation = False # TODO

    def get_adj_list(adjacency_dict):
        for from_node_idx, to_node_idxs in adjacency_dict.items():
            from_node_idx = int(from_node_idx)
            for to_idx in to_node_idxs:
                yield (from_node_idx, to_idx)

    edges = {}
    for edge_type, adj_dict in typilus_graph["edges"].items():
        adj_list: List[Tuple[int, int]] = list(get_adj_list(adj_dict))
        if len(adj_list) > 0:
            edges[edge_type] = np.array(adj_list, dtype=np.int32)
        else:
            edges[edge_type] = np.zeros((0, 2), dtype=np.int32)

    supernode_idxs_with_ground_truth: List[int] = []
    supernode_annotations: List[str] = []
    for supernode_idx, supernode_data in typilus_graph["supernodes"].items():
        if supernode_data["annotation"] in IGNORED_TYPES:
            continue
        if (
                not __tensorize_samples_with_no_annotation
                and supernode_data["annotation"] is None
        ):
            continue
        elif supernode_data["annotation"] is None:
            supernode_data["annotation"] = "??"
        supernode_idxs_with_ground_truth.append(int(supernode_idx))
        supernode_annotations.append(enforce_not_None(supernode_data["annotation"]))

    return (
        GraphData[str](
            node_information=typilus_graph["nodes"],
            edges=edges,
            reference_nodes={
                "token-sequence": typilus_graph["token-sequence"],
                "supernodes": supernode_idxs_with_ground_truth,
            },
        ),
        supernode_annotations,
    )
