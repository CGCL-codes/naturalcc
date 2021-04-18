import os
import numpy as np
from typing import Dict, Generic, List, NamedTuple, Tuple, TypeVar
TTensorizedNodeData = TypeVar("TTensorizedNodeData")


def enforce_not_None(e):
    """Enforce non-nullness of input. Used for typechecking and runtime safety."""
    if e is None:
        raise Exception("Input is None.")
    return e


class TensorizedGraphData(Generic[TTensorizedNodeData]):
    __slots__ = ("num_nodes", "node_tensorized_data", "adjacency_lists", "reference_nodes")

    def __init__(
        self,
        num_nodes: int,
        node_tensorized_data: List[TTensorizedNodeData],
        adjacency_lists: List[Tuple[np.ndarray, np.ndarray]],
        reference_nodes: Dict[str, np.ndarray],
    ):
        self.num_nodes = num_nodes
        self.node_tensorized_data = node_tensorized_data
        self.adjacency_lists = adjacency_lists
        self.reference_nodes = reference_nodes