import ast
from typing import Union, Optional, Dict, Tuple

import networkx as nx
from func_timeout import FunctionTimedOut, func_timeout
from networkx.algorithms.similarity import optimize_graph_edit_distance

from metrics_evaluation.metrics.ruby.util import get_ast_node_label, get_ast_children


def convert_ast_to_graph(
    root: Union[
        str,
        ast.AST,
    ]
) -> nx.DiGraph:
    g = nx.DiGraph()
    nodes = []
    edges = []
    num_nodes = 0

    def add_node() -> int:
        nonlocal num_nodes
        new_node = num_nodes
        num_nodes += 1
        return new_node

    def traverse(cur_node: Union[str, ast.AST], parent: Optional[int] = None):
        label = get_ast_node_label(cur_node)
        node_index = add_node()

        nodes.append((node_index, {"label": label}))
        if parent is not None:
            edges.append((parent, node_index, {"label": "AST"}))

        for child in get_ast_children(cur_node):
            traverse(child, node_index)

    traverse(root)
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def convert_dict_to_graph(dict_g: Dict) -> nx.DiGraph:
    g = nx.DiGraph()
    nodes = [(i, {"label": node}) for i, node in enumerate(dict_g["nodes"])]
    edges = [
        (int(v), int(u), {"label": edge_type})
        for edge_type in dict_g["edges"]
        for v, us in dict_g["edges"][edge_type].items()
        for u in us
    ]
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def compute_ged(
    sample_graph: nx.DiGraph, reference_graph: nx.DiGraph, use_edge_cost: bool = True
) -> Tuple[float, float]:

    ged_generator = optimize_graph_edit_distance(
        sample_graph,
        reference_graph,
        node_match=lambda v, u: v["label"] == u["label"],
        edge_match=lambda e1, e2: e1["label"] == e2["label"],
        edge_ins_cost=lambda e: 1 if use_edge_cost else 0,
        edge_del_cost=lambda e: 1 if use_edge_cost else 0,
    )
    total_size = sample_graph.number_of_nodes() + reference_graph.number_of_nodes()
    if use_edge_cost:
        total_size += sample_graph.number_of_edges() + reference_graph.number_of_edges()
    ged = total_size + 1

    if ged > total_size:
        while True:
            try:
                new_ged = func_timeout(1, next, args=(ged_generator,))
                ged = new_ged
            except (FunctionTimedOut, StopIteration):
                break

    if ged > total_size:
        return -1, -1

    return ged, total_size
