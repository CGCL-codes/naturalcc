import heapq
from typing import Iterable

import networkx as nx

from dependency_graph.models.graph_data import Node


def lexicographical_cyclic_topological_sort(G: nx.DiGraph, key=None) -> Iterable[Node]:
    """
    Generate the nodes in the unique lexicographical topological sort order for both DAGs and cyclic directed graphs.

    This function extends the capabilities of `networkx.algorithms.dag.lexicographical_topological_sort`
    to support not only directed acyclic graphs (DAGs) but also cyclic directed graphs.

    This function generates a unique ordering of nodes by first sorting topologically and then
    additionally by sorting lexicographically. Unlike standard topological sorting algorithms
    which are only defined for Directed Acyclic Graphs (DAGs), this modified algorithm also
    supports graphs that contain cycles. The result is a unique ordering of nodes in the graph.

    Lexicographical sorting is used to break ties in the topological sort and to determine a single,
    unique ordering. This can be particularly useful in comparing sort results.

    The lexicographical order can be customized by providing a function to the `key` parameter. The
    definition of the key function is the same as used in Python's built-in `sort()`. The function takes
    a single argument and returns a key to use for sorting purposes.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed graph, which may be acyclic (DAG) or cyclic.

    key : function, optional
        A function of one argument that converts a node name to a comparison key.
        It defines and resolves ambiguities in the sort order. Defaults to the identity function.

    Yields
    ------
    nodes
        Yields the nodes of G in unique topological sort order, taking cycles into account.

    Raises
    ------
    NetworkXError
        Raised if the graph `G` is undirected.

    RuntimeError
        If `G` is changed while the returned iterator is being processed.

    TypeError
        Raised if node names are un-sortable.
        Consider using the `key` parameter to resolve ambiguities.

    Examples
    --------
    >>> DG = nx.DiGraph([(2, 1), (2, 5), (1, 3), (1, 4), (5, 4)])
    >>> list(lexicographical_cyclic_topological_sort(DG))
    [2, 1, 3, 5, 4]
    >>> list(lexicographical_cyclic_topological_sort(DG, key=lambda x: -x))
    [2, 5, 1, 4, 3]

    >>> DG = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]) # cyclic directed graph
    >>> list(lexicographical_cyclic_topological_sort(DG))
    ["A", "B", "C", "D"]

    The sort will fail for any graph with integer and string nodes as comparison of integer to strings
    is not defined in Python. Is 3 greater or less than 'red'?

    >>> DG = nx.DiGraph([(1, "red"), (3, "red"), (1, "green"), (2, "blue")])
    >>> list(lexicographical_cyclic_topological_sort(DG))
    Traceback (most recent call last):
    ...
    TypeError: '<' not supported between instances of 'str' and 'int'
    ...

    Incomparable nodes can be resolved using a `key` function. This example function
    allows comparison of integers and strings by returning a tuple where the first
    element is True for `str`, False otherwise. The second element is the node name.
    This groups the strings and integers separately so they can be compared only among themselves.

    >>> key = lambda node: (isinstance(node, str), node)
    >>> list(lexicographical_cyclic_topological_sort(DG, key=key))
    [1, 2, 3, 'blue', 'green', 'red']

    Notes
    -----
    This algorithm is adapted from `networkx.algorithms.dag.lexicographical_topological_sort` and extended
    to support cyclic directed graphs by handling strongly connected components.
    """
    if not G.is_directed():
        msg = "Topological sort not defined on undirected graphs."
        raise nx.NetworkXError(msg)

    if key is None:

        def key(node):
            return node

    nodeid_map = {n: i for i, n in enumerate(G)}

    def create_tuple(node):
        return key(node), nodeid_map[node], node

    indegree_map = {v: d for v, d in G.in_degree()}
    G_copy: nx.MultiDiGraph = G.copy()

    while G_copy.nodes:
        # These nodes have min indegree and ready to be returned.
        min_indegree = min(d for _, d in G_copy.in_degree())
        min_indegree_heap = [
            create_tuple(v) for v, d in G_copy.in_degree() if d == min_indegree
        ]

        heapq.heapify(min_indegree_heap)
        # If min indegree is greater than 0, it means the graph contains cycles
        # We should just try to take the first node on the heap to break the cycle
        if min_indegree > 0:
            _, _, node = heapq.heappop(min_indegree_heap)
            min_indegree_heap = [create_tuple(node)]

        while min_indegree_heap:
            _, _, node = heapq.heappop(min_indegree_heap)

            if node not in G_copy:
                # If node is not in the graph, it means the node pointed to itself.
                continue
            for _, child in G_copy.edges(node):
                try:
                    indegree_map[child] -= 1
                except KeyError as err:
                    raise RuntimeError("Graph changed during iteration") from err
                if indegree_map[child] == 0:
                    try:
                        child_tuple = create_tuple(child)
                        if child_tuple not in min_indegree_heap:
                            heapq.heappush(min_indegree_heap, child_tuple)
                    except TypeError as err:
                        raise TypeError(
                            f"{err}\nConsider using `key=` parameter to resolve ambiguities in the sort order."
                        )

            G_copy.remove_node(node)
            yield node
