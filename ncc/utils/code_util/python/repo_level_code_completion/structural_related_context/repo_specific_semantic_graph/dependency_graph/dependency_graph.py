import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set, Tuple

import networkx as nx
from dependency_graph.models import PathLike, VirtualPath
from dependency_graph.models.graph_data import Edge, EdgeRelation, Node, NodeType
from dependency_graph.models.language import Language
from dependency_graph.utils.digraph import lexicographical_cyclic_topological_sort
from dependency_graph.utils.intervals import find_innermost_interval

if sys.version_info < (3, 9):

    def is_relative_to(self, *other):
        try:
            self.relative_to(*other)
            return True
        except ValueError:
            return False

    # Patch the method in OriginalPath
    Path.is_relative_to = is_relative_to


class DependencyGraph:
    def __init__(self, repo_path: PathLike, *languages: Language) -> None:
        # See https://networkx.org/documentation/stable/reference/classes/multidigraph.html
        # See also https://stackoverflow.com/questions/26691442/how-do-i-add-a-new-attribute-to-an-edge-in-networkx
        self.graph = nx.MultiDiGraph()
        self.repo_path = Path(repo_path) if isinstance(repo_path, str) else repo_path
        # De-duplicate the languages and convert to tuple
        self.languages = tuple(set([Language(lang) for lang in languages]))

        self._update_callbacks: Set[Callable] = set()
        # Clear the cache of self.get_edges when the graph is updated
        self.register_update_callback(self.get_edges.cache_clear)

    def as_retriever(self) -> "DependencyGraphContextRetriever":
        return DependencyGraphContextRetriever(graph=self)

    def register_update_callback(self, callback):
        if callback not in self._update_callbacks:
            self._update_callbacks.add(callback)

    def _notify_update(self):
        for callback in self._update_callbacks:
            callback()

    def add_node(self, node: Node):
        if self.graph.has_node(node):
            return
        self.graph.add_node(node)
        self._notify_update()

    def add_nodes_from(self, nodes: Iterable[Node]):
        self.graph.add_nodes_from(nodes)
        self._notify_update()

    def add_relational_edge(
        self, n1: Node, n2: Node, r1: Edge, r2: Optional[Edge] = None
    ):
        """Add a relational edge between two nodes.
        r2 can be None to indicate this is a unidirectional edge."""
        self.add_node(n1)
        self.add_node(n2)
        self.graph.add_edge(n1, n2, relation=r1)
        if r2 is not None:
            self.graph.add_edge(n2, n1, relation=r2)

    def add_relational_edges_from(
        self, edges: Iterable[Tuple[Node, Node, Edge, Optional[Edge]]]
    ):
        """Add relational edges.
        r2 can be None to indicate this is a unidirectional edge."""
        for e in edges:
            assert len(e) in (3, 4), f"Invalid edges length: {e}, should be 3 or 4"
            self.add_relational_edge(*e)

    def get_related_edges(
        self, *relations: EdgeRelation
    ) -> List[Tuple[Node, Node, Edge]]:
        filtered_edges = self.get_edges(
            edge_filter=lambda in_node, out_node, edge: edge.relation in relations
        )
        # Sort by edge's location
        return sorted(filtered_edges, key=lambda e: e[2].location.__str__())

    def get_related_edges_by_node(
        self, node: Node, *relations: EdgeRelation
    ) -> Optional[List[Tuple[Node, Node, Edge]]]:
        """Get the related edges of the given node by the given relations.
        If the given node is not in the graph, return None."""
        if node not in self.graph:
            return None

        return [
            edge
            for edge in self.graph.edges(node, data="relation")
            if edge[2].relation in relations
        ]

    def get_related_subgraph(self, *relations: EdgeRelation) -> "DependencyGraph":
        """Get a subgraph that contains all the nodes and edges that are related to the given relations.
        This subgraph is a new sub-copy of the original graph."""
        edges = self.get_related_edges(*relations)
        sub_graph = DependencyGraph(self.repo_path, *self.languages)
        sub_graph.add_relational_edges_from(edges)
        return sub_graph

    @lru_cache(maxsize=128)
    def get_edges(
        self,
        # the edge_filter parameter should also be hashable
        edge_filter: Callable[[Node, Node, Edge], bool] = None,
    ) -> List[Tuple[Node, Node, Edge]]:
        # self.graph.edges(data="relation") is something like:
        # [(1, 2, Edge(...), (1, 2, Edge(...)), (3, 4, Edge(...)]
        if edge_filter is None:
            return list(self.graph.edges(data="relation"))

        return [
            edge for edge in self.graph.edges(data="relation") if edge_filter(*edge)
        ]

    @lru_cache(maxsize=128)
    def get_nodes(
        self,
        # the node_filter parameter should also be hashable
        node_filter: Callable[[Node], bool] = None,
    ) -> List[Node]:
        if node_filter is None:
            return list(self.graph.nodes())

        return list(filter(node_filter, self.graph.nodes()))

    def get_edge(self, n1: Node, n2: Node) -> Optional[List[Edge]]:
        if self.graph.has_edge(n1, n2):
            return [data["relation"] for data in self.graph[n1][n2].values()]

    def get_topological_sorting(self, relation: EdgeRelation = None) -> Iterable[Node]:
        """
        Get the topological sorting of the graph.
        """
        if relation is None:
            G = self.graph
        else:
            G = self.get_related_subgraph(relation).graph
        yield from lexicographical_cyclic_topological_sort(G, key=lambda n: str(n))

    def compose_all(self, *graphs: "DependencyGraph"):
        """Merge the given graphs into this graph and return ."""
        all_graphs = [self.graph] + [graph.graph for graph in graphs]
        self.graph = nx.compose_all(all_graphs)

        language_set = set(self.languages)
        for graph in graphs:
            language_set.update(graph.languages)
        self.languages = tuple(language_set)

        self._notify_update()

    def to_dict(self) -> dict:
        edge_list = self.get_edges()
        return {
            "repo_path": str(self.repo_path),
            "languages": self.languages,
            "edges": [
                (edge[0].to_dict(), edge[1].to_dict(), edge[2].to_dict())
                for edge in edge_list
            ],
        }

    def to_json(self, indent=None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(obj_dict: dict) -> "DependencyGraph":
        edges = [
            (Node.from_dict(edge[0]), Node.from_dict(edge[1]), Edge.from_dict(edge[2]))
            for edge in obj_dict["edges"]
        ]
        graph = DependencyGraph(obj_dict["repo_path"], *obj_dict["languages"])
        graph.add_relational_edges_from(edges)
        return graph

    @staticmethod
    def from_json(json_str: str) -> "DependencyGraph":
        obj_dict = json.loads(json_str)
        return DependencyGraph.from_dict(obj_dict)


class DependencyGraphContextRetriever:
    """
    DependencyGraphRetriever provides a class to retrieve code snippets from a Repo-Specific Semantic Graph in context level.
    The difference between this and the DependencyGraph is that this retrieves the context of a code, it is not dealing
    with a graph problem, while the DependencyGraph is.
    """

    def __init__(self, graph: DependencyGraph):
        self.graph = graph

    def _Path(self, file_path: PathLike) -> Path:
        if isinstance(self.graph.repo_path, VirtualPath):
            return VirtualPath(self.graph.repo_path.fs, file_path)
        elif isinstance(self.graph.repo_path, Path):
            return Path(file_path)
        else:
            return Path(file_path)

    def _get_innermost_node_by_line(
        self,
        file_path: PathLike,
        start_line: int,
    ) -> Optional[Node]:
        # Statement nodes are not taken into account for now
        file_path = self._Path(file_path)
        nodes = self.graph.get_nodes(
            node_filter=lambda n: n.location
            and n.location.file_path == file_path
            and n.type != NodeType.STATEMENT
        )
        intervals = [
            (node.location.start_line, node.location.end_line, node)
            for node in nodes
            if node.location.start_line and node.location.end_line
        ]
        innermost_interval = find_innermost_interval(intervals, start_line)
        return innermost_interval[2] if innermost_interval else None

    def get_related_edges_by_innermost_node_between_line(
        self,
        file_path: PathLike,
        start_line: int,
        *relations: EdgeRelation,
    ) -> Optional[List[Tuple[Node, Node, Edge]]]:
        file_path = self._Path(file_path)
        node = self._get_innermost_node_by_line(file_path, start_line)
        related_edge_list = self.graph.get_related_edges_by_node(
            node,
            *relations,
        )
        return related_edge_list

    def is_node_from_cross_file(self, node: Node, file_path: PathLike) -> bool:
        file_path = self._Path(file_path)
        return (
            node.location is not None
            and node.location.file_path is not None
            and node.location.file_path != file_path
            and node.location.file_path.is_relative_to(self.graph.repo_path)
        )

    def is_node_from_in_file(self, node: Node, file_path: PathLike) -> bool:
        file_path = self._Path(file_path)
        return (
            node.location is not None
            and node.location.file_path is not None
            and node.location.file_path == file_path
        )

    def get_cross_file_context(
        self,
        file_path: PathLike,
        edge_filter: Callable[[Node, Node, Edge], bool] = None,
    ) -> List[Tuple[Node, Node, Edge]]:
        """
        Construct the cross-file context of a file
        - The in node should be located in the repo and be cross-file
        - The out node should be in the same file
        """
        file_path = self._Path(file_path)
        # Don't feel guilty, self.graph.get_edges is cached!
        edge_list = self.graph.get_edges(
            edge_filter=lambda in_node, out_node, edge: self.is_node_from_cross_file(
                in_node, file_path
            )
            and self.is_node_from_in_file(out_node, file_path)
        )

        # This custom edge_filter is applied after the edge_list is constructed by self.graph.get_edges
        # because we want more cache hits on the filter called above.
        if edge_filter is not None:
            return [edge for edge in edge_list if edge_filter(*edge)]

        # Sort by edge's location
        return sorted(edge_list, key=lambda e: e[2].location.__str__())

    def get_cross_file_definition_by_line(
        self,
        file_path: PathLike,
        start_line: int,
    ) -> Optional[List[Tuple[Node, Node, Edge]]]:
        """
        Construct the cross-file definition of a file by line.
        It will return the cross-file definition of the innermost scope(func/class) located between the start_line.
        Usually it is the out-node we are interested in.
        """
        file_path = self._Path(file_path)
        edge_list = []

        related_edge_list = self.get_related_edges_by_innermost_node_between_line(
            file_path,
            start_line,
            EdgeRelation.Construct,
            EdgeRelation.BaseClassOf,
            EdgeRelation.Overrides,
            EdgeRelation.Calls,
            EdgeRelation.Instantiates,
            EdgeRelation.Uses,
            # EdgeRelation.Defines,
        )
        if related_edge_list:
            edge_list += related_edge_list

        # Find the module node that is related to the file
        module_node = self.graph.get_nodes(
            node_filter=lambda n: n.location
            and n.location.file_path
            and n.location.file_path == file_path
            and n.type == NodeType.MODULE
        )
        # assert (
        #     len(module_node) <= 1
        # ), f"There should be at most 1 module node related to the file: {file_path}"

        # Find the importation edges
        if module_node:
            importation_edge_list = self.graph.get_related_edges_by_node(
                module_node[0],
                EdgeRelation.Imports,
            )
            edge_list += importation_edge_list

        cross_file_related_edge_list = []
        for edge in edge_list:
            if self.is_node_from_cross_file(edge[1], file_path):
                if (
                    edge[2].relation in (EdgeRelation.DefinedBy, EdgeRelation.DefinedBy)
                    and edge[1].location
                    and edge[1].location.start_line < start_line
                ):
                    cross_file_related_edge_list.append(edge)
                elif edge[2].location and edge[2].location.start_line < start_line:
                    # Filter the cross-file related edges, the edge location start_line should be less than the
                    # start_line
                    cross_file_related_edge_list.append(edge)

        # Sort by edge's location
        return sorted(
            cross_file_related_edge_list, key=lambda e: e[2].location.__str__()
        )

    def get_cross_file_reference_by_line(
        self,
        file_path: PathLike,
        start_line: int,
    ) -> Optional[List[Tuple[Node, Node, Edge]]]:
        """
        Construct the cross-file usage of a file by line
        It will return the cross-file usage of the innermost scope(func/class) located between the start_line.
        Usually it is the out-node we are interested in.
        """
        file_path = self._Path(file_path)
        related_edge_list = self.get_related_edges_by_innermost_node_between_line(
            file_path,
            start_line,
            EdgeRelation.ConstructedBy,
            EdgeRelation.DerivedClassOf,
            EdgeRelation.OverriddenBy,
            EdgeRelation.CalledBy,
            EdgeRelation.InstantiatedBy,
            EdgeRelation.UsedBy,
            EdgeRelation.ImportedBy,
            # EdgeRelation.DefinedBy,
        )
        if not related_edge_list:
            return None

        cross_file_related_edge_list = [
            edge
            for edge in related_edge_list
            if self.is_node_from_cross_file(edge[1], file_path)
        ]

        # Sort by edge's location
        return sorted(
            cross_file_related_edge_list, key=lambda e: e[2].location.__str__()
        )

    def get_related_edges_by_file(
        self, file_path: PathLike, *relations: EdgeRelation
    ) -> List[Tuple[Node, Node, Edge]]:
        """
        Get all related edges of a file and return them
        """
        file_path = self._Path(file_path)
        return self.graph.get_edges(
            edge_filter=lambda in_node, out_node, edge: self.is_node_from_in_file(
                in_node, file_path
            )
            and edge.relation in relations
        )
