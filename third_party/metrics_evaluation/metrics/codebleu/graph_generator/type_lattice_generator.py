from typing import Any, List, Dict, FrozenSet, Set, Tuple, Optional
from collections import defaultdict, ChainMap
from itertools import chain
from functools import lru_cache
import json

from .typeparsing import (
    TypeAnnotationNode,
    NameAnnotationNode,
    parse_type_annotation_node,
)
from .typeparsing import RewriteRuleVisitor
from .typeparsing import DirectInheritanceRewriting
from .typeparsing import EraseOnceTypeRemoval
from .typeparsing import PruneAnnotationVisitor
from .typeparsing import AliasReplacementVisitor
from .typeparsing.rewriterules import RemoveStandAlones
from .typeparsing.rewriterules import RemoveRecursiveGenerics
from .typeparsing.rewriterules import RemoveUnionWithAnys
from .typeparsing.rewriterules import RemoveGenericWithAnys


class TypeLatticeGenerator:

    ANY_TYPE = parse_type_annotation_node("typing.Any")

    def __init__(
        self, typing_rules_path: str, max_depth_size: int = 2, max_list_size: int = 2
    ):
        self.__to_process = []
        self.__processed = set()
        self.new_type_rules = defaultdict(set)  # [new type, ref]
        self.module_naming_rules = {}  # [short, long.version]
        self.__all_types = {self.ANY_TYPE: 0}
        self.__ids_to_nodes = [self.ANY_TYPE]
        self.__type_reprs = {repr(self.ANY_TYPE): 0}

        # Rewrites
        with open(typing_rules_path, "r") as f:
            rules = json.load(f)
        # alias -> default name
        self.__aliases = {
            parse_type_annotation_node(k): parse_type_annotation_node(v)
            for k, v in rules["aliasing_rules"]
        }
        for type_annotation in chain(self.__aliases.keys(), self.__aliases.values()):
            self.__annotation_to_id(type_annotation)
        self.__project_specific_aliases = {}

        self.__max_annotation_depth = max_depth_size
        self.__max_depth_pruning_visitor = PruneAnnotationVisitor(
            self.ANY_TYPE, max_list_size
        )

        # [specialized type, general type]
        self.is_a_edges = defaultdict(set)  # type: Dict[int, Set[int]]
        self.project_is_a = defaultdict(set)  # type: Dict[int, Set[int]]

        # type -> supertypes
        for k, v in rules["is_a_relations"]:
            self.__add_is_a_relationship(
                parse_type_annotation_node(k), parse_type_annotation_node(v)
            )

        for known_type in self.__all_types:
            known_type_id = self.__annotation_to_id(known_type)
            if known_type != self.ANY_TYPE and len(self.is_a_edges[known_type_id]) == 0:
                self.is_a_edges[known_type_id].add(0)

        self.__compute_non_generic_types()

        self.__type_erasure = EraseOnceTypeRemoval()
        self.__direct_inheritance_rewriting = DirectInheritanceRewriting(
            self.__is_a_relationships, self.__non_generic_types
        )

        self.__rewrites_verbose_annotations = RewriteRuleVisitor(
            [
                RemoveUnionWithAnys(),
                RemoveStandAlones(),
                RemoveRecursiveGenerics(),
                RemoveGenericWithAnys(),
            ]
        )
        assert len(self.__ids_to_nodes) == len(
            set(repr(r) for r in self.__ids_to_nodes)
        )

    def create_alias_replacement(
        self, imported_symbols: Dict[TypeAnnotationNode, TypeAnnotationNode]
    ) -> AliasReplacementVisitor:
        return AliasReplacementVisitor(
            ChainMap(imported_symbols, self.__project_specific_aliases, self.__aliases)
        )

    def __compute_non_generic_types(self):
        # Now get all the annotations that are *not* generics
        child_edges = defaultdict(set)
        for parent, children in self.is_a_edges.items():
            for child in children:
                child_edges[child].add(parent)

        generic_transitive_closure = set()
        to_visit = [
            self.__all_types[parse_type_annotation_node("typing.Generic")],
            self.__all_types[parse_type_annotation_node("typing.Tuple")],
            self.__all_types[parse_type_annotation_node("typing.Callable")],
        ]
        while len(to_visit) > 0:
            next_node = to_visit.pop()
            generic_transitive_closure.add(next_node)
            to_visit.extend(
                (
                    t
                    for t in child_edges[next_node]
                    if t not in generic_transitive_closure
                )
            )

        non_generic_types = set(self.__all_types.values()) - generic_transitive_closure

        # Add special objects
        non_generic_types.add(
            self.__annotation_to_id(parse_type_annotation_node("abc.ABC"))
        )

        self.__non_generic_types = frozenset(
            (self.__ids_to_nodes[i] for i in non_generic_types)
        )

    def __annotation_to_id(self, annotation: TypeAnnotationNode) -> int:
        annotation_idx = self.__all_types.get(annotation)
        if annotation_idx is None:
            if repr(annotation) in self.__type_reprs:
                return self.__type_reprs[repr(annotation)]
            annotation_idx = len(self.__all_types)
            self.__all_types[annotation] = annotation_idx
            self.__type_reprs[repr(annotation)] = annotation_idx
            self.__ids_to_nodes.append(annotation)

        return annotation_idx

    def __all_reachable_from(self, type_idx: int) -> FrozenSet[int]:
        reachable = set()
        to_visit = [type_idx]  # type: List[int]
        while len(to_visit) > 0:
            next_type_idx = to_visit.pop()
            reachable.add(next_type_idx)
            to_visit.extend(
                (
                    parent_type_idx
                    for parent_type_idx in self.is_a_edges[next_type_idx]
                    if parent_type_idx not in reachable
                )
            )
        return frozenset(reachable)

    def __add_is_a_relationship(
        self, from_type: TypeAnnotationNode, to_type: TypeAnnotationNode
    ) -> None:
        from_node_idx = self.__annotation_to_id(from_type)
        to_node_idx = self.__annotation_to_id(to_type)

        if from_node_idx == to_node_idx:
            return

        reachable_from_idx = self.__all_reachable_from(from_node_idx)
        if to_node_idx in reachable_from_idx:
            # This is already reachable, ignore direct is-a relationship.
            return

        all_reachable = self.__all_reachable_from(to_node_idx)
        if from_node_idx in all_reachable:
            print(f"The {from_node_idx}<->{to_node_idx} would be a circle. Ignoring.")
            return

        self.is_a_edges[from_node_idx].add(to_node_idx)

    def __is_a_relationships(
        self, from_type: TypeAnnotationNode
    ) -> FrozenSet[TypeAnnotationNode]:
        from_node_idx = self.__annotation_to_id(from_type)
        return frozenset(
            (self.__ids_to_nodes[t] for t in self.is_a_edges[from_node_idx])
        )

    def add_type(
        self,
        annotation: TypeAnnotationNode,
        imported_symbols: Dict[TypeAnnotationNode, TypeAnnotationNode],
    ):
        annotation = annotation.accept_visitor(
            self.create_alias_replacement(imported_symbols)
        )
        if annotation in self.__all_types:
            return
        pruned = annotation.accept_visitor(
            self.__max_depth_pruning_visitor, self.__max_annotation_depth
        )
        if pruned != annotation:
            self.__add_is_a_relationship(annotation, pruned)
            self.__to_process.append(pruned)
        else:
            self.__to_process.append(annotation)

    @lru_cache(16384)
    def __compute_erasures(
        self, type_annotation: TypeAnnotationNode
    ) -> Tuple[List[TypeAnnotationNode], bool]:
        return type_annotation.accept_visitor(self.__type_erasure)

    @lru_cache(16384)
    def __rewrite_verbose(
        self, type_annotation: TypeAnnotationNode
    ) -> TypeAnnotationNode:
        return type_annotation.accept_visitor(self.__rewrites_verbose_annotations, None)

    def build_graph(self):
        print(
            "Building type graph for project... (%s elements to process)"
            % len(self.__to_process)
        )
        i = 0

        while len(self.__to_process) > 0:
            next_type = self.__to_process.pop()

            if next_type in self.__processed:
                continue

            i += 1
            if i > 500:
                print(
                    "Building type graph for project... (%s elements to process)"
                    % len(self.__to_process)
                )
                i = 0
                if len(self.__to_process) > 3000:
                    print(f"Queue quite long. Current element {next_type}")

            all_erasures, erasure_happened = self.__compute_erasures(next_type)
            if erasure_happened:
                for erased_type in all_erasures:
                    erased_type = self.__rewrite_verbose(erased_type)
                    erased_type_has_been_proceesed = erased_type in self.__all_types

                    self.__add_is_a_relationship(next_type, erased_type)
                    if not erased_type_has_been_proceesed:
                        self.__to_process.append(erased_type)

            all_inherited_types_and_self = next_type.accept_visitor(
                self.__direct_inheritance_rewriting
            )
            was_rewritten = len(all_inherited_types_and_self) > 1

            if was_rewritten:
                if (
                    not erasure_happened
                    or len(self.__to_process) < 5000
                    or len(all_inherited_types_and_self) < 5
                ):
                    for type_annotation in all_inherited_types_and_self:
                        type_annotation = type_annotation.accept_visitor(
                            self.__max_depth_pruning_visitor,
                            self.__max_annotation_depth,
                        )
                        type_annotation = self.__rewrite_verbose(type_annotation)
                        type_has_been_seen = type_annotation in self.__all_types

                        if not type_has_been_seen:
                            self.__add_is_a_relationship(next_type, type_annotation)
                            self.__to_process.append(type_annotation)

            if not was_rewritten and not erasure_happened:
                # Add a rule to Any
                self.__add_is_a_relationship(next_type, self.ANY_TYPE)
            self.__processed.add(next_type)

        # Clean up project-specific aliases
        self.__project_specific_aliases.clear()
        print("Done building type graph")

    def add_class(self, class_name: str, parents: List[TypeAnnotationNode]) -> None:
        class_name = NameAnnotationNode(class_name)
        for parent in parents:
            if parent is None:
                continue
            assert isinstance(parent, TypeAnnotationNode), (parent, type(parent))
            self.__add_is_a_relationship(class_name, parent)

    def add_type_alias(
        self, new_annotation: TypeAnnotationNode, ref_annotation: TypeAnnotationNode
    ) -> None:
        self.__project_specific_aliases[new_annotation] = ref_annotation

    def canonicalize_annotation(
        self,
        annotation: TypeAnnotationNode,
        local_aliases: Dict[TypeAnnotationNode, TypeAnnotationNode],
    ) -> Optional[TypeAnnotationNode]:
        if annotation is None:
            return None
        return annotation.accept_visitor(self.create_alias_replacement(local_aliases))

    def return_json(self) -> Dict[str, Any]:
        edges = []
        for from_type_idx, to_type_idxs in self.is_a_edges.items():
            for to_type_idx in to_type_idxs:
                edges.append((from_type_idx, to_type_idx))

        assert len(self.__ids_to_nodes) == len(
            set(repr(r) for r in self.__ids_to_nodes)
        )
        return {
            "nodes": list(
                (repr(type_annotation) for type_annotation in self.__ids_to_nodes)
            ),
            "edges": edges,
        }
