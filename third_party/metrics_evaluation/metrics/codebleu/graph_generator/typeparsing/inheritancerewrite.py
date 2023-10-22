from itertools import product
from typing import Callable, Iterator, Set
import random

from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.nodes import (
    TypeAnnotationNode,
    SubscriptAnnotationNode,
    TupleAnnotationNode,
    ListAnnotationNode,
    AttributeAnnotationNode,
    IndexAnnotationNode,
    ElipsisAnnotationNode,
)
from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.visitor import TypeAnnotationVisitor

__all__ = ["DirectInheritanceRewriting"]


class DirectInheritanceRewriting(TypeAnnotationVisitor):
    """Replace Nodes their direct is-a relationships"""

    def __init__(
        self,
        is_a_info: Callable[[TypeAnnotationNode], Iterator[TypeAnnotationNode]],
        non_generic_types: Set[TypeAnnotationNode],
        limit_combinations_to: int = 10000,
    ):
        self.__is_a = is_a_info
        self.__non_generic_types = non_generic_types
        self.__limit_combinations_to = limit_combinations_to

    def visit_subscript_annotation(self, node: SubscriptAnnotationNode):
        value_node_options = node.value.accept_visitor(self)
        if node.slice is None:
            slice_node_options = [None]
        else:
            slice_node_options = node.slice.accept_visitor(self)

        all_children = []
        for v in value_node_options:
            if v in self.__non_generic_types:
                all_children.append(v)
                continue
            for s in slice_node_options:
                all_children.append(SubscriptAnnotationNode(v, s))

        return all_children

    def visit_tuple_annotation(self, node: TupleAnnotationNode):
        all_elements_options = [e.accept_visitor(self) for e in node.elements]
        r = [TupleAnnotationNode(t) for t in product(*all_elements_options)]

        if len(r) > self.__limit_combinations_to:
            random.shuffle(r)
            return r[: self.__limit_combinations_to]
        return r

    def visit_name_annotation(self, node):
        return [node] + list(self.__is_a(node))

    def visit_list_annotation(self, node: ListAnnotationNode):
        all_elements_options = [e.accept_visitor(self) for e in node.elements]
        r = [ListAnnotationNode(t) for t in product(*all_elements_options)]
        if len(r) > self.__limit_combinations_to:
            random.shuffle(r)
            return r[: self.__limit_combinations_to]
        return r

    def visit_attribute_annotation(self, node: AttributeAnnotationNode):
        v = [node] + list(self.__is_a(node))
        return v

    def visit_index_annotation(self, node: IndexAnnotationNode):
        next_values = node.value.accept_visitor(self)
        return [IndexAnnotationNode(v) for v in next_values]

    def visit_elipsis_annotation(self, node: ElipsisAnnotationNode):
        return [node]

    def visit_name_constant_annotation(self, node):
        return [node]

    def visit_unknown_annotation(self, node):
        return [node]
