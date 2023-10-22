from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.nodes import (
    TypeAnnotationNode,
    SubscriptAnnotationNode,
    TupleAnnotationNode,
    ElipsisAnnotationNode,
    ListAnnotationNode,
    AttributeAnnotationNode,
    IndexAnnotationNode,
)
from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.visitor import TypeAnnotationVisitor

__all__ = ["PruneAnnotationVisitor"]


class PruneAnnotationVisitor(TypeAnnotationVisitor):
    """Prune Long Annotations"""

    def __init__(self, replacement_node: TypeAnnotationNode, max_list_size: int):
        self.__max_list_size = max_list_size
        self.__replacement_node = replacement_node

    def visit_subscript_annotation(
        self, node: SubscriptAnnotationNode, current_remaining_depth: int
    ):
        if current_remaining_depth == 0:
            return self.__replacement_node

        if node.slice is None:
            pruned_slice = None
        elif current_remaining_depth <= 2:  # Remove the subscript completely.
            return node.value.accept_visitor(self, current_remaining_depth - 1)
        else:
            pruned_slice = node.slice.accept_visitor(self, current_remaining_depth - 1)

        return SubscriptAnnotationNode(
            value=node.value.accept_visitor(self, current_remaining_depth - 1),
            slice=pruned_slice,
        )

    def visit_tuple_annotation(
        self, node: TupleAnnotationNode, current_remaining_depth: int
    ):
        if len(node.elements) > self.__max_list_size:
            elements = node.elements[: self.__max_list_size - 1] + (
                ElipsisAnnotationNode(),
            )
        else:
            elements = node.elements

        if len(node.elements) == 0:
            pruned = node.elements
        elif current_remaining_depth <= 1 and len(node.elements) == 1:
            pruned = [self.__replacement_node]
        elif current_remaining_depth <= 1:
            pruned = [ElipsisAnnotationNode()]
        else:
            pruned = (
                e.accept_visitor(self, current_remaining_depth - 1) for e in elements
            )

        return TupleAnnotationNode(pruned)

    def visit_name_annotation(self, node, current_remaining_depth: int):
        return node

    def visit_list_annotation(
        self, node: ListAnnotationNode, current_remaining_depth: int
    ):
        if len(node.elements) > self.__max_list_size:
            pruned = node.elements[: self.__max_list_size - 1] + (
                ElipsisAnnotationNode(),
            )

        if len(node.elements) == 0:
            pruned = node.elements
        elif current_remaining_depth <= 1 and len(node.elements) == 1:
            pruned = [self.__replacement_node]
        elif current_remaining_depth <= 1:
            pruned = [ElipsisAnnotationNode()]
        else:
            pruned = (
                e.accept_visitor(self, current_remaining_depth - 1)
                for e in node.elements
            )

        return ListAnnotationNode(pruned)

    def visit_attribute_annotation(
        self, node: AttributeAnnotationNode, current_remaining_depth: int
    ):
        return node

    def visit_index_annotation(
        self, node: IndexAnnotationNode, current_remaining_depth: int
    ):
        if current_remaining_depth == 0:
            return self.__replacement_node

        return IndexAnnotationNode(
            node.value.accept_visitor(self, current_remaining_depth)
        )

    def visit_elipsis_annotation(self, node, current_remaining_depth: int):
        return node

    def visit_name_constant_annotation(self, node, current_remaining_depth: int):
        return node

    def visit_unknown_annotation(self, node, current_remaining_depth: int):
        return node
