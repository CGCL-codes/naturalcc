from typing import Dict, Tuple

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

__all__ = ["AliasReplacementVisitor"]


class AliasReplacementVisitor(TypeAnnotationVisitor):
    """Replace Nodes with Aliases. Assumes recursion has been resolved in replacement_map"""

    def __init__(self, replacement_map: Dict[TypeAnnotationNode, TypeAnnotationNode]):
        self.__replacement_map = replacement_map

    def __replace_full(
        self, node: TypeAnnotationNode
    ) -> Tuple[TypeAnnotationNode, bool]:
        replaced = False
        seen_names = {node}
        while node in self.__replacement_map:
            node = self.__replacement_map[node]
            replaced = True
            if node in seen_names:
                print(
                    f"WARNING: Circle between {seen_names}. Picking the {node} for now."
                )
                break
            else:
                seen_names.add(node)
        return node, replaced

    def visit_subscript_annotation(self, node: SubscriptAnnotationNode):
        replacement, replaced = self.__replace_full(node)
        if replaced:
            return replacement
        return SubscriptAnnotationNode(
            value=node.value.accept_visitor(self),
            slice=node.slice.accept_visitor(self) if node.slice is not None else None,
        )

    def visit_tuple_annotation(self, node: TupleAnnotationNode):
        replacement, replaced = self.__replace_full(node)
        if replaced:
            return replacement
        return TupleAnnotationNode((e.accept_visitor(self) for e in node.elements))

    def visit_name_annotation(self, node):
        return self.__replace_full(node)[0]

    def visit_list_annotation(self, node: ListAnnotationNode):
        replacement, replaced = self.__replace_full(node)
        if replaced:
            return replacement
        return ListAnnotationNode((e.accept_visitor(self) for e in node.elements))

    def visit_attribute_annotation(self, node: AttributeAnnotationNode):
        replacement, replaced = self.__replace_full(node)
        if replaced:
            return replacement
        return AttributeAnnotationNode(node.value.accept_visitor(self), node.attribute)

    def visit_index_annotation(self, node: IndexAnnotationNode):
        replacement, replaced = self.__replace_full(node)
        if replaced:
            return replacement
        return IndexAnnotationNode(node.value.accept_visitor(self))

    def visit_elipsis_annotation(self, node: ElipsisAnnotationNode):
        return node

    def visit_name_constant_annotation(self, node):
        return self.__replace_full(node)[0]

    def visit_unknown_annotation(self, node):
        return node
