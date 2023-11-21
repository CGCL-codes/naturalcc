from typing import List

from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.nodes import (
    TypeAnnotationNode,
    SubscriptAnnotationNode,
    TupleAnnotationNode,
    ListAnnotationNode,
    AttributeAnnotationNode,
    IndexAnnotationNode,
    ElipsisAnnotationNode,
)
from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.rewriterules import RewriteRule
from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.visitor import TypeAnnotationVisitor


class RewriteRuleVisitor(TypeAnnotationVisitor):
    """Replace Nodes based on a list of rules."""

    def __init__(self, rules: List[RewriteRule]):
        self.__rules = rules

    def __apply_on_match(
        self, original_node: TypeAnnotationNode, parent: TypeAnnotationNode
    ) -> TypeAnnotationNode:
        for rule in self.__rules:
            if rule.matches(original_node, parent):
                return rule.apply(original_node)
        return original_node

    def visit_subscript_annotation(
        self, node: SubscriptAnnotationNode, parent: TypeAnnotationNode
    ) -> SubscriptAnnotationNode:
        node = SubscriptAnnotationNode(
            value=node.value.accept_visitor(self, node),
            slice=node.slice.accept_visitor(self, node)
            if node.slice is not None
            else None,
        )
        return self.__apply_on_match(node, parent)

    def visit_tuple_annotation(
        self, node: TupleAnnotationNode, parent: TypeAnnotationNode
    ) -> TupleAnnotationNode:
        node = TupleAnnotationNode(
            (e.accept_visitor(self, node) for e in node.elements)
        )
        return self.__apply_on_match(node, parent)

    def visit_name_annotation(self, node, parent: TypeAnnotationNode):
        return self.__apply_on_match(node, parent)

    def visit_list_annotation(
        self, node: ListAnnotationNode, parent: TypeAnnotationNode
    ):
        node = ListAnnotationNode((e.accept_visitor(self, node) for e in node.elements))
        return self.__apply_on_match(node, parent)

    def visit_attribute_annotation(
        self, node: AttributeAnnotationNode, parent: TypeAnnotationNode
    ):
        node = AttributeAnnotationNode(
            node.value.accept_visitor(self, node), node.attribute
        )
        return self.__apply_on_match(node, parent)

    def visit_index_annotation(
        self, node: IndexAnnotationNode, parent: TypeAnnotationNode
    ):
        node = IndexAnnotationNode(node.value.accept_visitor(self, node))
        return self.__apply_on_match(node, parent)

    def visit_elipsis_annotation(
        self, node: ElipsisAnnotationNode, parent: TypeAnnotationNode
    ):
        return self.__apply_on_match(node, parent)

    def visit_name_constant_annotation(self, node, parent: TypeAnnotationNode):
        return self.__apply_on_match(node, parent)

    def visit_unknown_annotation(self, node, parent: TypeAnnotationNode):
        return self.__apply_on_match(node, parent)
