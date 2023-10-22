from typing import Optional

from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.nodes import (
    parse_type_annotation_node,
    TypeAnnotationNode,
    SubscriptAnnotationNode,
    IndexAnnotationNode,
    TupleAnnotationNode,
)
from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.rewriterules import RewriteRule

__all__ = ["RemoveRecursiveGenerics"]


class RemoveRecursiveGenerics(RewriteRule):

    GENERIC_NODE = parse_type_annotation_node("typing.Generic")

    def matches(
        self, node: TypeAnnotationNode, parent: Optional[TypeAnnotationNode]
    ) -> bool:
        if not isinstance(node, SubscriptAnnotationNode):
            return False
        if node.value != self.GENERIC_NODE:
            return False

        slice = node.slice
        if not isinstance(slice, IndexAnnotationNode):
            return False
        slice = slice.value

        if isinstance(slice, TupleAnnotationNode):
            return any(
                s == self.GENERIC_NODE
                or (
                    isinstance(s, SubscriptAnnotationNode)
                    and s.value == self.GENERIC_NODE
                )
                for s in slice.elements
            )

        return False

    def apply(self, matching_node: TypeAnnotationNode) -> TypeAnnotationNode:
        slice = matching_node.slice
        if not isinstance(slice, IndexAnnotationNode):
            return matching_node
        slice = slice.value

        next_slice = set()
        for s in slice.elements:
            if s == self.GENERIC_NODE:
                pass  # has no arguments
            elif (
                isinstance(s, SubscriptAnnotationNode) and s.value == self.GENERIC_NODE
            ):
                if isinstance(s.slice.value, TupleAnnotationNode):
                    next_slice |= set(s.slice.value.elements)
                else:
                    next_slice.add(s.slice.value)
            else:
                next_slice.add(s)
        return SubscriptAnnotationNode(
            matching_node.value, IndexAnnotationNode(TupleAnnotationNode(next_slice))
        )
