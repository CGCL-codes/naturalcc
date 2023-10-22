from typing import Optional

from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.nodes import (
    parse_type_annotation_node,
    TypeAnnotationNode,
    SubscriptAnnotationNode,
)
from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.rewriterules import RewriteRule

__all__ = ["RemoveStandAlones"]


class RemoveStandAlones(RewriteRule):

    UNION_NODE = parse_type_annotation_node("typing.Union")
    OPTIONAL_NODE = parse_type_annotation_node("typing.Optional")
    GENERIC_NODE = parse_type_annotation_node("typing.Generic")
    ANY_NODE = parse_type_annotation_node("typing.Any")

    def matches(
        self, node: TypeAnnotationNode, parent: Optional[TypeAnnotationNode]
    ) -> bool:
        if (
            not node == self.UNION_NODE
            and not node == self.OPTIONAL_NODE
            and not node == self.GENERIC_NODE
        ):
            return False
        return not isinstance(parent, SubscriptAnnotationNode)

    def apply(self, matching_node: TypeAnnotationNode) -> TypeAnnotationNode:
        return self.ANY_NODE
