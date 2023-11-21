from abc import ABC, abstractmethod

from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.nodes import TypeAnnotationNode


class RewriteRule(ABC):
    @abstractmethod
    def matches(self, node: TypeAnnotationNode, parent: TypeAnnotationNode) -> bool:
        pass

    @abstractmethod
    def apply(self, matching_node: TypeAnnotationNode) -> TypeAnnotationNode:
        pass
