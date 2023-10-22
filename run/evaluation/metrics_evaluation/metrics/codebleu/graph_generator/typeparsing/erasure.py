from itertools import product

from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.nodes import (
    SubscriptAnnotationNode,
    TupleAnnotationNode,
    ListAnnotationNode,
    AttributeAnnotationNode,
    IndexAnnotationNode,
    ElipsisAnnotationNode,
)
from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.visitor import TypeAnnotationVisitor

__all__ = ["EraseOnceTypeRemoval"]


class EraseOnceTypeRemoval(TypeAnnotationVisitor):
    """Replace Nodes with Aliases. Assumes recursion has been resolved in replacement_map"""

    def __init__(self):
        pass

    def visit_subscript_annotation(self, node: SubscriptAnnotationNode):
        if node.slice is None:
            erasure_happened_at_a_slice = False
        else:
            next_slices, erasure_happened_at_a_slice = node.slice.accept_visitor(self)

        if not erasure_happened_at_a_slice:
            return [node, node.value], True  # Erase type parameters

        return [
            SubscriptAnnotationNode(value=node.value, slice=s) for s in next_slices
        ], True

    def visit_tuple_annotation(self, node: TupleAnnotationNode):
        elements = [e.accept_visitor(self) for e in node.elements]

        erasure_happened_before = any(e[1] for e in elements)
        return [
            TupleAnnotationNode(t) for t in product(*(e[0] for e in elements))
        ], erasure_happened_before

    def visit_name_annotation(self, node):
        return [node], False

    def visit_list_annotation(self, node: ListAnnotationNode):
        elements = [e.accept_visitor(self) for e in node.elements]

        erasure_happened_before = any(e[1] for e in elements)
        return [
            ListAnnotationNode(t) for t in product(*(e[0] for e in elements))
        ], erasure_happened_before

    def visit_attribute_annotation(self, node: AttributeAnnotationNode):
        return [node], False

    def visit_index_annotation(self, node: IndexAnnotationNode):
        next_values, erasure_happened = node.value.accept_visitor(self)
        return [IndexAnnotationNode(v) for v in next_values], erasure_happened

    def visit_elipsis_annotation(self, node: ElipsisAnnotationNode):
        return [node], False

    def visit_name_constant_annotation(self, node):
        return [node], False

    def visit_unknown_annotation(self, node):
        return [node], False
