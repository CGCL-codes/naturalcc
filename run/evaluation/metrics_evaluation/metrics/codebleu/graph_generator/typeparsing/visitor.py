from abc import ABC
from typing import Any

__all__ = ["TypeAnnotationVisitor"]


class TypeAnnotationVisitor(ABC):
    def visit_subscript_annotation(self, node, *args) -> Any:
        pass

    def visit_tuple_annotation(self, node, *args) -> Any:
        pass

    def visit_name_annotation(self, node, *args) -> Any:
        pass

    def visit_list_annotation(self, node, *args) -> Any:
        pass

    def visit_attribute_annotation(self, node, *args) -> Any:
        pass

    def visit_index_annotation(self, node, *args) -> Any:
        pass

    def visit_elipsis_annotation(self, node, *args) -> Any:
        pass

    def visit_name_constant_annotation(self, node, *args) -> Any:
        pass

    def visit_unknown_annotation(self, node, *args) -> Any:
        pass
