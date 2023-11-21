from abc import ABC, abstractmethod
from typing import Any, Optional, Iterator

import typed_ast
from typed_ast.ast3 import parse

from metrics_evaluation.metrics.codebleu.graph_generator.typeparsing.visitor import TypeAnnotationVisitor

__all__ = [
    "FaultyAnnotation",
    "TypeAnnotationNode",
    "SubscriptAnnotationNode",
    "TupleAnnotationNode",
    "NameAnnotationNode",
    "ListAnnotationNode",
    "AttributeAnnotationNode",
    "IndexAnnotationNode",
    "ElipsisAnnotationNode",
    "NameConstantAnnotationNode",
    "UnknownAnnotationNode",
    "parse_type_annotation_node",
    "parse_type_comment",
]


class FaultyAnnotation(Exception):
    pass


class TypeAnnotationNode(ABC):
    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def accept_visitor(self, visitor: TypeAnnotationVisitor, *args) -> Any:
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @staticmethod
    @abstractmethod
    def parse(node) -> "SubscriptAnnotationNode":
        pass


class SubscriptAnnotationNode(TypeAnnotationNode):
    def __init__(self, value: TypeAnnotationNode, slice: Optional[TypeAnnotationNode]):
        self.value = value
        self.slice = slice

    def size(self) -> int:
        size = 1 + self.value.size()
        if self.slice is not None:
            size += self.slice.size()
        return size

    def accept_visitor(self, visitor: TypeAnnotationVisitor, *args) -> Any:
        return visitor.visit_subscript_annotation(self, *args)

    def __repr__(self):
        return repr(self.value) + "[" + repr(self.slice) + "]"

    def __hash__(self):
        return hash(self.value) ^ (hash(self.slice) + 13)

    def __eq__(self, other):
        if not isinstance(other, SubscriptAnnotationNode):
            return False
        else:
            return self.value == other.value and self.slice == other.slice

    @staticmethod
    def parse(node) -> "SubscriptAnnotationNode":
        assert hasattr(node, "value")
        assert hasattr(node, "slice")

        v = _parse_recursive(node.value)
        s = _parse_recursive(node.slice)
        assert v is not None
        return SubscriptAnnotationNode(v, s)


class TupleAnnotationNode(TypeAnnotationNode):
    def __init__(self, elements: Iterator[TypeAnnotationNode]):
        self.elements = tuple(elements)

    def size(self) -> int:
        return sum(e.size() for e in self.elements) + 1

    def accept_visitor(self, visitor: TypeAnnotationVisitor, *args) -> Any:
        return visitor.visit_tuple_annotation(self, *args)

    def __repr__(self):
        return ", ".join(repr(e) for e in self.elements)

    def __hash__(self):
        if len(self.elements) > 0:
            return hash(self.elements)
        else:
            return 1

    def __eq__(self, other):
        if not isinstance(other, TupleAnnotationNode):
            return False
        else:
            return self.elements == other.elements

    @staticmethod
    def parse(node) -> "TupleAnnotationNode":
        assert hasattr(node, "elts")
        return TupleAnnotationNode((_parse_recursive(el) for el in node.elts))


class NameAnnotationNode(TypeAnnotationNode):
    def __init__(self, identifier: str):
        self.identifier = identifier

    def size(self) -> int:
        return 1

    def accept_visitor(self, visitor: TypeAnnotationVisitor, *args) -> Any:
        return visitor.visit_name_annotation(self, *args)

    def __repr__(self):
        return self.identifier

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        if not isinstance(other, NameAnnotationNode):
            return False
        return self.identifier == other.identifier

    @staticmethod
    def parse(node) -> "NameAnnotationNode":
        assert hasattr(node, "id")
        return NameAnnotationNode(node.id)


class ListAnnotationNode(TypeAnnotationNode):
    def __init__(self, elements: Iterator[TypeAnnotationNode]):
        self.elements = tuple(elements)

    def size(self) -> int:
        return sum(e.size() for e in self.elements) + 1

    def accept_visitor(self, visitor: TypeAnnotationVisitor, *args) -> Any:
        return visitor.visit_list_annotation(self, *args)

    def __repr__(self):
        return "[" + ", ".join(repr(e) for e in self.elements) + "]"

    def __hash__(self):
        if len(self.elements) > 0:
            return hash(self.elements)
        else:
            return 2

    def __eq__(self, other):
        if not isinstance(other, ListAnnotationNode):
            return False
        return self.elements == other.elements

    @staticmethod
    def parse(node) -> "ListAnnotationNode":
        assert hasattr(node, "elts")
        return ListAnnotationNode((_parse_recursive(el) for el in node.elts))


class AttributeAnnotationNode(TypeAnnotationNode):
    def __init__(self, value: TypeAnnotationNode, attribute: str):
        self.value = value
        assert isinstance(attribute, str), type(attribute)
        self.attribute = attribute

    def size(self) -> int:
        return 1 + self.value.size()

    def accept_visitor(self, visitor: TypeAnnotationVisitor, *args) -> Any:
        return visitor.visit_attribute_annotation(self, *args)

    def __repr__(self):
        return repr(self.value) + "." + self.attribute

    def __hash__(self):
        return hash(self.attribute) ^ (hash(self.value) + 13)

    def __eq__(self, other):
        if not isinstance(other, AttributeAnnotationNode):
            return False
        else:
            return self.attribute == other.attribute and self.value == other.value

    @staticmethod
    def parse(node) -> "AttributeAnnotationNode":
        assert hasattr(node, "value")
        assert hasattr(node, "attr")
        return AttributeAnnotationNode(_parse_recursive(node.value), node.attr)


class IndexAnnotationNode(TypeAnnotationNode):
    def __init__(self, value: TypeAnnotationNode):
        self.value = value

    def size(self) -> int:
        return 1 + self.value.size()

    def accept_visitor(self, visitor: TypeAnnotationVisitor, *args) -> Any:
        return visitor.visit_index_annotation(self, *args)

    def __repr__(self):
        return repr(self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, IndexAnnotationNode):
            return False
        return self.value == other.value

    @staticmethod
    def parse(node) -> "IndexAnnotationNode":
        assert hasattr(node, "value")
        return IndexAnnotationNode(_parse_recursive(node.value))


class ElipsisAnnotationNode(TypeAnnotationNode):
    def __init__(self):
        pass

    def size(self) -> int:
        return 1

    def accept_visitor(self, visitor: TypeAnnotationVisitor, *args) -> Any:
        return visitor.visit_elipsis_annotation(self, *args)

    def __repr__(self):
        return "..."

    def __hash__(self):
        return 3

    def __eq__(self, other):
        return isinstance(other, ElipsisAnnotationNode)

    @staticmethod
    def parse(node) -> "ElipsisAnnotationNode":
        return ElipsisAnnotationNode()


class NameConstantAnnotationNode(TypeAnnotationNode):
    def __init__(self, value: Any):
        self.value = value

    def size(self) -> int:
        return 1

    def accept_visitor(self, visitor: TypeAnnotationVisitor, *args) -> Any:
        return visitor.visit_name_constant_annotation(self, *args)

    def __repr__(self):
        return repr(self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, NameConstantAnnotationNode):
            return False
        return self.value == other.value

    @staticmethod
    def parse(node) -> "NameConstantAnnotationNode":
        if hasattr(node, "value"):
            return NameConstantAnnotationNode(node.value)
        return NameConstantAnnotationNode(None)


class UnknownAnnotationNode(TypeAnnotationNode):
    def __init__(self):
        import pdb

        pdb.set_trace()
        pass

    def size(self) -> int:
        return 1

    def accept_visitor(self, visitor: TypeAnnotationVisitor, *args) -> Any:
        return visitor.visit_unknown_annotation(self, *args)

    def __repr__(self):
        return "%UNKNOWN%"

    def __hash__(self):
        return 4

    def __eq__(self, other):
        return isinstance(other, UnknownAnnotationNode)

    @staticmethod
    def parse(node) -> "UnknownAnnotationNode":
        raise NotImplementedError()


def _parse_string_annotation(node):
    assert hasattr(node, "s")
    try:
        node = parse(node.s, "", mode="eval")
        return _parse_recursive(node.body)
    except SyntaxError:
        return None


def _parse_recursive(node) -> TypeAnnotationNode:
    if isinstance(node, typed_ast._ast3.Subscript):  # pytype: disable=module-attr
        return SubscriptAnnotationNode.parse(node)
    elif isinstance(node, typed_ast._ast3.Tuple):  # pytype: disable=module-attr
        return TupleAnnotationNode.parse(node)
    elif isinstance(node, typed_ast._ast3.Name):  # pytype: disable=module-attr
        return NameAnnotationNode.parse(node)
    elif isinstance(node, typed_ast._ast3.List):  # pytype: disable=module-attr
        return ListAnnotationNode.parse(node)
    elif isinstance(node, typed_ast._ast3.Attribute):  # pytype: disable=module-attr
        return AttributeAnnotationNode.parse(node)
    elif isinstance(node, typed_ast._ast3.Str):  # pytype: disable=module-attr
        return _parse_string_annotation(node)
    elif isinstance(node, typed_ast._ast3.Index):  # pytype: disable=module-attr
        return IndexAnnotationNode.parse(node)
    elif isinstance(node, typed_ast._ast3.Ellipsis):  # pytype: disable=module-attr
        return ElipsisAnnotationNode.parse(node)
    elif isinstance(node, typed_ast._ast3.NameConstant):  # pytype: disable=module-attr
        return NameConstantAnnotationNode.parse(node)
    elif isinstance(node, typed_ast._ast3.Num):  # pytype: disable=module-attr
        return NameConstantAnnotationNode.parse(node)
    else:
        raise Exception("Unparsable type node.")


def parse_type_annotation_node(node) -> Optional[TypeAnnotationNode]:
    """
    Processes the node containing the type annotation and return the object corresponding to the node type.
    """
    try:
        if isinstance(node, str):
            r = parse_type_comment(node)
        else:
            r = _parse_recursive(node)
        return r
    except Exception as e:
        pass
    return None


def parse_type_comment(annotation: str) -> Optional[TypeAnnotationNode]:
    try:
        node = parse(annotation, "", mode="eval")
    except SyntaxError:
        return None
    try:
        return _parse_recursive(node.body)
    except Exception as e:
        return None
