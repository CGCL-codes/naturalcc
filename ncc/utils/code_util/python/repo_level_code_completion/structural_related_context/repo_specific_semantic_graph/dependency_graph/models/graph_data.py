import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dataclasses_json import dataclass_json, config

from dependency_graph.models.language import Language
from dependency_graph.utils.log import setup_logger
from dependency_graph.utils.read_file import read_file_to_string
from dependency_graph.utils.text import slice_text

# Initialize logging
logger = setup_logger()


class EdgeRelation(enum.Enum):
    """The relation between two nodes"""

    # (x, y, z) represents (Relationship Category ID, Category Internal ID, Relationship Direction)
    # 1. Syntax relations
    ParentOf = (1, 0, 0)
    ChildOf = (1, 0, 1)
    Construct = (1, 1, 0)
    ConstructedBy = (1, 1, 1)
    # 2. Import relations
    Imports = (2, 0, 0)
    ImportedBy = (2, 0, 1)
    # 3. Inheritance relations
    BaseClassOf = (3, 0, 0)
    DerivedClassOf = (3, 0, 1)
    # 4. Method override relations
    Overrides = (4, 0, 0)
    OverriddenBy = (4, 0, 1)
    # 5. Method call relations
    Calls = (5, 0, 0)
    CalledBy = (5, 0, 1)
    # 6. Object instantiation relations
    Instantiates = (6, 0, 0)
    InstantiatedBy = (6, 0, 1)
    # 7. Field use relations
    Uses = (7, 0, 0)
    UsedBy = (7, 0, 1)
    # 8. Def-use relations
    Defines = (8, 0, 0)
    DefinedBy = (8, 0, 1)

    def __str__(self):
        return self.name

    @classmethod
    def __getitem__(cls, name):
        return cls[name]

    def get_inverse_kind(self) -> "EdgeRelation":
        new_value = [*self.value]
        new_value[2] = 1 - new_value[2]
        return EdgeRelation(tuple(new_value))

    def is_inverse_relationship(self, other) -> bool:
        return (
            self.value[0] == other.value[0]
            and self.value[1] == other.value[1]
            and self.value[2] != other.value[2]
        )


@dataclass_json
@dataclass
class Location:
    def __str__(self) -> str:
        signature = f"{self.file_path}"
        loc = [self.start_line, self.start_column, self.end_line, self.end_column]
        if any([l is not None for l in loc]):
            signature += f":{self.start_line}:{self.start_column}-{self.end_line}:{self.end_column}"

        return signature

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_text(self) -> Optional[str]:
        if self.file_path is None:
            return None

        content = read_file_to_string(self.file_path)
        loc = [self.start_line, self.start_column, self.end_line, self.end_column]
        if any([l is None for l in loc]):
            return None

        return slice_text(
            content, self.start_line, self.start_column, self.end_line, self.end_column
        )

    file_path: Optional[Path] = field(
        default=None,
        metadata=config(
            encoder=lambda v: str(v),
            decoder=lambda v: v if isinstance(v, Path) else Path(v),
        ),
    )
    """The file path"""
    start_line: Optional[int] = None
    """The start line number, 1-based"""
    start_column: Optional[int] = None
    """The start column number, 1-based"""
    end_line: Optional[int] = None
    """The end line number, 1-based"""
    end_column: Optional[int] = None
    """The end column number, 1-based"""


class NodeType(str, enum.Enum):
    # TODO should nest a language to mark different type for different language
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    VARIABLE = "variable"
    STATEMENT = "statement"

    def __str__(self):
        return self.value


@dataclass_json
@dataclass
class Node:
    def __str__(self) -> str:
        return f"{self.name}:{self.type.value}@{self.location}"

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_text(self) -> Optional[str]:
        return self.location.get_text()

    def get_stub(
        self, language: Language, include_comments: bool = False
    ) -> Optional[str]:
        if language == Language.Python:
            from dependency_graph.utils.mypy_stub import generate_python_stub

            return generate_python_stub(self.get_text())
        elif language == Language.Java:
            from dependency_graph.utils.tree_sitter_stub import generate_java_stub

            return generate_java_stub(
                self.get_text(), include_comments=include_comments
            )
        elif language == Language.CSharp:
            from dependency_graph.utils.tree_sitter_stub import (
                generate_c_sharp_stub,
            )

            return generate_c_sharp_stub(
                self.get_text(), include_comments=include_comments
            )
        elif language in (Language.TypeScript, Language.JavaScript):
            from dependency_graph.utils.tree_sitter_stub import (
                generate_ts_js_stub,
            )

            return generate_ts_js_stub(
                self.get_text(), include_comments=include_comments
            )
        else:
            logger.warning(f"Stub generation is not supported for {language}")
            return None

    type: NodeType = field(
        metadata=config(
            encoder=lambda v: NodeType(v).value, decoder=lambda v: NodeType(v)
        )
    )
    """The type of the node"""
    name: str
    """The name of the node"""
    location: Location
    """The location of the node"""


@dataclass_json
@dataclass
class Edge:
    def __str__(self) -> str:
        signature = f"{self.relation}"
        if self.location:
            signature += f"@{self.location}"
        return signature

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_text(self) -> Optional[str]:
        return self.location.get_text()

    def get_inverse_edge(self) -> "Edge":
        return Edge(
            relation=self.relation.get_inverse_kind(),
            location=self.location,
        )

    relation: EdgeRelation = field(
        metadata=config(encoder=lambda v: str(v), decoder=lambda v: EdgeRelation[v])
    )
    """The relation between two nodes"""
    location: Optional[Location] = None
    """The location of the edge"""
