import typing
from enum import Enum, auto
from typing import Optional, NamedTuple, List, Dict
from pprint import PrettyPrinter

from .typeparsing import TypeAnnotationNode


class EdgeType(Enum):
    CHILD = auto()
    NEXT = auto()
    LAST_LEXICAL_USE = auto()
    NEXT_USE = auto()
    COMPUTED_FROM = auto()
    RETURNS_TO = auto()
    OCCURRENCE_OF = auto()
    SUBTOKEN_OF = auto()


class TokenNode:
    """A wrapper around token nodes, such that an object-identity is used for comparing nodes."""

    def __init__(
        self, token: str, lineno: Optional[int] = None, col_offset: Optional[int] = None
    ):
        assert isinstance(token, str)
        self.token = token
        self.lineno = lineno
        self.col_offset = col_offset

    def __str__(self):
        return self.token


class StrSymbol:
    def __init__(self, name: str):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, StrSymbol):
            return False
        return self.name == other.name

    def __str__(self):
        return "Symbol: " + self.name


class SymbolInformation(NamedTuple):
    name: str
    locations: List[typing.Tuple[int, int]]
    annotatable_locations: Dict[typing.Tuple[int, int], Optional[TypeAnnotationNode]]
    symbol_type: str

    @classmethod
    def create(cls, name: str, symbol_type: str) -> "SymbolInformation":
        return SymbolInformation(name, [], {}, symbol_type)


def prettyprint_graph(g: Dict):
    g["token-sequence"] = [f"{ind}_{g['nodes'][ind]}" for ind in g["token-sequence"]]
    g["edges"] = {
        edge_type: {
            f"{v}_{g['nodes'][v]}": [f"{u}_{g['nodes'][u]}" for u in us]
            for v, us in g["edges"][edge_type].items()
        }
        for edge_type in g["edges"]
    }
    PrettyPrinter().pprint(g)
