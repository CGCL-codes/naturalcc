from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ParseTreeInfo:
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    text: str
    type: str
    parent: Optional["ParseTreeInfo"] = None


@dataclass
class RegexInfo:
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    text: str
