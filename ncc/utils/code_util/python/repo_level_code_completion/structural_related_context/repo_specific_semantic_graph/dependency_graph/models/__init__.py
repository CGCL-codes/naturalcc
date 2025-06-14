from pathlib import Path
from typing import Union

from dependency_graph.models.virtual_fs.virtual_path import VirtualPath

PathLike = Union[str, Path, VirtualPath]
