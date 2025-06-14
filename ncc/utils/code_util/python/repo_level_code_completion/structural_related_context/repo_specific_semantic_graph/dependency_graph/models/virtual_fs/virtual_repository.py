from collections import namedtuple
from typing import List

from fs.memoryfs import MemoryFS

from dependency_graph.models import PathLike
from dependency_graph.models.language import Language
from dependency_graph.models.repository import Repository
from dependency_graph.models.virtual_fs.virtual_path import VirtualPath

# Define the VirtualFile named tuple
VirtualFile = namedtuple("VirtualFile", ["relative_path", "content"])


class VirtualRepository(Repository):
    def __init__(
        self,
        repo_path: PathLike,
        language: Language,
        virtual_files: List[VirtualFile],  # Use the named tuple for typing
    ):
        self.fs = MemoryFS()
        # Make sure the repo path is absolute
        self.repo_path = VirtualPath(self.fs, "/", repo_path)
        self.repo_path.mkdir(parents=True)

        for file_path, content in virtual_files:
            # Strip the leading slash on the file path
            p = VirtualPath(self.fs, self.repo_path / file_path.lstrip("/"))
            # Create all files in the file system
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

        super().__init__(self.repo_path, language)
