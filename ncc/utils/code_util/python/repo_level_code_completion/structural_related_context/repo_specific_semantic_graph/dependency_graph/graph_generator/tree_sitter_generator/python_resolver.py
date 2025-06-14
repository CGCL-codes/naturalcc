import os
from pathlib import Path
from typing import Optional

from importlab.import_finder import is_builtin
from importlab.resolve import (
    convert_to_path,
    ImportException,
)


class Resolver:
    """
    Inspired and modified from importlab.resolve.Resolver to support Path/VirtualPath
    """

    def __init__(self, repo_path: Path, current_module: Path):
        self.repo_path = repo_path
        self.current_module = current_module
        self.current_directory = self.current_module.parent

    def _find_file(self, name):
        init = name / "__init__.py"
        # Check if there is a name to prevent `Path('/').with_suffix('.py')` error
        if not name.name:
            return None
        py = name.with_suffix(".py")
        for file in [init, py]:
            if file.exists():
                return file
        return None

    def resolve_import(self, item) -> Optional[Path]:
        """Simulate how Python resolves imports.

        Returns the filename of the source file Python would load
        when processing a statement like 'import name' in the module
        we're currently under.

        Args:
            item: An instance of ImportItem

        Returns:
            A filename

        Raises:
            ImportException: If the module doesn't exist.
        """
        name = item.name
        # The last part in `from a.b.c import d` might be a symbol rather than a
        # module, so we try a.b.c and a.b.c.d as names.
        short_name = None
        if item.is_from and not item.is_star:
            if "." in name.lstrip("."):
                # The name is something like `a.b.c`, so strip off `.c`.
                rindex = name.rfind(".")
            else:
                # The name is something like `..c`, so strip off just `c`.
                rindex = name.rfind(".") + 1
            short_name = name[:rindex]

        if is_builtin(name):
            return None

        filename, level = convert_to_path(name)
        if level:
            # This is a relative import; we need to resolve the filename
            # relative to the importing file path.
            filename = str((self.current_directory / filename).resolve())

        if not short_name:
            try_filename = True
            try_short_filename = False
        elif item.source:
            # If the import has a source path, we can use it to eliminate
            # filenames that don't match.
            source_filename, _ = os.path.splitext(item.source)
            dirname, basename = os.path.split(source_filename)
            if basename == "__init__":
                source_filename = dirname
            try_filename = source_filename.endswith(filename)
            try_short_filename = not try_filename
        else:
            try_filename = try_short_filename = True

        try_filename_files = set()
        try_short_filename_files = set()
        if try_filename:
            try_filename_files.add((name, self.repo_path / filename))
            try_filename_files.add((name, self.current_directory / filename))

            # Add parent directories
            for parent in self.current_directory.parents:
                potential_path = parent / filename
                if potential_path.is_relative_to(
                    self.repo_path
                ):  # Ensure the path is within repo_path
                    try_filename_files.add((name, potential_path))

        if try_short_filename:
            short_filename = (self.repo_path / filename).parent
            try_short_filename_files.add((short_name, short_filename))
            short_filename = (self.current_directory / filename).parent
            try_short_filename_files.add((short_name, short_filename))

            # Add parent directories
            for parent in self.current_directory.parents:
                potential_path = (parent / filename).parent
                if potential_path.is_relative_to(
                    self.repo_path
                ):  # Ensure the path is within repo_path
                    try_short_filename_files.add((short_name, potential_path))

        # Try filename first then short filename
        files = list(try_filename_files) + list(try_short_filename_files)

        for module_name, path in files:
            f = self._find_file(path)
            if not f or f == self.current_module:
                # We cannot import a file from itself.
                continue
            # if item.is_relative():
            #     package_name = get_package_name(self.current_module)
            #     if package_name is None:
            #         # Relative import in non-package
            #         raise ImportException(name)
            #     module_name = get_absolute_name(package_name, module_name)
            return f

        raise ImportException(name)
