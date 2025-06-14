import _imp
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import (
    SourceFileLoader,
    ModuleSpec,
    ExtensionFileLoader,
    SourcelessFileLoader,
    BYTECODE_SUFFIXES,
)
from importlib.util import spec_from_loader
from pathlib import Path

from fs.base import FS

from dependency_graph.models import VirtualPath

SOURCE_SUFFIXES = [".py"]
_POPULATE = sys.path


def _get_supported_file_loaders():
    """Copied from importlib._bootstrap_external._get_supported_file_loaders

    Returns a list of file-based module loaders.

    Each item is a tuple (loader, suffixes).
    """
    extensions = ExtensionFileLoader, _imp.extension_suffixes()
    source = SourceFileLoader, SOURCE_SUFFIXES
    bytecode = SourcelessFileLoader, BYTECODE_SUFFIXES
    return [extensions, source, bytecode]


def spec_from_file_location(
    name,
    location: VirtualPath = None,
    *,
    loader=None,
    submodule_search_locations=_POPULATE,
):
    """Copied and modified from importlib._bootstrap_external.spec_from_file_location to support VirtualPath

    Return a module spec based on a file location.

    To indicate that the module is a package, set
    submodule_search_locations to a list of directory paths.  An
    empty list is sufficient, though its not otherwise useful to the
    import system.

    The loader must take a spec as its only __init__() arg.

    """
    if location is None:
        # The caller may simply want a partially populated location-
        # oriented spec.  So we set the location to a bogus value and
        # fill in as much as we can.
        location = "<unknown>"
        if hasattr(loader, "get_filename"):
            # ExecutionLoader
            try:
                location = loader.get_filename(name)
            except ImportError:
                pass
    else:
        location = Path(location).__fspath__()
        try:
            location = Path(location).absolute()
        except OSError:
            pass

    # If the location is on the filesystem, but doesn't actually exist,
    # we could return None here, indicating that the location is not
    # valid.  However, we don't have a good way of testing since an
    # indirect location (e.g. a zip file or URL) will look like a
    # non-existent file relative to the filesystem.

    spec = ModuleSpec(name, loader, origin=location)
    spec._set_fileattr = True

    # Pick a loader if one wasn't provided.
    if loader is None:
        for loader_class, suffixes in _get_supported_file_loaders():
            if location.endswith(tuple(suffixes)):
                loader = loader_class(name, location)
                spec.loader = loader
                break
        else:
            return None

    # Set submodule_search_paths appropriately.
    if submodule_search_locations is _POPULATE:
        # Check the loader.
        if hasattr(loader, "is_package"):
            try:
                is_package = loader.is_package(name)
            except ImportError:
                pass
            else:
                if is_package:
                    spec.submodule_search_locations = []
    else:
        spec.submodule_search_locations = submodule_search_locations
    if spec.submodule_search_locations == []:
        if location:
            dirname = str(location.parent)
            spec.submodule_search_locations.append(dirname)

    return spec


def spec_from_loader(name, loader, *, origin=None, is_package=None):
    """Return a module spec based on various loader methods."""
    if origin is None:
        origin = getattr(loader, "_ORIGIN", None)

    if not origin and hasattr(loader, "get_filename"):
        if is_package is None:
            return spec_from_file_location(name, loader=loader)
        search = [] if is_package else None
        return spec_from_file_location(
            name, loader=loader, submodule_search_locations=search
        )

    if is_package is None:
        if hasattr(loader, "is_package"):
            try:
                is_package = loader.is_package(name)
            except ImportError:
                is_package = None  # aka, undefined
        else:
            # the default
            is_package = False

    return ModuleSpec(name, loader, origin=origin, is_package=is_package)


class VirtualFSLoader(SourceFileLoader):
    """
    A loader that uses a PyFilesystem instance to load modules
    """

    def __init__(self, fs: FS, fullname, path):
        super().__init__(fullname, path)
        self.fs = fs

    def __hash__(self):
        return hash(self.fs) ^ hash(self.name) ^ hash(self.path)

    def get_data(self, path):
        """Return the data from path as raw bytes."""
        return self.fs.readbytes(path)

    def get_source(self, fullname):
        return self.fs.readtext(self.path)

    def get_filename(self, fullname) -> VirtualPath:
        """Return the path to the source file as found by the finder.
        !!!It is very important to return as VirtualPath instead of str because Jedi will use it to find the Path
        in the cache, see jedi.parser_utils.get_parso_cache_node.
        If we return str, Jedi will initialize it as Path(see parso.file_io.FileIO.__init__), not VirtualPath, causing
        the cache to be missed.
        """
        return VirtualPath(self.fs, self.path)

    def is_package(self, fullname):
        """Concrete implementation of InspectLoader.is_package by checking if
        the path returned by get_filename has a filename of '__init__.py'."""
        filename = self.get_filename(fullname).name
        filename_base = filename.rsplit(".", 1)[0]
        tail_name = fullname.rpartition(".")[2]
        return filename_base == "__init__" and tail_name != "__init__"


class VirtualFSFinder(MetaPathFinder):
    """
    A meta path finder that uses a PyFilesystem instance to find modules.
    It will loop through all paths in sys.path and try to find the module in the fs.
    """

    def __init__(self, fs: FS):
        self.fs = fs
        # Cache to avoid creating multiple loaders for the same module
        self.memory_loader_cache = {}

    def find_spec(self, fullname, path=None, target=None):
        # Default search in sys.path if path is not provided
        search_paths = sys.path if path is None else path

        # Search through all paths in sys.path
        for p in search_paths:
            # Filter out non-existent sys path in the fs
            if not self.fs.exists(p):
                continue

            # Transform module name to file path
            module_rel_path = f"{fullname.replace('.', '/')}"
            module_abs_path = f"{p}/{module_rel_path}".replace("//", "/")

            is_package = False
            if self.fs.isdir(module_abs_path):
                module_abs_path = f"{module_abs_path}/__init__.py"
                is_package = True
            else:
                module_abs_path = f"{module_abs_path}.py"

            if self.fs.exists(module_abs_path):
                # Use the cache to avoid re-creating MemoryFSLoader instances
                if fullname not in self.memory_loader_cache:
                    loader = VirtualFSLoader(self.fs, fullname, module_abs_path)
                    self.memory_loader_cache[fullname] = loader
                else:
                    loader = self.memory_loader_cache[fullname]

                spec = spec_from_loader(fullname, loader)
                if is_package:
                    spec.submodule_search_locations = [
                        # VirtualPath(self.fs, module_abs_path)
                        module_abs_path,
                    ]

                return spec

        return None
