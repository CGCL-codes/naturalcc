import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

from importlab.parsepy import ImportStatement
from importlab.resolve import ImportException

from dependency_graph.graph_generator.tree_sitter_generator import ImportFinder
from dependency_graph.graph_generator.tree_sitter_generator.info import (
    RegexInfo,
    ParseTreeInfo,
)
from dependency_graph.graph_generator.tree_sitter_generator.python_resolver import (
    Resolver,
)
from dependency_graph.models import VirtualPath, PathLike
from dependency_graph.models.language import Language
from dependency_graph.models.repository import Repository
from dependency_graph.utils.log import setup_logger

# Initialize logging
logger = setup_logger()

if sys.version_info < (3, 9):

    def is_relative_to(self, *other):
        try:
            self.relative_to(*other)
            return True
        except ValueError:
            return False

    # Patch the method in OriginalPath
    Path.is_relative_to = is_relative_to


class ImportResolver:
    def __init__(self, repo: Repository):
        self.repo = repo

    def _Path(self, file_path: PathLike) -> Path:
        """
        Convert the str file path to handle both physical and virtual paths
        """
        if isinstance(self.repo.repo_path, VirtualPath):
            return VirtualPath(self.repo.repo_path.fs, file_path)
        elif isinstance(self.repo.repo_path, Path):
            return Path(file_path)
        else:
            return Path(file_path)

    def resolve_import(
        self,
        import_symbol_node: Union[ParseTreeInfo, RegexInfo],
        module_map: Dict[str, List[Path]],
        importer_file_path: Path,
    ) -> List[Path]:
        resolved_path_list = []
        if isinstance(import_symbol_node, RegexInfo):
            assert (
                self.repo.language in ImportFinder.languages_using_regex
            ), f"import_symbol_node {import_symbol_node} of type RegexInfo is only supported for {ImportFinder.languages_using_regex}, not {self.repo.language}"

        if self.repo.language in (Language.Java, Language.Kotlin):
            import_symbol_name = import_symbol_node.text
            # Deal with star import: `import xxx.*`
            if ".*" in import_symbol_node.parent.text:
                for module_name, path_list in module_map.items():
                    # Use rpartition to split the string at the rightmost '.'
                    package_name, _, _ = module_name.rpartition(".")
                    if package_name == import_symbol_name:
                        resolved_path_list.extend(path_list)
            else:
                resolved_path_list.extend(module_map.get(import_symbol_name, []))
        elif self.repo.language == Language.CSharp:
            import_symbol_name = import_symbol_node.text
            resolved_path_list.extend(module_map.get(import_symbol_name, []))
        elif self.repo.language in (Language.TypeScript, Language.JavaScript):
            resolved_path_list.extend(
                self.resolve_ts_js_import(
                    import_symbol_node, module_map, importer_file_path
                )
            )
        elif self.repo.language == Language.Python:
            resolved_path_list.extend(
                self.resolve_python_import(import_symbol_node, importer_file_path)
            )
        elif self.repo.language == Language.PHP:
            resolved_path_list.extend(
                self.resolve_php_import(import_symbol_node, importer_file_path)
            )
        elif self.repo.language == Language.Ruby:
            resolved_path_list.extend(
                self.resolve_ruby_import(import_symbol_node, importer_file_path)
            )
        elif self.repo.language in (Language.C, Language.CPP):
            resolved_path_list.extend(
                self.resolve_cfamily_import(import_symbol_node, importer_file_path)
            )
        elif self.repo.language == Language.Go:
            resolved_path_list.extend(self.resolve_go_import(import_symbol_node))
        elif self.repo.language == Language.Swift:
            resolved_path_list.extend(
                self.resolve_swift_import(import_symbol_node, importer_file_path)
            )
        elif self.repo.language == Language.Rust:
            resolved_path_list.extend(
                self.resolve_rust_import(import_symbol_node, importer_file_path)
            )
        elif self.repo.language == Language.Lua:
            resolved_path_list.extend(
                self.resolve_lua_import(import_symbol_node, importer_file_path)
            )
        elif self.repo.language == Language.Bash:
            resolved_path_list.extend(
                self.resolve_bash_import(import_symbol_node, importer_file_path)
            )
        elif self.repo.language == Language.R:
            resolved_path_list.extend(
                self.resolve_r_import(import_symbol_node, importer_file_path)
            )
        else:
            raise NotImplementedError(f"Language {self.repo.language} is not supported")

        # Resolve the path so that relative file path is normalized. This is important for the node identification in the graph
        resolved_path_list = [path.resolve() for path in resolved_path_list]
        # De-duplicate the resolved path
        path_list = set(resolved_path_list)
        resolved_path_list = []
        # Remove file not in the repo
        for resolved_path in path_list:
            if (
                resolved_path.is_relative_to(self.repo.repo_path)
                and resolved_path.is_file()
            ):
                resolved_path_list.append(resolved_path)

        return resolved_path_list

    def resolve_ts_js_import(
        self,
        import_symbol_node: ParseTreeInfo,
        module_map: Dict[str, List[Path]],
        importer_file_path: Path,
    ) -> List[Path]:
        def _search_file(search_path: Path, module_name: str) -> List[Path]:
            result_path = []
            for ext in extension_list:
                if (search_path / f"{module_name}{ext}").exists():
                    result_path.append(search_path / f"{module_name}{ext}")
                elif (search_path / f"{module_name}").is_dir():
                    """
                    In case the module is a directory, we should search for the `module_dir/index.{js|ts}` file
                    """
                    for ext in extension_list:
                        if (search_path / f"{module_name}" / f"index{ext}").exists():
                            result_path.append(
                                search_path / f"{module_name}" / f"index{ext}"
                            )
                    break
            return result_path

        import_symbol_name = import_symbol_node.text
        extension_list = (
            Repository.code_file_extensions[Language.TypeScript]
            + Repository.code_file_extensions[Language.JavaScript]
        )

        # Find the module path
        # e.g. './Descriptor' -> './Descriptor.ts'; '../Descriptor' -> '../Descriptor.ts'
        if (
            import_symbol_name.startswith("./")
            or ".." in Path(import_symbol_name).parts
            or "." in Path(import_symbol_name).parts
        ):
            """If the path is relative, then search in the filesystem"""
            suffix = self._Path(import_symbol_name).suffix
            if suffix:
                """If there is a suffix in the name, then search in the filesystem"""
                path = importer_file_path.parent / import_symbol_name
                if path.exists():
                    result_path = [path]
                else:
                    result_path = _search_file(
                        importer_file_path.parent, import_symbol_name
                    )
            else:
                result_path = _search_file(
                    importer_file_path.parent, import_symbol_name
                )
            return result_path
        else:
            return module_map.get(import_symbol_name, [])

    def resolve_python_import(
        self,
        import_symbol_node: ParseTreeInfo,
        importer_file_path: Path,
    ) -> List[Path]:
        def analyze_import_statement(
            import_statement: str, is_from_import: bool = None
        ) -> List[Tuple[str, str, str, bool]]:
            # Deal with Python break lines
            import_statement = import_statement.replace("\n", "").replace("\\", "")
            # Regular expression to match from ... import ... as ... or import ... as ...
            from_pattern = r"from\s+([\w\.]+)\s+import\s*(\(.*?\)|[^\(\),]+(?:\s+as\s+\w+)?(?:\s*,\s*[^\(\),]+(?:\s+as\s+\w+)?)*)"
            import_pattern = (
                r"import\s+([\w\.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w\.]+(?:\s+as\s+\w+)?)*)"
            )

            if is_from_import is True:
                match = re.search(from_pattern, import_statement)
                if not match:
                    raise ValueError(
                        f"Invalid parsing Python import statement: {import_statement}"
                    )
                module_name = match.group(1)  # Module name
                items_str = match.group(2)

                # Remove parentheses if present
                items_str = items_str.strip("()")

                items = [item.strip() for item in items_str.split(",")]
                parsed_items = []

                for imported_name in items:
                    asname = None
                    # Handle aliasing with 'as'
                    if " as " in imported_name:
                        imported_name, asname = imported_name.split(" as ")
                        imported_name = imported_name.strip()
                        asname = asname.strip()
                    if imported_name:  # Ensure item is not empty
                        parsed_items.append(
                            (
                                module_name,
                                imported_name,
                                asname,
                                "*" in imported_name,  # is_wildcard_import
                            )
                        )
                return parsed_items

            elif is_from_import is False:
                match = re.search(import_pattern, import_statement)
                if not match:
                    raise ValueError(
                        f"Invalid parsing Python import statement: {import_statement}"
                    )
                modules_str = match.group(1)  # Module names

                # Split the modules by the comma and process each one
                modules = [module.strip() for module in modules_str.split(",")]
                parsed_modules = []

                for module in modules:
                    asname = None
                    if " as " in module:
                        module, asname = module.split(" as ")
                        module = module.strip()
                        asname = asname.strip()
                    parsed_modules.append(
                        (
                            module,
                            None,  # imported_name
                            asname,
                            False,  # is_wildcard_import
                        )
                    )
                return parsed_modules

            raise ValueError("is_from_import must be True or False")

        assert import_symbol_node.type in (
            "import_statement",
            "import_from_statement",
        ), "import_symbol_node type is not import_statement or import_from_statement"

        resolver = Resolver(self.repo.repo_path, importer_file_path)
        resolved_path_list = []
        is_from_import = import_symbol_node.type == "import_from_statement"
        parsed_items = analyze_import_statement(import_symbol_node.text, is_from_import)
        for module_name, imported_name, asname, is_star in parsed_items:
            if module_name == ".":
                # Convert `from . import qux` to `.qux`
                name = f".{imported_name}"
            elif module_name == "..":
                # Convert `from .. import qux` to `..qux`
                name = f"..{imported_name}"
            elif imported_name and imported_name != "*":
                name = f"{module_name}.{imported_name}"
            else:
                name = module_name

            imp = ImportStatement(name, asname, is_from_import, is_star, None)

            try:
                resolved_path = resolver.resolve_import(imp)
                if resolved_path:
                    resolved_path_list.append(resolved_path)
            except ImportException:
                pass

        return resolved_path_list

    def resolve_php_import(
        self,
        import_symbol_node: ParseTreeInfo,
        importer_file_path: Path,
    ) -> List[Path]:
        import_symbol_name = import_symbol_node.text
        # Strip double and single quote
        import_symbol_name = import_symbol_name.strip('"').strip("'")
        # Find the module path
        result_path = []
        import_path = self._Path(import_symbol_name)
        if import_path.is_absolute() and import_path.exists():
            result_path.append(import_path)
        else:
            path = importer_file_path.parent / import_symbol_name
            if path.exists():
                result_path.append(path)
        return result_path

    def resolve_ruby_import(
        self,
        import_symbol_node: ParseTreeInfo,
        importer_file_path: Path,
    ) -> List[Path]:
        import_symbol_name = import_symbol_node.text
        # Strip double and single quote
        import_symbol_name = import_symbol_name.strip('"').strip("'")

        import_path = self._Path(import_symbol_name)

        # Heuristics to search for the header file
        search_paths = [
            import_path,
        ]

        # Add parent directories to the search path
        for parent in importer_file_path.parents:
            potential_path = parent / import_path
            if potential_path.is_relative_to(
                self.repo.repo_path
            ):  # Ensure the path is within repo_path
                search_paths.append(potential_path)

        # Find the module path
        result_path = []
        # Check if any of these paths exist
        extension_list = Repository.code_file_extensions[Language.Ruby]
        for path in search_paths:
            if path.exists():
                result_path.append(path)
            else:
                """Directly add the extension to the end no matter if it has suffix or not"""
                for ext in extension_list:
                    path = path.parent / f"{path.name}{ext}"
                    if path.exists():
                        result_path.append(path)

        return result_path

    def resolve_cfamily_import(
        self,
        import_symbol_node: ParseTreeInfo,
        importer_file_path: Path,
    ) -> List[Path]:

        import_symbol_name = import_symbol_node.text
        # Strip double quote and angle bracket
        import_symbol_name = import_symbol_name.strip('"').lstrip("<").rstrip(">")
        import_path = self._Path(import_symbol_name)

        # Heuristics to search for the header file
        search_paths = [
            # Common practice to have headers in 'include' directory
            self.repo.repo_path / "include" / import_path,
            # Relative path from the C file's directory
            importer_file_path.parent / import_path,
            # Common practice to have headers in 'src' directory
            self.repo.repo_path / "src" / import_path,
            # Absolute/relative path as given in the include statement
            import_path,
        ]

        # Add parent directories of the C file path
        for parent in importer_file_path.parents:
            potential_path = parent / import_path
            if potential_path.is_relative_to(
                self.repo.repo_path
            ):  # Ensure the path is within repo_path
                search_paths.append(potential_path)

        # Add sibling directories of each directory component of importer_file_path
        for parent in importer_file_path.parents:
            for sibling in parent.iterdir():
                if sibling.is_dir() and sibling != importer_file_path:
                    potential_path = sibling / import_path
                    if potential_path.is_relative_to(
                        self.repo.repo_path
                    ):  # Ensure the path is within repo_path
                        search_paths.append(potential_path)

        # Find the module path
        result_path = []
        # Check if any of these paths exist
        extension_list = (
            Repository.code_file_extensions[Language.C]
            + Repository.code_file_extensions[Language.CPP]
        )
        for path in search_paths:
            if path.exists():
                result_path.append(path)
            else:
                """Directly add the extension to the end no matter if it has suffix or not"""
                for ext in extension_list:
                    path = path.parent / f"{path.name}{ext}"
                    if path.exists():
                        result_path.append(path)

        return result_path

    def resolve_go_import(self, import_symbol_node: ParseTreeInfo) -> List[Path]:
        def parse_go_mod(go_mod_path: Path) -> Tuple[str, Dict[str, Path]]:
            """
            Parses the go.mod file and returns the module path and replacements.
            :param go_mod_path: The path to the go.mod file.
            :return: A tuple containing the module path and replacements.
            """
            module_path = None
            replacements = {}

            for line in go_mod_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("module "):
                    module_path = line.split()[1]
                elif line.startswith("replace "):
                    parts = line.split()
                    if len(parts) >= 4 and parts[2] == "=>":
                        replacements[parts[1]] = self._Path(parts[3])

            return module_path, replacements

        def search_fallback_paths(import_stmt: str, base_path: Path):
            """Searches various fallback paths within the project directory."""
            search_paths = [
                base_path / import_stmt.replace("/", os.sep),
                base_path / "src" / import_stmt.replace("/", os.sep),
                base_path / "vendor" / import_stmt.replace("/", os.sep),
                base_path / "pkg" / import_stmt.replace("/", os.sep),
            ]
            found_files = []

            for path in search_paths:
                if path.is_dir():
                    go_files = list(path.glob("*.go"))
                    if go_files:
                        found_files.extend(go_files)
                elif path.with_suffix(".go").is_file():
                    found_files.append(path.with_suffix(".go"))

            return found_files

        # Parse the go.mod file
        go_mod_path = self.repo.repo_path / "go.mod"
        if go_mod_path.exists():
            module_path, replacements = parse_go_mod(go_mod_path)
        else:
            module_path, replacements = None, {}

        # Find corresponding paths for the imported packages
        imported_paths = []

        import_stmt = import_symbol_node.text
        import_stmt = import_stmt.strip('"')

        # Resolve the import path using replacements or the module path
        resolved_paths = []
        if import_stmt in replacements:
            resolved_path = replacements[import_stmt]
            resolved_paths.append(resolved_path)
        elif module_path and import_stmt.startswith(module_path):
            resolved_path = self.repo.repo_path / import_stmt[len(module_path) + 1 :]
            resolved_paths.append(resolved_path)
        else:
            # Fallback logic: Try to resolve based on project directory structure
            resolved_paths.extend(
                search_fallback_paths(import_stmt, self.repo.repo_path)
            )

        for resolved_path in resolved_paths:
            if resolved_path and resolved_path.is_dir():
                # Try to find a .go file in the directory
                go_files = list(resolved_path.glob("*.go"))
                if go_files:
                    imported_paths.extend(go_files)

        return imported_paths

    def resolve_swift_import(
        self, import_symbol_node: ParseTreeInfo, importer_file_path: Path
    ) -> List[Path]:
        import_symbol_name = import_symbol_node.text
        if "." in import_symbol_name:
            # Handle individual declarations importing such as `import kind module.symbol`
            # In this case, we extract the module name from the import statement
            import_symbol_name = ".".join(import_symbol_name.split(".")[:-1])

        import_symbol_name = import_symbol_name.replace(".", os.sep)
        import_path = self._Path(import_symbol_name)

        # Heuristic search for source files corresponding to the imported modules
        search_paths = [
            self.repo.repo_path / "Sources" / import_symbol_name,
            self.repo.repo_path / "Tests" / import_symbol_name,
            self.repo.repo_path / "Modules" / import_symbol_name,
        ]

        # Add parent directories of the Swift file path
        for parent in importer_file_path.parents:
            search_paths.append(parent / import_path)

        # Add sibling directories of each directory component of importer_file_path
        for parent in importer_file_path.parents:
            for sibling in parent.iterdir():
                if sibling.is_dir() and sibling != importer_file_path:
                    search_paths.append(sibling / import_path)

        # Heuristic search for source files corresponding to the imported modules
        result_files = []

        for path in search_paths:
            extension_list = Repository.code_file_extensions[Language.Swift]
            if (
                path.is_relative_to(self.repo.repo_path)
                and path.exists()
                and path.is_dir()
            ):
                for ext in extension_list:
                    for swift_file in path.glob(f"**/*{ext}"):
                        result_files.append(swift_file)

        # Return list of Path objects corresponding to the imported files
        return result_files

    def resolve_rust_import(
        self, import_symbol_node: ParseTreeInfo, importer_file_path: Path
    ) -> List[Path]:
        def find_import_path(
            project_root: Path,
            file: Path,
            module_path: List[str],
        ) -> Optional[Path]:
            """
            Given the project root, the file containing the import, and the module path,
            heuristically find the corresponding file path for the imported module.

            :param project_root: The root directory of the Rust project.
            :param file: The file (pathlib.Path) containing the import statement.
            :param module_path: A list of module components (e.g., ["my_module", "sub_module"]).
            :return: The pathlib.Path object for the corresponding file or None if not found.
            """
            # Determine if the import is absolute
            is_absolute = module_path[0] == "crate"

            if is_absolute:
                module_path = module_path[1:]  # Remove the leading "crate"
                # Start from the project root if the path is absolute
                current_dir = project_root / "src"
            elif module_path[0] == "super":
                module_path = module_path[1:]  # Remove the leading "super"
                # Start from the file's parent if the path is super
                current_dir = file.parent
            else:
                current_dir = file.parent

            for i, part in enumerate(module_path):
                if part == "*":
                    break

                # Check if the module is a directory with a mod.rs or a file <module_name>.rs
                dir_path = current_dir / part
                mod_file_path = dir_path / "mod.rs"
                file_path = current_dir / f"{part}.rs"

                if (
                    mod_file_path.is_relative_to(self.repo.repo_path)
                    and mod_file_path.exists()
                ):
                    current_dir = dir_path
                    # If it is the last part or the next part is star, return the mod.rs
                    if i == len(module_path) - 1 or module_path[i + 1] == "*":
                        return mod_file_path

                elif (
                    file_path.is_relative_to(self.repo.repo_path) and file_path.exists()
                ):
                    return file_path

                else:
                    # If not found, check further up the directory hierarchy for relative imports
                    if not is_absolute:
                        found = False
                        for ancestor in current_dir.parents:
                            ancestor_dir_path = ancestor / part
                            ancestor_mod_file_path = ancestor_dir_path / "mod.rs"
                            ancestor_file_path = ancestor / f"{part}.rs"

                            if not ancestor_file_path.is_relative_to(
                                self.repo.repo_path
                            ):
                                continue

                            if ancestor_mod_file_path.exists():
                                # If it is the last part or the next part is star, return the mod.rs
                                if (
                                    i == len(module_path) - 1
                                    or module_path[i + 1] == "*"
                                ):
                                    return mod_file_path

                                current_dir = ancestor_dir_path
                                found = True
                                break
                            elif ancestor_file_path.exists():
                                return ancestor_file_path

                        if not found:
                            return None
                    else:
                        return None

            last_part = module_path[-1]
            if module_path[-1] == "*" and len(module_path) > 1:
                last_part = module_path[-2]
            # If we reach here, assume the last module part is a file
            final_file = current_dir / f"{last_part}.rs"
            if final_file.is_relative_to(self.repo.repo_path) and final_file.exists():
                return final_file
            else:
                return None

        if import_symbol_node.type == "scoped_use_list":
            """
            Parse the import statement for a scoped use list
            e.g. Parsing `use super::{sub_module::sub_function as sub, bar::*};`, we should split the use list into
            the following module_path:
            - `['super', 'sub_module', 'sub_function']`
            - `[super', 'bar', '*']`

            Then we can attempt to find the imported file based on heuristics
            """

            def parse_scoped_use_list(statement):
                # Deal with multi-line use list
                statement = statement.replace("\n", "")
                # Regular expression to match the scoped use list
                pattern = r"([\w:]+)::({.*})"

                # Find matches
                matches = re.findall(pattern, statement)

                result = []
                for base, scoped in matches:
                    # Remove curly braces and split by comma
                    scoped_items = scoped.strip("{}").split(", ")
                    for item in scoped_items:
                        # Remove alias (e.g. " as sub") and split by "::"
                        module_path = base.split("::") + [
                            i.strip().split(" as ")[0] for i in item.split("::")
                        ]
                        result.append(module_path)

                return result

            scoped_use_list_module_paths = parse_scoped_use_list(
                import_symbol_node.text
            )

            imported_files = []
            # Attempt to find the imported file based on heuristics
            for module_path in scoped_use_list_module_paths:
                imported_file = find_import_path(
                    self.repo.repo_path, importer_file_path, module_path
                )
                if imported_file:
                    imported_files.append(imported_file)

            return imported_files
        else:
            # Decode the symbol name and split into module path components
            import_symbol_name = import_symbol_node.text
            module_path = import_symbol_name.split("::")
            # Attempt to find the imported file based on heuristics
            imported_file = find_import_path(
                self.repo.repo_path, importer_file_path, module_path
            )
            return [imported_file] if imported_file else []

    def resolve_lua_import(
        self,
        import_symbol_node: Union[ParseTreeInfo, RegexInfo],
        importer_file_path: Path,
    ) -> List[Path]:
        import_symbol_name = import_symbol_node.text

        import_symbol_name = import_symbol_name.strip('"').strip("'")
        extension_list = Repository.code_file_extensions[Language.Lua]

        # Here, we make sure in case of `dofile("module4.lua")`, the `.lua` suffix is preserved.
        if all(ext not in import_symbol_name for ext in extension_list):
            """
            In case of `require("submodule.module2")`, the Lua file is expected to be located at
            `submodule/module2.lua`. But before replacing the `.` with `/`, we need to make sure in case of
            `dofile("module4.lua")`, the `.lua` suffix is preserved.
            """
            import_symbol_name = import_symbol_name.replace(".", os.sep)

        resolved_path = importer_file_path.parent / import_symbol_name

        if resolved_path.is_relative_to(self.repo.repo_path) and resolved_path.exists():
            return [resolved_path]

        for ext in extension_list:
            path = resolved_path.with_suffix(ext)
            if path.is_relative_to(self.repo.repo_path) and path.exists():
                return [resolved_path.with_suffix(ext)]
        return []

    def resolve_bash_import(
        self,
        import_symbol_node: Union[ParseTreeInfo, RegexInfo],
        importer_file_path: Path,
    ) -> List[Path]:
        import_symbol_name = import_symbol_node.text
        import_path = self._Path(import_symbol_name)
        if import_path.is_relative_to(self.repo.repo_path) and import_path.exists():
            return [self._Path(import_symbol_name)]
        else:
            resolved_path = importer_file_path.parent / import_symbol_name
            if (
                resolved_path.is_relative_to(self.repo.repo_path)
                and resolved_path.exists()
            ):
                return [resolved_path]
        return []

    def resolve_r_import(
        self,
        import_symbol_node: Union[ParseTreeInfo, RegexInfo],
        importer_file_path: Path,
    ) -> List[Path]:
        import_symbol_name = import_symbol_node.text
        import_symbol_name = import_symbol_name.strip('"').strip("'")

        import_path = self._Path(import_symbol_name)
        if import_path.is_relative_to(self.repo.repo_path) and import_path.exists():
            return [self._Path(import_symbol_name)]
        else:
            resolved_path = importer_file_path.parent / import_symbol_name
            if (
                resolved_path.is_relative_to(self.repo.repo_path)
                and resolved_path.exists()
            ):
                return [resolved_path]
        return []
