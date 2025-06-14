import traceback
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Union

from tqdm import tqdm

from dependency_graph.dependency_graph import DependencyGraph
from dependency_graph.graph_generator import BaseDependencyGraphGenerator
from dependency_graph.graph_generator.tree_sitter_generator.import_finder import (
    ImportFinder,
)
from dependency_graph.graph_generator.tree_sitter_generator.info import (
    RegexInfo,
    ParseTreeInfo,
)
from dependency_graph.graph_generator.tree_sitter_generator.resolve_import import (
    ImportResolver,
)
from dependency_graph.models import PathLike
from dependency_graph.models.graph_data import (
    Node,
    NodeType,
    Location,
    EdgeRelation,
    Edge,
)
from dependency_graph.models.language import Language
from dependency_graph.models.repository import Repository
from dependency_graph.utils.log import setup_logger
from dependency_graph.utils.read_file import read_file_to_string
from dependency_graph.utils.text import get_position

# Initialize logging
logger = setup_logger()


class TreeSitterDependencyGraphGenerator(BaseDependencyGraphGenerator):
    supported_languages: Tuple[Language] = (
        Language.Python,
        Language.Java,
        Language.CSharp,
        Language.TypeScript,
        Language.JavaScript,
        Language.Kotlin,
        Language.PHP,
        Language.Ruby,
        Language.C,
        Language.CPP,
        Language.Go,
        Language.Swift,
        Language.Rust,
        Language.Lua,
        Language.Bash,
        Language.R,
    )

    def __init__(self, max_lines_to_read: int = None):
        """
        Initialize TreeSitterDependencyGraphGenerator
        :param max_lines_to_read: The maximum number of lines to read from a file. Default is None.
        Tree-sitter parser may fail and more seriously, causes memory leak or deadlock if the file is too large.
        So if max_lines_to_read is set, the file is read by limited line to workaround ths.
        """
        self.max_lines_to_read = max_lines_to_read
        super().__init__()

    def read_file_to_string_with_limited_line(self, file_path: PathLike) -> str:
        # Use read_file_to_string here to avoid non-UTF8 decoding issue
        content = read_file_to_string(
            file_path, max_lines_to_read=self.max_lines_to_read
        )
        return content

    def generate_file(
        self,
        repo: Repository,
        code: str = None,
        file_path: PathLike = None,
    ) -> DependencyGraph:
        raise NotImplementedError("generate_file is not implemented")

    def generate(self, repo: Repository) -> DependencyGraph:
        D = DependencyGraph(repo.repo_path, repo.language)
        module_map: Dict[str, List[Path]] = defaultdict(list)
        # The key is (file_path, class_name)
        import_map: Dict[
            Tuple[Path, str], Union[List[ParseTreeInfo], List[RegexInfo]]
        ] = defaultdict(list)

        finder = ImportFinder(repo.language)
        resolver = ImportResolver(repo)

        for file_path in tqdm(repo.files, desc="Finding imports"):
            try:
                content = self.read_file_to_string_with_limited_line(file_path)
                name = finder.find_module_name(file_path)
                if name:
                    module_map[name].append(file_path)
                nodes = finder.find_imports(content)
                import_map[(file_path, name)].extend(nodes)
            except Exception as e:
                tb_str = "\n".join(traceback.format_tb(e.__traceback__))
                logger.error(
                    f"Error {e} finding import in {file_path}, will ignore: {tb_str}"
                )

        for (
            importer_file_path,
            importer_module_name,
        ), import_symbol_nodes in tqdm(import_map.items(), desc="Resolving imports"):
            for import_symbol_node in import_symbol_nodes:
                try:
                    resolved = resolver.resolve_import(
                        import_symbol_node, module_map, importer_file_path
                    )
                except Exception as e:
                    tb_str = "\n".join(traceback.format_tb(e.__traceback__))
                    logger.error(
                        f"Error {e} resolving import `{import_symbol_node.text}` in {importer_file_path}, will ignore: {tb_str}"
                    )
                    continue

                for importee_file_path in resolved:
                    # Use read_file_to_string here to avoid non-UTF8 decoding issue
                    importer_file_location = get_position(
                        read_file_to_string(
                            importer_file_path, max_lines_to_read=self.max_lines_to_read
                        )
                    )

                    importee_file_location = get_position(
                        read_file_to_string(
                            importee_file_path, max_lines_to_read=self.max_lines_to_read
                        )
                    )

                    importee_module_name = None
                    for module_name, file_paths in module_map.items():
                        if importee_file_path in file_paths:
                            importee_module_name = module_name
                            break

                    from_node = Node(
                        type=NodeType.MODULE,
                        name=importer_module_name,
                        location=Location(
                            file_path=importer_file_path,
                            start_line=importer_file_location[0][0],
                            start_column=importer_file_location[0][1],
                            end_line=importer_file_location[1][0],
                            end_column=importer_file_location[1][1],
                        ),
                    )
                    to_node = Node(
                        type=NodeType.MODULE,
                        name=importee_module_name,
                        location=Location(
                            file_path=importee_file_path,
                            start_line=importee_file_location[0][0],
                            start_column=importee_file_location[0][1],
                            end_line=importee_file_location[1][0],
                            end_column=importee_file_location[1][1],
                        ),
                    )
                    import_location = Location(
                        file_path=importer_file_path,
                        start_line=import_symbol_node.start_point[0] + 1,
                        start_column=import_symbol_node.start_point[1] + 1,
                        end_line=import_symbol_node.end_point[0] + 1,
                        end_column=import_symbol_node.end_point[1] + 1,
                    )
                    D.add_relational_edge(
                        from_node,
                        to_node,
                        Edge(
                            relation=EdgeRelation.Imports,
                            location=import_location,
                        ),
                        Edge(
                            relation=EdgeRelation.ImportedBy,
                            location=import_location,
                        ),
                    )

        return D
