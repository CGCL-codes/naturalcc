import argparse
import json
import logging
from dataclasses import dataclass, field
from multiprocessing import Lock, Manager
from pathlib import Path
from typing import Optional
from dataclasses_json import dataclass_json, config
from joblib import Parallel, delayed
from tqdm import tqdm

from dependency_graph import (
    DependencyGraph,
    Repository,
    Language,
    construct_dependency_graph,
    GraphGeneratorType,
)
from dependency_graph.models.graph_data import Node, Edge, NodeType


@dataclass_json
@dataclass
class RetrievedChunk:
    def __hash__(self):
        return hash(f"{self.filename}:{self.retrieved_chunk}")

    retrieved_chunk: str
    filename: str
    # In degree of the retrieved node
    score: float
    node_type: str
    relation: str


@dataclass_json
@dataclass
class CrossfileDefinitionByDependencyGraph:
    text: str = (
        "# Here are some relevant code fragments from other files of the repo:\n"
    )
    _list: list[RetrievedChunk] = field(
        default_factory=list, metadata=config(field_name="list")
    )


@dataclass_json
@dataclass
class CrossfileReferenceByDependencyGraph:
    text: str = (
        "# Here are some relevant code fragments from other files of the repo:\n"
    )
    _list: list[RetrievedChunk] = field(
        default_factory=list, metadata=config(field_name="list")
    )


def construct_cross_file_definition_context(
    graph: DependencyGraph,
    repo_path: Path,
    file_path: Path,
    start_line: int,
    language: str,
) -> CrossfileDefinitionByDependencyGraph:
    """
    Construct the context of a file by the Repo-Specific Semantic Graph.
    """
    edge_list: list[tuple[Node, Node, Edge]] = (
        graph.as_retriever().get_cross_file_definition_by_line(file_path, start_line)
    )

    context_list = []

    for edge in edge_list:
        # Determine the retrieved_chunk content based on conditions
        if edge[1].type != NodeType.FUNCTION:
            retrieved_chunk_content = edge[1].get_stub(Language(language))
            if retrieved_chunk_content is None:
                retrieved_chunk_content = edge[1].get_text()
        else:
            retrieved_chunk_content = edge[1].get_text()

        # Create a RetrievedChunk object
        context = RetrievedChunk(
            retrieved_chunk=retrieved_chunk_content,
            filename=str(edge[1].location.file_path.relative_to(repo_path)),
            score=graph.graph.in_degree(edge[1]),
            node_type=edge[1].type.value,
            relation=edge[2].relation.name,
        )

        # Append the RetrievedChunk object to the context_list
        context_list.append(context)

    # Remove duplicate
    context_list = list(set(context_list))

    return CrossfileDefinitionByDependencyGraph(_list=context_list)


def construct_cross_file_reference_context(
    graph: DependencyGraph,
    repo_path: Path,
    file_path: Path,
    start_line: int,
) -> CrossfileReferenceByDependencyGraph:
    """
    Construct the context of a file by the Repo-Specific Semantic Graph.
    """
    edge_list: list[tuple[Node, Node, Edge]] = (
        graph.as_retriever().get_cross_file_reference_by_line(file_path, start_line)
    )
    if not edge_list:
        edge_list = []

    context_list = [
        RetrievedChunk(
            # Generate the stub only if the node is not a function
            retrieved_chunk=edge[1].get_text(),
            filename=str(edge[1].location.file_path.relative_to(repo_path)),
            score=graph.graph.in_degree(edge[1]),
            node_type=edge[1].type.value,
            relation=edge[2].relation.name,
        )
        for edge in edge_list
    ]
    # Remove duplicate
    context_list = list(set(context_list))

    return CrossfileReferenceByDependencyGraph(_list=context_list)


def process_data(
    data: dict,
    repository_suite_path: Path,
    language: str,
    output_path: Path,
    dependency_graph_dict: dict[str, DependencyGraph],
    lock: Lock,
    dependency_graph_suite_path: Optional[Path] = None,
    line_number_start_from_1: Optional[bool] = False,
):
    repository = data["metadata"]["repository"]
    file = data["metadata"]["file"]
    groundtruth_start_lineno = data["metadata"]["groundtruth_start_lineno"]

    # repoEval starts from 1 already
    if not line_number_start_from_1:
        # Convert to 1-based
        groundtruth_start_lineno += 1

    repo_path = repository_suite_path / f"{repository}"

    if language == "python":
        dependency_graph_generator = GraphGeneratorType.JEDI
    elif language in {"java", "typescript", "javascript", "c_sharp"}:
        dependency_graph_generator = GraphGeneratorType.TREE_SITTER
    else:
        raise ValueError(f"Unsupported language: {language}")

    with lock:
        if repository not in dependency_graph_dict:
            if dependency_graph_suite_path and dependency_graph_suite_path.is_dir():
                dependency_graph_path = (
                    dependency_graph_suite_path / f"{repository}.json"
                )
                dependency_graph_dict[repository] = DependencyGraph.from_json(
                    dependency_graph_path.read_text()
                )
            else:
                repo = Repository(repo_path, Language(language))
                try:
                    dependency_graph_dict[repository] = construct_dependency_graph(
                        repo, dependency_graph_generator
                    )
                except FileNotFoundError:
                    pass

    file_path = repo_path / f"{file}"
    if not file_path.exists() and "fpath_tuple" in data["metadata"]:
        # Remove first component in fpath_tuple
        file_path = repo_path / Path(*data["metadata"]["fpath_tuple"][1:])

    try:
        graph = dependency_graph_dict[repository]
        def_context = construct_cross_file_definition_context(
            graph, repo_path, file_path, groundtruth_start_lineno, language
        )
        ref_context = construct_cross_file_reference_context(
            graph, repo_path, file_path, groundtruth_start_lineno
        )
        data["crossfile_definition_by_dependency_graph"] = def_context.to_dict()
        data["crossfile_reference_by_dependency_graph"] = ref_context.to_dict()
    except Exception as e:
        logging.error(f"Error occurred when processing {file_path}: {e}")
        data["crossfile_definition_by_dependency_graph"] = (
            CrossfileDefinitionByDependencyGraph().to_dict()
        )
        data["crossfile_reference_by_dependency_graph"] = (
            CrossfileReferenceByDependencyGraph().to_dict()
        )

    with output_path.open("a") as f:
        f.write(json.dumps(data) + "\n")


def main(
    data_path: Path,
    repository_suite_path: Path,
    language: str,
    output_path: Path,
    max_workers: int = 8,
    dependency_graph_suite_path: Optional[Path] = None,
    line_number_start_from_1: Optional[bool] = False,
) -> None:
    """
    Construct CrossCodeEval data from code Repo-Specific Semantic Graph
    :param data_path:
    :param repository_suite_path:
    :param language:
    :param output_path:
    :param max_workers:
    :param dependency_graph_suite_path: if provided, load pre-generated Repo-Specific Semantic Graph
    from `dependency_graph_suite_path/{repository}.json`
    :return:
    """
    cceval_data = data_path.read_text().splitlines()

    # Use a Manager to create a shared dictionary and lock for multi-processing
    with (
        Manager() as manager,
        tqdm(desc=f"Processing {data_path.name} with language {language}") as pbar,
    ):
        dependency_graph_dict = manager.dict()
        lock = manager.Lock()

        # Use joblib to process each line in the batch
        Parallel(n_jobs=max_workers, verbose=10)(
            delayed(process_data)(
                json.loads(d),
                repository_suite_path,
                language,
                output_path,
                dependency_graph_dict,
                lock,
                dependency_graph_suite_path,
                line_number_start_from_1,
            )
            for d in cceval_data
        )
        pbar.update()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct CrossCodeEval data from code Repo-Specific Semantic Graph"
    )
    parser.add_argument(
        "-d",
        "--data-path",
        help="The CrossCodeEval data path.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-r",
        "--repository-suite-path",
        help="The CrossCodeEval code repository suite path.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-g",
        "--dependency-graph-suite-path",
        help="The Repo-Specific Semantic Graph suite path.",
        required=False,
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-l",
        "--language",
        choices=["python", "java", "typescript", "javascript", "c_sharp"],
        help="Repository language.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output-path",
        help="The output path.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-j",
        "--jobs",
        help="The maximum number of workers.",
        required=False,
        type=int,
        default=8,
    )
    return parser.parse_args()


if __name__ == "__main__":
    # args = parse_args()
    # data_path = args.data_path
    # repository_suite_path = args.repository_suite_path
    # language = args.language
    # output_path = args.output_path
    # jobs = args.jobs
    # dependency_graph_suite_path = args.dependency_graph_suite_path

    data_files = [
        "/home/wanyao/talentan/cceval/data/python/line_completion.jsonl"
    ]
    output_files = [
        "/home/wanyao/talentan/RepoFuse/data/cross_code_eval/line_completion_dependency_graph.jsonl",
    ]
    
    repository_suite_path = Path("/home/wanyao/talentan/cceval/raw_data/crosscodeeval_rawdata")
    language = "python"
    jobs = 10
    dependency_graph_suite_path = None

    for data_file, output_file in zip(data_files, output_files):
        data_path = Path(data_file)
        output_path = Path(output_file)
        
        main(
            data_path,
            repository_suite_path,
            language,
            output_path,
            jobs,
            dependency_graph_suite_path,
        )