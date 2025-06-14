import argparse
from datetime import datetime
from pathlib import Path

from dependency_graph import (
    construct_dependency_graph,
    output_dependency_graph,
)
from dependency_graph.graph_generator import (
    GraphGeneratorType,
)
from dependency_graph.models.repository import Repository
from dependency_graph.utils.log import setup_logger

# Initialize logging
logger = setup_logger()


OUTPUT_FORMATS = ["edgelist", "pyvis", "ipysigma"]

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construct Repo-Specific Semantic Graph for a given project."
    )
    parser.add_argument(
        "-r",
        "--repo",
        type=Path,
        required=True,
        help="The path to a local repository.",
    )

    parser.add_argument(
        "-l", "--lang", help="The language of the parsed file.", required=True
    )

    parser.add_argument(
        "-g",
        "--graph-generator",
        type=GraphGeneratorType,
        default=GraphGeneratorType.JEDI,
        help=f"The code agent type to use. Should be one of the {[g.value for g in GraphGeneratorType]}. Defaults to {GraphGeneratorType.JEDI.value}.",
    )

    parser.add_argument(
        "-f",
        "--output-format",
        help="The format of the output.",
        default="edgelist",
        choices=OUTPUT_FORMATS,
        required=False,
    )

    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        help="The path to the output file. If not specified, will print to stdout.",
        default=None,
        required=False,
    )

    args = parser.parse_args()

    lang = args.lang
    graph_generator = args.graph_generator
    repo = Repository(args.repo, lang)
    output_file: Path = args.output_file
    output_format: str = args.output_format

    if output_file is not None and output_file.is_dir():
        raise IsADirectoryError(f"{output_file} is a directory.")

    start_time = datetime.now()
    graph = construct_dependency_graph(repo, graph_generator)
    end_time = datetime.now()

    elapsed_time = (end_time - start_time).total_seconds()
    logger.info(f"Finished constructing the Repo-Specific Semantic Graph in {elapsed_time} sec")

    output_dependency_graph(graph, output_format, output_file)
