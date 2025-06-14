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
from ncc.utils.code_util.python.repo_level_code_completion.structural_related_context.construct_cceval_data import *
from dependency_graph import (
    DependencyGraph,
    Repository,
    Language,
    construct_dependency_graph,
    GraphGeneratorType,
)
from dependency_graph.models.graph_data import Node, Edge, NodeType

if __name__ == "__main__":
    # args = parse_args()
    # data_path = args.data_path
    # repository_suite_path = args.repository_suite_path
    # language = args.language
    # output_path = args.output_path
    # jobs = args.jobs
    # dependency_graph_suite_path = args.dependency_graph_suite_path

    data_files = [
        "/home/wanyao/talentan/cceval/data_mini/python/line_completion.jsonl"
    ]
    output_files = [
        "/home/wanyao/talentan/naturalcc/examples/test_repo_context/line_completion_dependency_graph.jsonl",
    ]
    
    repository_suite_path = Path("/home/wanyao/talentan/cceval/data_mini/python/repos")
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