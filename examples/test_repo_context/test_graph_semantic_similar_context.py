import os
import json
from tqdm import tqdm
from networkx.readwrite import json_graph
from ncc.utils.code_util.python.repo_level_code_completion.graph_semantic_similar_context.ccg.utils import CONSTANTS, CodeGenTokenizer
from ncc.utils.code_util.python.repo_level_code_completion.graph_semantic_similar_context.ccg.slicing import Slicing
from ncc.utils.code_util.python.repo_level_code_completion.graph_semantic_similar_context.ccg.ccg import create_graph
from ncc.utils.code_util.python.repo_level_code_completion.graph_semantic_similar_context.ccg.utils import iterate_repository_file, make_needed_dir, set_default, dump_jsonl, graph_to_json
from concurrent.futures import ThreadPoolExecutor, as_completed
from ncc.utils.code_util.python.repo_level_code_completion.graph_semantic_similar_context.build_graph_database import *

from concurrent.futures import ThreadPoolExecutor
import os
from ncc.utils.code_util.python.repo_level_code_completion.graph_semantic_similar_context.ccg.utils import CONSTANTS, dump_jsonl, json_to_graph, CodeGenTokenizer, load_jsonl, make_needed_dir
import copy
import networkx as nx
import queue
import Levenshtein
import argparse
import time
from ncc.utils.code_util.python.repo_level_code_completion.graph_semantic_similar_context.ccg.metrics import hit
from functools import partial
from ncc.utils.code_util.python.repo_level_code_completion.graph_semantic_similar_context.build_query_graph import build_query_subgraph
from ncc.utils.code_util.python.repo_level_code_completion.graph_semantic_similar_context.search_code import *
if __name__ == '__main__':
    graph_db_builder = GraphDatabaseBuilder()
    repos = CONSTANTS.repos
    # 使用线程池并行处理多个仓库
    with ThreadPoolExecutor(max_workers=60) as executor:
        future_to_repo = {executor.submit(graph_db_builder.build_slicing_graph_database, repo): repo for repo in repos}
        for future in as_completed(future_to_repo):
            repo = future_to_repo[future]
            try:
                future.result()
            except Exception as exc:
                print(f"仓库 {repo} 处理出错: {exc}")
            else:
                print(f"仓库 {repo} 处理完成。")


    build_query_subgraph("line_completion.jsonl")
    query_cases = load_jsonl(os.path.join(CONSTANTS.query_graph_save_dir, "line_completion.jsonl"))
    
    # modes = ['coarse', 'fine', 'coarse2fine']
    # gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    modes = ['coarse2fine']
    gammas = [0.1]

    for mode in modes:
        for gamma in gammas:
            save_path = os.path.join(f"{CONSTANTS.search_res_dir}/line_completion.{mode}.{int(gamma*100)}.search_res.jsonl")
            make_needed_dir(save_path)

            all_start_time = time.time()
            searcher = CodeSearchWorker(query_cases, save_path, mode, gamma=gamma)
            searcher.run()
            all_end_time = time.time()

            running_time = all_end_time - all_start_time
            search_cases = load_jsonl(save_path)
            hit1, hit5, hit10 = hit(search_cases, hits=[1, 5, 10])

            print('-'*20 + "Parameters" + '-'*20)
            print(f'mode: {mode}')
            print(f'gamma: {gamma}')
            print('-' * 20 + "Results" + '-' * 20)
            print(f'save_path: {save_path}')
            print(f'hit1: {hit1:.4f}')
            print(f'hit5: {hit5:.4f}')
            print(f'hit10: {hit10:.4f}')
            print(f'runtime: {running_time:.4f}\n')
