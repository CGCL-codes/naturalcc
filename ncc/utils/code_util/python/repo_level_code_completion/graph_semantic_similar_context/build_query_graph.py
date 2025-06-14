import os
import copy
from tqdm import tqdm
import networkx as nx
from .ccg.ccg import create_graph
from .ccg.slicing import Slicing
from .ccg.utils import load_jsonl, make_needed_dir, graph_to_json, CONSTANTS, dump_jsonl, CodeGenTokenizer


def last_n_context_lines_graph(graph: nx.MultiDiGraph):
    """
    获取控制依赖图（CCG）中最后一个节点的前向依赖切片。

    参数:
    graph (nx.MultiDiGraph): 控制依赖图对象。

    返回:
    tuple: 包含前向依赖上下文、前向依赖行号列表和前向依赖图的元组。
    """
    max_line = 0
    last_node_id = 0
    slicer = Slicing()
    # 遍历图中的所有节点，找到行号最大的节点
    for v in graph.nodes:
        if graph.nodes[v]['startRow'] > max_line:
            max_line = graph.nodes[v]['startRow']
            last_node_id = v
    # 获取最后一个节点的前向依赖切片
    return slicer.forward_dependency_slicing(last_node_id, graph, contain_node=True)


def build_query_subgraph(task_name):
    """
    构建查询子图，并将其保存到指定路径。

    参数:
    task_name (str): 任务名称，对应数据集文件名。

    返回:
    None
    """
    test_cases = load_jsonl(os.path.join(CONSTANTS.dataset_dir, task_name))
    graph_test_cases = []
    tokenizer = CodeGenTokenizer()
    # 使用进度条显示进度
    with tqdm(total=len(test_cases)) as pbar:
        for case in test_cases:
            # 读取完整的查询上下文
            case_path = os.path.join(case['metadata']['repository'], case['metadata']['file'])
            line_no = case['metadata']['groundtruth_start_lineno']
            repo_name = case['metadata']['repository']
            if repo_name not in CONSTANTS.repos:
                continue
            full_path = os.path.join(CONSTANTS.repo_base_dir, case_path)
            # 如果文件不存在，则跳过
            if not os.path.exists(full_path):
                continue
            with open(full_path, 'r', encoding='utf-8') as f:
                src_lines = f.readlines()
            query_context = src_lines[:line_no]
            # 创建控制依赖图（CCG）
            ccg = create_graph(query_context, repo_name)
            # 获取最后一个节点的前向依赖切片
            query_ctx, query_line_list, query_graph = last_n_context_lines_graph(ccg)
            graph_case = dict()
            graph_case['query_forward_graph'] = graph_to_json(query_graph)
            graph_case['query_forward_context'] = query_ctx
            graph_case['query_forward_encoding'] = tokenizer.tokenize(query_ctx)
            context_lines = case['prompt'].splitlines(keepends=True)
            graph_case['context'] = context_lines
            graph_case['groundtruth'] = case['groundtruth']
            graph_case['metadata'] = copy.deepcopy(case['metadata'])
            graph_case['metadata']['forward_context_line_list'] = query_line_list
            graph_test_cases.append(copy.deepcopy(graph_case))
            pbar.update(1)

    save_path = os.path.join(CONSTANTS.query_graph_save_dir, task_name)
    make_needed_dir(save_path)
    dump_jsonl(graph_test_cases, save_path)
    return


if __name__ == "__main__":
    tasks_name = ["api_level.java.test", 
                  "line_level.java.test",
                  "api_level.python.test",
                  "line_level.python.test"]
    for task_name in tasks_name:
        build_query_subgraph(task_name)



