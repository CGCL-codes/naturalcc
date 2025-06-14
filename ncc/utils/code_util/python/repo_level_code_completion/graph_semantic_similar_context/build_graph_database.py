import os
import json
from tqdm import tqdm
from networkx.readwrite import json_graph
from .ccg.utils import CONSTANTS, CodeGenTokenizer
from .ccg.slicing import Slicing
from .ccg.ccg import create_graph
from .ccg.utils import iterate_repository_file, make_needed_dir, set_default, dump_jsonl, graph_to_json
from concurrent.futures import ThreadPoolExecutor, as_completed

class GraphDatabaseBuilder:
    # 初始化方法，设置仓库基础目录和图数据库保存目录
    def __init__(self, repo_base_dir=CONSTANTS.repo_base_dir,
                 graph_database_save_dir=CONSTANTS.graph_database_save_dir):
        self.repo_base_dir = repo_base_dir
        self.graph_database_save_dir = graph_database_save_dir
        return

    # 构建完整的图数据库
    def build_full_graph_database(self, repo_name):
        # 获取代码文件列表
        code_files = iterate_repository_file(self.repo_base_dir, repo_name)
        file_num = 0
        # 创建所需的目录
        make_needed_dir(os.path.join(self.graph_database_save_dir, repo_name))
        with tqdm(total=len(code_files)) as pbar:
            for file in code_files:
                # 读取文件内容
                with open(file, 'r', encoding='utf-8') as f:
                    src_lines = f.readlines()
                # 创建图
                ccg = create_graph(src_lines, repo_name)
                if ccg is None:
                    pbar.update(1)
                    continue
                # 保存图到文件
                save_path = os.path.join(self.graph_database_save_dir, repo_name, f"{file_num}.json")
                file_num += 1
                make_needed_dir(save_path)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(json_graph.node_link_data(ccg), f, ensure_ascii=False, default=set_default)

                pbar.update(1)
        return

    # 构建切片图数据库
    def build_slicing_graph_database(self, repo_name):
        save_name = os.path.join(self.graph_database_save_dir, f"{repo_name}.jsonl")

        # 如果已经存在这个文件，就打印提示并直接返回，避免重复处理
        if os.path.exists(save_name):
            print(f"Skipping repo {repo_name} because {save_name} already exists.")
            return
        
        slicer = Slicing()
        repo_dict = []

        # 获取所有文件
        code_files = iterate_repository_file(self.repo_base_dir, repo_name)
        repo_base_dir_len = len(self.repo_base_dir.split('/'))
        tokenizer = CodeGenTokenizer()
        # 使用进度条显示进度
        with tqdm(total=len(code_files)) as pbar:
            for file in code_files:
                pbar.set_description(file)

                # print(f"开始处理文件：{file}")
                with open(file, 'r', encoding='utf-8') as f:
                    src_lines = f.readlines()
                # print(f"文件 {file} 内容读取完成")

                ccg = create_graph(src_lines, repo_name)
                if ccg is None:
                    pbar.update(1)
                    continue

                # 对每个语句进行切片
                for v in ccg.nodes:
                    if ccg.nodes[v]['nodeType'] == 'file_node':
                        continue
                    # print(f"正在处理节点 {v}（文件：{file}）")
                    # 初始化一个字典来存储当前节点的信息
                    curr_dict = dict()
                    
                    # 执行前向依赖切片，获取前向依赖上下文、行号列表和子图
                    forward_context, forward_line, forward_graph = slicer.forward_dependency_slicing(v, ccg, contain_node=False)
                    
                    # 将前向依赖子图转换为 JSON 格式并存储在字典中
                    curr_dict['key_forward_graph'] = graph_to_json(forward_graph)
                    
                    # 将前向依赖上下文存储在字典中
                    curr_dict['key_forward_context'] = forward_context
                    
                    # 对前向依赖上下文进行编码并存储在字典中
                    curr_dict['key_forward_encoding'] = tokenizer.tokenize(forward_context)
                    
                    # 将当前节点的源代码行拼接成字符串并存储在字典中
                    curr_dict['statement'] = "".join(ccg.nodes[v]['sourceLines'])
                    
                    # 获取当前节点的起始行号
                    statement_line_row = ccg.nodes[v]['startRow']
                    
                    # 计算上下文的起始行号和结束行号
                    start_line_row = max(0, statement_line_row - 5)
                    end_line_row = min(statement_line_row + 5, len(src_lines))
                    
                    # 将上下文的源代码行拼接成字符串并存储在字典中
                    curr_dict['val'] = "".join(src_lines[start_line_row:end_line_row])
                    
                    # 将文件路径转换为相对路径并存储在字典中
                    curr_dict['fpath_tuple'] = file.split('/')[repo_base_dir_len:]
                    
                    # 初始化最大前向行号
                    max_forward_line = 0
                    
                    # 如果前向行号列表不为空，获取其中的最大值
                    if len(forward_line) != 0:
                        max_forward_line = max(forward_line)
                    
                    # 将最大行号存储在字典中
                    curr_dict['max_line_no'] = max(max_forward_line, end_line_row)
                    
                    # 将当前节点的信息字典添加到 repo_dict 列表中
                    repo_dict.append(curr_dict.copy())
                pbar.update(1)

        # 创建所需的目录并保存结果
        make_needed_dir(save_name)
        dump_jsonl(repo_dict, save_name)
        return

if __name__ == '__main__':
    graph_db_builder = GraphDatabaseBuilder()
    repos = CONSTANTS.repos
    # target_repos = ["yoheinakajima-babyagi-d10b08c", "iryna-kondr-scikit-llm-38ad5e9", "DAMO-NLP-SG-MT-LLaMA-c72f7db", "RobertoCorti-gptravel-bcf49dd"]
    # for repo in target_repos:
    #     graph_db_builder.build_slicing_graph_database(repo)
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
