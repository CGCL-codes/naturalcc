import json
from pathlib import Path

def process_files(completion_path, search_res_path, output_path):
    # 构建task_id到search_res条目的映射
    search_data = {}
    with open(search_res_path) as f:
        for line in f:
            entry = json.loads(line)
            task_id = entry["metadata"]["task_id"]
            search_data[task_id] = entry["top_k_context"]

    # 处理line_completion.jsonl
    output = []
    with open(completion_path) as f:
        for line in f:
            data = json.loads(line)
            task_id = data["metadata"]["task_id"]
            
            # 提取并转换crossfile_context
            cross_list = []
            for item in search_data.get(task_id, []):
                # 处理filename路径
                code_snippet, trigger_line, key_forward_context, file_paths, line_num, score = item
                
                cross_list.append({
                    "retrieved_chunk": code_snippet,
                    "filename": file_paths,
                    "score": score
                })
            
            # 添加新字段
            data["crossfile_context"] = {"list": cross_list}
            output.append(data)

    # 写入新文件
    with open(output_path, "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")

# 使用示例
# process_files(
#     "/home/wanyao/talentan/RepoFuse/data_mini/python/line_completion.jsonl",
#     "/home/wanyao/talentan/RepoFuse/data_mini/python/search_res/line_completion.coarse.10.search_res.jsonl",
#     "/home/wanyao/talentan/RepoFuse/data_mini/python/line_completion_rg1_subgraph_sim.jsonl"
# )


import sys
sys.path.extend([
    "/home/wanyao/talentan",
    "/home/wanyao/talentan/cceval",
    "/home/wanyao/talentan/cceval/prompt_builder"
])
import json
from pathlib import Path
from collections import defaultdict
from cceval.prompt_builder.rerank_utils import SemanticReranking

QUERY_LENGTH = 15

def merge_crossfile_contexts(completion_path, strategy_paths, output_path):
    # 创建多层存储结构：task_id -> 检索方法 -> 检索结果列表
    strategy_data = defaultdict(lambda: defaultdict(list))
    MODEL_CONFIGS = [
        {"name": "unixcoder", "field": "unixcoder_score", "model_type": "unixcoder", "max_length": 256},
        {"name": "codebert", "field": "codebert_score", "model_type": "codebert", "max_length": 256}
    ]
    
    # 初始化多个模型
    model_rankers = {
        cfg["name"]: SemanticReranking(
            model_type=cfg["model_type"],
            max_sequence_length=cfg["max_length"]
        )
        for cfg in MODEL_CONFIGS
    }

    # 第一阶段加载策略结果
    for strategy_name, file_path in strategy_paths.items():
        with open(file_path) as f:
            for line in f:
                entry = json.loads(line)
                task_id = entry["metadata"]["task_id"]
                
                # 根据策略名称选择字段
                context_field = "crossfile_definition_by_dependency_graph" if strategy_name == "Related" else "crossfile_context"
                
                converted_items = []
                for item in entry.get(context_field, {}).get("list", []):
                    converted_items.append({
                        "retrieved_chunk": item["retrieved_chunk"],
                        "filename": item["filename"],
                        "score": item["score"]
                    })
                
                strategy_data[task_id][strategy_name] = converted_items

    # 第二阶段：合并到主文件
    merged_records = []
    with open(completion_path) as f:
        for line in f:
            record = json.loads(line)
            task_id = record["metadata"]["task_id"]
            
            prompt_lines = record["prompt"].split('\n')[-QUERY_LENGTH:]
            query_text = '\n'.join(prompt_lines)

            # 保留原始crossfile_context结构
            merged_context = record.get("crossfile_context", {})
            
            # 添加各策略的检索结果
            merged_context["strategies"] = {
                strategy: strategy_data[task_id].get(strategy, [])
                for strategy in strategy_paths.keys()
            }
            
            record["crossfile_context"] = merged_context

            all_queries = []
            all_chunks = []
            strategy_chunk_map = []
            # 修改chunk处理部分（移除分块逻辑）
            for strategy_name in strategy_paths.keys():
                strategy_chunks = record["crossfile_context"]["strategies"][strategy_name]
                for idx, chunk in enumerate(strategy_chunks):
                    all_queries.append(query_text)
                    all_chunks.append(chunk["retrieved_chunk"])  # 直接使用完整代码片段
                    strategy_chunk_map.append((strategy_name, idx))
            
            # 修改分数计算方式（对齐rerank_utils的query处理）
            QUERY_LINES = 15  # 与rerank_utils保持一致的最后n行设置
            query_lines = [l for l in query_text.split('\n') if l.strip()][-QUERY_LINES:]
            aligned_query = '\n'.join(query_lines)
            
            # 修改模型计算部分
            if len(all_queries) > 0:
                for model_cfg in MODEL_CONFIGS:
                    ranker = model_rankers[model_cfg["name"]]
                    # 使用对齐后的query（最后n行）
                    _, _, scores = ranker.rerank(
                        query=aligned_query,  # 对齐后的查询
                        docs=all_chunks,      # 使用完整代码片段
                        gpu_id=5
                    )
                    
                    # 修改分数聚合方式（直接赋值，无需分块取最大值）
                    for (s_name, orig_idx), score in zip(strategy_chunk_map, scores):
                        record["crossfile_context"]["strategies"][s_name][orig_idx][model_cfg["field"]] = score
            merged_records.append(record)

    # 写入合并后的文件
    with open(output_path, "w") as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# 使用示例
base_path = Path("/home/wanyao/talentan/RepoFuse/data_mini/python")

strategy_files  = {
    "UnixCoder": base_path / "line_completion_rg1_unixcoder_cosine_sim.jsonl",
    "CodeBert": base_path / "line_completion_rg1_codebert_cosine_sim.jsonl",
    "Jaccard": base_path / "line_completion_rg1_jaccard_sim.jsonl",
    "Edit": base_path / "line_completion_rg1_edit_distance.jsonl",
    "BM25": base_path / "line_completion_rg1_bm25.jsonl",
    "TF_IDF": base_path / "line_completion_rg1_tfidf.jsonl",
    # "Graph": base_path / "line_completion_rg1_subgraph_sim.jsonl",
    "Related": base_path / "line_completion_rg1_related.jsonl"
}

merge_crossfile_contexts(
    completion_path=base_path / "line_completion.jsonl",
    strategy_paths=strategy_files,
    output_path=base_path / "merged_line_completion.jsonl"
)