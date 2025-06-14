# 仓库上下文提取测试
本目录包含用于测试不同类型仓库上下文的代码。

## 目录结构
test_repo_context/
├── line_completion_dependency_graph.jsonl    # 依赖图生成的测试数据
├── line_completion_rg1_tfidf.jsonl           # TF-IDF相似性测试数据
├── test_common_similar_context.py            # 通用相似性上下文检索测试（支持
├── test_graph_semantic_similar_context.py    # 图语义结构相似性上下文检索测试
└── test_structural_related_context.py        # 结构相关上下文检索测试

## 测试类型

1. **通用相似性测试(test_common_similar_context.py)**
   - 基于文本相似度的上下文匹配
   - 支持UnixCoder\CodeBert\BM25\TF-IDF\编辑相似度等多种相似度检索方法
   - eg.测试命令：基于TF-IDF的上下文检索
python augment_with_cfc.py \
--language python \
--rerank True \
--ranker sparse \
--ranking_fn tfidf \
--crossfile_distance 100 \
--maximum_chunk_to_rerank 1000 \
--maximum_cross_files 1000 \
--maximum_cross_file_chunk 10 \
--use_next_chunk_as_cfc True \
--skip_if_no_cfc False \
--output_file_suffix rg1

2. **图语义相似性测试(test_graph_semantic_similar_context.py)**  
   - 基于代码语义图的相似性分析
   - 考虑变量、函数调用等语义关系
   - 测试命令：python test_graph_semantic_similar_context.py

3. **结构相关性测试(test_structural_related_context.py)**
   - 分析代码结构相关性
   - 考虑类继承、接口实现等结构关系
   - 测试命令：python test_structural_related_context.py
