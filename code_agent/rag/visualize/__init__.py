# -*- coding: utf-8 -*-
"""RAG knowledge graph generation and visualization package."""

from __future__ import annotations

# 图生成与可视化分成两个阶段，解析源码的入口由 generate_graph 提供。
# C 与 Java 使用各自的解析入口，但向可视化层交付同类图数据。
from code_agent.rag.visualize.generate_graph import (
    main as generate_graph_main,
    parse_c_project,
    parse_java_project,
)
# 可视化阶段负责加载图、转换节点边并生成 HTML，不在此入口执行任务。
from code_agent.rag.visualize.visualize import (
    generate_html,
    load_rag_graph,
    main as visualize_main,
    rag_to_vis,
)

# main 的别名区分两个命令行入口，避免导出时互相覆盖。
# 公开函数也可由后端直接调用，无需通过命令行重新启动进程。
__all__ = [
    "generate_graph_main",
    "generate_html",
    "load_rag_graph",
    "parse_c_project",
    "parse_java_project",
    "rag_to_vis",
    "visualize_main",
]
