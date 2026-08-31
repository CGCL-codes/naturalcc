"""把 RAG 解析出的 C 项目符号图导入 Neo4j。

配置（环境变量）：
    NEO4J_URI       默认 bolt://localhost:7687
    NEO4J_USER      默认 neo4j
    NEO4J_PASSWORD  必填，无默认值
    CGRAPH_JSON     待导入的符号图 JSON 路径
"""
import json
import os
import sys

from py2neo import Graph, Node, Relationship

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
GRAPH_JSON = os.environ.get("CGRAPH_JSON")

if not NEO4J_PASSWORD:
    sys.exit("请设置环境变量 NEO4J_PASSWORD。")
if not GRAPH_JSON:
    sys.exit("请设置环境变量 CGRAPH_JSON，指向待导入的符号图 JSON 文件。")

# 连接 Neo4j
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

with open(GRAPH_JSON, "r") as f:
    data = json.load(f)

for file_path, symbols in data.items():
    module_node = Node("Module", name=file_path, file_path=symbols[file_path]["file_path"])
    graph.merge(module_node, "Module", "name")

    for sym_name, sym_data in symbols.items():
        if sym_name == file_path:
            continue
        node = Node(sym_data["type"], name=sym_name, defn=sym_data.get("def"), sline=sym_data.get("sline"))
        graph.merge(node, sym_data["type"], "name")
        graph.merge(Relationship(module_node, "DECLARES", node))

        # include 关系
        if "include" in sym_data:
            for inc in sym_data["include"]:
                if inc:
                    inc_node = Node("Module", name=inc)
                    graph.merge(inc_node, "Module", "name")
                    graph.merge(Relationship(module_node, "INCLUDE", inc_node))

        # rels 关系
        if "rels" in sym_data:
            for target, _, rel_type in sym_data["rels"]:
                target_node = Node("Symbol", name=target)
                graph.merge(target_node, "Symbol", "name")
                graph.merge(Relationship(node, rel_type.upper(), target_node))
