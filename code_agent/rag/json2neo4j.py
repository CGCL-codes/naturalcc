import json
from py2neo import Graph, Node, Relationship

# 连接 Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

with open("ccoder/CEval/c_graph/t113-system-prj.json", "r") as f:
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
