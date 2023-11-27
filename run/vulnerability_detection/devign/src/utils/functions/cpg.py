from collections import OrderedDict
from ..objects.cpg.function import Function


def order_nodes(nodes, max_nodes):
    # sorts nodes by line and column

    nodes_by_column = sorted(nodes.items(), key=lambda n: n[1].get_column_number())
    nodes_by_line = sorted(nodes_by_column, key=lambda n: n[1].get_line_number())

    for i, node in enumerate(nodes_by_line):
        node[1].order = i

    if len(nodes) > max_nodes:
        print(f"CPG cut - original nodes: {len(nodes)} to max: {max_nodes}")
        return OrderedDict(nodes_by_line[:max_nodes])

    return OrderedDict(nodes_by_line)


def filter_nodes(nodes):
    return {n_id: node for n_id, node in nodes.items() if node.has_code() and
            node.has_line_number() and
            node.label not in ["Comment", "Unknown"]}


def parse_to_nodes(cpg, max_nodes=500):
    nodes = {}
    for function in cpg["functions"]:
        func = Function(function)
        # Only nodes with code and line number are selected
        filtered_nodes = filter_nodes(func.get_nodes())
        nodes.update(filtered_nodes)

    return order_nodes(nodes, max_nodes)
