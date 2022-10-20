from .node import Node


class AST:
    def __init__(self, nodes, indentation):
        self.size = len(nodes)
        self.indentation = indentation + 1
        self.nodes = {node["id"].split(".")[-1]: Node(node, self.indentation) for node in nodes}

    def __str__(self):
        indentation = self.indentation * "\t"
        nodes_str = ""

        for node in self.nodes:
            nodes_str += f"{indentation}{self.nodes[node]}"

        return f"\n{indentation}Size: {self.size}\n{indentation}Nodes:{nodes_str}"

    def get_nodes_type(self):
        return {n_id: node.type for n_id, node in self.nodes.items()}
