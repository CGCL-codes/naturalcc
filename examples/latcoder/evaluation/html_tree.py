from bs4 import BeautifulSoup
from graphviz import Digraph
from bs4.element import Tag
from typing import Union


class HTMLMulNode:
    def __init__(self, name):
        self.childs = []
        self.name = name
        self.parent = None
        self.depth = 0

    def add_child(self, ch):
        self.childs.append(ch)
        ch.parent = self
        ch.depth = self.depth + 1      

def html2tree(html: str, drop_leaves=True):
    soup = BeautifulSoup(html, "html.parser")
    nodes = []

    def dfs(html_element: Tag, parent: HTMLMulNode):
        name = html_element.name if html_element.name else str(html_element.strip())
        new_node = HTMLMulNode(name)
        nodes.append(new_node)
        if parent is None:
            new_node.depth = 0
        else:
            parent.add_child(new_node)
        if html_element.name and html_element.contents:
            for child in html_element.contents:
                if child and str(child).strip() and html_element is not child:
                    if not (drop_leaves and child.name is None):
                        dfs(child, new_node)
        return new_node

    if soup.html:
        dfs(soup.html, None)         
    return nodes

def subtree_copy(src: HTMLMulNode, parent: HTMLMulNode, height=0):
    new_node = HTMLMulNode(src.name)
    if parent:
        parent.add_child(new_node)
    if height > 0:
        for ch in src.childs:
            subtree_copy(ch, new_node, height - 1)
    return new_node


def tree2dot(tree: HTMLMulNode):
    node_id = [0]
    def dfs(root: Union[HTMLMulNode], graph: Digraph, id, pid):
        graph.node(str(id), root.name)
        if pid is not None:
            graph.edge(str(pid), str(id))
        childs = root.childs
        for child in childs:
            node_id[0] += 1
            dfs(child, graph, node_id[0], id)

    # Create a Digraph object
    dot = Digraph()
    # Traverse the BeautifulSoup object and build the graph
    dfs(tree, dot, node_id[0], None)
    return dot