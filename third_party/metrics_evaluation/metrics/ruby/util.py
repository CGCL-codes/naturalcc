import tokenize
import ast
import re

from io import BytesIO
from typing import List, Optional, Union, Any, Dict

from metrics_evaluation.metrics.codebleu.graph_generator.graphgenerator import AstGraphGenerator
from metrics_evaluation.metrics.codebleu.graph_generator.type_lattice_generator import TypeLatticeGenerator


def tokenize_builtin(code: str) -> List[str]:
    try:
        tokens = list(tokenize.tokenize(BytesIO(code.encode("utf-8")).readline))[1:-1]
        tokens = [token.string for token in tokens]
        return tokens
    except tokenize.TokenError:
        return tokenize_tranx(code)


def tokenize_tranx(code: str) -> List[str]:
    """The tokenizer taken from https://github.com/pcyin/tranX
    Originally from Wang Ling et al.,
    Latent Predictor Networks for Code Generation (2016)
    @param code: string containing a code snippet
    @return: list of code tokens
    """
    code = re.sub(r"([^A-Za-z0-9_])", r" \1 ", code)
    code = re.sub(r"([a-z])([A-Z])", r"\1 \2", code)
    code = re.sub(r"\s+", " ", code)
    code = code.replace('"', "`")
    code = code.replace("'", "`")
    tokens = [t for t in code.split(" ") if t]

    return tokens


def create_ast(code: str) -> Optional[ast.AST]:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def create_graph(code: str) -> Optional[Dict]:
    try:
        lattice = TypeLatticeGenerator("build/typingRules.json")
        generator = AstGraphGenerator(code, lattice)
        graph = generator.build()
        return graph
    except SyntaxError:
        return None


def get_ast_children(node: Union[Any, ast.AST]) -> List[Union[Any, ast.AST]]:
    if not isinstance(node, ast.AST):
        return []

    def wrap(node_field: Union[list, Union[Any, ast.AST]]) -> List[Union[Any, ast.AST]]:
        if isinstance(node_field, list):
            return node_field
        return [node_field]

    children = [child for field in node._fields for child in wrap(getattr(node, field))]
    return children


def get_ast_node_label(node: Union[Any, ast.AST]) -> str:
    if not isinstance(node, ast.AST):
        return str(node)
    return str(type(node))


def ast_labels_distance(label1: str, label2: str) -> float:
    if label1 == label2:
        return 0.0
    return 1.0


def get_ast_size(node: Union[str, ast.AST]) -> int:
    if isinstance(node, int):
        return 1
    return sum(1 for _ in ast.walk(node))
