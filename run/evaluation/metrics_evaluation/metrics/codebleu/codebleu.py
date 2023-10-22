import tokenize
import re
import keyword
import json
from os.path import isfile

from .bleu_score import compute_bleu
import itertools
from collections import defaultdict
from tree_sitter import Language, Parser
from .graph_generator import graphgenerator as gg
from .graph_generator import graphgenutils as gu
from .graph_generator.type_lattice_generator import TypeLatticeGenerator
import networkx as nx
import math

from io import BytesIO
from typing import List, Optional, Union, Any

keyword_weight = 4  # not sure about the keyword weight: in CodeBLEU paper it is 5 in one place and 4 in another. Real keyword_weight is keyword_weight + 1 (see e.g. realization in _dictize)

# General functions
def _inlist(el, lst):
    if type(lst) == list:
        return el in lst
    else:
        return el == lst


# Data flow related part


def _split_identifier(
    identifier,
):  # The format in which the variables are stored in typilus is slightly unusual: they store variable as a str of format locationInCode_VariableName; it is convenient for me to break this string into a tuple
    lst = identifier.split("_", 1)
    lst[0] = int(lst[0])
    return lst


def _fix_identifier(identifier):
    lst = identifier.split("_", 1)
    lst[0] = int(lst[0]) - 1
    return str(lst[0]) + "_" + lst[1]


def _parse_occurences(occur_dict):
    definitions_list = []
    for key, value in occur_dict.items():
        curr_identifier = _split_identifier(key)
        for item in value:
            curr_identifier = _split_identifier(item)
            curr_identifier[0] -= 1  # to match the standard notation of the variables
            if curr_identifier not in definitions_list:
                definitions_list.append(curr_identifier)

    definitions_list.sort(key=lambda x: x[0])
    variables_list = [
        str(item[0]) + "_" + item[1] for item in definitions_list
    ]  # the part is basically to sort variables according to their location in code; gotta simplify that when won't be this tired
    variables_dict = dict()
    for i, item in enumerate(variables_list):
        variables_dict[item] = "var_" + str(
            i
        )  # renaming the variables to a common standard to be name-independent
    for key, value in occur_dict.items():
        if len(value) > 1:
            variables_dict[key] = [
                variables_dict[_fix_identifier(item)] for item in value
            ]
        else:
            variables_dict[key] = variables_dict[_fix_identifier(value[0])]
    return variables_dict


def _find_node(
    G, id
):  # to find a node in the graph by the corresponding variable id (same to location)
    for node in list(G.nodes):
        if G.nodes[node]["id"] == id:
            return node
    return None


def _find_edge(
    G, edge_data
):  # find the edge from the candidate graph in the reference graph
    edge_list = list(G.edges)
    edge_list.sort(key=lambda x: x[0])
    for edge in edge_list:
        if (G.nodes[edge[0]]["data"] == edge_data[0]) and (
            G.nodes[edge[1]]["data"] == edge_data[1]
        ):
            return edge
    return None


def _find_matching_variable(
    variables_dict, thekey
):  # if there are more than one variables with the same name in the code (e.g. we have several functions with same names of local variables), we wanna parse, to which variable do we refer now
    lookup = _split_identifier(thekey)
    match_var = [0, ""]
    for key, value in variables_dict.items():
        checked = _split_identifier(key)
        if (
            (lookup[1] == checked[1])
            and (lookup[0] >= checked[0])
            and (match_var[0] < checked[0])
        ):
            match_var = checked
    return [lookup[0], str(match_var[0]) + "_" + match_var[1]]


def _create_graph(next_use_dict, variables_dict, occur_dict):
    G = nx.DiGraph()
    nodes_list = []
    for key, value in occur_dict.items():
        toadd = _find_matching_variable(variables_dict, key)
        if toadd[1] != "0_":  # to disregard the attributes -- afaiu, we don't need them
            nodes_list.append(toadd)
    nodes_list.sort(key=lambda x: x[0])
    for i, item in enumerate(nodes_list):
        G.add_node(i, id=item[0], data=variables_dict[item[1]])
    for key, value in next_use_dict.items():
        from_node = _find_node(G, _split_identifier(key)[0])
        for item in value:
            to_node = _find_node(G, _split_identifier(item)[0])
            G.add_edge(from_node, to_node)
    return G


def _compare_graphs(
    ref_G, cand_G
):  # we look up edges from candidate one by one, if they are in the reference; if they are -- increase # of matches by one and pop the corresponding edge from list of ref edges
    matched = 0
    cand_edges = list(cand_G.edges)
    cand_edges.sort(key=lambda x: x[0])
    for edge in cand_edges:
        edge_data = (cand_G.nodes[edge[0]]["data"], cand_G.nodes[edge[1]]["data"])
        matched_edge = _find_edge(ref_G, edge_data)
        if matched_edge is not None:
            ref_G.remove_edge(*matched_edge)
            matched += 1
    return matched


def _fix_graph(g):  # pretty print by Egor Bogomolov
    g["token-sequence"] = [f"{ind}_{g['nodes'][ind]}" for ind in g["token-sequence"]]
    g["edges"] = {
        edge_type: {
            f"{v}_{g['nodes'][v]}": [f"{u}_{g['nodes'][u]}" for u in us]
            for v, us in g["edges"][edge_type].items()
        }
        for edge_type in g["edges"]
    }
    return g


def dfg_match(reference, candidate, lattice):  # dfg part of the metric
    try:
        ref_parsed = _fix_graph(gg.AstGraphGenerator(reference, lattice).build())
    except:
        return (
            -1
        )  # we return -1 if it is impossible to do some step for reference snippet and return 0 if it is possible for reference, but not for candidate
    try:
        cand_parsed = _fix_graph(gg.AstGraphGenerator(candidate, lattice).build())
    except:
        return 0
    if ("NEXT_USE" not in ref_parsed["edges"]) or (
        "OCCURRENCE_OF" not in ref_parsed["edges"]
    ):
        return -1
    if ("NEXT_USE" not in cand_parsed["edges"]) or (
        "OCCURRENCE_OF" not in cand_parsed["edges"]
    ):
        return 0
    ref_variables = _parse_occurences(ref_parsed["edges"]["OCCURRENCE_OF"])
    cand_variables = _parse_occurences(cand_parsed["edges"]["OCCURRENCE_OF"])
    ref_G = _create_graph(
        ref_parsed["edges"]["NEXT_USE"],
        ref_variables,
        ref_parsed["edges"]["OCCURRENCE_OF"],
    )
    cand_G = _create_graph(
        cand_parsed["edges"]["NEXT_USE"],
        cand_variables,
        cand_parsed["edges"]["OCCURRENCE_OF"],
    )
    if len(ref_G.edges) == 0:
        return -1
    if len(cand_G.edges) == 0:
        return 0
    max_match = len(ref_G.edges)
    return _compare_graphs(ref_G, cand_G) / max_match


# AST-related part
def _compare_nodes(node1, node2):
    return node1.type == node2.type


def _compare_ast(node1, node2):
    if not _compare_nodes(node1, node2):
        return False
    if len(node1.children) != len(node2.children):
        return False
    if len(node1.children) > 0:
        return all(itertools.starmap(_compare_ast, zip(node1.children, node2.children)))
    else:
        return True


def _find_subtrees(node, trees_list):
    trees_list.append(node)
    for item in node.children:
        _find_subtrees(item, trees_list)


def _comp_trees(
    ref_list, cand_list
):  # comparing list of subtrees of candidate and reference
    common_subtrees = 0
    for cand_str in cand_list:
        for ref_str in ref_list:
            if _compare_ast(ref_str, cand_str):
                common_subtrees += 1
                ref_list.remove(ref_str)
                break
    return common_subtrees


def ast_match(reference, candidate, parser):  # ast part of the metric
    try:
        ref_tree = parser.parse(bytes(reference, "utf8"))
    except:
        return -1
    try:
        cand_tree = parser.parse(bytes(candidate, "utf8"))
    except:
        return 0
    ref_list = []
    cand_list = []
    _find_subtrees(ref_tree.root_node, ref_list)
    _find_subtrees(cand_tree.root_node, cand_list)
    max_subtrees = len(ref_list)
    common_subtrees = _comp_trees(ref_list, cand_list)
    return common_subtrees / max_subtrees


# The BLEU-related part


def _dictize(
    lst, kwd
):  # a dictionary with keys being tokens and values being # of times the tokens appear in the snippet (weight of the token is taken into account)
    out_dict = dict()
    for item in lst:
        out_dict[item] = out_dict.get(item, 0) + 1 + keyword_weight * _inlist(item, kwd)
    return out_dict


def _token_overlap(dict_ref, dict_can):  # computing token overlap
    matches = 0
    for key in dict_ref:
        matches += min(dict_ref[key], dict_can.get(key, 0))
    return matches


def tokenize_builtin(code: str) -> List[str]:  # function by Egor Bogomolov
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


def pure_bleu(reference, candidate, max_order=4, smooth=False):
    return compute_bleu(
        [[tokenize_builtin(reference)]],
        [tokenize_builtin(candidate)],
        max_order,
        smooth,
    )[
        0
    ]  # for some reason we need this exact amount of brackets for compute_bleu to work; I don't fully understand why and this might be related to the issue with the wrong BLEU computation


def weighted_bleu(reference, candidate):  # the wighted bleu part of the metric
    keywords = []
    for e in dir(__builtins__):  # getting the list of keywords
        keywords.append(e)
    for e in keyword.kwlist:
        keywords.append(e)

    ref_tokens = tokenize_builtin(reference)
    can_tokens = tokenize_builtin(candidate)
    ref_dict = _dictize(ref_tokens, keywords)
    can_dict = _dictize(can_tokens, keywords)

    ratio = len(can_tokens) / len(ref_tokens)
    if ratio > 1.0:
        bp = 1.0
    else:
        if ratio == 0.0:
            bp = 0.0
        else:
            bp = math.exp(1 - 1.0 / ratio)

    possible_matches = 0
    for token in ref_tokens:
        possible_matches += 1 + keyword_weight * _inlist(token, keywords)
    matches = _token_overlap(ref_dict, can_dict)

    return matches * bp / possible_matches


def get_python_parser():
    if not isfile("build/my-languages.so"):
        Language.build_library(
            'build/my-languages.so',
            ['build/tree-sitter-python']
        )
    parser = Parser()
    PY_LANGUAGE = Language("build/my-languages.so", "python")
    parser.set_language(PY_LANGUAGE)
    return parser


def codebleu(reference, candidate, weights=[0.1, 0.1, 0.4, 0.4]):
    lattice = TypeLatticeGenerator("build/typingRules.json")
    parser = get_python_parser()
    scores = (
        pure_bleu(reference, candidate),
        weighted_bleu(reference, candidate),
        ast_match(reference, candidate, parser),
        dfg_match(reference, candidate, lattice),
    )
    final_score = 0.0
    norm = 0.0
    for i, item in enumerate(scores):
        if (
            item != -1
        ):  # if we can't compute some metric, we shouldn't include it in the score
            final_score += item * weights[i]
            norm += weights[i]
    final_score = final_score / norm
    return final_score
