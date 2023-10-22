from typing import Tuple

import editdistance
import math

from .nx_graphs import compute_ged, convert_ast_to_graph, convert_dict_to_graph
from .util import tokenize_tranx, create_ast, create_graph
from metrics_evaluation.metrics.codebleu.bleu_score import compute_bleu


def string_similarity(sample: str, reference: str) -> float:
    sample_tokens = tokenize_tranx(sample)
    reference_tokens = tokenize_tranx(reference)
    sample_len = len(sample_tokens)
    reference_len = len(reference_tokens)
    if sample_len == 0 and reference_len == 0:
        return 1.0
    distance = editdistance.eval(sample_tokens, reference_tokens) / max(
        sample_len, reference_len
    )
    return 1.0 - distance


def tree_similarity(sample: str, reference: str, return_size=False):
    sample_tree = create_ast(sample)
    if sample_tree is None:
        return None

    reference_tree = create_ast(reference)
    if reference_tree is None:
        return None

    tree_edit_distance, total_size = compute_ged(
        convert_ast_to_graph(sample_tree),
        convert_ast_to_graph(reference_tree),
        use_edge_cost=False,
    )

    if (total_size == -1) or (total_size == 0):
        return None

    if return_size:
        return (1.0 - tree_edit_distance / total_size, total_size)

    return 1.0 - tree_edit_distance / total_size


def graph_similarity(sample: str, reference: str, return_size=False):
    sample_graph = create_graph(sample)
    if sample_graph is None:
        return None

    reference_graph = create_graph(reference)
    if reference_graph is None:
        return None

    graph_edit_distance, total_size = compute_ged(
        convert_dict_to_graph(sample_graph),
        convert_dict_to_graph(reference_graph),
        use_edge_cost=True,
    )

    if (total_size == -1) or (total_size == 0):
        return None

    if return_size:
        return (1.0 - graph_edit_distance / total_size, total_size)

    return 1.0 - graph_edit_distance / total_size


def bleu(sample: str, reference: str):
    hyp = [tokenize_tranx(sample.strip())]
    ref = [tokenize_tranx(reference.strip())]
    return compute_bleu([ref], hyp)


def ruby(sample: str, reference: str) -> Tuple[float, str]:
    graph_sim = graph_similarity(sample, reference)
    if graph_sim is not None:
        return graph_sim, "graph"
    tree_sim = tree_similarity(sample, reference)
    if tree_sim is not None:
        return tree_sim, "tree"
    return string_similarity(sample, reference), "string"


def rubybleu(sample: str, reference: str, weight: int, use_tree=False):
    if use_tree:
        ruby_sim = tree_similarity(sample, reference, return_size=True)
    else:
        ruby_sim = graph_similarity(sample, reference, return_size=True)
    bleu_sim = bleu(sample, reference)
    if ruby_sim is None:
        return round(bleu_sim[0], 4)
    else:
        print(ruby_sim, bleu_sim)
        score = ruby_sim[0] * math.exp(-weight / ruby_sim[1]) + bleu_sim[0] * (
            1 - math.exp(-weight / ruby_sim[1])
        )
        return score
