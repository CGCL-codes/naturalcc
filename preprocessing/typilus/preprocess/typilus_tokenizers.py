import ujson
import itertools
from preprocessing.typilus.utils import (
    TokenEmbedder,
    ignore_type_annotation,
)
from ncc.tokenizers import tokenization


def nodes_tokenizer(line, **kwargs):
    nodes = ujson.loads(line)
    subtokens = []
    if nodes:
        for node in nodes:
            infer_node = TokenEmbedder.filter_literals(node)
            if infer_node != node:
                subtokens.append(node)
            else:
                subtokens.extend(tokenization.split_identifier_into_parts(node))
    return subtokens


def edges_tokenizer(line, **kwargs):
    graph = ujson.loads(line)
    subtokens = list(graph.keys())
    # subtokens += [f'_{e}' for e in subtokens]
    return subtokens


def nodes_binarizer_tokenizer(line, vocab, max_num_subtokens=5):
    nodes = ujson.loads(line)
    subtokens = []
    if nodes:
        for node in nodes:
            infer_node = TokenEmbedder.filter_literals(node)
            if infer_node == node:
                node_subtokens = tokenization.split_identifier_into_parts(node)[:max_num_subtokens]
                node_subtokens = node_subtokens + [vocab.pad_word] * (max_num_subtokens - len(node_subtokens))
            elif node in vocab:
                node_subtokens = [node] + [vocab.pad_word] * (max_num_subtokens - 1)
            else:
                node_subtokens = [infer_node] + [vocab.pad_word] * (max_num_subtokens - 1)
            subtokens.append(node_subtokens)
    subtokens = list(itertools.chain(*subtokens))
    return subtokens


def annotation_tokenizer(line, **kwargs):
    supernodes = ujson.loads(line)
    if supernodes:
        annotations = [node['annotation'] for node in supernodes.values() \
                       if not ignore_type_annotation(node['annotation'])]
        annotations = [ann.split("[")[0] for ann in annotations]
        return annotations
    else:
        return []


def annotation_binarizer_tokenizer(line, **kwargs):
    supernodes = ujson.loads(line)
    if supernodes:
        node_ids, annotations, raw_annotations = [], [], []
        for idx, snode in supernodes.items():
            annotation = snode['annotation']
            # if kwargs.get('is_train', True) and ignore_type_annotation(annotation):
            if True and ignore_type_annotation(annotation):
                continue
            node_ids.append(int(idx))
            raw_annotations.append(annotation)
            annotations.append(annotation.split("[")[0])
        if len(node_ids) > 0:
            return node_ids, annotations, raw_annotations
    return [], [], []
