# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ncc.data.tools import data_utils
from ncc.data.ncc_dataset import NccDataset
import jsbeautifier
from dataset.augmented_javascript.utils.util import normalize_program
import torch
from ncc.data.type_prediction.utils import enforce_not_None
import torch
from ncc.utils import utils
from typing import Dict, Generic, List, NamedTuple, Tuple, TypeVar, OrderedDict
import gzip
import json
from ncc.data.type_prediction.utils import enforce_not_None
from ncc.data.type_prediction.utils import TensorizedGraphData
from ncc import LOGGER

TTensorizedNodeData = TypeVar("TTensorizedNodeData")


def collate(samples, pad_idx, no_type_id):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
        )
    B = len(samples)
    # X, Y = zip(*batch)
    X = merge('subword_ids')
    # Y = merge('label_segments')
    Y = [s['label_segments'] for s in samples]

    # X = pad_sequence(X, batch_first=True, padding_value=pad_id)
    L = X.size(1)

    # Create tensor of sequence lengths, [B]
    # lengths = torch.tensor([len(x) for x in X], dtype=torch.long)
    lengths = torch.LongTensor([s['subword_ids'].numel() for s in samples])
    # Make masks for each label interval
    labels = torch.zeros(B, L, dtype=torch.long)
    labels.fill_(no_type_id)
    output_attn = torch.zeros(B, L, L, dtype=torch.float)
    for i, y in enumerate(Y):
        for label_id, label_start, label_end in y:
            labels[i, label_start] = label_id
            output_attn[i, label_start, label_start:label_end] = 1.0 / (label_end.item() - label_start.item())

    # batch = {
    #     "node_data": self.__node_embedding_model.finalize_minibatch(
    #         accumulated_minibatch_data["node_data_mb"], device
    #     ),
    #     "adjacency_lists": [
    #         (
    #             torch.tensor(np.concatenate(adjFrom), dtype=torch.int64, device=device),
    #             torch.tensor(np.concatenate(adjTo), dtype=torch.int64, device=device),
    #         )
    #         for adjFrom, adjTo in accumulated_minibatch_data["adjacency_lists"]
    #     ],
    #     "node_to_graph_idx": torch.tensor(
    #         list(
    #             self.__create_node_to_graph_idx(
    #                 accumulated_minibatch_data["num_nodes_per_graph"]
    #             )
    #         ),
    #         dtype=torch.int64,
    #         device=device,
    #     ),
    #     "reference_node_graph_idx": {
    #         ref_name: torch.tensor(ref_node_graph_idx, dtype=torch.int64, device=device)
    #         for ref_name, ref_node_graph_idx in accumulated_minibatch_data[
    #             "reference_node_graph_idx"
    #         ].items()
    #     },
    #     "reference_node_ids": {
    #         ref_name: torch.tensor(
    #             np.concatenate(ref_node_idxs).astype(np.int32), dtype=torch.int64, device=device
    #         )
    #         for ref_name, ref_node_idxs in accumulated_minibatch_data[
    #             "reference_node_ids"
    #         ].items()
    #     },
    #     "num_graphs": len(accumulated_minibatch_data["num_nodes_per_graph"]),
    # }
    batch = None
    return batch


# def _tokenize(js_tokens, labels, sp, tgt_dict, max_length, split_source_targets_by_tab=False):
#     """Given a line from the .txt data files in DeepTyper, format and
#     tokenize the code into subwords while retaining a mapping between
#     type labels and the subwords.
#
#     Returns:
#         Beautified program
#         List of subword IDs
#         List of (label_id, label_start, label_end) tuples where label_start/end specify a range of subword IDs"""
#     assert TYPED_MARKER_START not in js_tokens
#     assert TYPED_MARKER_MID not in js_tokens
#     assert TYPED_MARKER_END not in js_tokens
#     cap_length = max_length != -1
#     #
#     # if split_source_targets_by_tab:
#     #     # Code tokens and type labels are delimeted by tab, as in .json files
#     #     js_tokens, labels = deeptyper_line.split("\t")
#     # else:
#     #     # Code tokens and type labels are delimited by space after </s>, as in .txt files
#     #     js_end = deeptyper_line.index("</s>") + len("</s>")
#     #     js_tokens = deeptyper_line[:js_end]
#     #     labels = deeptyper_line[js_end + 1:]
#
#     # Split code by spaces to get DeepTyper tokens, excluding <s>, </s>
#     # print('js_tokens: ', js_tokens)
#     # print('labels: ', labels)
#     js_tokens = js_tokens.split(" ")[1:-1]
#     labels = labels.split(" ")[1:-1]
#     # labels = labels[1:-1]
#     assert len(js_tokens) == len(labels)
#
#     # Add markers to each labeled token: tokens without no-type label
#     js_tokens_with_markers = []
#     for tok, label in zip(js_tokens, labels):
#         if label != "O":
#             tok = f"{TYPED_MARKER_START}{tok}{TYPED_MARKER_END}"
#         js_tokens_with_markers.append(tok)
#
#     # Normalize program by beautifying
#     js_beautified = jsbeautifier.beautify(" ".join(js_tokens_with_markers))
#     js_beautified_norm = normalize_program(js_beautified)
#     # js_beautified_norm = js_beautified_norm
#
#     # Subword tokenize, separately tokenizing each marked identifier
#     js_buffer = js_beautified_norm
#     subword_ids = [sp.PieceToId("<s>")]
#     label_segments = []
#
#     last_identifier_end = 0
#     start = js_buffer.find(TYPED_MARKER_START)
#     labels = list(filter(lambda l: l != "O", labels))
#     label_i = 0
#     if start < 0:
#         # No labeled words in this text, just tokenize the full program
#         buffer_ids = sp.EncodeAsIds(js_buffer)
#         subword_ids.extend(buffer_ids[: max_length - 2] if cap_length else buffer_ids)
#     while start >= 0:
#         # buffer: "stuff [start ptr]__LS__typedIdentifier__LE__..."
#         # Tokenize stuff before the typed identifier
#         buffer_ids = sp.EncodeAsIds(js_buffer[last_identifier_end:start])
#         if cap_length and len(subword_ids) + len(buffer_ids) + 1 > max_length:  # +1 for </s>
#             break
#         subword_ids.extend(buffer_ids)
#
#         # Tokenize typed identifier and record label
#         start = start + len(TYPED_MARKER_START)
#         end = js_buffer.index(TYPED_MARKER_END, start)
#         assert end > start, "Zero-length identifier"
#         identifier = js_buffer[start:end]
#         identifier_ids = sp.EncodeAsIds(identifier)
#         if cap_length and len(subword_ids) + len(identifier_ids) + 1 > max_length:  # +1 for </s>
#             break
#         # A segment is (label_id, label_start, label_end)
#         # label_id = target_to_id.get(labels[label_i], target_to_id["$any$"])
#         label_id = tgt_dict.indices.get(labels[label_i], tgt_dict.indices['$any$'])
#
#         label_segments.append((label_id, len(subword_ids), len(subword_ids) + len(identifier_ids)))
#         subword_ids.extend(identifier_ids)
#
#         start = js_buffer.find(TYPED_MARKER_START, start + 1)
#         last_identifier_end = end + len(TYPED_MARKER_END)
#         label_i += 1
#         # print('len(subword_ids): ', len(subword_ids))
#     subword_ids.append(sp.PieceToId("</s>"))
#     assert subword_ids[0] == sp.PieceToId("<s>")
#     assert subword_ids[-1] == sp.PieceToId("</s>")
#
#     return js_beautified, subword_ids, label_segments

def __iterate_edge_types(data_to_load, edge_types):
    for edge_type in edge_types: # self.__edge_idx_to_type:
        adjacency_list = data_to_load.edges.get(edge_type)
        if adjacency_list is not None and len(adjacency_list) > 0:
            adj = np.array(adjacency_list, dtype=np.int32)
            yield adj[:, 0], adj[:, 1]
        else:
            yield np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)


def gnn_tensorize(datapoint, source_dictionary, edge_types):
    tensorized_data = TensorizedGraphData(
        adjacency_lists=list(__iterate_edge_types(datapoint, edge_types)),
        node_tensorized_data=[
            # enforce_not_None(self.__node_embedding_model.tensorize(ni))
            source_dictionary.index(ni)
            for ni in datapoint.node_information
        ],
        reference_nodes={
            n: np.array(np.array(refs, dtype=np.int32))
            for n, refs in datapoint.reference_nodes.items()
        },
        num_nodes=len(datapoint.node_information),
    )

    if tensorized_data.num_nodes > 80000:
        LOGGER.warning("Dropping graph with %s nodes." % tensorized_data.num_nodes)
        return None

    num_edges = sum(len(adj) for adj in tensorized_data.adjacency_lists)
    if num_edges > 100000:
        LOGGER.warning("Dropping graph with %s edges." % num_edges)
        return None

    return tensorized_data


class CodeTypeDataset(NccDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
    """

    def __init__(
            self, src_tgt, src_sizes, src_dict, tgt_dict, edge_types,
            left_pad_source=False, left_pad_target=False,
            max_source_positions=1024,
            max_target_positions=1024,
            shuffle=True,
    ):
        self.src_tgt = src_tgt
        self.src_sizes = np.array(src_sizes)
        self.src_dict = src_dict
        self.edge_types = edge_types
        # self.tgt = tgt
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle

    def __getitem__(self, index):
        # Append EOS to end of tgt sentence if it does not have an EOS
        # and remove EOS from end of src sentence if it exists.
        # This is useful when we use existing datasets for opposite directions
        #   i.e., when we want to use tgt_dataset as src_dataset and vice versa
        # src_item = self.src[index]
        # tgt_item = self.tgt[index]
        src_item, tgt_item = self.src_tgt[index]
        src_item = gnn_tensorize(src_item, self.src_dict, self.edge_types)

        # line = self.lines[idx]
        # _, subword_ids, label_segments = _tokenize(
        #     src_item, tgt_item, self.sp, self.tgt_dict, self.max_source_positions)
        # if self.max_source_positions != -1:
        #     assert len(subword_ids) <= self.max_source_positions
        # subword_ids = torch.tensor(subword_ids, dtype=torch.long)
        # label_segments = torch.tensor(label_segments, dtype=torch.long)

        example = {
            'graph': src_item,
            'supernode_target_classes': tgt_item,
        }

        return example
        # return (subword_ids, label_segments)


        # node_id = self.node_ids[index]
        # extend = self.extends[index]
        # example = {
        #     'id': index,
        #     'source': src_item,
        #     'target': tgt_item,
        #     'node_id': node_id,
        #     'extend': extend,
        # }
        # return example

    def __len__(self):
        return len(self.tgt)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.tgt_dict.pad(), no_type_id=self.tgt_dict.index('O'),
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.src_sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.src_sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices#[np.argsort(self.src_sizes[indices], kind='mergesort')] # TODO: debug