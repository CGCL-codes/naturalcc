import numpy as np
import torch
from ncc.data.tools import data_utils
from ncc.data.ncc_dataset import NccDataset
from ncc import LOGGER
import dgl


def collate(
    samples, pad_idx, max_subtoken_len,
):
    if len(samples) == 0:
        return {}
    id = torch.LongTensor([s['id'] for s in samples])

    graph_batch, tgt_ids, tgt_labels, tgt_cls = [], [], [], []
    start_idx = 0
    for idx, s in enumerate(samples):
        # source
        graph = s['source']['edges']
        graph.ndata['subtoken'] = s['source']['nodes'].view(-1, max_subtoken_len)
        graph_batch.append(graph)
        # target
        tgt_idx = s['target']['supernodes.annotation.node'] + start_idx
        start_idx += s['source']['edges'].number_of_nodes()
        tgt_ids.append(tgt_idx)
        tgt_labels.append(s['target']['supernodes.annotation.type'])
        tgt_cls.extend(s['target']['supernodes.annotation.type.json'])
    graph_batch = dgl.batch(graph_batch)
    tgt_ids = torch.cat(tgt_ids, dim=0)
    tgt_labels = torch.cat(tgt_labels, dim=0)

    adjacent_matrix = torch.zeros([len(tgt_cls), len(tgt_cls)]).float()
    for i in range(len(tgt_cls)):
        for j in range(i + 1, len(tgt_cls)):
            if tgt_cls[i] == tgt_cls[j]:
                adjacent_matrix[i, j] = 1.
                adjacent_matrix[j, i] = 1.
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': len(tgt_ids),
        'net_input': {
            'src_graphs': graph_batch,
        },
        'target_ids': tgt_ids,
        'target': tgt_labels,
        'target_equal_ids': adjacent_matrix,
    }
    return batch


class LanguagePairDataset(NccDataset):

    def __init__(
        self, srcs, src_sizes, src_dicts,
        tgts, tgt_sizes, tgt_dicts,
        max_subtoken_len, pad=None,
        shuffle=True, **kwargs
    ):
        self.srcs = srcs
        self.src_sizes = src_sizes
        self.src_dicts = src_dicts
        self.tgts = tgts
        self.tgt_sizes = tgt_sizes
        self.tgt_dicts = tgt_dicts
        self.pad = pad
        self.shuffle = shuffle
        self.max_subtoken_len = max_subtoken_len

        self.length = None
        for src, src_dataset in self.srcs.items():
            if len(src_dataset) is not None:
                self.length = len(src_dataset)
                break

    def __getitem__(self, index):
        src_items = {src: src_data[index] for src, src_data in self.srcs.items()}
        tgt_items = {tgt: tgt_data[index] for tgt, tgt_data in self.tgts.items()}

        example = {
            'id': index,
            'source': src_items,
            'target': tgt_items,
        }
        return example

    def __len__(self):
        return self.length

    def collater(self, samples):
        return collate(samples, pad_idx=self.pad, max_subtoken_len=self.max_subtoken_len)

    # def num_tokens(self, index):
    #     """Return the number of tokens in a sample. This value is used to
    #     enforce ``--max-tokens`` during batching."""
    #     return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            {src: src_size[index] for src, src_size in self.src_sizes},
            {tgt: tgt_size[index] for tgt, tgt_size in self.tgt_sizes}
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        sizes_gt_0 = np.arange(len(self)).astype(bool)
        for sizes in list(self.src_sizes.values()) + list(self.tgt_sizes.values()):
            if sizes is not None:
                sizes_gt_0 &= sizes > 0
        indices = np.arange(len(self))[sizes_gt_0]
        if self.shuffle:
            np.random.shuffle(indices)
        return indices  # [np.argsort(self.src_sizes[indices], kind='mergesort')] # TODO: debug
