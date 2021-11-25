# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import torch
from ncc.data.tools import data_utils
from ncc.data.ncc_dataset import NccDataset


def collate_paths(values, sizes, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of list of 1d tensors into a padded 3d tensor."""
    res = torch.zeros(*sizes).long()

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][sizes[1] - len(v):] if left_pad else res[i][:len(v)])

    return res


def collate(
    samples, num_sample, subtoken_len, path_len,
    pad_idx, eos_idx, bos_idx,
    left_pad_source=False, left_pad_target=False, input_feeding=True,
):
    """
    src_seq: [t1, t2, ...]
    src_seq => path encoder => path hidden state

    tgt_seq: [t1, t2, ..., <EOS>] => append <BOS> and remove <EOS> => prev_output_tokens: [<BOS>, t1, t2, ...]
    prev_output_tokens => decoder => predicted_seq => loss <= tgt_seq
    """
    if len(samples) == 0:
        return {}

    def merge_path(key, sizes, left_pad, move_eos_to_beginning=False):
        return torch.stack(
            [
                collate_paths(s[key], sizes, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
                for s in samples
            ],
            dim=0
        )

    def merge_tokens(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def merge_tokens_add_bos_remove_eos(key, left_pad):
        values = [s[key] for s in samples]

        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)
        res[:, 0].fill_(bos_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v[:-1], res[i][size - len(v) + 1:] if left_pad else res[i][1:len(v)])
        return res

    id = torch.LongTensor([s['id'] for s in samples])

    heads = merge_path(key='heads', sizes=(num_sample, subtoken_len), left_pad=left_pad_source)
    tails = merge_path(key='tails', sizes=(num_sample, subtoken_len), left_pad=left_pad_source)
    bodies = merge_path(key='bodies', sizes=(num_sample, path_len), left_pad=left_pad_source)
    body_lens = bodies.ne(pad_idx).sum(dim=-1)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge_tokens('target', left_pad=left_pad_target)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge_tokens(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': [heads, bodies, tails],
            'src_lengths': body_lens,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    return batch


class PathDataset(NccDataset):
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
        self, src, src_sizes, src_dict, type_dict, src_szs,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        append_bos=False, eos=None,
        path_num=None, max_subtoken_len=None, max_path_len=None,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_szs = src_szs
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.type_dict = type_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos
        self.eos = (eos if eos is not None else src_dict.eos())
        self.path_num = path_num
        self.max_subtoken_len = max_subtoken_len
        self.max_path_len = max_path_len

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        src_sizes = self.src_szs[index]
        src_item = src_item.split(src_sizes.tolist(), dim=0)

        paths = [src_item[idx:idx + 3] for idx in range(0, len(src_item), 3)]
        if len(paths) > self.path_num:
            paths = random.sample(paths, self.path_num)
        else:
            if self.path_num // len(paths) > 1:
                paths = paths * (self.path_num // len(paths))
            paths += random.sample(paths, self.path_num - len(paths))

        heads, bodies, tails = zip(*paths)
        heads = [h[:self.max_subtoken_len] for h in heads]
        tails = [t[:self.max_subtoken_len] for t in tails]
        new_bodies = []
        for b in bodies:
            b = torch.cat((b[:self.max_path_len], torch.Tensor([self.eos]).long()), dim=0)
            new_bodies.append(b)

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

        example = {
            'id': index,
            'heads': heads,
            'bodies': new_bodies,
            'tails': tails,
            'target': tgt_item,
        }
        return example

    def __len__(self):
        return len(self.src)

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
            samples, num_sample=self.path_num, subtoken_len=self.max_subtoken_len, path_len=self.max_path_len + 1,
            pad_idx=self.src_dict.pad(), eos_idx=self.eos, bos_idx=self.src_dict.bos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        indices = np.random.permutation(len(self))
        return indices

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)
