import numpy as np
import torch

from ncc.data.ncc_dataset import NccDataset
from ncc.data.tools import data_utils


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        # tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target'][1:]) for s in samples)  # skip bos
        prev_output_tokens = target
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    return batch


class BELanguagePairDataset(NccDataset):
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
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        align_dataset=None,
        bos=None, eos=None,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.bos = bos if bos is not None else src_dict.eos()
        self.eos = eos if eos is not None else src_dict.eos()

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]

        example = {
            'id': index,
            'source': src_item,
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
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos,
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
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices  # [np.argsort(self.tgt_sizes[indices], kind='mergesort')]# TODO: debug
        return indices  # [np.argsort(self.src_sizes[indices], kind='mergesort')] # TODO: debug

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
