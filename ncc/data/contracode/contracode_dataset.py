# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from ncc.data.ncc_dataset import NccDataset
from torch.nn.utils.rnn import pad_sequence

def mask_mlm(seq, pad_id, mask_id, vocab_start_range, vocab_end_range):
    # The training data generator chooses 15% of the token positions at random for prediction.
    # If the i-th token is chosen, we replace the i-th token with
    # (0) not masked
    # (1) the [MASK] token 80% of the time (0.12)
    # (2) a random token 10% of the time (0.015)
    # (3) the unchanged i-th token 10% of the time (0.015)
    #
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py#L63
    torch.manual_seed(1)
    rand_replacements = torch.zeros_like(seq, dtype=torch.long).random_(vocab_start_range, vocab_end_range)
    torch.manual_seed(1)
    masked_tokens = (torch.rand_like(seq, dtype=torch.float) < 0.15) & (seq != pad_id)
    mask_type_prob = torch.rand_like(seq, dtype=torch.float)
    mask_token_prob = (mask_type_prob < 0.8) & masked_tokens
    random_token_prob = (mask_type_prob < 0.9) & (mask_type_prob >= 0.8) & masked_tokens
    identity_token_prob = (mask_type_prob >= 0.9) & masked_tokens
    assert torch.sum(masked_tokens) == torch.sum(mask_token_prob | random_token_prob | identity_token_prob)

    targets = torch.zeros_like(seq).fill_(pad_id)
    targets[masked_tokens] = seq[masked_tokens]

    seq[mask_token_prob] = mask_id
    seq[random_token_prob] = rand_replacements[random_token_prob]
    return seq, targets


def collate(samples, src_dict,  program_mode='contrastive', loss_mode='hybrid', left_pad_source=True, left_pad_target=False):
    if len(samples) == 0:
        return {}

    B = len(samples)
    if program_mode == "contrastive":
        samples_ = [[sample['code_q'], sample['code_k']] for sample in samples]
        X1, X2 = zip(*samples_)
        tokens = X1 + X2
    else:
        samples_ = [sample['code_q'] for sample in samples]
        tokens = samples_

    # Create tensor of sequence lengths, [B] or [2B]
    lengths = torch.tensor([len(x) for x in tokens], dtype=torch.long)

    # Create padded tensor for batch, [B, T] or [2B, T]
    tokens = pad_sequence(tokens, batch_first=True, padding_value=src_dict.pad())

    if program_mode == "contrastive":
        # Reshape X to [B, 2, T]
        T = tokens.size(-1)
        tokens = torch.reshape(tokens, (2, B, -1))
        tokens = torch.transpose(tokens, 0, 1)
        assert tokens.shape == (B, 2, T)
        lengths = torch.reshape(lengths, (2, B)).transpose(0, 1)
        assert lengths.shape == (B, 2)

        id = torch.LongTensor([s['id'] for s in samples])

        tokens_q, tokens_k = tokens[:, 0, :], tokens[:, 1, :]
        lengths_q, lengths_k = lengths[:, 0], lengths[:, 1]
        if loss_mode == 'hybrid':
            tokens_k, mlm_targets = mask_mlm(tokens_k, pad_id=src_dict.pad(), mask_id=src_dict.indices['[MASK]'], vocab_start_range=0, vocab_end_range=7999)
        else:
            mlm_targets = None

        example = {
            'id': id,
            'net_input': {
                'tokens_q': tokens_q,
                'tokens_k': tokens_k,
                'lengths_q': lengths_q,
                'lengths_k': lengths_k,
            },
            'mlm_targets': mlm_targets,
            'loss_mode': loss_mode,
        }
    elif program_mode == "identity":
        id = torch.LongTensor([s['id'] for s in samples])

        if loss_mode == 'mlm':
            tokens, mlm_targets = mask_mlm(tokens, pad_id=src_dict.pad(), mask_id=src_dict.indices['[MASK]'],
                                             vocab_start_range=8, vocab_end_range=7999)
        else:
            mlm_targets = None

        example = {
            'id': id,
            'net_input': {
                'tokens': tokens,
                'lengths': lengths,
            },
            'mlm_targets': mlm_targets,
            'loss_mode': loss_mode,
        }

    return example


class ContraCodeDataset(NccDataset):
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
            self, src, src_sizes, src_dict, program_mode='contrastive', loss_mode='hybrid',
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True,
            remove_eos_from_source=False, append_eos_to_target=False,
            append_bos=False, eos=None,
    ):
        self.src = src
        self.src_sizes = np.array(src_sizes)
        self.src_dict = src_dict
        self.program_mode = program_mode
        self.loss_mode = loss_mode
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.append_bos = append_bos
        self.eos = (eos if eos is not None else src_dict.eos())

    def __getitem__(self, index):
        src_item = self.src[index]
        n_alt = len(src_item)
        if self.program_mode == "identity":
            # return self.program2tensor(src_item[0])
            example = {
                'id': index,
                'code_q': src_item[0]
            }
        elif self.program_mode == "augmentation":
            i = np.random.randint(n_alt)
            # return self.program2tensor(src_item[i])
            example = {
                'id': index,
                'code_q': src_item[i]
            }
        elif self.program_mode == "contrastive":
            np.random.seed(1)   # TODO debug
            i = np.random.randint(n_alt)
            j = i
            if n_alt > 1:
                while j == i:
                    j = np.random.randint(n_alt)
            # return self.program2tensor(src_item[i]), self.program2tensor(src_item[j])
            example = {
                'id': index,
                'code_q': src_item[i],
                'code_k': src_item[j]
            }
        else:
            raise ValueError(f"Invalid program mode {self.program_mode}")

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
            samples, src_dict=self.src_dict, program_mode=self.program_mode, loss_mode=self.loss_mode,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        # return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        return self.src_sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        # return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        return self.src_sizes[index]

    @property
    def sizes(self):
        return self.src_sizes

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        # if self.tgt_sizes is not None:
        #     indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices  # [np.argsort(self.src_sizes[indices], kind='mergesort')] # TODO

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
