# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np

from ncc import LOGGER
from ncc.data import constants
from ncc.data.tools import data_utils
from ncc.data.wrappers import (
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    SortDataset,
    TokenBlockDataset,
)
from ncc.data.wrappers.prepend_token_dataset import PrependTokenDataset
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.tokenizers.utils import get_whole_word_mask
from ncc.utils import utils


def load_masked_traverse_dataset_roberta(args, epoch, data_path, split, source_dictionary, combine, ):
    split_path = os.path.join(data_path, '{}.ast_trav_df'.format(split))
    dataset = data_utils.load_indexed_dataset(
        path=split_path,
        dictionary=source_dictionary,
        dataset_impl=args['dataset']['dataset_impl'],
        combine=combine,
    )
    if dataset is None:
        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

    # # create continuous blocks of tokens
    # dataset = TokenBlockDataset(
    #     dataset,
    #     dataset.sizes,
    #     args['task']['tokens_per_sample'] - 1,  # one less for <s>
    #     pad=source_dictionary.pad(),
    #     eos=source_dictionary.eos(),
    #     break_mode=args['task']['sample_break_mode'],
    # )
    # LOGGER.info('loaded {} blocks from: {}'.format(len(dataset), split_path))

    # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
    dataset = PrependTokenDataset(dataset, source_dictionary.bos())  # .source_dictionary.bos()

    # create masked input and targets
    mask_whole_words = get_whole_word_mask(args, source_dictionary) \
        if args['task']['mask_whole_words'] else None

    src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
        dataset,
        source_dictionary,
        pad_idx=source_dictionary.pad(),
        mask_idx=source_dictionary.index(constants.MASK),  # self.mask_idx,
        seed=args['common']['seed'],
        mask_prob=args['task']['mask_prob'],
        leave_unmasked_prob=args['task']['leave_unmasked_prob'],
        random_token_prob=args['task']['random_token_prob'],
        freq_weighted_replacement=args['task']['freq_weighted_replacement'],
        mask_whole_words=mask_whole_words,
    )

    with data_utils.numpy_seed(args['common']['seed'] + epoch):
        shuffle = np.random.permutation(len(src_dataset))

    return SortDataset(
        NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': PadDataset(
                        src_dataset,
                        pad_idx=source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
                'target': PadDataset(
                    tgt_dataset,
                    pad_idx=source_dictionary.pad(),
                    left_pad=False,
                ),
                'nsentences': NumSamplesDataset(),
                'ntokens': NumelDataset(src_dataset, reduce=True),
            },
            sizes=[src_dataset.sizes],
        ),
        sort_order=[
            shuffle,
            src_dataset.sizes,
        ],
    )


@register_task('masked_traverse_roberta')
class MaskedTraverseRobertaTask(NccTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args['common']['seed']

        # add mask token
        self.mask_idx = self.dictionary.add_symbol(constants.MASK)

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        dictionary = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.json'.format(args['task']['source_lang'])))
        LOGGER.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        self.datasets[split] = load_masked_traverse_dataset_roberta(self.args, epoch, data_path, split,
                                                                    self.source_dictionary, combine)

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = PadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args['task']['tokens_per_sample'] - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode='eos',
            ),
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': src_dataset,
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
