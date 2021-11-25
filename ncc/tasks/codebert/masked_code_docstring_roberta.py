# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np

from ncc.data import constants
from ncc.data.wrappers.mask_tokens_dataset import MaskTokensDataset
from ncc.data.tools import data_utils
from ncc.data.wrappers import (
    NumSamplesDataset,
    NestedDictionaryDataset,
    IdDataset,
    ConcatSentencesDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
)
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.tokenizers.utils import get_whole_word_mask
from ncc.utils import utils


def load_masked_code_docstring_dataset_roberta(args, epoch,
                                               data_path, split,
                                               src, src_dict,
                                               tgt, tgt_dict,
                                               combine, dataset_impl, upsample_primary,
                                               left_pad_source, left_pad_target, max_source_positions,
                                               max_target_positions, prepend_bos=False, load_alignments=False,
                                               truncate_source=False, append_source_id=False):
    source_path = os.path.join(data_path, '{}.code'.format(split))
    target_path = os.path.join(data_path, '{}.docstring'.format(split))

    # source_dataset
    source_dataset = data_utils.load_indexed_dataset(source_path, 'text', src_dict, tokenizer=None,
                                                     dataset_impl=dataset_impl)
    if source_dataset is None:
        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, source_path))
    # target_dataset
    target_dataset = data_utils.load_indexed_dataset(target_path, 'text', tgt_dict, tokenizer=None,
                                                     dataset_impl=dataset_impl)
    if target_dataset is None:
        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, target_path))

    # concate dataset
    dataset = ConcatSentencesDataset([source_dataset, target_dataset])
    # create continuous blocks of tokens
    dataset = TokenBlockDataset(
        dataset,
        dataset.sizes,
        args['task']['tokens_per_sample'] - 1,  # one less for <s>
        pad=src_dict.pad(),
        eos=src_dict.eos(),
        break_mode=args['task']['sample_break_mode'],
    )
    # LOGGER.info('loaded {} blocks from: {}'.format(len(dataset), split_path))

    # # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
    dataset = PrependTokenDataset(dataset, src_dict.bos())  # .source_dictionary.bos()
    #
    # # create masked input and targets
    mask_whole_words = get_whole_word_mask(args, src_dict) \
        if args['task']['mask_whole_words'] else None

    src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
        dataset,
        src_dict,
        pad_idx=src_dict.pad(),
        mask_idx=src_dict.index(constants.T_MASK),  # self.mask_idx,
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
                        pad_idx=src_dict.pad(),
                        left_pad=False,
                    ),
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
                'target': PadDataset(
                    tgt_dataset,
                    pad_idx=src_dict.pad(),
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


@register_task('masked_code_docstring_roberta')
class MaskedCodeDocstringRoberataTask(NccTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        # self.dictionary = dictionary
        self.args = args
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.seed = args['common']['seed']

        # add mask token
        # self.mask_idx = dictionary.add_symbol('<mask>')

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        # dictionary = Dictionary.load(os.path.join(paths[0], 'dict.code.txt'))
        if args['dataset']['joined_dictionary']:
            src_dict = cls.load_dictionary(
                os.path.join(paths[0],
                             '{}.dict.txt'.format(args['task']['source_lang'])))  # args['task']['source_lang']
            tgt_dict = src_dict
        else:
            src_dict = cls.load_dictionary(
                os.path.join(paths[0],
                             '{}.dict.txt'.format(args['task']['source_lang'])))  # args['task']['source_lang']
            tgt_dict = cls.load_dictionary(
                os.path.join(paths[0], '{}.dict.txt'.format(args['task']['target_lang'])))

        src_dict.add_symbol(constants.S_SEP)
        src_dict.add_symbol(constants.S2S_SEP)
        src_dict.add_symbol(constants.CLS)
        src_dict.add_symbol(constants.T_MASK)
        src_dict.add_symbol(constants.SEP)

        tgt_dict.add_symbol(constants.S2S_BOS)
        tgt_dict.add_symbol(constants.T_MASK)
        tgt_dict.add_symbol(constants.SEP)
        print('<T_MASK> id is', src_dict.index('<T_MASK>'))
        print('<T_MASK> id is', tgt_dict.index('<T_MASK>'))

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args['task']['source_lang'], self.args['task']['target_lang']

        self.datasets[split] = load_masked_code_docstring_dataset_roberta(self.args, epoch,
                                                                          data_path, split, src, self.src_dict, tgt,
                                                                          self.tgt_dict,
                                                                          combine=combine,
                                                                          dataset_impl=self.args['dataset'][
                                                                              'dataset_impl'],
                                                                          upsample_primary=self.args['task'][
                                                                              'upsample_primary'],
                                                                          left_pad_source=self.args['task'][
                                                                              'left_pad_source'],
                                                                          left_pad_target=self.args['task'][
                                                                              'left_pad_target'],
                                                                          max_source_positions=self.args['task'][
                                                                              'max_source_positions'],
                                                                          max_target_positions=self.args['task'][
                                                                              'max_target_positions'],
                                                                          load_alignments=self.args['task'][
                                                                              'load_alignments'],
                                                                          truncate_source=self.args['task'][
                                                                              'truncate_source'],
                                                                          )

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
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict
