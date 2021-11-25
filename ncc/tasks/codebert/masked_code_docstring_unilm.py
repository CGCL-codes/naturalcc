# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from ncc.data import constants
from ncc.data.codebert.mask_code_docstring_pair_dataset import MaskCodeDocstringPairDataset
from ncc.data.tools import data_utils
from ncc.data.wrappers import (
    NumelDataset,
    IdDataset,
    NestedDictionaryDataset,
    TokenBlockDataset,
    PrependTokenDataset,
)
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.utils import utils


def load_masked_code_docstring_dataset_unilm(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    max_src_len=0, max_tgt_len=0,
    truncate_source=False, append_source_id=False):
    source_path = os.path.join(data_path, '{}.code'.format(split))
    target_path = os.path.join(data_path, '{}.docstring'.format(split))

    # source_dataset
    source_dataset = data_utils.load_indexed_dataset(source_path, 'text', src_dict, dataset_impl)
    if source_dataset is None:
        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, source_path))

    # target_dataset
    target_dataset = data_utils.load_indexed_dataset(target_path, 'text', tgt_dict, dataset_impl)
    if target_dataset is None:
        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, target_path))

    eos = None
    align_dataset = None
    target_dataset_sizes = target_dataset.sizes if target_dataset is not None else None

    return MaskCodeDocstringPairDataset(
        source_dataset, source_dataset.sizes, src_dict,
        target_dataset, target_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset, eos=eos,
        skipgram_prb=0.0,
        skipgram_size=0.0,
    )


@register_task('masked_code_docstring_unilm')
class MaskedCodeDocstringUnilmTask(NccTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        # self.dictionary = dictionary
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.seed = args['common']['seed']
        # add mask token
        # self.mask_idx = dictionary.add_symbol('<mask>')

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
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
        src_dict.add_symbol(constants.T_MASK)
        src_dict.add_symbol(constants.CLS)
        src_dict.add_symbol(constants.S2S_SEP)

        tgt_dict.add_symbol(constants.S_SEP)
        tgt_dict.add_symbol(constants.T_MASK)
        tgt_dict.add_symbol(constants.S2S_BOS)
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

        self.datasets[split] = load_masked_code_docstring_dataset_unilm(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args['dataset']['dataset_impl'],
            upsample_primary=self.args['task']['upsample_primary'],
            left_pad_source=self.args['task']['left_pad_source'],
            left_pad_target=self.args['task']['left_pad_target'],
            max_source_positions=self.args['task']['max_source_positions'],
            max_target_positions=self.args['task']['max_target_positions'],
            load_alignments=self.args['task']['load_alignments'],
            truncate_source=self.args['task']['truncate_source'],
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
