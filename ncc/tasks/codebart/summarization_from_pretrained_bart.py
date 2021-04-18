# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ncc.data.summarization.language_pair_dataset import LanguagePairDataset

# from .translation import load_langpair_dataset, TranslationTask
from ncc.tasks.summarization.summarization import SummarizationTask
from ncc.tasks import register_task
from ncc.data import indexed_dataset
from ncc.data.bpe_dictionary import (BPE_Dictionary, SENTENCEPIECE)
from ncc.data.tools import data_utils
from ncc.data.wrappers.append_token_dataset import AppendTokenDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.data.wrappers.portion_dataset import PortionDataset
from ncc.data.wrappers.prepend_token_dataset import PrependTokenDataset
from ncc.data.summarization.language_pair_dataset import LanguagePairDataset
from ncc.tokenizers import tokenization
from functools import lru_cache
import torch
from ncc.data.ncc_dataset import NccDataset
from ncc import LOGGER
import numpy as np
import os

EVAL_BLEU_ORDER = 4


class IndexedRawTextDataset(NccDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.lines.append(line.strip('\n'))
                tokens = dictionary.encode_line(
                    line, tokenization._space_tokenizer, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)


def _load_dataset(path, impl, dict):
    if impl == 'raw':
        src_dataset = IndexedRawTextDataset(path=path, dictionary=dict)
    elif impl == 'mmap':
        # mmap dataset has been numberized, no need for dict
        src_dataset = indexed_dataset.MMapIndexedDataset(path=path)
    else:
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    return src_dataset


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    dataset_impl,
    # combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target,
    max_source_positions, max_target_positions,
    prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False,
    truncate_target=False,
    append_eos_to_target=False,
    portion=None,
):
    # load source dataset
    src_path = os.path.join(data_path, '{}.{}'.format(split, src))
    src_dataset = _load_dataset(path=src_path, impl=dataset_impl, dict=src_dict)

    if truncate_source:
        LOGGER.info('truncate {}.{} to {}'.format(split, src, max_source_positions))
        src_dataset = TruncateDataset(src_dataset, max_source_positions)

    if portion is not None and split == 'train':
        LOGGER.info('set {}.{} portion to {}'.format(split, src, portion))
        src_dataset = PortionDataset(src_dataset, portion)

    # load target dataset
    tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
    tgt_dataset = _load_dataset(path=tgt_path, impl=dataset_impl, dict=tgt_dict)
    if truncate_target:
        LOGGER.info('truncate {}.{} to {}'.format(split, tgt, max_target_positions))
        tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    if portion is not None and split == 'train':
        LOGGER.info('set {}.{} portion to {}'.format(split, tgt, portion))
        tgt_dataset = PortionDataset(tgt_dataset, portion)

    # align_dataset = None
    # if load_alignments:
    #     align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
    #     if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
    #         align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    LOGGER.info('loaded {} examples from: {}'.format(len(src_dataset), src_path))
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_dataset), tgt_path))
    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=None, eos=eos,
        remove_eos_from_source=True,
        append_eos_to_target=append_eos_to_target,
        shuffle=True,
        # shuffle=False,  # debug
    )


@register_task('summarization_from_pretrained_bart')
class SummarizationFromPretrainedBARTTask(SummarizationTask):
    """
    Translate from source language to target language with a model initialized with a multilingual pretrain.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        SummarizationTask.add_args(parser)
        parser.add_argument('--langs', required=True, metavar='LANG',
                            help='comma-separated list of monolingual language, for example, "en,de,fr"'
                                 'be careful these langs are what you used for pretraining (the same order),'
                                 'not for finetuning.'
                                 'you should always add all pretraining language idx during finetuning.')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.langs = args.langs.split(',')
        for d in [src_dict, tgt_dict]:
            for l in self.langs:
                d.add_symbol('[{}]'.format(l))
            d.add_symbol('<mask>')

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            dataset_impl=self.args['dataset']['dataset_impl'],
            left_pad_source=self.args['task']['left_pad_source'],
            left_pad_target=self.args['task']['left_pad_target'],
            max_source_positions=self.args['task']['max_source_positions'],
            max_target_positions=self.args['task']['max_target_positions'],
            load_alignments=self.args['task']['load_alignments'],
            truncate_source=self.args['task']['truncate_source'],
            truncate_target=self.args['task']['truncate_target'],
            append_eos_to_target=self.args['task']['append_eos_to_target'],
            portion=self.args['dataset'].get('portion', None),
        )

    # def build_generator(self, args):
    #     if getattr(args, 'score_reference', False):
    #         from fairseq.sequence_scorer import SequenceScorer
    #         return SequenceScorer(
    #             self.target_dictionary,
    #             eos=self.tgt_dict.index('[{}]'.format(self.target_lang))
    #         )
    #     else:
    #         from fairseq.sequence_generator import SequenceGenerator
    #         return SequenceGenerator(
    #             self.target_dictionary,
    #             beam_size=getattr(args, 'beam', 5),
    #             max_len_a=getattr(args, 'max_len_a', 0),
    #             max_len_b=getattr(args, 'max_len_b', 200),
    #             min_len=getattr(args, 'min_len', 1),
    #             normalize_scores=(not getattr(args, 'unnormalized', False)),
    #             len_penalty=getattr(args, 'lenpen', 1),
    #             unk_penalty=getattr(args, 'unkpen', 0),
    #             temperature=getattr(args, 'temperature', 1.),
    #             match_source_len=getattr(args, 'match_source_len', False),
    #             no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
    #             eos=self.tgt_dict.index('[{}]'.format(self.args.target_lang))
    #         )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src_lang_id = self.source_dictionary.index('[{}]'.format(self.args.source_lang))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)
        return dataset
