# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

from ncc import LOGGER
from ncc.data import constants
from ncc.data import indexed_dataset
from ncc.data.summarization.language_pair_dataset import LanguagePairDataset
from ncc.data.wrappers.append_token_dataset import AppendTokenDataset
from ncc.data.wrappers.portion_dataset import PortionDataset
from ncc.data.wrappers.prepend_token_dataset import PrependTokenDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.data.wrappers.strip_token_dataset import StripTokenDataset
from ncc.tasks import register_task
from ncc.tasks.summarization.summarization import SummarizationTask
from ncc.utils import utils


def _load_dataset(path, impl, dict):
    if impl == 'raw':
        # src_dataset = IndexedRawTextDataset(path=path, dictionary=dict)
        raise NotImplementedError
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
    src_lang=None, tgt_lang=None,
    portion=None,
):
    # load source dataset
    src_path = os.path.join(data_path, '{}.{}'.format(split, src))
    src_dataset = _load_dataset(path=src_path, impl=dataset_impl, dict=src_dict)

    if truncate_source:
        LOGGER.info('truncate {}.{} to {}'.format(split, src, max_source_positions))
        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(src_dataset, src_dict.eos()),
                max_source_positions - (2 + int(prepend_bos))  # <eos>, [lang]
            ),
            src_dict.eos()
        )

    if portion is not None and split == 'train':
        LOGGER.info('set {}.{} portion to {}'.format(split, src, portion))
        src_dataset = PortionDataset(src_dataset, portion)

    # load target dataset
    tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
    tgt_dataset = _load_dataset(path=tgt_path, impl=dataset_impl, dict=tgt_dict)
    if truncate_target:
        LOGGER.info('truncate {}.{} to {}'.format(split, tgt, max_target_positions))
        tgt_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(tgt_dataset, tgt_dict.eos()),
                max_target_positions - (2 + int(prepend_bos))  # <eos>, [lang]
            ),
            src_dict.eos()
        )

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src_lang)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt_lang)))
        eos = tgt_dict.index('[{}]'.format(tgt_lang))

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
        remove_eos_from_source=False,
        append_eos_to_target=append_eos_to_target,
        shuffle=(split == 'train'),
        # shuffle=False,  # debug
    )


@register_task('translation_from_pretrained_bart')
class TranslationFromPretrainedBARTTask(SummarizationTask):
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

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.langs = args['task']['langs']  # .split(',')
        for d in [src_dict, tgt_dict]:
            for l in self.langs:
                d.add_symbol('[{}]'.format(l))
            d.add_symbol(constants.MASK)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # paths = self.args['task']['data'] #.split(':')
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args['task']['source_lang'], self.args['task']['target_lang']

        # max_source_positions = self.args['task']['max_source_positions']
        # max_target_positions = self.args['task']['max_target_positions']

        # max_source_positions = 402
        # max_target_positions = 32

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            dataset_impl=self.args['dataset']['dataset_impl'],
            left_pad_source=self.args['task']['left_pad_source'],
            left_pad_target=self.args['task']['left_pad_target'],
            max_source_positions=self.args['task']['max_source_positions'],
            max_target_positions=self.args['task']['max_target_positions'],
            # max_source_positions=max_source_positions,
            # max_target_positions=max_target_positions,
            load_alignments=self.args['task']['load_alignments'],
            truncate_source=self.args['task']['truncate_source'],
            truncate_target=self.args['task']['truncate_target'],
            prepend_bos=self.args['task']['prepend_bos'],
            append_eos_to_target=self.args['task']['append_eos_to_target'],
            append_source_id=self.args['task']['append_source_id'],
            src_lang=self.args['task']['src_lang'], tgt_lang=self.args['task']['tgt_lang'],
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
