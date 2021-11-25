# -*- coding: utf-8 -*-


import json
import os
from functools import lru_cache

import numpy as np
import torch

from ncc import (
    tokenizers,
    LOGGER,
)
from ncc.data import (
    indexed_dataset,
)
from ncc.data.dictionary import Dictionary
from ncc.data.ncc_dataset import NccDataset
from ncc.data.summarization.path_dataset import PathDataset
from ncc.data.wrappers.append_token_dataset import AppendTokenDataset
from ncc.data.wrappers.portion_dataset import PortionDataset
from ncc.data.wrappers.prepend_token_dataset import PrependTokenDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.eval.summarization import summarization_metrics
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.tokenizers import tokenization
from ncc.utils import utils
from ncc.utils.logging import metrics
from .summarization import SummarizationTask

EVAL_BLEU_ORDER = 4


def _load_dataset(path, impl, dict):
    if impl == 'raw':
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    elif impl == 'mmap':
        # mmap dataset has been numberized, no need for dict
        src_dataset = indexed_dataset.MMapIndexedDataset(path=path)
    else:
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    return src_dataset


def load_langpair_dataset(
    data_path, split,
    src, src_dict, type_dict,
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
    path_num=None, max_subtoken_len=None, max_path_len=None,
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

    src_sz_path = os.path.join(data_path, '{}.{}.sz'.format(split, src))
    src_sz_dataset = _load_dataset(path=src_sz_path, impl=dataset_impl, dict=src_dict)

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

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    LOGGER.info('loaded {} examples from: {}'.format(len(src_dataset), src_path))
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_dataset), tgt_path))
    return PathDataset(
        src_dataset, src_dataset.sizes, src_dict, type_dict, src_sz_dataset,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=None, eos=eos,
        remove_eos_from_source=True,
        append_eos_to_target=append_eos_to_target,
        path_num=path_num, max_subtoken_len=max_subtoken_len, max_path_len=max_path_len,
        shuffle=(split == 'train'),
    )


@register_task('path_summarization')
class PathSummarizationTask(SummarizationTask):

    def __init__(self, args, src_dict, type_dict, tgt_dict):
        """
        src_dict: subtoken of codes
        type_dict: type of ast nodes
        tgt_dict: method name/docstring dict
        """
        super().__init__(args, src_dict, tgt_dict)
        self.type_dict = type_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0

        dict = args['task'].get('dict', None)
        dict_type = args['task'].get('dict_type', None)
        if dict is None and dict_type is None:
            # load dictionaries
            src_dict = cls.load_dictionary(os.path.join(paths[0], 'subtoken.dict.jsonl'))
            tgt_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.jsonl'.format(args['task']['target_lang'])))
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
            type_dict = cls.load_dictionary(os.path.join(paths[0], 'type.dict.jsonl'))
            LOGGER.info('[subtoken] dictionary: {} types'.format(len(src_dict)))
            LOGGER.info('[type] dictionary: {} types'.format(len(type_dict)))
            LOGGER.info('[{}] dictionary: {} types'.format(args['task']['target_lang'], len(tgt_dict)))
        else:
            raise NotImplementedError
        return cls(args, src_dict, type_dict, tgt_dict)

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

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, self.type_dict, tgt, self.tgt_dict,
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
            path_num=self.args['dataset'].get(f'{split}_path_num', 200),
            max_subtoken_len=self.args['dataset'].get(f'max_subtoken_len', None),
            max_path_len=self.args['dataset'].get(f'max_path_len', None),
        )
