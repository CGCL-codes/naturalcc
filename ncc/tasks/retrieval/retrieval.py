# -*- coding: utf-8 -*-

import os
from collections import (
    OrderedDict,
)

from ncc import LOGGER
from ncc.data import indexed_dataset
from ncc.data.retrieval.deepcs_pair_dataset import DeepCSLanguagePairDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccComplTask
from ncc.utils import utils


def _load_dataset(path, impl, dict=None):
    if impl == 'raw':
        raise NotImplementedError(impl)
    elif impl == 'mmap':
        # mmap dataset has been numberized, no need for dict
        dataset = indexed_dataset.MMapIndexedDataset(path=path)
    else:
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    return dataset


def load_langpair_dataset(
    data_path, split,
    srcs, src_dicts,
    tgts, tgt_dicts,
    dataset_impl,
    src_max_tokens, tgt_max_tokens,
    **kwargs,
):
    # load source dataset
    src_datasets, src_sizes = OrderedDict(), OrderedDict()
    for idx, src in enumerate(srcs):
        src_path = os.path.join(data_path, '{}.{}'.format(split, src))
        src_datasets[src] = _load_dataset(path=src_path, impl=dataset_impl, dict=src_dicts[src])
        src_datasets[src] = TruncateDataset(src_datasets[src], src_max_tokens[idx])
        src_sizes[src] = src_datasets[src].sizes
    # load target dataset
    tgt_datasets, tgt_sizes = OrderedDict(), OrderedDict()
    for idx, tgt in enumerate(tgts):
        tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
        tgt_datasets[tgt] = _load_dataset(path=tgt_path, impl=dataset_impl, dict=tgt_dicts[tgt])
        tgt_datasets[tgt] = TruncateDataset(tgt_datasets[tgt], tgt_max_tokens[idx])
        tgt_sizes[tgt] = tgt_datasets[tgt].sizes

    return DeepCSLanguagePairDataset(
        src_datasets, src_sizes, src_dicts,
        tgt_datasets, tgt_sizes, tgt_dicts,
        pad=src_dicts[srcs[0]].pad(),
        shuffle=False,
    )


@register_task('retrieval')
class RetrievalTask(NccComplTask):
    """
    Task for code retrieval models (e.g., code and docstring).
    A simple implementation. Only consider (single modality) code-comment retrieval task.
    """

    def __init__(self, args, src_dicts, tgt_dicts):
        super().__init__(args)
        self.args = args
        self.src_dicts = src_dicts
        self.tgt_dicts = tgt_dicts

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        assert len(paths) > 0
        # load dictionaries
        src_dicts = OrderedDict()
        for lang in args['task']['source_langs']:
            src_dicts[lang] = cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format(lang)))
            LOGGER.info('[{}] dictionary: {} types'.format(lang, len(src_dicts[lang]) if lang != 'edges' else 0))
        tgt_dicts = OrderedDict()
        for lang in args['task']['target_langs']:
            tgt_dicts[lang] = cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format(lang)))
            LOGGER.info('[{}] dictionary: {} types'.format(lang, len(tgt_dicts[lang])))
        return cls(args, src_dicts, tgt_dicts)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        srcs, tgts = self.args['task']['source_langs'], self.args['task']['target_langs']

        self.datasets[split] = load_langpair_dataset(
            data_path, split, srcs, self.src_dicts, tgts, self.tgt_dicts,
            dataset_impl=self.args['dataset']['dataset_impl'],
            src_max_tokens=self.args['dataset']['src_max_tokens'],
            tgt_max_tokens=self.args['dataset']['tgt_max_tokens'],
        )
