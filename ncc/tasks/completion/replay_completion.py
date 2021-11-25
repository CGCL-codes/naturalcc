import os
import re

import numpy as np
import torch

from ncc import LOGGER
from ncc.data import constants
from ncc.data import indexed_dataset
from ncc.data.completion.replay_completion_dataset import ReplayCompletionDataset
from ncc.data.completion.completion_dataset import CompletionDataset
from ncc.data.completion.completion_dictionary import CompletionDictionary as Dictionary
from .completion import CompletionTask
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.data.wrappers.slice_dataset import SliceDataset
from ncc.data.wrappers.concat_dataset import ConcatDataset
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.utils import utils
from ncc.utils.logging import metrics


def _load_dataset(path, impl, dict=None):
    if impl == 'raw':
        raise NotImplementedError(impl)
    elif impl == 'mmap':
        # mmap dataset has been numberized, no need for dict
        src_dataset = indexed_dataset.MMapIndexedDataset(path=path)
    else:
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    return src_dataset


def load_token_dataset(
    data_path, split, tgt, tgt_dict, dataset_impl,
    truncate_target=False, max_target_positions=None,
    # lifelong learning
    prev_tasks=[], cur_task=None,
):
    # load tokens
    tgt_path = os.path.join(data_path, cur_task, '{}.{}'.format(split, tgt))
    tgt_dataset = _load_dataset(tgt_path, dataset_impl)
    if truncate_target:
        tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions)
        LOGGER.info('Truncate dataset into max length: {}'.format(max_target_positions))

    prev_tgt_datasets, prev_tgt_dataset_sizes = [], []
    for prev_task in prev_tasks:
        prev_tgt_path = os.path.join(data_path, prev_task, '{}.{}'.format(split, tgt))
        prev_dataset = _load_dataset(prev_tgt_path, dataset_impl)
        if truncate_target:
            prev_dataset = TruncateDataset(prev_dataset, max_target_positions)
        prev_tgt_datasets.append(prev_dataset)
        prev_tgt_dataset_sizes.append(prev_dataset.sizes)

    LOGGER.info('loaded {} examples from: [{}](current task) + {}(previous tasks)'. \
                format(len(tgt_dataset), cur_task, prev_tasks))

    return ReplayCompletionDataset(
        tgt_dataset, tgt_dataset.sizes,
        prev_tgt_datasets, prev_tgt_dataset_sizes,
        tgt_dict, max_target_positions=max_target_positions,
    )


def load_inference_token_dataset(
    data_paths, split, tgt, tgt_dict, dataset_impl,
    attrs=None, attr_dict=None,
    attrs_mapping=None, reversed_attrs_mapping=None,
    truncate_target=False, max_target_positions=None,
):
    # load tokens
    tgt_dataset = []
    for path in data_paths:
        tgt_path = os.path.join(path, '{}.{}'.format(split, tgt))
        tgt_dataset.append(_load_dataset(tgt_path, dataset_impl))
    tgt_dataset = ConcatDataset(tgt_dataset)
    if truncate_target:
        tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions)
        LOGGER.info('Truncate dataset into max length: {}'.format(max_target_positions))
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_dataset), data_paths))
    return CompletionDataset(
        tgt_dataset, tgt_dataset.sizes, tgt_dict, extends=None,
        attrs=None, attr_indices=None, attr_dict=None,
        attrs_mapping=None, reversed_attrs_mapping=None,
        max_target_positions=max_target_positions,
    )


@register_task('replay_completion')
class ReplayCompletionTask(CompletionTask):
    """Lifelong learning"""

    def __init__(self, args, dictionary, token_dictionary=None):
        super().__init__(args, dictionary, token_dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        if self.args['task']['target_lang'] == 'code_tokens' and self.args['task'].get('code_types', False):
            attrs_mapping = {
                'attr': {self.token_dictionary.index('attr')},
                'num': {self.token_dictionary.index('Num')},
                'name': {self.token_dictionary.index('NameStore'),
                         self.token_dictionary.index('NameLoad')},
                'param': {self.token_dictionary.index('arg'),
                          self.token_dictionary.index('kwarg'),
                          self.token_dictionary.index('vararg')},
            }
        elif self.args['task']['target_lang'] == 'ast' and self.args['task'].get('code_types', False):
            attrs_mapping = {
                'attr': {self.token_dictionary.index('attr')},
                'num': {self.token_dictionary.index('Num')},
                'name': {self.token_dictionary.index('NameStore'),
                         self.token_dictionary.index('NameLoad')},
                'param': {self.token_dictionary.index('NameParam')},
            }
        else:
            attrs_mapping = None

        if attrs_mapping:
            reversed_attrs_mapping = {}
            for k, vs in attrs_mapping.items():
                if len(vs) > 1:
                    for v in vs:
                        reversed_attrs_mapping[v] = k
                else:
                    reversed_attrs_mapping[list(vs)[0]] = k
        else:
            reversed_attrs_mapping = None

        task_idx = kwargs.get('task_idx', 1)
        if split == 'train':
            cur_task = self.args['task']['task_pipeline'][task_idx]
            init_from_scratch = task_idx == 1 and self.args['task']['task_pipeline'][0] == 'scratch'
            if not init_from_scratch:
                # 1) sample protion data from previous tasks, and 2) first task is not scratch
                prev_tasks = self.args['task']['task_pipeline'][:task_idx]
            else:
                prev_tasks = []
            self.datasets[split] = load_token_dataset(
                data_path, split, self.args['task']['target_lang'], self.target_dictionary,
                dataset_impl=self.args['dataset']['dataset_impl'],
                truncate_target=self.args['dataset'].get('truncate_target', False),
                max_target_positions=self.max_positions(),
                # lifelong
                prev_tasks=prev_tasks, cur_task=cur_task,
            )
        else:
            data_paths = [os.path.join(data_path, self.args['task']['task_pipeline'][task_idx])]
            self.datasets[split] = load_inference_token_dataset(
                data_paths, split, self.args['task']['target_lang'], self.target_dictionary,
                attrs_mapping=attrs_mapping, reversed_attrs_mapping=reversed_attrs_mapping,
                attrs=self.args['task'].get('code_types', None),
                attr_dict=self.token_dictionary,
                dataset_impl=self.args['dataset']['dataset_impl'],
                truncate_target=self.args['dataset'].get('truncate_target', False),
                max_target_positions=self.max_positions(),
            )
