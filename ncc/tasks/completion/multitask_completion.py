import os
import re

import numpy as np
import torch

from ncc import LOGGER
from ncc.data import constants
from ncc.data import indexed_dataset
from ncc.data.completion.completion_dataset import CompletionDataset
from ncc.data.completion.completion_dictionary import CompletionDictionary as Dictionary
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.data.wrappers.concat_dataset import ConcatDataset
from ncc.utils import utils
from ncc.utils.logging import metrics
from .completion import CompletionTask


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
    data_paths, split, tgt, tgt_dict, dataset_impl,
    attrs=None, attr_dict=None,
    attrs_mapping=None, reversed_attrs_mapping=None,
    truncate_target=False, max_target_positions=None,
):
    # load tokens
    tgt_dataset = []
    for data_path in data_paths:
        tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
        tgt_dataset.append(_load_dataset(tgt_path, dataset_impl))
    tgt_dataset = ConcatDataset(tgt_dataset)
    if truncate_target:
        tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions)
        LOGGER.info('Truncate dataset into max length: {}'.format(max_target_positions))
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_dataset), data_paths))
    # load tokens.ext
    tgt_ext_paths = [os.path.join(data_path, '{}.{}.ext'.format(split, tgt)) for data_path in data_paths]
    if all(indexed_dataset.SeqIndexedDataset.exists(tgt_ext_path) for tgt_ext_path in tgt_ext_paths):
        tgt_ext_dataset = indexed_dataset.SeqIndexedDataset(tgt_ext_paths[0])
        for tgt_ext_path in tgt_ext_paths[1:]:
            tgt_ext_dataset.append(indexed_dataset.SeqIndexedDataset(tgt_ext_path))
        if truncate_target:
            tgt_ext_dataset.clip(max_position=max_target_positions)
        assert len(tgt_dataset) == len(tgt_ext_dataset), (len(tgt_dataset), len(tgt_ext_dataset))
    else:
        tgt_ext_dataset = None
    # load attrs
    if attrs is None:
        attr_dataset = None
    else:
        attr_dataset = []
        for data_path in data_paths:
            attr_path = os.path.join(data_path, '{}.code_types'.format(split))
            attr_dataset.append(_load_dataset(attr_path, dataset_impl))
        attr_dataset = ConcatDataset(attr_dataset)
        if truncate_target:
            tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions)
            LOGGER.info('Truncate dataset\'s attributes into max length: {}'.format(max_target_positions))
        LOGGER.info('loaded {} examples from: {}'.format(len(attr_dataset), data_path))
        # load attr.ext
        attr_ext_paths = [os.path.join(data_path, '{}.code_types.ext'.format(split)) for data_path in data_paths]
        if all(indexed_dataset.SeqIndexedDataset.exists(attr_ext_path) for attr_ext_path in attr_ext_paths):
            attr_ext_dataset = indexed_dataset.SeqIndexedDataset(attr_ext_paths[0])
            for attr_ext_path in attr_ext_paths[1:]:
                attr_ext_dataset.append(indexed_dataset.SeqIndexedDataset(attr_ext_path))
            if truncate_target:
                attr_ext_dataset.clip(max_position=max_target_positions)
            assert np.all(tgt_ext_dataset == attr_ext_dataset)
            del attr_ext_dataset

    return CompletionDataset(
        tgt_dataset, tgt_dataset.sizes, tgt_dict, extends=tgt_ext_dataset,
        attrs=attrs, attr_indices=attr_dataset, attr_dict=attr_dict,
        attrs_mapping=attrs_mapping, reversed_attrs_mapping=reversed_attrs_mapping,
        max_target_positions=max_target_positions,
    )


@register_task('multi_task_completion')
class MultiTaskCompletionTask(CompletionTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, dictionary, token_dictionary=None):
        super().__init__(args, dictionary, token_dictionary)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        # load dictionaries
        dict_file = os.path.join(paths[0], '{}.dict.jsonl'.format(args['task']['target_lang']))
        dictionary = cls.load_dictionary(dict_file)
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['target_lang'], len(dictionary)))
        token_file = os.path.join(paths[0], 'code_types.dict.jsonl')
        if os.path.exists(token_file):
            token_dictionary = cls.load_dictionary(token_file)
            LOGGER.info('[code_tokens] dictionary: {} types'.format(len(token_dictionary)))
        else:
            token_dictionary = None
        return cls(args, dictionary, token_dictionary)

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

        data_paths = [os.path.join(data_path, task_name) for task_name in self.args['task']['task_pipeline']]

        self.datasets[split] = load_token_dataset(
            data_paths, split, self.args['task']['target_lang'], self.target_dictionary,
            attrs_mapping=attrs_mapping, reversed_attrs_mapping=reversed_attrs_mapping,
            attrs=self.args['task'].get('code_types', None),
            attr_dict=self.token_dictionary,
            dataset_impl=self.args['dataset']['dataset_impl'],
            truncate_target=self.args['dataset'].get('truncate_target', False),
            max_target_positions=self.max_positions()
        )
