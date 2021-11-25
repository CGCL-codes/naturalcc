import os
import re

import numpy as np
import torch

from ncc import LOGGER
from ncc.data import constants
from ncc.data import indexed_dataset
from ncc.data.completion.completion_dataset import CompletionDataset
from ncc.data.completion.reg_completion_dataset import RegCompletionDataset
from .completion import CompletionTask
from ncc.data.completion.completion_dictionary import CompletionDictionary as Dictionary
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.data.kd.teacher_out_dataset import TeacherOutDataset
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
    attrs=None, attr_dict=None,
    attrs_mapping=None, reversed_attrs_mapping=None,
    truncate_target=False, max_target_positions=None,
    model=None, shuffle=True,
):
    # load tokens
    tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
    tgt_dataset = _load_dataset(tgt_path, dataset_impl)
    if truncate_target:
        tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions)
        LOGGER.info('Truncate dataset into max length: {}'.format(max_target_positions))
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_dataset), tgt_path))
    # load tokens.ext
    tgt_ext_path = os.path.join(data_path, '{}.{}.ext'.format(split, tgt))
    if indexed_dataset.SeqIndexedDataset.exists(tgt_ext_path):
        tgt_ext_dataset = indexed_dataset.SeqIndexedDataset(tgt_ext_path)
        if truncate_target:
            tgt_ext_dataset.clip(max_position=max_target_positions)
        assert len(tgt_dataset) == len(tgt_ext_dataset), (len(tgt_dataset), len(tgt_ext_dataset))
    else:
        tgt_ext_dataset = None
    # load attrs
    if attrs is None:
        attr_dataset = None
    else:
        attr_path = os.path.join(data_path, '{}.code_types'.format(split))
        attr_dataset = _load_dataset(attr_path, dataset_impl)
        if truncate_target:
            tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions)
            LOGGER.info('Truncate dataset\'s attributes into max length: {}'.format(max_target_positions))
        LOGGER.info('loaded {} examples from: {}'.format(len(attr_dataset), attr_path))
        # load attr.ext
        attr_ext_path = os.path.join(data_path, '{}.code_types.ext'.format(split))
        if indexed_dataset.SeqIndexedDataset.exists(attr_ext_path):
            attr_ext_dataset = indexed_dataset.SeqIndexedDataset(attr_ext_path)
            if truncate_target:
                attr_ext_dataset.clip(max_position=max_target_positions)
            assert np.all(tgt_ext_dataset == attr_ext_dataset)
            del attr_ext_dataset

    return RegCompletionDataset(
        tgt_dataset, tgt_dataset.sizes, tgt_dict, extends=tgt_ext_dataset,
        attrs=attrs, attr_indices=attr_dataset, attr_dict=attr_dict,
        attrs_mapping=attrs_mapping, reversed_attrs_mapping=reversed_attrs_mapping,
        max_target_positions=max_target_positions,
        shuffle=shuffle,
    )


@register_task('reg_completion')
class RegCompletionTask(CompletionTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, dictionary, token_dictionary=None):
        super().__init__(args, dictionary, token_dictionary)

    def gen_hidden_features(self, model_path):
        from ncc.utils import checkpoint_utils
        assert os.path.exists(model_path), FileNotFoundError(model_path)

        models, _ = checkpoint_utils.load_model_ensemble(
            utils.split_paths(model_path),
            arg_overrides=eval(self.args['eval']['model_overrides']),
            task=self,
        )
        self.teacher = models[0]

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
        data_path = os.path.join(data_path, self.args['task']['task_pipeline'][task_idx])
        self.datasets[split] = load_token_dataset(
            data_path, split, self.args['task']['target_lang'], self.target_dictionary,
            attrs_mapping=attrs_mapping, reversed_attrs_mapping=reversed_attrs_mapping,
            attrs=self.args['task'].get('code_types', None),
            attr_dict=self.token_dictionary,
            dataset_impl=self.args['dataset']['dataset_impl'],
            truncate_target=self.args['dataset'].get('truncate_target', False),
            max_target_positions=self.max_positions(),
            shuffle=(split == 'train'),
        )
