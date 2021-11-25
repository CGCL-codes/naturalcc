# -*- coding: utf-8 -*-


import os

from ncc import LOGGER
from ncc.data import indexed_dataset
from ncc.data.retrieval.retrieval_dataset import RetrievalDataset
from ncc.data.wrappers.concat_dataset import ConcatDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.tasks import register_task
from ncc.tasks.retrieval.simple_retrieval import SimpleRetrievalTask
from ncc.utils import utils


def _load_dataset(paths, impl, dict=None):
    if impl == 'raw':
        raise NotImplementedError(impl)
    elif impl == 'mmap':
        # mmap dataset has been numberized, no need for dict
        dataset = [indexed_dataset.MMapIndexedDataset(path=path) for path in paths]
    else:
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    return dataset


def load_tokens_dataset(
    data_path, split, src, src_dict, tgt, tgt_dict, dataset_impl, src_max_tokens=None, tgt_max_tokens=None,
    src_aux=None, src_aux_dict=None, tgt_aux=None, tgt_aux_dict=None, src_aux_max_tokens=None, tgt_aux_max_tokens=None,
    fraction_using_func_name=0., labels=None, shuffle=False,
):
    if len(labels) == 1 and os.path.exists(os.path.join(data_path, '{}.{}.idx'.format(split, src))):
        src_paths = [os.path.join(data_path, '{}.{}'.format(split, src))]
    else:
        src_paths = [os.path.join(data_path, '{}.{}.{}'.format(split, lbl, src)) for lbl in labels]
    src_datasets = _load_dataset(src_paths, dataset_impl)
    src_datasets = [TruncateDataset(ds, src_max_tokens) for ds in src_datasets]
    src_datasets = ConcatDataset(src_datasets, labels=labels)

    if len(labels) == 1 and os.path.exists(os.path.join(data_path, '{}.{}.idx'.format(split, tgt))):
        tgt_paths = [os.path.join(data_path, '{}.{}'.format(split, tgt))]
    else:
        tgt_paths = [os.path.join(data_path, '{}.{}.{}'.format(split, lbl, tgt)) for lbl in labels]
    tgt_datasets = _load_dataset(tgt_paths, dataset_impl)
    tgt_datasets = [TruncateDataset(ds, tgt_max_tokens) for ds in tgt_datasets]
    tgt_datasets = ConcatDataset(tgt_datasets, labels=labels)

    LOGGER.info('loaded {} examples from: {}'.format(len(src_datasets), src_paths))
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_datasets), tgt_paths))

    if split == 'train' and src_aux is not None:
        if len(labels) == 1 and os.path.exists(os.path.join(data_path, '{}.{}.idx'.format(split, src_aux))):
            src_aux_paths = [os.path.join(data_path, '{}.{}'.format(split, src_aux))]
        else:
            src_aux_paths = [os.path.join(data_path, '{}.{}.{}'.format(split, lbl, src_aux)) for lbl in labels]
        src_aux_datasets = _load_dataset(src_aux_paths, dataset_impl)
        if src_aux_max_tokens is None:
            src_aux_max_tokens = src_max_tokens
        src_aux_datasets = [TruncateDataset(ds, src_aux_max_tokens) for ds in src_aux_datasets]
        src_aux_datasets = ConcatDataset(src_aux_datasets, labels=labels)
        LOGGER.info('loaded {} examples from: {}'.format(len(src_aux_datasets), src_aux_paths))
    else:
        src_aux_datasets = None

    if split == 'train' and tgt_aux is not None:
        if len(labels) == 1 and os.path.exists(os.path.join(data_path, '{}.{}.idx'.format(split, tgt_aux))):
            tgt_aux_paths = [os.path.join(data_path, '{}.{}'.format(split, tgt_aux))]
        else:
            tgt_aux_paths = [os.path.join(data_path, '{}.{}.{}'.format(split, lbl, tgt_aux)) for lbl in labels]
        tgt_aux_datasets = _load_dataset(tgt_aux_paths, dataset_impl)
        if tgt_aux_max_tokens is None:
            tgt_aux_max_tokens = tgt_max_tokens
        tgt_aux_datasets = [TruncateDataset(ds, tgt_aux_max_tokens) for ds in tgt_aux_datasets]
        tgt_aux_datasets = ConcatDataset(tgt_aux_datasets, labels=labels)
        LOGGER.info('loaded {} examples from: {}'.format(len(tgt_aux_datasets), tgt_aux_paths))
    else:
        tgt_aux_datasets = None

    return RetrievalDataset(
        src_datasets, src_datasets.sizes, src_dict,
        tgt_datasets, tgt_datasets.sizes, tgt_dict,
        max_source_positions=src_max_tokens, max_target_positions=tgt_max_tokens,
        src_aux=src_aux_datasets,
        src_aux_sizes=None if src_aux_datasets is None else src_aux_datasets.sizes,
        src_aux_dict=src_dict if src_aux_dict is None else src_aux_dict,
        tgt_aux=tgt_aux_datasets,
        tgt_aux_sizes=None if tgt_aux_datasets is None else tgt_aux_datasets.sizes,
        tgt_aux_dict=tgt_dict if tgt_aux_dict is None else tgt_aux_dict,
        fraction_using_func_name=fraction_using_func_name,
        shuffle=shuffle,
        labels=labels,
    )


@register_task('fr_simple_retrieval')
class FTSimpleRetrievalTask(SimpleRetrievalTask):
    """
    Task for code retrieval models (e.g., code and docstring).
    A simple implementation. Only consider (single modality) code-comment retrieval task.
    """

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        src, tgt = self.args['task']['source_lang'], self.args['task']['target_lang']
        if split == 'train':
            src_aux, tgt_aux = self.args['task']['source_aux_lang'], self.args['task']['target_aux_lang']
        else:
            src_aux, tgt_aux = None, None

        task_idx = kwargs.get('task_idx', 1)
        if split == 'valid':
            labels = self.args['dataset']['langs'][:task_idx + 1]
        else:
            labels = [self.args['dataset']['langs'][task_idx]]
        self.datasets[split] = load_tokens_dataset(
            data_path, split, src, self.source_dictionary, tgt, self.target_dictionary,
            dataset_impl=self.args['dataset']['dataset_impl'],
            src_max_tokens=self.args['dataset']['code_max_tokens'],
            tgt_max_tokens=self.args['dataset']['query_max_tokens'],
            src_aux=src_aux, tgt_aux=tgt_aux,
            fraction_using_func_name=self.args['task']['fraction_using_func_name'],
            labels=labels,
            shuffle=(split == 'train'),
        )
