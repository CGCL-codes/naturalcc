# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from ncc import LOGGER
from ncc.logging import metrics
from ncc.data.dictionary import Dictionary
from ncc.data.retrieval.retrieval_dictionary import RetrievalDictionary
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.data.retrieval.retrieval_dataset import RetrievalDataset
from ncc.utils import utils

from ncc.data.wrappers.truncate_dataset import TruncateDataset

from ncc.data import indexed_dataset


def _load_dataset(path, impl, dict=None):
    if impl == 'raw':
        raise NotImplementedError(impl)
    elif impl == 'mmap':
        # mmap dataset has been numberized, no need for dict
        src_dataset = indexed_dataset.MMapIndexedDataset(path=path)
    else:
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    return src_dataset


def load_tokens_dataset(
    data_path, split, src, src_dict, tgt, tgt_dict, dataset_impl, src_max_tokens=None, tgt_max_tokens=None,
    src_aux=None, src_aux_dict=None, tgt_aux=None, tgt_aux_dict=None, src_aux_max_tokens=None, tgt_aux_max_tokens=None,
    fraction_using_func_name=0.,
):
    src_path = os.path.join(data_path, '{}.{}'.format(split, src))
    src_dataset = _load_dataset(src_path, dataset_impl)
    src_dataset = TruncateDataset(src_dataset, src_max_tokens)

    tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
    tgt_dataset = _load_dataset(tgt_path, dataset_impl)
    tgt_dataset = TruncateDataset(tgt_dataset, tgt_max_tokens)

    LOGGER.info('loaded {} examples from: {}'.format(len(src_dataset), src_path))
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_dataset), tgt_path))

    if src_aux is not None:
        src_aux_path = os.path.join(data_path, '{}.{}'.format(split, src_aux))
        src_aux_dataset = _load_dataset(src_aux_path, dataset_impl)
        if src_aux_max_tokens is None:
            src_aux_max_tokens = src_max_tokens
        src_aux_dataset = TruncateDataset(src_aux_dataset, src_aux_max_tokens)
        LOGGER.info('loaded {} examples from: {}'.format(len(src_aux_dataset), src_aux_path))

    if tgt_aux is not None:
        tgt_aux_path = os.path.join(data_path, '{}.{}'.format(split, tgt_aux))
        tgt_aux_dataset = _load_dataset(tgt_aux_path, dataset_impl)
        if tgt_aux_max_tokens is None:
            tgt_aux_max_tokens = tgt_max_tokens
        tgt_aux_dataset = TruncateDataset(tgt_aux_dataset, tgt_aux_max_tokens)
        LOGGER.info('loaded {} examples from: {}'.format(len(tgt_aux_dataset), tgt_aux_path))

    return RetrievalDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        max_source_positions=src_max_tokens, max_target_positions=tgt_max_tokens,
        src_aux=src_aux_dataset,
        src_aux_sizes=None if src_aux is None else src_aux_dataset.sizes,
        src_aux_dict=src_dict if src_aux_dict is None else src_aux_dict,
        tgt_aux=tgt_aux_dataset,
        tgt_aux_sizes=None if tgt_aux is None else tgt_aux_dataset.sizes,
        tgt_aux_dict=tgt_dict if tgt_aux_dict is None else tgt_aux_dict,
        fraction_using_func_name=fraction_using_func_name,
    )


@register_task('retrieval')
class RetrievalTask(NccTask):
    """
    Task for code retrieval models (e.g., code and docstring).
    A simple implementation. Only consider (single modality) code-comment retrieval task.
    """

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.args = args
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        # self.seed = args['common']['seed']

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        if filename.endswith('.txt'):
            return Dictionary.load(filename)
        else:
            is_bpe = os.path.basename(filename).split('.')[-3] == 'bpe'
            if is_bpe:
                return RetrievalDictionary.load_json(filename)
            else:
                return Dictionary.load_json(filename)

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        if args['dataset']['joined_dictionary']:
            modalities = sorted(args['task']['source_lang'] + args['task']['target_lang'])
            src_dict = tgt_dict = \
                cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format('_'.join(modalities))))
        else:
            src_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format(args['task']['source_lang'])))
            tgt_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format(args['task']['target_lang'])))
        return cls(args, src_dict, tgt_dict)

    @classmethod
    def build_dictionary(
        cls, filenames, tokenize_func=None,
        workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = Dictionary()

        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, tokenize_func, num_workers=workers)

        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def build_bpe_dictionary(
        cls, filenames, tokenize_func=None,
        workers=1, threshold=-1, nwords=10000, padding_factor=8
    ):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = RetrievalDictionary()

        for filename in filenames:
            RetrievalDictionary.add_bpe_token_to_dictionary(
                filename, d, nwords, tokenize_func, workers
            )

        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        src, tgt = self.args['task']['source_lang'], self.args['task']['target_lang']
        src_aux, tgt_aux = self.args['task']['source_aux_lang'], self.args['task']['target_aux_lang']

        if self.args['model']['arch'] in ['nbow', 'conv1d_res', 'birnn', 'self_attn']:
            self.datasets[split] = load_tokens_dataset(
                data_path, split, src, self.source_dictionary, tgt, self.target_dictionary,
                dataset_impl=self.args['dataset']['dataset_impl'],
                src_max_tokens=self.args['dataset']['code_max_tokens'],
                tgt_max_tokens=self.args['dataset']['query_max_tokens'],
                src_aux=src_aux, tgt_aux=tgt_aux,
                fraction_using_func_name=self.args['task']['fraction_using_func_name'],
            )
        else:
            raise NotImplementedError

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def build_model(self, args):
        model = super().build_model(args)
        return model

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.NccDataset`.
            model (~fairseq.models.BaseNccModel): the model
            criterion (~fairseq.criterions.NccCriterion): the criterion
            optimizer (~fairseq.optim.NccOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        with torch.no_grad():
            if self.args['task']['eval_mrr']:
                def mean_logs(key):
                    return sum(log.get(key, 0) for log in logging_outputs) / len(logging_outputs)

                metrics.log_scalar('mrr', mean_logs('mrr'))
