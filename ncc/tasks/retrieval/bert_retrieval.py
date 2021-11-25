# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import os
from collections import Counter

import torch

from ncc import LOGGER
from ncc.data import (
    indexed_dataset,
    constants,
)
from ncc.data.dictionary import Dictionary
from ncc.data.retrieval.bert_dataset import BertDataset
from ncc.data.wrappers.concat_dataset import ConcatDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.data.wrappers.prepend_token_dataset import PrependTokenDataset
from ncc.data.wrappers.append_token_dataset import AppendTokenDataset
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.utils import utils
from ncc.utils.logging import metrics


def _load_dataset(path, impl, dict=None):
    if impl == 'raw':
        raise NotImplementedError(impl)
    elif impl == 'mmap':
        # mmap dataset has been numberized, no need for dict
        dataset = indexed_dataset.MMapIndexedDataset(path=path)
    else:
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    return dataset


def load_tokens_dataset(
    data_path, split, src, src_dict, tgt, tgt_dict, dataset_impl,
    max_source_positions=None, max_target_positions=None, max_positions=None,
    append_source_eos=False, append_target_eos=False,
    shuffle=False,
):
    src_path = os.path.join(data_path, '{}.{}'.format(split, src))
    src_dataset = _load_dataset(src_path, dataset_impl)
    if max_source_positions is not None:
        src_dataset = TruncateDataset(src_dataset, max_source_positions)
    LOGGER.info('loaded {} examples from: {}'.format(len(src_dataset), src_path))

    tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
    tgt_dataset = _load_dataset(tgt_path, dataset_impl)
    if max_target_positions is not None:
        tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions)
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_dataset), tgt_path))

    return BertDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        max_source_positions=max_source_positions, max_target_positions=max_target_positions,
        max_positions=max_positions,
        append_source_eos=append_source_eos, append_target_eos=append_target_eos,
        shuffle=shuffle,
    )


@register_task('bert_retrieval')
class BertRetrievalTask(NccTask):
    """
    load code/docstring as input
    """

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.args = args
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename)

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        if args['dataset']['joined_dictionary']:
            modalities = sorted(args['task']['source_lang'] + args['task']['target_lang'])
            src_dict = tgt_dict = cls.load_dictionary(
                os.path.join(paths[0], '{}.dict.json'.format('_'.join(modalities))))
        else:
            src_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.jsonl'.format(args['task']['source_lang'])))
            tgt_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.jsonl'.format(args['task']['target_lang'])))
        return cls(args, src_dict, tgt_dict)

    @classmethod
    def build_dictionary(
        cls, filenames, tokenize_func,
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
            Dictionary.add_token_to_dictionary(
                filename, d, tokenize_func, workers
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

        self.datasets[split] = load_tokens_dataset(
            data_path, split, src, self.source_dictionary, tgt, self.target_dictionary,
            dataset_impl=self.args['dataset']['dataset_impl'],
            max_source_positions=self.args['dataset']['max_source_positions'],
            max_target_positions=self.args['dataset']['max_target_positions'],
            max_positions=self.args['dataset']['max_positions'],
            append_source_eos=self.args['dataset']['append_source_eos'],
            append_target_eos=self.args['dataset']['append_target_eos'],
            shuffle=(split == 'train'),
        )

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

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
                def sum_logs(key):
                    return sum(log.get(key, 0) for log in logging_outputs)

                metrics.log_scalar('mrr', sum_logs('mrr'))
                metrics.log_scalar('sample_size', sum_logs('sample_size'))

    def encode_query_input(self, query):
        from ncc.tokenizers import tokenization
        if query[-1] == '.':
            query = query[:-1] + ' .'
        query = query.split()
        query_ids = self.tgt_dict.encode_line(query, tokenization._lower_tokenizer,
                                              func_name=None, min_func_len=None)
        query_ids = query_ids[:self.args['dataset']['query_max_tokens']]
        return query_ids
