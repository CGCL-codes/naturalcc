# -*- coding: utf-8 -*-


import os
from collections import Counter

import torch

from ncc import LOGGER
from ncc.data import indexed_dataset
from ncc.data.dictionary import Dictionary
from ncc.data.retrieval.retrieval_dataset import RetrievalDataset
from ncc.data.retrieval.hybrid.hybrid_retrieval_dictionary import HybridRetrievalDictionary
from ncc.data.wrappers.concat_dataset import ConcatDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.utils import utils
from ncc.utils.logging import metrics


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


@register_task('simple_retrieval')
class SimpleRetrievalTask(NccTask):
    """
    Task for code retrieval models (e.g., code and docstring).
    A simple implementation. Only consider (single modality) code-comment retrieval task.
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
        return HybridRetrievalDictionary.load(filename)

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

    @classmethod
    def build_bpe_dictionary(
        cls, filenames, tokenize_func,
        workers=1, threshold=-1, nwords=-1, padding_factor=8,
        **special_symbols,
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
        bpe_portion = special_symbols.get('bpe_portion', 0.5)
        bpetoken_num = int(nwords * bpe_portion)
        subtoken_num = nwords - bpetoken_num
        # subtoken
        from ncc.data import constants
        subtoken_d = Dictionary(
            pad=special_symbols.get('pad', constants.PAD),
            bos=special_symbols.get('bos', constants.BOS),
            eos=special_symbols.get('eos', constants.EOS),
            unk=special_symbols.get('unk', constants.UNK),
            extra_special_symbols=special_symbols.get('extra_special_symbols', None),
        )
        for filename in filenames:
            Dictionary.add_token_to_dictionary(
                filename, subtoken_d, tokenize_func, workers
            )
        remaining_tokens = Counter({sym: c for sym, c in zip(subtoken_d.symbols, subtoken_d.count)})
        subtoken_d.finalize(threshold=threshold, nwords=subtoken_num, padding_factor=padding_factor)
        remaining_tokens = Counter({sym: c for sym, c in remaining_tokens.items() if sym not in subtoken_d})
        # bpetoken
        from ncc.data.retrieval.word_bpe_dictionary import WordBpeDicionary
        bpetoken_d = WordBpeDicionary()
        bpetoken_d.learn_bpe_vocab(remaining_tokens.elements(), bpetoken_num)
        bpetoken_d.finalize(threshold=0, nwords=bpetoken_num, padding_factor=padding_factor)
        from ncc.data.retrieval.hybrid.hybrid_retrieval_dictionary import HybridRetrievalDictionary
        return HybridRetrievalDictionary(subtoken_d, bpetoken_d)

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

        self.datasets[split] = load_tokens_dataset(
            data_path, split, src, self.source_dictionary, tgt, self.target_dictionary,
            dataset_impl=self.args['dataset']['dataset_impl'],
            src_max_tokens=self.args['dataset']['code_max_tokens'],
            tgt_max_tokens=self.args['dataset']['query_max_tokens'],
            src_aux=src_aux, tgt_aux=tgt_aux,
            fraction_using_func_name=self.args['task']['fraction_using_func_name'],
            labels=self.args['dataset'].get('langs', None),
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
                def sum_logs(key):
                    return sum(log.get(key, 0) for log in logging_outputs)

                metrics.log_scalar('mrr', sum_logs('mrr'))
                metrics.log_scalar('sample_size', sum_logs('sample_size'))
