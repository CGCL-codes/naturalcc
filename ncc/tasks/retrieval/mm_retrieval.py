# -*- coding: utf-8 -*-


import os
from collections import Counter

import torch

from ncc import LOGGER
from ncc.data import indexed_dataset
from ncc.data.dictionary import Dictionary
from ncc.data.retrieval.mm_retrieval_dataset import MultiModalitiesRetrievalDataset
from ncc.data.retrieval.hybrid.hybrid_retrieval_dictionary import HybridRetrievalDictionary
from ncc.data.wrappers.concat_dataset import ConcatDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.utils import utils
from ncc.utils.logging import metrics
from collections import OrderedDict


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
    data_path, split, srcs, src_dicts, tgts, tgt_dicts, dataset_impl,
    max_srcs=None, max_tgts=None,
    shuffle=False, sample_neg=False, **kwargs,
):
    langs = kwargs.get('langs', None)
    # source langs
    src_paths = OrderedDict()
    for src in srcs:
        if langs is None:
            src_paths[src] = [os.path.join(data_path, f'{split}.{src}')]
        else:
            src_paths[src] = [os.path.join(data_path, f'{split}.{lang}.{src}') for lang in langs]
    src_datasets, src_sizes = OrderedDict(), OrderedDict()
    for idx, (src, paths) in enumerate(src_paths.items()):
        datasets = _load_dataset(paths, dataset_impl)
        if max_srcs is not None:
            datasets = [TruncateDataset(ds, max_srcs[idx]) for ds in datasets]
        datasets = ConcatDataset(datasets, labels=langs)
        src_datasets[src] = datasets
        src_sizes[src] = datasets.sizes
    LOGGER.info('loaded {} modality(ies) from: {}'.format(len(src_datasets), src_paths))
    # target langs
    tgt_paths = OrderedDict()
    for tgt in tgts:
        if langs is None:
            tgt_paths[tgt] = [os.path.join(data_path, f'{split}.{tgt}')]
        else:
            tgt_paths[tgt] = [os.path.join(data_path, f'{split}.{lang}.{tgt}') for lang in langs]
    tgt_datasets, tgt_sizes = OrderedDict(), OrderedDict()
    for idx, (tgt, paths) in enumerate(tgt_paths.items()):
        datasets = _load_dataset(paths, dataset_impl)
        if max_tgts is not None:
            datasets = [TruncateDataset(ds, max_tgts[idx]) for ds in datasets]
        datasets = ConcatDataset(datasets, labels=langs)
        tgt_datasets[tgt] = datasets
        tgt_sizes[tgt] = datasets.sizes
    LOGGER.info('loaded {} modality(ies) from: {}'.format(len(tgt_datasets), tgt_paths))

    return MultiModalitiesRetrievalDataset(
        src_datasets, src_sizes, src_dicts,
        tgt_datasets, tgt_sizes, tgt_dicts,
        max_source_positions=max_srcs, max_target_positions=max_tgts,
        fraction_using_func_name=kwargs.get('fraction_using_func_name', None),
        shuffle=shuffle,
        labels=langs, sample_neg=sample_neg,
    )


@register_task('mm_retrieval')
class MultiModalitiesRetrievalTask(NccTask):
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
            modalities = sorted(args['task']['source_langs'] + args['task']['target_langs'])
            src_dicts = tgt_dicts = cls.load_dictionary(
                os.path.join(paths[0], '{}.dict.json'.format('_'.join(modalities))))
        else:
            src_dicts = {
                lang: cls.load_dictionary(os.path.join(paths[0], f'{lang}.dict.jsonl'))
                for lang in args['task']['source_langs']
            }
            tgt_dicts = {
                lang: cls.load_dictionary(os.path.join(paths[0], f'{lang}.dict.jsonl'))
                for lang in args['task']['target_langs']
            }
        return cls(args, src_dicts, tgt_dicts)

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
        srcs, tgts = self.args['task']['source_langs'], self.args['task']['target_langs']
        assert len(self.args['dataset']['max_srcs']) == len(srcs) and \
               len(self.args['dataset']['max_tgts']) == len(tgts)
        self.datasets[split] = load_tokens_dataset(
            data_path, split, srcs, self.source_dictionaries, tgts, self.target_dictionaries,
            dataset_impl=self.args['dataset']['dataset_impl'],
            max_srcs=self.args['dataset']['max_srcs'],
            max_tgts=self.args['dataset']['max_tgts'],
            langs=self.args['dataset']['langs'],
            shuffle=(split == 'train'),
            fraction_using_func_name=self.args['task']['fraction_using_func_name'],
            sample_neg=(self.args['optimization'].get('sample_neg', False))
        )

    @property
    def source_dictionaries(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dicts

    @property
    def target_dictionaries(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dicts

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

    def encode_query_input(self, query):
        from ncc.tokenizers import tokenization
        if query[-1] == '.':
            query = query[:-1] + ' .'
        query = query.split()
        query_ids = self.tgt_dict.encode_line(query, tokenization._lower_tokenizer,
                                              func_name=None, min_func_len=None)
        query_ids = query_ids[:self.args['dataset']['query_max_tokens']]
        return query_ids
