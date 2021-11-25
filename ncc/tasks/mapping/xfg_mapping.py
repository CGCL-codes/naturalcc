# -*- coding: utf-8 -*-

import os
import pickle
from collections import OrderedDict

import torch
from .mapping import Mapping

from ncc import (
    LOGGER,
)
from ncc.data import (
    indexed_dataset,
)
from ncc.data.mapping.xfg_dictionary import XFGDicionary as Dictionary
from ncc.data.mapping.language_pair_dataset import LanguagePairDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.eval.summarization import summarization_metrics
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.utils import utils


def _load_dataset(path, impl, dict):
    if impl == 'raw':
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    elif impl == 'mmap':
        dataset = indexed_dataset.MMapIndexedDataset(path=path)
    else:
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    return dataset


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    dataset_impl,
    left_pad_source,
    max_source_positions,
    src_aux=None,
):
    # load source dataset
    src_path = os.path.join(data_path, '{}.{}'.format(split, src))
    src_dataset = _load_dataset(path=src_path, impl=dataset_impl, dict=src_dict)
    src_dataset = TruncateDataset(src_dataset, truncation_length=max_source_positions, truncate_prefix=0)

    # load target dataset
    tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
    tgt_dataset = _load_dataset(path=tgt_path, impl=dataset_impl, dict=tgt_dict)

    # load auxiliary dataset
    aux_datasets = OrderedDict()
    for aux in src_aux:
        aux_path = os.path.join(data_path, '{}.{}'.format(split, aux))
        with open(aux_path, 'rb') as reader:
            aux_datasets[aux] = pickle.load(reader)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    LOGGER.info('loaded {} examples from: {}'.format(len(src_dataset), src_path))
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_dataset), tgt_path))
    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict, src_aux=aux_datasets,
        tgt=tgt_dataset, tgt_sizes=tgt_dataset_sizes, tgt_dict=tgt_dict,
        left_pad_source=left_pad_source,
        max_source_positions=max_source_positions,
        shuffle=(split == 'train'),
    )


@register_task('xfg_mapping')
class XFGMapping(Mapping):

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename)

    @classmethod
    def build_dictionary(
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
        from ncc.data import constants
        d = Dictionary(
            pad=special_symbols.get('pad', constants.PAD),
            bos=special_symbols.get('bos', constants.BOS),
            eos=special_symbols.get('eos', constants.EOS),
            unk=special_symbols.get('unk', constants.UNK),
            extra_special_symbols=special_symbols.get('extra_special_symbols', None),
        )

        for filename in filenames:
            Dictionary.add_xfg_to_dictionary(
                filename, d, tokenize_func, eos_word=None, num_workers=workers,
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

        # infer langcode
        src, tgt = self.args['task']['source_lang'], self.args['task']['target_lang']

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            dataset_impl=self.args['dataset']['dataset_impl'],
            left_pad_source=self.args['task']['left_pad_source'],
            max_source_positions=self.args['task']['max_source_positions'],
            src_aux=self.args['task']['source_aux'],
        )
