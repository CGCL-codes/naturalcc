# -*- coding: utf-8 -*-

import os
import pickle
from collections import OrderedDict

import torch

from ncc import (
    LOGGER,
)
from ncc.data import (
    indexed_dataset,
)
from ncc.data.dictionary import Dictionary
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


@register_task('mapping')
class Mapping(NccTask):

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.jsonl'.format(args['task']['source_lang'])))
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['source_lang'], len(src_dict)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.jsonl'.format(args['task']['target_lang'])))
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['target_lang'], len(tgt_dict)))
        return cls(args, src_dict, tgt_dict)

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

        # infer langcode
        src, tgt = self.args['task']['source_lang'], self.args['task']['target_lang']

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            dataset_impl=self.args['dataset']['dataset_impl'],
            left_pad_source=self.args['task']['left_pad_source'],
            max_source_positions=self.args['task']['max_source_positions'],
            src_aux=self.args['task']['source_aux'],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

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

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args['task']['max_source_positions'],)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_score(self, hyps, refs, ids):
        hypotheses, references = dict(), dict()

        for key, pred, tgt in zip(ids, hyps, refs):
            hypotheses[key] = [pred]
            references[key] = tgt if isinstance(tgt, list) else [tgt]

        bleu, rouge_l, meteor = summarization_metrics.eval_accuracies(hypotheses, references)

        return bleu, rouge_l, meteor

    def encode_input(self, input, tokenizer=None):
        if tokenizer is not None:
            input = ''.join(char if str.isalnum(char) else ' ' for char in input)  # for python_wan dataset
            input = tokenizer(input)
        input = input[:self.args['task']['max_source_positions']]
        input = [self.src_dict.index(token) for token in input] + [self.src_dict.eos()]
        input = torch.Tensor(input).long()  # [bsz, len]
        input = {
            'net_input': {
                'src_tokens': input.unsqueeze(dim=0),
                'src_lengths': torch.LongTensor([input.numel()]),
            },
        }
        return input

    def decode_output(self, output):
        output = output[0][0]['tokens']
        output = self.tgt_dict.string(output)
        if not str.endswith(output, "."):
            output += "."
        return output
