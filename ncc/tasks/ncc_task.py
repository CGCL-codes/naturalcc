# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict

import torch

from ncc.data import (
    constants,
    iterators,
)
from ncc.data.dictionary import Dictionary
from ncc.data.ncc_dataset import NccDataset
from ncc.data.tools import data_utils
from ncc.tokenizers import tokenization
from ncc.utils import utils
from ncc.utils.logging import metrics


class NccTask(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        pass

    @staticmethod
    def logging_outputs_can_be_summed(criterion) -> bool:
        """
        Whether the logging outputs returned by `train_step` and `valid_step` can
        be summed across workers prior to calling `aggregate_logging_outputs`.
        Setting this to True will improves distributed training speed.
        """
        return criterion.logging_outputs_can_be_summed()

    def __init__(self, args):
        self.args = args
        self.datasets = {}
        self.dataset_to_epoch_iter = {}

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename)

    @classmethod
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8,
        **kwargs,
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
        d = Dictionary(
            pad=kwargs.get('pad', constants.PAD),
            bos=kwargs.get('bos', constants.BOS),
            eos=kwargs.get('eos', constants.EOS),
            unk=kwargs.get('unk', constants.UNK),
            extra_special_symbols=kwargs.get('extra_special_symbols', None),
        )
        for filename in filenames:
            Dictionary.add_file_to_dictionary(
                filename, d, tokenization.tokenize_line, d.eos_word, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args, **kwargs)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        raise NotImplementedError

    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~fairseq.data.NccDataset` corresponding to *split*
        """
        # from ncc.data.fairseq_dataset import NccDataset

        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        if not isinstance(self.datasets[split], NccDataset):
            raise TypeError("Datasets are expected to be of type NccDataset")
        return self.datasets[split]

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.NccDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # For default fairseq task, return same iterator across epochs
        # as datasets are not dynamic, can be overridden in task specific
        # setting.
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]
        assert isinstance(dataset, NccDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices,
                dataset,
                max_positions,
                raise_exception=(not ignore_invalid_inputs),
            )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices,
            dataset.num_tokens,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
        self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter

    def build_model(self, args, config=None):
        """
        Build the :class:`~fairseq.models.BaseNccModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.models.BaseNccModel` instance
        """
        from ncc import models
        # assert 0
        return models.build_model(args, config, self)

    def build_criterion(self, args):
        """
        Build the :class:`~fairseq.criterions.NccCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.NccCriterion` instance
        """
        from ncc import criterions

        return criterions.build_criterion(args, self)

    def build_generator(self, models, args, **extra_gen_cls_kwargs):
        if args['model']['arch'] in ['codenn']:
            from ncc.eval.summarization.codenn_generator import CodeNNGenerator
            return CodeNNGenerator(self.target_dictionary, **extra_gen_cls_kwargs)
        # elif 'transformer' in args['model']['arch']:
        #     from ncc.eval.summarization.transformer_generator import TransformerGenerator
        #     return TransformerGenerator(self.target_dictionary, **extra_gen_cls_kwargs)
        # elif args['model']['arch'] in ['deepcom']:
        #     from ncc.eval.generator.deepcom_generator import DeepComGenerator
        #     return DeepComGenerator(self.target_dictionary, **extra_gen_cls_kwargs)
        else:
            from ncc.eval.summarization.sequence_generator_fair import SequenceGenerator
            from ncc.eval.summarization import search
            search_strategy = search.BeamSearch(self.target_dictionary)
            return SequenceGenerator(
                self.target_dictionary,
                beam_size=extra_gen_cls_kwargs.get('beam', args['eval']['beam']),
                max_len_b=extra_gen_cls_kwargs.get('max_len_b', args['eval']['max_len_b']),
                search_strategy=search_strategy,
            )

    def build_completor(self, models, args):
        from ncc.eval.completion.sequence_completor import SequenceCompletor

        return SequenceCompletor()

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
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, **kwargs):
        with torch.no_grad():
            return generator.generate(models, sample, **kwargs)

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        pass

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        """[deprecated] Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            "The aggregate_logging_outputs API is deprecated. "
            "Please use the reduce_metrics API instead."
        )
        with metrics.aggregate() as agg:
            self.reduce_metrics(logging_outputs, criterion)
            return agg.get_smoothed_values()

    def reduce_metrics(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        # backward compatibility for tasks that override aggregate_logging_outputs
        base_func = NccTask.aggregate_logging_outputs
        self_func = getattr(self, "aggregate_logging_outputs").__func__
        if self_func is not base_func:
            utils.deprecation_warning(
                "Tasks should implement the reduce_metrics API. "
                "Falling back to deprecated aggregate_logging_outputs API."
            )
            agg_logging_outputs = self.aggregate_logging_outputs(
                logging_outputs, criterion
            )
            for k, v in agg_logging_outputs.items():
                metrics.log_scalar(k, v)
            return

        if not any("ntokens" in log for log in logging_outputs):
            warnings.warn(
                "ntokens not found in Criterion logging outputs, cannot log wpb or wps"
            )
        else:
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
            metrics.log_scalar("wpb", ntokens, priority=180, round=1)
            metrics.log_speed("wps", ntokens, priority=90, round=1)

        if not any("nsentences" in log for log in logging_outputs):
            warnings.warn(
                "nsentences not found in Criterion logging outputs, cannot log bsz"
            )
        else:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
            metrics.log_scalar("bsz", nsentences, priority=190, round=1)

        criterion.__class__.reduce_metrics(logging_outputs)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    def encode_input(self, input):
        """Return the processed input to be fed into task (for interface).
        Examples:
            in Seq2Seq model of summarization task, users input "def addition ..."
            encode the input into [index("def"), index("addition"), ..., <eos>]
        """
        raise NotImplementedError(f"encode_input function of {self} has not been implemented.")

    def decode_output(self, output):
        """Return the output (for interface)
        Examples:
            in Seq2Seq model of summarization task, the model return of the output [xx, xx, xx] of the code "def addition ..."
            decode the outputput into "addition operator"
        """
        raise NotImplementedError(f"decode_output function of {self} has not been implemented.")


class NccComplTask(NccTask):

    @classmethod
    def load_dictionaries(cls, filenames):
        dicts = OrderedDict({})
        for key, filename in filenames.items():
            dicts[key] = cls.load_dictionary(filename)
        return dicts

    @property
    def source_dictionary(self, key):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    @property
    def target_dictionary(self, key):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError
