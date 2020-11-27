# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import torch
from ncc.utils import utils
from ncc.logging import metrics
from ncc.data.tools import data_utils
from ncc.data import iterators
from ncc.data.dictionary import Dictionary
from ncc.data.ncc_dataset import NccDataset
from ncc.eval import search
from ncc.data.tokenizers import space_tokenizer


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
        if filename.endswith('.txt'):
            return Dictionary.load(filename)
        else:
            return Dictionary.load_json(filename)

    @classmethod
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
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
            Dictionary.add_file_to_dictionary(
                filename, d, space_tokenizer, d.eos_word, workers
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

    def build_tokenizer(self, args):
        """
        Build the :class:`~fairseq.criterions.NccCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.NccCriterion` instance
        """
        from ncc.data import tokenizers

        return tokenizers.build_tokenization(args, self)

    def build_generator(self, args, extra_gen_cls_kwargs=None):
        if args['model']['arch'] in ['neural_transformer_summarization']:
            from ncc.eval.sequence_generator import TransformerSequenceGenerator
            return TransformerSequenceGenerator(
                self.target_dictionary,
                beam_size=args['eval']['beam'],  # getattr(args, "beam", 5),
                max_len_a=args['eval']['max_len_a'],  # getattr(args, "max_len_a", 0),
                max_len_b=args['eval']['max_len_b'],  # getattr(args, "max_len_b", 200),
                min_len=args['eval']['min_len'],  # getattr(args, "min_len", 1),
                normalize_scores=(not args['eval']['unnormalized']),  # (not getattr(args, "unnormalized", False)),
                len_penalty=args['eval']['lenpen'],  # getattr(args, "lenpen", 1),
                unk_penalty=args['eval']['unkpen'],  # getattr(args, "unkpen", 0),
                temperature=args['eval']['temperature'],  # getattr(args, "temperature", 1.0),
                match_source_len=args['eval']['match_source_len'],  # getattr(args, "match_source_len", False),
                no_repeat_ngram_size=args['eval']['no_repeat_ngram_size'],  # getattr(args, "no_repeat_ngram_size", 0),
                # search_strategy=search_strategy,
                # **extra_gen_cls_kwargs,
            )
        elif args['model']['arch'] in ['seq2seq']:
            from ncc.eval.sequence_generator import LSTMSequenceGenerator
            return LSTMSequenceGenerator(
                self.target_dictionary,
                beam_size=args['eval']['beam'],  # getattr(args, "beam", 5),
                max_len_a=args['eval']['max_len_a'],  # getattr(args, "max_len_a", 0),
                max_len_b=args['eval']['max_len_b'],  # getattr(args, "max_len_b", 200),
                min_len=args['eval']['min_len'],  # getattr(args, "min_len", 1),
                normalize_scores=(not args['eval']['unnormalized']),  # (not getattr(args, "unnormalized", False)),
                len_penalty=args['eval']['lenpen'],  # getattr(args, "lenpen", 1),
                unk_penalty=args['eval']['unkpen'],  # getattr(args, "unkpen", 0),
                temperature=args['eval']['temperature'],  # getattr(args, "temperature", 1.0),
                match_source_len=args['eval']['match_source_len'],  # getattr(args, "match_source_len", False),
                no_repeat_ngram_size=args['eval']['no_repeat_ngram_size'],  # getattr(args, "no_repeat_ngram_size", 0),
                # search_strategy=search_strategy,
                # **extra_gen_cls_kwargs,
            )

    def build_generator_fair(
        self, args,
        seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if args['eval']['score_reference']:
            from ncc.eval.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args['eval'], "print_alignment", False),
            )

        from ncc.eval.sequence_generator_fair import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = args['eval']['sampling']  # getattr(args, "sampling", False)
        sampling_topk = args['eval']['sampling_topk']  # getattr(args, "sampling_topk", -1)
        sampling_topp = args['eval']['sampling_topp']  # getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = args['eval']['diverse_beam_groups']  # getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = args['eval']['diverse_beam_strength']  # getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = args['eval']['match_source_len']  # getattr(args, "match_source_len", False)
        diversity_rate = args['eval']['diversity_rate']  # getattr(args, "diversity_rate", -1)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        if seq_gen_cls is None:
            if getattr(args['eval'], "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment  # TBC
            else:
                seq_gen_cls = SequenceGenerator
        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        return seq_gen_cls(
            self.target_dictionary,
            beam_size=args['eval']['beam'],  # getattr(args, "beam", 5),
            max_len_a=args['eval']['max_len_a'],  # getattr(args, "max_len_a", 0),
            max_len_b=args['eval']['max_len_b'],  # getattr(args, "max_len_b", 200),
            min_len=args['eval']['min_len'],  # getattr(args, "min_len", 1),
            normalize_scores=(not args['eval']['unnormalized']),  # (not getattr(args, "unnormalized", False)),
            len_penalty=args['eval']['lenpen'],  # getattr(args, "lenpen", 1),
            unk_penalty=args['eval']['unkpen'],  # getattr(args, "unkpen", 0),
            temperature=args['eval']['temperature'],  # getattr(args, "temperature", 1.0),
            match_source_len=args['eval']['match_source_len'],  # getattr(args, "match_source_len", False),
            no_repeat_ngram_size=args['eval']['no_repeat_ngram_size'],  # getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def build_completor(self, models, args):
        from ncc.eval.sequence_completor import SequenceCompletor

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

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate(models, sample)  # , prefix_tokens=prefix_tokens

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
