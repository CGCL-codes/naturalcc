# -*- coding: utf-8 -*-

import json
import os
from functools import lru_cache

import numpy as np
import torch
from dataclasses import dataclass, field

import logging
from typing import Optional

from ncc import (
    tokenizers,
    LOGGER,
)
from ncc.data import (
    indexed_dataset,
)
from ncc.dataclass import ChoiceEnum, NccDataclass
from ncc.data.dictionary import Dictionary
from ncc.data.ncc_dataset import NccDataset
from ncc.data.summarization.language_pair_dataset import LanguagePairDataset
from ncc.data.wrappers.append_token_dataset import AppendTokenDataset
from ncc.data.wrappers.portion_dataset import PortionDataset
from ncc.data.wrappers.prepend_token_dataset import PrependTokenDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.eval.summarization import summarization_metrics
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.tokenizers import tokenization
from ncc.utils import utils
from ncc.utils.logging import metrics

from omegaconf import II


EVAL_BLEU_ORDER = 4



def _load_dataset(path, impl, dict):
    if impl == 'raw':
        src_dataset = indexed_dataset.IndexedRawTextDataset(path=path, dictionary=dict, tokenization=tokenization._space_tokenizer)
    elif impl == 'mmap':
        # mmap dataset has been numberized, no need for dict
        src_dataset = indexed_dataset.MMapIndexedDataset(path=path)
    else:
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    return src_dataset


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    dataset_impl,
    # combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target,
    max_source_positions, max_target_positions,
    prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False,
    truncate_target=False,
    append_eos_to_target=False,
    portion=None,
):
    # load source dataset
    src_path = os.path.join(data_path, '{}.{}'.format(split, src))
    print("!!!!" + src_path)
    src_dataset = _load_dataset(path=src_path, impl=dataset_impl, dict=src_dict)

    if truncate_source:
        LOGGER.info('truncate {}.{} to {}'.format(split, src, max_source_positions))
        src_dataset = TruncateDataset(src_dataset, max_source_positions)

    if portion is not None and split == 'train':
        LOGGER.info('set {}.{} portion to {}'.format(split, src, portion))
        src_dataset = PortionDataset(src_dataset, portion)

    # load target dataset
    tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
    tgt_dataset = _load_dataset(path=tgt_path, impl=dataset_impl, dict=tgt_dict)
    if truncate_target:
        LOGGER.info('truncate {}.{} to {}'.format(split, tgt, max_target_positions))
        tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    if portion is not None and split == 'train':
        LOGGER.info('set {}.{} portion to {}'.format(split, tgt, portion))
        tgt_dataset = PortionDataset(tgt_dataset, portion)

    # align_dataset = None
    # if load_alignments:
    #     align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
    #     if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
    #         align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    LOGGER.info('loaded {} examples from: {}'.format(len(src_dataset), src_path))
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_dataset), tgt_path))
    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=None, eos=eos,
        remove_eos_from_source=True,
        append_eos_to_target=append_eos_to_target,
        shuffle=(split == 'train'),
    )

@dataclass
class SummarizationConfig(NccDataclass):
    data: str = field(
        default = "none", 
        metadata = {"help": "path to data directory"}
    )
    dataset_impl: Optional[ChoiceEnum(indexed_dataset.get_available_dataset_impl())] = field(
        default = "mmap",
        metadata = {"help": "implementation of dataset"}
    )
    dict: Optional[str] = field(
        default = None,
        metadata = {"help": "path to already existed dict"}
    )
    dict_type: Optional[str] = field(
        default = None,
        metadata = {"help": "type of existed dict"}
    )
    source_lang: str = field(
        default = "code_tokens",
        metadata = {"help": "source language"}
    )
    target_lang: str = field(
        default = "docstring_tokens",
        metadata = {"help": "target language"}
    )
    load_alignments: bool = field(
        default = False,
        metadata = {"help": "load the binarized alignments"},
    )
    left_pad_source: bool = field(
        default = True,
        metadata = {"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default = False,
        metadata = {"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default = 1024,
        metadata = {"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default = 1024,
        metadata = {"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default = 1,
        metadata = {"help": "amount to upsample primary dataset"}
    )
    truncate_source: bool = field(
        default = False,
        metadata = {"help": "truncate source to max-source-positions"},
    )
    truncate_target: bool = field(
        default = False,
        metadata = {"help": "truncate target to max-target-positions"},
    )
    eval_bleu: bool = field(
        default = False,
        metadata = {"help": "evaluation with BLEU scores"}
    )
    eval_bleu_detok: str = field(
        default = 'space',
        metadata = {"help": "detokenizer before computing BLEU. use 'space' to disable detokenization"}
    )
    eval_bleu_detok_args: str = field(
        default = '',
        metadata = {"help": "args for building the tokenizer, if needed"}
    )
    eval_tokenized_bleu: bool = field(
        default = False,
        metadata = {"help": "if setting, we compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: bool = field(
        default = False,
        metadata = {"help": "remove BPE before computing BLEU"}
    )
    eval_bleu_args: str = field(
        default = '',
        metadata = {
            "help": "generation args for BLUE scoring",
            "e.g.": '\'{"beam": 4, "lenpen": 0.6}\''
        }
    )
    eval_bleu_print_samples: bool = field(
        default = False,
        metadata = {"help": "print sample generations during validation"}
    )
    eval_with_sacrebleu: bool = field(
        default = False,
        metadata = {"help": "evalate with sacrebleu"}
    )
    append_eos_to_target: bool = field(
        default = True,
        metadata = {"help": "whether to append eos to target"}
    )
    portion: Optional[float] = field(
        default = None
    )
    
    
    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    use_plasma_view: bool = II("common.use_plasma_view")
    plasma_path: str = II("common.plasma_path")

@register_task('summarization', dataclass=SummarizationConfig)
class SummarizationTask(NccTask):
    """
    This task`SummarizationTask` will handle file as follows:
        1) truncate source/target sentence
        2) append eos for target sentence for offset
        3) move eos of target sentence to the head of it, e.g.
            decoder input: a b c
            ground truth: <eos> a b c
    """

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.args = args

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        dict = args.dict
        dict_type = args.dict_type
        if dict is None and dict_type is None:
            # load dictionaries
            src_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.jsonl'.format(args.source_lang)))
            tgt_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.jsonl'.format(args.target_lang)))
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
            LOGGER.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
            LOGGER.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        else:
            raise NotImplementedError
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
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            dataset_impl=self.args.dataset_impl,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            truncate_target=self.args.truncate_target,
            append_eos_to_target=self.args.append_eos_to_target,
            portion=self.args.portion,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_model(self, args):
        model = super().build_model(args)
        if args.eval_bleu:
            assert args.eval_bleu_detok is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            # detok_args = args['task']['eval_bleu_detok_args'] if args['task']['eval_bleu_detok_args'] else '{}'
            # if args['bpe'] is not None:
            #     self.tokenizer = tokenizers.build_bpe(
            #         dict(bpe=args['task'].get('eval_bleu_detok', '{}'), **detok_args)
            #     )
            # else:
            #     self.tokenizer = tokenizers.build_tokenizer(
            #         dict(tokenizer=args['task'].get('eval_bleu_detok', '{}'), **detok_args)
            #     )
            detok_args = json.loads(
                args.eval_bleu_detok_args if args.eval_bleu_detok_args else '{}'
            )
            self.tokenizer = tokenizers.build_tokenizer(
                dict(tokenizer=args.eval_bleu_detok or '{}', **detok_args)
            )
            self.sequence_generator = self.build_generator([model], args)
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

        def decode(toks, escape_unk=False, trunc_eos=True):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                escape_unk=escape_unk,
                trunc_eos=trunc_eos,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            if len(s) == 0:
                s = '0'  # if predict sentence is null, use '0'
            return s

        if self.args.eval_bleu:
            gen_out = self.inference_step(self.sequence_generator, [model], sample)
            ids = sample['id'].tolist()
            hyps, refs = [], []
            for i in range(len(gen_out)):
                hyps.append(decode(gen_out[i][0]['tokens']))
                refs.append(decode(
                    utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                ))
            if self.args.eval_with_sacrebleu:
                import sacrebleu
                tokenize = sacrebleu.DEFAULT_TOKENIZER if not self.args.eval_tokenized_bleu else 'none'
                bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)
                logging_output['_bleu_sys_len'] = bleu.sys_len
                logging_output['_bleu_ref_len'] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                    logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
            else:
                bleu, rouge_l, meteor = self._inference_score(hyps, refs, ids)
                logging_output['bleu'] = round(bleu, 4)
                logging_output['rouge_l'] = round(rouge_l, 4)
                logging_output['meteor'] = round(meteor, 4)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            if self.args.eval_with_sacrebleu:
                def sum_logs(key):
                    import torch
                    result = sum(log.get(key, 0) for log in logging_outputs)
                    if torch.is_tensor(result):
                        result = result.cpu()
                    return result

                counts, totals = [], []
                for i in range(EVAL_BLEU_ORDER):
                    counts.append(sum_logs('_bleu_counts_' + str(i)))
                    totals.append(sum_logs('_bleu_totals_' + str(i)))

                if max(totals) > 0:
                    # log counts as numpy arrays -- log_scalar will sum them correctly
                    metrics.log_scalar('_bleu_counts', np.array(counts))
                    metrics.log_scalar('_bleu_totals', np.array(totals))
                    metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                    metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                    def compute_bleu(meters):
                        import inspect
                        import sacrebleu
                        fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                        if 'smooth_method' in fn_sig:
                            smooth = {'smooth_method': 'exp'}
                        else:
                            smooth = {'smooth': 'exp'}
                        bleu = sacrebleu.compute_bleu(
                            correct=meters['_bleu_counts'].sum,
                            total=meters['_bleu_totals'].sum,
                            sys_len=meters['_bleu_sys_len'].sum,
                            ref_len=meters['_bleu_ref_len'].sum,
                            **smooth
                        )
                        return round(bleu.score, 6)

                    metrics.log_derived('bleu', compute_bleu)
            else:

                def sum_logs(key):
                    return sum(log.get(key, 0) for log in logging_outputs)

                metrics.log_scalar('bleu', sum_logs('bleu'), round=6)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

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
        input = input[:self.args.max_source_positions]
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
