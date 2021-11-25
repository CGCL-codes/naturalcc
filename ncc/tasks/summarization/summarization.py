# -*- coding: utf-8 -*-

import json
import os
from functools import lru_cache

import numpy as np
import torch

from ncc import (
    tokenizers,
    LOGGER,
)
from ncc.data import (
    indexed_dataset,
)
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

EVAL_BLEU_ORDER = 4


class IndexedRawTextDataset(NccDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.lines.append(line.strip('\n'))
                tokens = dictionary.encode_line(
                    line, tokenization._space_tokenizer, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)


def _load_dataset(path, impl, dict):
    if impl == 'raw':
        src_dataset = IndexedRawTextDataset(path=path, dictionary=dict)
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


@register_task('summarization')
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

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0

        dict = args['task'].get('dict', None)
        dict_type = args['task'].get('dict_type', None)
        if dict is None and dict_type is None:
            # load dictionaries
            src_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.jsonl'.format(args['task']['source_lang'])))
            tgt_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.jsonl'.format(args['task']['target_lang'])))
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
            LOGGER.info('[{}] dictionary: {} types'.format(args['task']['source_lang'], len(src_dict)))
            LOGGER.info('[{}] dictionary: {} types'.format(args['task']['target_lang'], len(tgt_dict)))
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
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args['task']['source_lang'], self.args['task']['target_lang']

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            dataset_impl=self.args['dataset']['dataset_impl'],
            left_pad_source=self.args['task']['left_pad_source'],
            left_pad_target=self.args['task']['left_pad_target'],
            max_source_positions=self.args['task']['max_source_positions'],
            max_target_positions=self.args['task']['max_target_positions'],
            load_alignments=self.args['task']['load_alignments'],
            truncate_source=self.args['task']['truncate_source'],
            truncate_target=self.args['task']['truncate_target'],
            append_eos_to_target=self.args['task']['append_eos_to_target'],
            portion=self.args['dataset'].get('portion', None),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_model(self, args):
        model = super().build_model(args)
        if args['task']['eval_bleu']:
            assert args['task']['eval_bleu_detok'] is not None, (
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
                args['task']['eval_bleu_detok_args'] if args['task']['eval_bleu_detok_args'] else '{}'
            )
            self.tokenizer = tokenizers.build_tokenizer(
                dict(tokenizer=args['task'].get('eval_bleu_detok', '{}'), **detok_args)
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
                self.args['task']['eval_bleu_remove_bpe'],
                escape_unk=escape_unk,
                trunc_eos=trunc_eos,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            if len(s) == 0:
                s = '0'  # if predict sentence is null, use '0'
            return s

        if self.args['task']['eval_bleu']:
            gen_out = self.inference_step(self.sequence_generator, [model], sample)
            ids = sample['id'].tolist()
            hyps, refs = [], []
            for i in range(len(gen_out)):
                hyps.append(decode(gen_out[i][0]['tokens']))
                refs.append(decode(
                    utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                ))
            if self.args['task']['eval_with_sacrebleu']:
                import sacrebleu
                tokenize = sacrebleu.DEFAULT_TOKENIZER if not self.args['task']['eval_tokenized_bleu'] else 'none'
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
        if self.args['task']['eval_bleu']:

            if self.args['task']['eval_with_sacrebleu']:
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
        return (self.args['task']['max_source_positions'], self.args['task']['max_target_positions'])

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
