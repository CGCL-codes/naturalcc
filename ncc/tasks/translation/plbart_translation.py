# -*- coding: utf-8 -*-

import json
import os
from functools import lru_cache

import numpy as np
import torch
import itertools

from ncc import (
    tokenizers,
    LOGGER,
)
from ncc.data import (
    indexed_dataset,
)
from ncc.data.dictionary import Dictionary
from ncc.data.ncc_dataset import NccDataset
from ncc.data.wrappers.append_token_dataset import AppendTokenDataset
from ncc.data.wrappers.portion_dataset import PortionDataset
from ncc.data.wrappers.prepend_token_dataset import PrependTokenDataset
from ncc.data.wrappers.strip_token_dataset import StripTokenDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.eval.summarization import summarization_metrics
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.tokenizers import tokenization
from ncc.utils import utils
from ncc.utils.logging import metrics
from ncc.data import constants
from ncc.data.translation.plbart_pair_dataset import PLBartPairDataset
from ncc.utils.file_ops import (
    file_io, json_io,
)

EVAL_BLEU_ORDER = 4


def _load_dataset(path, impl, dict):
    if impl == 'mmap':
        # mmap dataset has been numberized, no need for dict
        src_dataset = indexed_dataset.MMapIndexedDataset(path=path)
    elif impl == 'pkl':
        src_dataset = file_io.open(f"{path}.pkl", 'rb')
    else:
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    return src_dataset


def load_langpair_dataset(
    data_path, split,
    src_dict, tgt_dict,
    src_lang, tgt_lang,
    dataset_impl,
    max_source_positions=None, max_target_positions=None,
):
    # load source dataset
    src_path = os.path.join(data_path, src_lang, f'{split}.code_tokens')
    src_tokens = _load_dataset(path=src_path, impl=dataset_impl, dict=src_dict)
    src_code = None
    if split != 'train':
        src_code = [src_dict.string(src_tokens[idx], bpe_symbol='sentencepiece') for idx in range(len(src_tokens))]
    src_tokens = AppendTokenDataset(
        TruncateDataset(
            StripTokenDataset(src_tokens, src_dict.eos()),
            max_source_positions - 2,
        ),
        src_dict.eos()
    )
    src_tokens = AppendTokenDataset(src_tokens, src_dict.index('[{}]'.format(src_lang)))
    LOGGER.info('truncate {}/{}.code_tokens to {}'.format(src_lang, split, max_source_positions))

    # load target dataset
    tgt_path = os.path.join(data_path, tgt_lang, f'{split}.code_tokens')
    tgt_tokens = _load_dataset(path=tgt_path, impl=dataset_impl, dict=tgt_dict)
    tgt_code = None
    if split != 'train':
        tgt_code = [tgt_dict.string(tgt_tokens[idx], bpe_symbol='sentencepiece') for idx in range(len(tgt_tokens))]
    tgt_tokens = AppendTokenDataset(
        TruncateDataset(
            StripTokenDataset(tgt_tokens, tgt_dict.eos()),
            max_target_positions - 2,
        ),
        tgt_dict.eos()
    )
    tgt_tokens = AppendTokenDataset(tgt_tokens, tgt_dict.index('[{}]'.format(tgt_lang)))
    LOGGER.info('truncate {}/{}.code_tokens to {}'.format(tgt_lang, split, max_target_positions))

    return PLBartPairDataset(
        src_dict, tgt_dict,
        src_tokens, src_tokens.sizes, src_code=src_code,
        tgt=tgt_tokens, tgt_sizes=tgt_tokens.sizes, tgt_code=tgt_code,
        src_lang=src_lang, tgt_lang=tgt_lang,
        max_source_positions=max_source_positions, max_target_positions=max_target_positions,
        eos=tgt_dict.index('[{}]'.format(tgt_lang)),
        shuffle=(split == 'train'),
    )


@register_task('plbart_translation')
class PLBartTranslationTask(NccTask):
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
        dict = cls.load_dictionary(os.path.join(paths[0], 'dict.jsonl'))
        for l in ['java', 'python', 'en_XX']:
            dict.add_symbol("[{}]".format(l))
        dict.add_symbol(constants.MASK)

        if f"[{args['task']['source_lang']}]" not in dict:
            dict.add_symbol(f"[{args['task']['source_lang']}]")
        if f"[{args['task']['target_lang']}]" not in dict:
            dict.add_symbol(f"[{args['task']['target_lang']}]")

        src_dict = tgt_dict = dict

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['source_lang'], len(src_dict)))
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['target_lang'], len(tgt_dict)))
        return cls(args, src_dict, tgt_dict)

    @classmethod
    def build_dictionary(
        cls, filenames, tokenize_func,
        workers=1, threshold=-1, nwords=-1, padding_factor=1,
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

        self.datasets[split] = load_langpair_dataset(
            data_path, split,
            src_dict=self.src_dict,
            tgt_dict=self.tgt_dict,
            src_lang=self.args['task']['source_lang'],
            tgt_lang=self.args['task']['target_lang'],
            dataset_impl=self.args['dataset']['dataset_impl'],
            max_source_positions=self.args['task']['max_source_positions'],
            max_target_positions=self.args['task']['max_target_positions'],
        )

    def build_model(self, args):
        model = super().build_model(args)
        if args['task']['eval_bleu']:
            assert args['task']['eval_bleu_detok'] is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(
                args['task']['eval_bleu_detok_args'] if args['task']['eval_bleu_detok_args'] else '{}'
            )
            self.tokenizer = tokenizers.build_tokenizer(
                dict(tokenizer=args['task'].get('eval_bleu_detok', '{}'), **detok_args)
            )

            gen_args = args['task']['eval_bleu_args'] or {}
            self.sequence_generator = self.build_generator(
                [model], args, **gen_args, eos=self.tgt_dict.index(f"[{self.args['task']['target_lang']}]")
            )
        return model

    @property
    def target_dictionary(self):
        return self.src_dict

    @property
    def source_dictionary(self):
        return self.tgt_dict

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        mode = self.args['dataset']['valid_subset']

        def decode(toks, escape_unk=False, trunc_eos=True):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args['task']['eval_bleu_remove_bpe'],
                extra_symbols_to_ignore=[self.dataset(mode).eos],
                escape_unk=escape_unk,
                trunc_eos=trunc_eos,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            if len(s) == 0:
                s = '0'  # if predict sentence is null, use '0'
            return s

        if self.args['task']['eval_bleu']:
            gen_out = self.inference_step(self.sequence_generator, [model], sample, bos_token=self.dataset(mode).eos)
            ids = sample['id'].tolist()
            hyps, refs = [], []
            for i in range(len(ids)):
                hyps.append(decode(gen_out[i][0]['tokens']))
                # refs.append(decode(
                #     utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                #     escape_unk=True,  # don't count <unk> as matches to the hypo
                # ))
                refs.append(
                    self.dataset(mode).tgt_code[sample['id'][i].item()]
                )
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

    def _inference_score(self, hyps, refs, ids):
        hypotheses, references = dict(), dict()

        for key, pred, tgt in zip(ids, hyps, refs):
            hypotheses[key] = [pred]
            references[key] = tgt if isinstance(tgt, list) else [tgt]

        bleu, rouge_l, meteor = summarization_metrics.eval_accuracies(hypotheses, references)

        return bleu, rouge_l, meteor

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
