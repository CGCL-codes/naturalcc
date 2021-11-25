# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import os
import json
import numpy as np
from ncc.utils.logging import metrics
from ncc import LOGGER
from ncc.data.dictionary import Dictionary
from ncc.tasks.ncc_task import NccTask
from ncc.tasks import register_task
from ncc.utils import utils
from ncc import tokenizers
from ncc.data import indexed_dataset
from ncc.data.tools import data_utils
from ncc.data.wrappers.append_token_dataset import AppendTokenDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.data.wrappers.strip_token_dataset import StripTokenDataset
from ncc.data.wrappers.concat_dataset import ConcatDataset
from ncc.data.wrappers.prepend_token_dataset import PrependTokenDataset
from ncc.utils.fed_utils import TeacherOutputDataset
from ncc.data.summarization.universal_dataset import UniversalDataset

EVAL_BLEU_ORDER = 4


def load_langpair_dataset(
    args,
    programming_langs,
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False, is_distill=False,
):
    def split_exists(split, src, data_path):
        filename = os.path.join(data_path, '{}.{}'.format(split, src))  # -{}.{} , tgt, lang
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    topk_idxs = []
    topk_probs = []
    expert_scores = []
    dataset_ids = []
    lng_borders = [0]
    is_train = split == 'train'

    for ds_idx, program_lang in enumerate(programming_langs):
        lang_data_path = os.path.join(data_path, program_lang)

        split_k = split
        # infer langcode
        if split_exists(split_k, src, lang_data_path):
            prefix = os.path.join(lang_data_path, '{}.'.format(split_k))  # {}-{}. , src, tgt
        elif split_exists(split_k, tgt, lang_data_path):
            prefix = os.path.join(lang_data_path, '{}.'.format(split_k))  # {}-{}. , tgt, src
        else:
            raise NotImplementedError('No data in {}'.format(lang_data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)
        length = len(src_dataset)
        lng_borders.append(lng_borders[-1] + length)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt, tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        for i in range(length):
            dataset_ids.append(ds_idx)

        if is_distill and is_train:  # distill only for train
            path = '{}_{}_{}_topk_idx'.format(lang_data_path, src, tgt)
            topk_idxs.append(TeacherOutputDataset(path))
            path = '{}_{}_{}_topk_prob'.format(lang_data_path, src, tgt)
            topk_probs.append(TeacherOutputDataset(path))
            expert_bleu = os.path.join(data_path, 'expert_bleu_{}_{}_{}.json'.format(program_lang, src, tgt))
            expert_bleu = json.load(open(expert_bleu))
            expert_scores.append(expert_bleu[f"bleu_{program_lang}"])

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    sample_ratios = [1] * len(src_datasets)
    sample_ratios[0] = upsample_primary
    src_dataset = ConcatDataset(src_datasets, sample_ratios)
    if len(tgt_datasets) > 0:
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
    else:
        tgt_dataset = None

    LOGGER.info('src data: {}, tgt data: {}'.format(len(src_dataset), len(tgt_dataset)))

    if is_distill and is_train:  # distill only for train
        topk_idx_dataset = ConcatDataset(topk_idxs)
        topk_probs_dataset = ConcatDataset(topk_probs)
        assert len(topk_probs_dataset) == len(src_dataset), (len(topk_probs_dataset), len(src_dataset))
        assert len(topk_idx_dataset) == len(src_dataset)
    else:
        topk_idx_dataset = None
        topk_probs_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    return UniversalDataset(
        args,
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        dataset_ids=dataset_ids, lng_borders=lng_borders, dataset_names=programming_langs,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,

        topk_idxs=topk_idx_dataset, topk_probs=topk_probs_dataset,
        expert_scores=expert_scores, is_train=is_train,
    )


@register_task('universal_summarization')
class UniversalSummarizationTask(NccTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.id2lng = sorted(['java', 'python', 'ruby', 'php', 'go', 'javascript', 'csharp'])
        self.lng2id = {v: k for k, v in enumerate(self.id2lng)}

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        # options.eval_bool
        args['task']['left_pad_source'] = bool(args['task']['left_pad_source'])
        args['task']['left_pad_target'] = bool(args['task']['left_pad_target'])
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        # find language pair automatically
        if args['task']['source_lang'] is None or args['task']['target_lang'] is None:
            # args['task'].source_lang, args['task'].target_lang = data_utils.infer_language_pair(args.data[0])
            args['task']['source_lang'], args['task']['target_lang'] = data_utils.infer_language_pair(paths[0])
        if args['task']['source_lang'] is None or args['task']['target_lang'] is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format(args['task']['source_lang'])))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format(args['task']['target_lang'])))

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['source_lang'], len(src_dict)))
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['target_lang'], len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

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
            self.args,
            self.args['task']['programming_langs'],
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args['dataset']['dataset_impl'],
            upsample_primary=self.args['task']['upsample_primary'],
            left_pad_source=self.args['task']['left_pad_source'],
            left_pad_target=self.args['task']['left_pad_target'],
            max_source_positions=self.args['task']['max_source_positions'],
            max_target_positions=self.args['task']['max_target_positions'],
            load_alignments=self.args['task']['load_alignments'],
            truncate_source=self.args['task']['truncate_source'],
            is_distill=self.args['kd']['is_distill'],
        )

    def build_model(self, args):
        model = super().build_model(args)
        if args['task']['eval_bleu']:
            assert args['task']['eval_bleu_detok'] is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            # detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            detok_args = json.loads(
                args['task']['eval_bleu_detok_args'] if args['task']['eval_bleu_detok_args'] else '{}')
            self.tokenizer = tokenizers.build_tokenizer(
                dict(
                    tokenizer=args['task']['eval_bleu_detok'] if args['task']['eval_bleu_detok'] else None,
                    # getattr(args, 'eval_bleu_detok', None),
                    **detok_args
                ))
            # The gen_args parameters have been set in the yml file
            # gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            # self.sequence_generator = self.build_generator(Namespace(**gen_args))
            self.sequence_generator = self.build_generator(args)
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args['task']['eval_bleu']:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args['task']['eval_bleu']:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

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
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args['model']['max_source_positions'], self.args['model']['max_target_positions'])

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args['task']['eval_bleu_remove_bpe'],
                escape_unk=escape_unk,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args['task']['eval_bleu_print_samples']:
            LOGGER.info('example hypothesis: ' + hyps[0])
            LOGGER.info('example reference: ' + refs[0])
        # tokenize = sacrebleu.DEFAULT_TOKENIZER if not self.args['task']['eval_tokenized_bleu'] else 'none'
        # return sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)
        if self.args['task']['eval_tokenized_bleu']:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
