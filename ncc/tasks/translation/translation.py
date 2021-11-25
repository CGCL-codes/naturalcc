# -*- coding: utf-8 -*-

import json
import os

import torch

from ncc import (
    tokenizers,
    LOGGER,
)
from ncc.data.summarization.language_pair_dataset import LanguagePairDataset
from ncc.data.tools import data_utils
from ncc.data.wrappers import (
    AppendTokenDataset,
    PortionDataset,
    PrependTokenDataset,
    TruncateDataset,
    StripTokenDataset,
)
from ncc.eval.summarization import summarization_metrics
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.utils import utils
from ncc.utils.logging import metrics


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    dataset_impl,
    left_pad_source, left_pad_target,
    max_source_positions, max_target_positions,
    prepend_bos=False,
    truncate_source=False, append_source_id=False,
    truncate_target=False,
    append_eos_to_target=False,
    portion=None,
):
    # load source dataset
    src_path = os.path.join(data_path, '{}.{}'.format(split, src))
    src_dataset = data_utils.load_indexed_dataset(src_path, dictionary=src_dict, dataset_impl=dataset_impl)

    # load target dataset
    tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
    tgt_dataset = data_utils.load_indexed_dataset(tgt_path, dictionary=tgt_dict, dataset_impl=dataset_impl)

    # few-shot learning
    if portion is not None and split == 'train':
        LOGGER.info('set {}.{} portion to {}'.format(split, src, portion))
        src_dataset = PortionDataset(src_dataset, portion)
        LOGGER.info('set {}.{} portion to {}'.format(split, tgt, portion))
        tgt_dataset = PortionDataset(tgt_dataset, portion)

    # prepend BOS
    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    # append EOS
    if truncate_source:
        LOGGER.info('truncate {}.{} to {}'.format(split, src, max_source_positions))
        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(src_dataset, src_dict.eos()),
                max_source_positions - 1 - int(append_source_id),
            ),
            src_dict.eos(),
        )

    if truncate_target:
        LOGGER.info('truncate {}.{} to {}'.format(split, tgt, max_target_positions))
        tgt_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(tgt_dataset, tgt_dict.eos()),
                max_target_positions - 1 - int(append_source_id),
            ),
            tgt_dict.eos(),
        )

    # append [lang]
    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index(f"[{src}]"))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index(f"[{tgt}]"))
        eos = tgt_dict.index(f"[{tgt}]")

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
        shuffle=(split == "train"),
    )


@register_task('translation')
class TranslationTask(NccTask):
    """
    This task`SummarizationTask` will handle file as follows:
        1) truncate source/target sentence
        2) append eos for target sentence for offset
        3) move eos of target sentence to the head of it, e.g.
            decoder input: <eos> a b c
            ground truth: a b c <eos>
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

        share_dict = args['task'].get('share_dict', False)
        if share_dict:
            src_dict = tgt_dict = cls.load_dictionary(os.path.join(paths[0], "dict.jsonl"))
        else:
            # load dictionaries
            src_dict = cls.load_dictionary(os.path.join(paths[0], f"{args['task']['source_lang']}.dict.jsonl"))
            tgt_dict = cls.load_dictionary(os.path.join(paths[0], f"{args['task']['target_lang']}.dict.jsonl"))
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
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            dataset_impl=self.args['dataset']['dataset_impl'],
            left_pad_source=self.args['task']['left_pad_source'],
            left_pad_target=self.args['task']['left_pad_target'],
            max_source_positions=self.args['task']['max_source_positions'],
            max_target_positions=self.args['task']['max_target_positions'],
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
            detok_args = json.loads(
                args['task']['eval_bleu_detok_args'] if args['task']['eval_bleu_detok_args'] else '{}'
            )
            self.tokenizer = tokenizers.build_tokenizer(
                dict(tokenizer=args['task'].get('eval_bleu_detok', '{}'), **detok_args)
            )
            self.sequence_generator = self.build_generator([model], args)
        return model

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

            bleu, rouge_l, meteor = self._inference_score(hyps, refs, ids)
            logging_output['bleu'] = round(bleu, 4)
            logging_output['rouge_l'] = round(rouge_l, 4)
            logging_output['meteor'] = round(meteor, 4)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args['task']['eval_bleu']:
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

    def encode_input(self, input, tokenize):
        if tokenize:
            input = ''.join(char if str.isalnum(char) else ' ' for char in input)  # for python_wan dataset
            input = tokenize(input)
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
