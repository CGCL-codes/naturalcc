import json
import os

import torch

from ncc import (
    tokenizers,
    LOGGER,
)
from ncc.data import indexed_dataset
from ncc.data.dictionary import Dictionary
from ncc.data.summarization.be_language_pair_dataset import BELanguagePairDataset
from ncc.data.wrappers.append_token_dataset import AppendTokenDataset
from ncc.data.wrappers.prepend_token_dataset import PrependTokenDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.eval.summarization import summarization_metrics
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.utils import utils
from ncc.utils.logging import metrics

EVAL_BLEU_ORDER = 4


def _load_dataset(path, impl, dict):
    if impl == 'raw':
        src_dataset = indexed_dataset.IndexedRawTextDataset(path=path, dictionary=dict)
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
    prepend_bos=True,
    append_eos=True,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    truncate_target=False,
):
    # truncate sentence for prepend <bos> and append <eos>
    max_target_positions -= int(prepend_bos) + int(append_eos)

    # load source dataset
    src_path = os.path.join(data_path, '{}.{}'.format(split, src))
    src_dataset = _load_dataset(path=src_path, impl=dataset_impl, dict=src_dict)

    if truncate_source:
        # sntn => sntn[:max_source_positions]
        LOGGER.info('truncate {}.{} to {}'.format(split, src, max_source_positions))
        src_dataset = TruncateDataset(src_dataset, max_source_positions)

    # load target dataset
    tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
    tgt_dataset = _load_dataset(path=tgt_path, impl=dataset_impl, dict=tgt_dict)
    if truncate_target:
        # sntn => sntn[:max_target_positions]
        LOGGER.info('truncate {}.{} to {}'.format(split, tgt, max_target_positions))
        tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions)  # 2 for BOS and EOS
    # sntn[:max_target_positions] => <bos> sntn[:max_target_positions]
    if prepend_bos:
        tgt_dataset = PrependTokenDataset(tgt_dataset, token=tgt_dict.bos())
    if append_eos:
        tgt_dataset = AppendTokenDataset(tgt_dataset, token=tgt_dict.eos())
    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    LOGGER.info('loaded {} examples from: {}'.format(len(src_dataset), src_path))
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_dataset), tgt_path))
    return BELanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=None,
        bos=src_dict.bos(),
        eos=src_dict.eos(),
        shuffle=(split == 'train'),
    )


@register_task('be_summarization')
class BESummarizationTask(NccTask):
    """
    This task`SummarizationTask` will handle file as follows:
        1) truncate source/target sentence
        2) append <eos>/<bos>
        3) move eos of target sentence to the head of it, e.g.
            decoder input: <bos> a b c <eos>
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
        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.jsonl'.format(args['task']['source_lang'])))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.jsonl'.format(args['task']['target_lang'])))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['source_lang'], len(src_dict)))
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
            left_pad_target=self.args['task']['left_pad_target'],
            max_source_positions=self.args['task']['max_source_positions'],
            max_target_positions=self.args['task']['max_target_positions'],
            load_alignments=self.args['task']['load_alignments'],
            truncate_source=self.args['task']['truncate_source'],
            truncate_target=self.args['task']['truncate_target'],
            prepend_bos=kwargs.get('prepend_bos', True),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return BELanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_model(self, args, **extra_gen_cls_kwargs):
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
            self.sequence_generator = self.build_generator([model], args, **extra_gen_cls_kwargs)

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
            if len(s) == 0:
                s = '0'  # if predict sentence is null, use '0'
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(self.sequence_generator, [model], sample, bos_token=self.target_dictionary.bos())
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

        # if self.args['model']['arch'] in ['codenn']:
        #     bleu, rouge_l, meteor = eval_utils.eval_accuracies(hypotheses, references, mode='test')
        # else:
        #     bleu, rouge_l, meteor = eval_utils.eval_accuracies(hypotheses, references)
        bleu, rouge_l, meteor = summarization_metrics.eval_accuracies(hypotheses, references)

        return bleu, rouge_l, meteor

    def encode_input(self, input, tokenizer=None):
        if tokenizer is not None:
            input = ''.join(char if str.isalnum(char) else ' ' for char in input)  # for python_wan dataset
            input = tokenizer(input)
        input = input[:self.args['task']['max_source_positions']]
        input = torch.Tensor([self.src_dict.index(token) for token in input]).long()
        input = {
            'net_input': {
                'src_tokens': input.unsqueeze(dim=0),
                'src_lengths': torch.LongTensor([input.numel()]),
            },
        }
        return input

    def decode_output(self, output):
        output = output[0][0]['tokens']
        output = utils.strip_eos(output, self.tgt_dict.eos())
        output = self.tgt_dict.string(output)
        if not str.endswith(output, "."):
            output += "."
        return output
