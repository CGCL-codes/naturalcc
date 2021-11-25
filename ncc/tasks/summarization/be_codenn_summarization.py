import json
import os

import torch

from ncc import (
    tokenizers,
    LOGGER,
)
from ncc.data import (
    indexed_dataset,
    iterators,
)
from ncc.data.dictionary import Dictionary
from ncc.data.ncc_dataset import NccDataset
from ncc.data.summarization.be_codenn_language_pair_dataset import BELanguagePairDataset
from ncc.data.tools import data_utils
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


def _load_ids(path):
    ids = []
    with open(path, 'r') as reader:
        for line in reader:
            ids.append(int(json.loads(line)))
    return ids


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

    # load tgt ids
    tgt_ids_path = os.path.join(data_path, '{}.id'.format(split))
    tgt_ids = _load_ids(tgt_ids_path)

    LOGGER.info('loaded {} examples from: {}'.format(len(src_dataset), src_path))
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_dataset), tgt_path))
    return BELanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        tgt_ids=tgt_ids,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=None,
        bos=src_dict.bos(),
        eos=src_dict.eos(),
        # shuffle=True,
        shuffle=False,  # debug
    )


@register_task('be_codenn_summarization')
class BECodeNNSummarizationTask(NccTask):
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
        workers=1, threshold=-1, nwords=-1, padding_factor=8
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
            self.sequence_generator = self.build_generator(
                [model], args, max_len_b=args['eval']['max_len_b'],
            )

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

        gen_out = self.inference_step(
            self.sequence_generator, [model], sample,
            bos_token=self.target_dictionary.bos(),
            # bos_token=self.target_dictionary.pad(),
        )
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

    def step_out(self, sample, model):
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

        gen_out = self.inference_step(self.sequence_generator, [model], sample,
                                      bos_token=self.target_dictionary.bos())
        src_ids = sample['src_ids']
        tgt_ids = sample['tgt_ids']
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        return hyps, refs, src_ids, tgt_ids

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args['task']['eval_bleu']:
            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            metrics.log_scalar('bleu', sum_logs('bleu'))

        if self.args['task']['eval_meteor']:
            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            metrics.log_scalar('meteor', sum_logs('meteor'))

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

    def encode_input(self, input, tokenize):
        if tokenize:
            input = ''.join(char if str.isalnum(char) else ' ' for char in input)  # for python_wan dataset
            input = tokenize(input)
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
        output = self.tgt_dict.string(output)
        if not str.endswith(output, "."):
            output += "."
        return output
