# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from ncc import LOGGER
import torch
import numpy as np
from ncc.logging import metrics
from ncc.tasks.ncc_task import NccTask
from ncc.tasks import register_task
from ncc.utils import utils
from ncc.data.completion.seqrnn_dataset import SeqRNNDataset
from ncc.data.completion.completion_dictionary import CompletionDictionary as Dictionary
from ncc.data import indexed_dataset
import re


def _load_dataset(path, impl, dict=None):
    if impl == 'raw':
        raise NotImplementedError(impl)
    elif impl == 'mmap':
        # mmap dataset has been numberized, no need for dict
        src_dataset = indexed_dataset.MMapIndexedDataset(path=path)
    else:
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    return src_dataset


def load_token_dataset(data_path, split, tgt, tgt_dict, ext, dataset_impl, max_target_positions):
    tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
    tgt_dataset = _load_dataset(tgt_path, dataset_impl)
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_dataset), tgt_path))

    if ext is not None:
        ext_path = os.path.join(data_path, '{}.{}'.format(split, ext))
        ext_dataset = _load_dataset(ext_path, dataset_impl)
        LOGGER.info('loaded {} examples from: {}'.format(len(ext_dataset), ext_path))
    else:
        ext_dataset = None

    return SeqRNNDataset(
        tgt_dataset, tgt_dataset.sizes, tgt_dict, extends=ext_dataset,
        max_target_positions=max_target_positions,
    )


@register_task('completion')
class CompletionTask(NccTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        # load dictionaries
        dictionary = cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format(args['task']['target_lang'])))
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['target_lang'], len(dictionary)))
        return cls(args, dictionary)

    @classmethod
    def build_dictionary(
        cls, filenames, tokenize_func=None,
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
            Dictionary.add_file_to_dictionary(filename, d, tokenize_func, num_workers=workers)

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

        if self.args['model']['arch'] == 'seqrnn':
            self.datasets[split] = load_token_dataset(
                data_path, split, self.args['task']['target_lang'], self.target_dictionary,
                dataset_impl=self.args['dataset']['dataset_impl'], ext=self.args['task']['ext'],
                max_target_positions=self.max_positions())

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return SeqRNNDataset(src_tokens, src_lengths, self.target_dictionary)  # TODO: bug

    def build_model(self, args):
        model = super().build_model(args)

        if args['task']['eval_accuracy'] or args['task']['eval_last_accuracy'] or args['task']['eval_mrr']:
            self.sequence_completor = self.build_completor([model], args)

        return model

    def valid_step(self, sample, model, criterion):
        # print('valid_step...')
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        with torch.no_grad():
            net_output = self.sequence_completor.complete([model], sample, prefix_tokens=None)

        # ignore pad
        idx = sample['net_input']['src_tokens'].view(-1) != self.target_dictionary.pad()
        # ignore UNK in tgt because predict UNK is meaningless
        # while feed UNK into modle and predict non-UNK tokens still make sense
        idx[sample['target'].view(-1) == self.target_dictionary.unk()] = 0
        # ignore overlapping tokens
        max_len = sample['target'].size(-1)
        for i, ext_i in enumerate(sample['extends']):
            idx[i * max_len:i * max_len + ext_i] = 0

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        last_lprobs = torch.stack([lprobs[idx, last_idx, :] for idx, last_idx in enumerate(sample['src_last_idx'])])
        lprobs = lprobs.view(-1, lprobs.size(-1))[idx]
        target = model.get_targets(sample, net_output).view(-1)[idx]

        rank = torch.argmax(lprobs, dim=-1)
        last_rank = torch.argmax(last_lprobs, dim=-1)
        accuracy = 1. * torch.sum(rank == target) / sample['ntokens']
        last_gt = torch.stack([sample['target'][idx, last_idx] for idx, last_idx in enumerate(sample['tgt_last_idx'])])
        last_accuracy = 1. * torch.sum(last_rank == last_gt) / len(last_rank)
        logging_output['accuracy'] = accuracy
        logging_output['last_accuracy'] = last_accuracy

        mrr = np.mean([1. / (r.item() + 1) for r in rank.view(-1)])
        logging_output['mrr'] = mrr

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            return sum(log.get(key, 0) for log in logging_outputs)

        if self.args['task']['eval_accuracy']:
            if sum_logs('accuracy') > 0:  # ==0: no accuracy items in the logging outputs, it means the training stage
                metrics.log_scalar('accuracy', sum_logs('accuracy'))
        if self.args['task']['eval_last_accuracy']:
            if sum_logs(
                'last_accuracy') > 0:  # ==0: no accuracy items in the logging outputs, it means the training stage
                metrics.log_scalar('last_accuracy', sum_logs('last_accuracy'))
        if self.args['task']['eval_mrr']:
            if sum_logs('mrr') > 0:
                metrics.log_scalar('mrr', sum_logs('mrr'))

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return self.args['task']['max_target_positions']

    def build_generator(self, args):
        return self.sequence_completor

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

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

    def encode_input(self, input):
        input = input.replace('lambda', ' ').replace('if', ' ').replace('is', ' ').replace('not', ' '). \
            replace('return', ' ')
        input = re.split(r'[\s|\.|)|(|,|:|\[|\]]+', input.strip())
        input = [token for token in input if len(token) > 1]
        input = [self.target_dictionary.index(token) for token in input]
        input = torch.Tensor(input).long().unsqueeze(dim=0)
        input = {
            'net_input': {
                'src_tokens': input,
            },
        }
        return input

    def decode_output(self, output, k=5):
        output = torch.softmax(output[0][0, -1, :], dim=0)
        topk_prob, topk_idx = output.topk(k=10)

        output = []
        for prob, idx in zip(topk_prob, topk_idx):
            token, prob = self.target_dictionary[idx], round(prob.item(), 5)
            # if token == 'None':
            #     continue
            output.append((token, prob))
        return output[:k]
