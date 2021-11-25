import os
import re

import numpy as np
import torch

from ncc import LOGGER
from ncc.data import constants
from ncc.data import indexed_dataset
from ncc.data.completion.completion_dataset import CompletionDataset
from ncc.data.completion.completion_dictionary import CompletionDictionary as Dictionary
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.utils import utils
from ncc.utils.logging import metrics


def _load_dataset(path, impl, dict=None):
    if impl == 'raw':
        raise NotImplementedError(impl)
    elif impl == 'mmap':
        # mmap dataset has been numberized, no need for dict
        src_dataset = indexed_dataset.MMapIndexedDataset(path=path)
    else:
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    return src_dataset


def load_token_dataset(
    data_path, split, tgt, tgt_dict, dataset_impl,
    attrs=None, attr_dict=None,
    attrs_mapping=None, reversed_attrs_mapping=None,
    truncate_target=False, max_target_positions=None,
    shuffle=True,
):
    # load tokens
    tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
    tgt_dataset = _load_dataset(tgt_path, dataset_impl)
    if truncate_target:
        tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions)
        LOGGER.info('Truncate dataset into max length: {}'.format(max_target_positions))
    LOGGER.info('loaded {} examples from: {}'.format(len(tgt_dataset), tgt_path))
    # load tokens.ext
    tgt_ext_path = os.path.join(data_path, '{}.{}.ext'.format(split, tgt))
    if indexed_dataset.SeqIndexedDataset.exists(tgt_ext_path):
        tgt_ext_dataset = indexed_dataset.SeqIndexedDataset(tgt_ext_path)
        if truncate_target:
            tgt_ext_dataset.clip(max_position=max_target_positions)
        assert len(tgt_dataset) == len(tgt_ext_dataset), (len(tgt_dataset), len(tgt_ext_dataset))
    else:
        tgt_ext_dataset = None
    # load attrs
    if attrs is None:
        attr_dataset = None
    else:
        attr_path = os.path.join(data_path, '{}.code_types'.format(split))
        attr_dataset = _load_dataset(attr_path, dataset_impl)
        if truncate_target:
            tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions)
            LOGGER.info('Truncate dataset\'s attributes into max length: {}'.format(max_target_positions))
        LOGGER.info('loaded {} examples from: {}'.format(len(attr_dataset), attr_path))
        # load attr.ext
        attr_ext_path = os.path.join(data_path, '{}.code_types.ext'.format(split))
        if indexed_dataset.SeqIndexedDataset.exists(attr_ext_path):
            attr_ext_dataset = indexed_dataset.SeqIndexedDataset(attr_ext_path)
            if truncate_target:
                attr_ext_dataset.clip(max_position=max_target_positions)
            assert np.all(tgt_ext_dataset == attr_ext_dataset)
            del attr_ext_dataset

    return CompletionDataset(
        tgt_dataset, tgt_dataset.sizes, tgt_dict, extends=tgt_ext_dataset,
        attrs=attrs, attr_indices=attr_dataset, attr_dict=attr_dict,
        attrs_mapping=attrs_mapping, reversed_attrs_mapping=reversed_attrs_mapping,
        max_target_positions=max_target_positions,
        shuffle=shuffle,
    )


@register_task('completion')
class CompletionTask(NccTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, dictionary, token_dictionary=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.token_dictionary = token_dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        # load dictionaries
        dict_file = os.path.join(paths[0], '{}.dict.jsonl'.format(args['task']['target_lang']))
        dictionary = cls.load_dictionary(dict_file)
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['target_lang'], len(dictionary)))
        token_file = os.path.join(paths[0], 'code_types.dict.jsonl')
        if os.path.exists(token_file):
            token_dictionary = cls.load_dictionary(token_file)
            LOGGER.info('[code_tokens] dictionary: {} types'.format(len(token_dictionary)))
        else:
            token_dictionary = None
        return cls(args, dictionary, token_dictionary)

    @classmethod
    def build_dictionary(
        cls, filenames, tokenize_func,
        workers=1, threshold=-1, nwords=-1, padding_factor=8,
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
            Dictionary.add_token_to_dictionary(
                filename, d, tokenize_func, workers,
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

        if self.args['task']['target_lang'] == 'code_tokens' and self.args['task'].get('code_types', False):
            attrs_mapping = {
                'attr': {self.token_dictionary.index('attr')},
                'num': {self.token_dictionary.index('Num')},
                'name': {self.token_dictionary.index('NameStore'),
                         self.token_dictionary.index('NameLoad')},
                'param': {self.token_dictionary.index('arg'),
                          self.token_dictionary.index('kwarg'),
                          self.token_dictionary.index('vararg')},
            }
        elif self.args['task']['target_lang'] == 'ast' and self.args['task'].get('code_types', False):
            attrs_mapping = {
                'attr': {self.token_dictionary.index('attr')},
                'num': {self.token_dictionary.index('Num')},
                'name': {self.token_dictionary.index('NameStore'),
                         self.token_dictionary.index('NameLoad')},
                'param': {self.token_dictionary.index('NameParam')},
            }
        else:
            attrs_mapping = None

        if attrs_mapping:
            reversed_attrs_mapping = {}
            for k, vs in attrs_mapping.items():
                if len(vs) > 1:
                    for v in vs:
                        reversed_attrs_mapping[v] = k
                else:
                    reversed_attrs_mapping[list(vs)[0]] = k
        else:
            reversed_attrs_mapping = None

        self.datasets[split] = load_token_dataset(
            data_path, split, self.args['task']['target_lang'], self.target_dictionary,
            attrs_mapping=attrs_mapping, reversed_attrs_mapping=reversed_attrs_mapping,
            attrs=self.args['task'].get('code_types', None),
            attr_dict=self.token_dictionary,
            dataset_impl=self.args['dataset']['dataset_impl'],
            truncate_target=self.args['dataset'].get('truncate_target', False),
            max_target_positions=self.max_positions(),
            shuffle=(split == 'train'),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return CompletionDataset(src_tokens, src_lengths, self.target_dictionary)  # TODO: bug

    def build_model(self, args):
        model = super().build_model(args)
        self.sequence_completor = self.build_completor([model], args)
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        with torch.no_grad():
            net_output = self.sequence_completor.complete([model], sample, prefix_tokens=None)

            # ignore pad
            idx = sample['net_input']['src_tokens'].view(-1) != self.target_dictionary.pad()
            # ignore overlapping tokens
            max_len = sample['target'].size(-1)
            for i, ext_i in enumerate(sample['extends']):
                idx[i * max_len:i * max_len + ext_i] = 0

            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))[idx]
            target = model.get_targets(sample, net_output).view(-1)[idx]

            rank = torch.argmax(lprobs, dim=-1)
            accuracy = rank == target
            logging_output['accuracy'] = accuracy.sum().float()
            # 1. / (lprobs >= torch.stack([lprobs[idx, gt] for idx, gt in enumerate(target.tolist())]).unsqueeze(dim=-1)).sum(-1)).sum()
            logging_output['mrr'] = (1. / (lprobs >= lprobs[:, target].diag().unsqueeze(dim=-1)).sum(-1)).sum()
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            return sum(log.get(key, 0) for log in logging_outputs)

        ntokens = sum_logs('ntokens')
        accuracy_sum = sum_logs('accuracy')
        metrics.log_scalar('accuracy', accuracy_sum / ntokens, ntokens, round=6)

        if self.args['task']['eval_mrr']:
            mrr_sum = sum_logs('mrr')
            metrics.log_scalar('mrr', mrr_sum / ntokens, ntokens, round=6)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return self.args['task']['max_target_positions']

    def build_generator(self, models, args):
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
        return Dictionary.load(filename)

    def encode_input(self, input, tokenizer=None):
        # input = input.replace('lambda', ' ').replace('if', ' ').replace('is', ' ').replace('not', ' '). \
        #     replace('return', ' ')
        # input = re.split(r'[\s|\.|)|(|,|:|\[|\]]+', input.strip())
        # input = re.split(r'[\s]+', input.strip())
        # input = [token for token in input]
        if tokenizer is not None:
            input = tokenizer(input)
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
        topk_prob, topk_idx = output.topk(k=k)

        output = []
        for prob, idx in zip(topk_prob, topk_idx):
            token, prob = self.target_dictionary[idx], round(prob.item(), 5)
            # if token == 'None':
            #     continue
            output.append((token, prob))
        return output
