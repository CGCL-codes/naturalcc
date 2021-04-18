# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import lru_cache

import sentencepiece as spm
import torch

from ncc.data import constants
from ncc.data.dictionary import Dictionary
from ncc.data.ncc_dataset import NccDataset
from ncc.data.type_prediction.codetype_dataset import CodeTypeDataset
from ncc.eval.summarization.summarization_metrics import eval_accuracies
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.utils import utils


class IndexedCodeDataset(NccDataset):
    def __init__(self, path, dictionary, append_eos=False, reverse_order=False):
        self.examples_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.examples_list)

    def read_data(self, path, dictionary, max_source_positions=1024):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                program = line.rstrip()
                self.lines.append(line)
                self.examples_list.append(program)
                self.sizes.append(len(program.split(' ')))  # TODO: simply calculate the length by space tokenization

        # with open(path, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         program = ujson.loads(line)
        #         # programs = []
        #         # for program in line:
        #         # TODO
        #         program = [dictionary.bos_word] + program[: max_source_positions-2] + [dictionary.eos_word]
        #         program = dictionary.encode_line(program, line_tokenizer=None, add_if_not_exist=False,
        #                                       append_eos=self.append_eos, reverse_order=self.reverse_order).long()
        #         # programs.append(program)
        #
        #         self.examples_list.append(program)
        #         self.sizes.append(len(program))

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.examples_list[i]

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


class IndexedTypeDataset(NccDataset):
    def __init__(self, path, dictionary, append_eos=False, reverse_order=False):
        self.examples_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.examples_list)

    def read_data(self, path, dictionary, max_source_positions=1024):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                type = line.rstrip().strip('\n')
                self.lines.append(line)
                self.examples_list.append(type)
                self.sizes.append(len(type.split(' ')))

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.examples_list[i]

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


def load_codetype_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict, sp,
    combine, dataset_impl,
    # upsample_primary,
    # left_pad_source, left_pad_target, max_source_positions,
    max_source_positions,
    # max_target_positions, prepend_bos=False, load_alignments=False,
    # truncate_source=False, append_source_id=False,
    # append_eos_to_target=False,
):
    src_data_path = os.path.join(data_path, '{}.{}'.format(split, src))
    # src_dataset = data_utils.load_indexed_dataset(src_data_path, 'codetype_code', src_dict, dataset_impl)
    src_dataset = IndexedCodeDataset(path=src_data_path, dictionary=src_dict, append_eos=False)

    tgt_data_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
    # tgt_dataset = data_utils.load_indexed_dataset(tgt_data_path, 'codetype_type', tgt_dict, dataset_impl)
    tgt_dataset = IndexedTypeDataset(path=tgt_data_path, dictionary=tgt_dict, append_eos=False)

    return CodeTypeDataset(
        src_dataset, src_dataset.sizes, src_dict, tgt_dataset, tgt_dict, sp, max_source_positions=max_source_positions,
        shuffle=False,
        # max_target_positions=max_target_positions,
    )


@register_task('type_prediction')
class TypePredictionTask(NccTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        if filename.endswith('.txt'):
            dictionary = Dictionary(
                extra_special_symbols=[constants.CLS, constants.SEP, constants.MASK,
                                       constants.EOL, constants.URL])
            dictionary.add_from_file(filename)
        else:
            dictionary = Dictionary(extra_special_symbols=[constants.CLS, constants.SEP, constants.MASK,
                                                           constants.EOL, constants.URL]).add_from_json_file(filename)
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        # load dictionaries
        # src_dict = cls.load_dictionary(os.path.join(paths[0], 'csnjs_8k_9995p_unigram_url.dict.txt'))
        src_dict = Dictionary(
            extra_special_symbols=[constants.CLS, constants.SEP, constants.MASK,
                                   constants.EOL, constants.URL])
        src_dict.add_from_file(args['dataset']['srcdict'])
        tgt_dict = Dictionary.load(args['dataset']['tgtdict'])

        # src_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.txt'.format(args['task']['source_lang'])))
        # tgt_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.txt'.format(args['task']['target_lang'])))
        # assert src_dict.pad() == tgt_dict.pad()
        # assert src_dict.eos() == tgt_dict.eos()
        # assert src_dict.unk() == tgt_dict.unk()
        # LOGGER.info('[{}] dictionary: {} types'.format(args['task']['source_lang'], len(src_dict)))
        # LOGGER.info('[{}] dictionary: {} types'.format(args['task']['target_lang'], len(tgt_dict)))

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

        sp = spm.SentencePieceProcessor()
        sp.load(self.args['dataset']['src_sp'])

        self.datasets[split] = load_codetype_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict, sp,
            combine=combine, dataset_impl=self.args['dataset']['dataset_impl'],
            # upsample_primary=self.args['task']['upsample_primary'],
            # left_pad_source=self.args['task']['left_pad_source'],
            # left_pad_target=self.args['task']['left_pad_target'],
            max_source_positions=self.args['task']['max_source_positions'],
            # max_target_positions=self.args['task']['max_target_positions'],
            # load_alignments=self.args['task']['load_alignments'],
            # truncate_source=self.args['task']['truncate_source'],
            # append_eos_to_target=self.args['task']['append_eos_to_target'],
        )
        # item = self.datasets[split].__getitem__(0)
        # print('item: ', item)

    # def build_dataset_for_inference(self, src_tokens, src_lengths):
    #     return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_model(self, args):
        model = super().build_model(args)
        if args['task']['eval_accuracy']:
            self.type_predictor = self.build_type_predictor([model], args)

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
        with torch.no_grad():
            net_output = self.type_predictor.predict([model], sample, prefix_tokens=None)
        print('net_output-models.type_prediction.py: ', net_output)
        exit()
        # selected = sample['node_ids']['leaf_ids']
        # max_len = sample['target'].size(-1)
        # node_ids = []
        # for i, lst in enumerate(selected):
        #     node_ids += [j + (max_len - 1) * i for j in lst]
        #
        # lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))[node_ids]
        # target = model.get_targets(sample, net_output).view(-1)[node_ids]
        #
        # rank = torch.argmax(lprobs, 1)
        # mrr = np.mean([1. / (r.item() + 1) for r in rank.view(-1)])
        #
        # ncorrect = torch.sum(rank == target)
        #
        # accuracy = ncorrect / sample['ntokens']
        # logging_output['accuracy'] = accuracy
        # logging_output['mrr'] = mrr
        #
        # return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        # if self.args['task']['eval_bleu']:
        #     def sum_logs(key):
        #         return sum(log.get(key, 0) for log in logging_outputs)
        #
        #     metrics.log_scalar('bleu', sum_logs('bleu'))

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

        bleu, rouge_l, meteor = eval_accuracies(hypotheses, references, filename='pred.txt')

        return bleu, rouge_l, meteor
