# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
from functools import lru_cache

import sentencepiece as spm
import torch

from ncc import LOGGER
from ncc.data import constants
from ncc.data.contracode.contracode_dataset import ContraCodeDataset
from ncc.data.dictionary import Dictionary
from ncc.data.ncc_dataset import NccDataset
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.tokenizers.tokenization import normalize_program
from ncc.utils import utils


class IndexedJavascriptAugmentedDataset(NccDataset):
    def __init__(self, path, dictionary, sp, append_eos=False, reverse_order=False):
        self.examples_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary, sp)
        self.size = len(self.examples_list)

    def read_data(self, path, dictionary, sp, max_source_positions=1024, min_alternatives=1):
        with open(path, 'rb') as f:
            # Option 1:
            lines = pickle.load(f)
            lines = list(map(list, lines))
            if min_alternatives:
                lines = list(filter(lambda ex: len(ex) >= min_alternatives, lines))

            for line in lines:
                # line = ujson.loads(line)
                programs = []
                for program in line:
                    program = normalize_program(program)
                    program = sp.EncodeAsIds(program)
                    program = torch.LongTensor(
                        [dictionary.bos()] + program[: (max_source_positions - 2)] + [dictionary.eos()])
                    # Option 2
                    # program = [dictionary.bos_word] + program[: max_source_positions - 2] + [dictionary.eos_word]
                    # program = dictionary.encode_line(program, line_tokenizer=None, add_if_not_exist=False,
                    #                                  append_eos=self.append_eos,
                    #                                  reverse_order=self.reverse_order).long()
                    programs.append(program)

                self.examples_list.append(programs)
                self.sizes.append(len(programs[0]))

            # Option 2: this is for ujson format
            # for line in f:
            #     line = ujson.loads(line)
            #     programs = []
            #     for program in line:
            #
            #         # Option 1 # TODO
            #         program = sp.EncodeAsIds(program)
            #         program = torch.LongTensor([dictionary.bos()] + program[: (max_source_positions - 2)] + [dictionary.eos()])
            #         # Option 2
            #         # program = [dictionary.bos_word] + program[: max_source_positions - 2] + [dictionary.eos_word]
            #         # program = dictionary.encode_line(program, line_tokenizer=None, add_if_not_exist=False,
            #         #                                  append_eos=self.append_eos,
            #         #                                  reverse_order=self.reverse_order).long()
            #         programs.append(program)
            #
            #     self.examples_list.append(programs)
            #     self.sizes.append(len(programs[0]))

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


def load_augmented_code_dataset_hybrid(args, epoch, data_path, split, source_dictionary, combine, ):
    # split_path = os.path.join(data_path, 'javascript_augmented_debug.sp.json') # '{}.code'.format(split)
    split_path = os.path.join(data_path, 'javascript_augmented_debug.pickle')  # '{}.code'.format(split)
    sp = spm.SentencePieceProcessor()
    sp.load(args['dataset']['src_sp'])
    dataset = IndexedJavascriptAugmentedDataset(path=split_path, dictionary=source_dictionary, sp=sp, append_eos=False)
    if dataset is None:
        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

    return ContraCodeDataset(
        dataset, dataset.sizes, source_dictionary, program_mode='contrastive',
        shuffle=False,
    )


@register_task('contracode_hybrid')
class ContraCodeHybrid(NccTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args['common']['seed']
        # self.sp = sp
        # add mask token
        # self.mask_idx = self.dictionary.add_symbol(constants.MASK)

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
            dictionary = Dictionary(
                extra_special_symbols=[constants.CLS, constants.SEP, constants.MASK,
                                       constants.EOL, constants.URL]).add_from_file(filename)
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        dictionary = cls.load_dictionary(os.path.join(paths[0], 'csnjs_8k_9995p_unigram_url.dict.txt'))
        LOGGER.info('dictionary: {} types'.format(len(dictionary)))
        # sp = spm.SentencePieceProcessor()
        # sp.Load(os.path.join(paths[0], 'csnjs_8k_9995p_unigram_url.model'))

        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        self.datasets[split] = load_augmented_code_dataset_hybrid(self.args, epoch, data_path, split,
                                                                  self.source_dictionary, combine)  # , self.sp,

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
