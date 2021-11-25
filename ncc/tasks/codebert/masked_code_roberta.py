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
from ncc.data.wrappers import (
    IdDataset,
    NestedDictionaryDataset,
    TokenBlockDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    SortDataset,

)
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask
from ncc.tokenizers.tokenization import normalize_program
from ncc.utils import utils


# def load_masked_code_dataset_roberta(args, epoch, data_path, split, source_dictionary, combine,):
#     split_path = os.path.join(data_path, '{}.code'.format(split))
#     dataset = data_utils.load_indexed_dataset(
#         path=split_path,
#         modality='code',
#         dictionary=source_dictionary,
#         dataset_impl=args['dataset']['dataset_impl'],
#         combine=combine,
#     )
#     if dataset is None:
#         raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))
#
#     # create continuous blocks of tokens
#     # dataset = TokenBlockDataset(
#     #     dataset,
#     #     dataset.sizes,
#     #     args['task']['tokens_per_sample'] - 1,  # one less for <s>
#     #     pad=source_dictionary.pad(),
#     #     eos=source_dictionary.eos(),
#     #     break_mode=args['task']['sample_break_mode'],
#     # )
#     # LOGGER.info('loaded {} blocks from: {}'.format(len(dataset), split_path))
#
#     # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
#     dataset = PrependTokenDataset(dataset, source_dictionary.bos())  # .source_dictionary.bos()
#
#     # create masked input and targets
#     mask_whole_words = get_whole_word_mask(args, source_dictionary) \
#         if args['task']['mask_whole_words'] else None
#
#     src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
#         dataset,
#         source_dictionary,
#         pad_idx=source_dictionary.pad(),
#         mask_idx=source_dictionary.index(constants.MASK),  # self.mask_idx,
#         seed=args['common']['seed'],
#         mask_prob=args['task']['mask_prob'],
#         leave_unmasked_prob=args['task']['leave_unmasked_prob'],
#         random_token_prob=args['task']['random_token_prob'],
#         freq_weighted_replacement=args['task']['freq_weighted_replacement'],
#         mask_whole_words=mask_whole_words,
#     )
#
#     # with data_utils.numpy_seed(args['common']['seed'] + epoch):
#     #     shuffle = np.random.permutation(len(src_dataset))
#     dataset = NestedDictionaryDataset(
#                 {
#                     'id': IdDataset(),
#                     'net_input': {
#                         'src_tokens': PadDataset(
#                             src_dataset,
#                             pad_idx=source_dictionary.pad(),
#                             left_pad=False,
#                         ),
#                         'src_lengths': NumelDataset(src_dataset, reduce=False),
#                     },
#                     'target': PadDataset(
#                         tgt_dataset,
#                         pad_idx=source_dictionary.pad(),
#                         left_pad=False,
#                     ),
#                     'nsentences': NumSamplesDataset(),
#                     'ntokens': NumelDataset(src_dataset, reduce=True),
#                 },
#                 sizes=[src_dataset.sizes],
#             )
#     return dataset

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


def load_code_dataset_mlm(args, epoch, data_path, split, source_dictionary, combine, ):
    # split_path = os.path.join(data_path, 'javascript_augmented_debug.sp.json') # '{}.code'.format(split)
    split_path = os.path.join(data_path, 'javascript_augmented_debug.pickle')  # '{}.code'.format(split)
    sp = spm.SentencePieceProcessor()
    sp.load(args['dataset']['src_sp'])
    dataset = IndexedJavascriptAugmentedDataset(path=split_path, dictionary=source_dictionary, sp=sp, append_eos=False)
    if dataset is None:
        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

    return ContraCodeDataset(
        dataset, dataset.sizes, source_dictionary, program_mode='identity', loss_mode='mlm',
        shuffle=False,
    )


@register_task('masked_code_roberta')
class MaskedCodeRobertaTask(NccTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args['common']['seed']

        # add mask token
        # self.mask_idx = self.dictionary.add_symbol(constants.MASK)

    # @classmethod
    # def load_dictionary(cls, filename):
    #     """Load the dictionary from the filename
    #
    #     Args:
    #         filename (str): the filename
    #     """
    #     if filename.endswith('.txt'):
    #         dictionary = Dictionary(
    #             extra_special_symbols=[constants.CLS, constants.SEP, constants.MASK,
    #                                    constants.EOL, constants.URL])
    #         dictionary.add_from_file(filename)
    #     else:
    #         dictionary = Dictionary(extra_special_symbols=[constants.CLS, constants.SEP, constants.MASK,
    #                                                        constants.EOL, constants.URL]).add_from_json_file(filename)
    #     return dictionary
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
        # dictionary = cls.load_dictionary(os.path.join(paths[0], 'codesearchnet_ruby.dict.txt'))
        dictionary = cls.load_dictionary(os.path.join(paths[0], 'csnjs_8k_9995p_unigram_url.dict.txt'))
        # dictionary = cls.load_dictionary(args['dataset']['srcdict'])
        LOGGER.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        self.datasets[split] = load_code_dataset_mlm(self.args, epoch, data_path, split, self.source_dictionary,
                                                     combine)

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = PadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args['task']['tokens_per_sample'] - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode='eos',
            ),
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': src_dataset,
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
