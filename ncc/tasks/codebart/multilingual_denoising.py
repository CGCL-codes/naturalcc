# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from functools import lru_cache
import numpy as np
from ncc.data.tools import data_utils
from ncc.data.wrappers.sort_dataset import SortDataset
from ncc.data.tools.token_block_dataset import TokenBlockDataset
from ncc.tasks import register_task
from ncc.tokenizers.utils import get_whole_word_mask
from ncc.data.wrappers.prepend_token_dataset import PrependTokenDataset
from ncc.data.dictionary import Dictionary
from ncc.tasks.codebart import DenoisingTask
from ncc.data.codebart.denoising_dataset import DenoisingDataset
from ncc.data.wrappers.append_token_dataset import AppendTokenDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.data.wrappers.strip_token_dataset import StripTokenDataset
# from ncc.data.wrappers.concat_sentences_dataset import ConcatSentencesDataset
from ncc.data.wrappers.concat_dataset import ConcatDataset
from ncc.data.wrappers.resampling_dataset import ResamplingDataset
from ncc.utils import utils
from ncc import LOGGER
from ncc.data import constants
from ncc.data.ncc_dataset import NccDataset
from ncc.tokenizers import tokenization
from ncc.data import indexed_dataset


class IndexedRawTextDataset(NccDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.lines.append(line.strip('\n'))
                tokens = dictionary.encode_line(
                    line, tokenization._space_tokenizer, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

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


def load_lang_dataset_denoising(path, impl, dict):
    if impl == 'raw':
        src_dataset = IndexedRawTextDataset(path=path, dictionary=dict)
    elif impl == 'mmap':
        # mmap dataset has been numberized, no need for dict
        src_dataset = indexed_dataset.MMapIndexedDataset(path=path)
    else:
        raise NotImplementedError("No such {} dataset implementation.".format(impl))
    return src_dataset


@register_task('multilingual_denoising')
class MultilingualDenoisingTask(DenoisingTask):
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task.
        """
        # paths = args.data.split(':')
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], 'dict.jsonl'))

        data_path = paths[0]
        if args['task']['langs'] is None:
            languages = sorted([
                name for name in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, name))
            ])
        else:
            languages = args['task']['langs']  # .split(',')

        if args['task']['add_lang_token']:
            for lang in languages:
                dictionary.add_symbol('[{}]'.format(lang))

        LOGGER.info("Loading dictionary: {} types".format(len(dictionary)))
        # if not hasattr(args, 'shuffle_instance'):
        #     args.shuffle_instance = False
        return cls(args, dictionary)

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.dictionary = dictionary
        self.seed = args['common']['seed']

        # add mask token
        if constants.MASK in self.dictionary:
            self.mask_idx = self.dictionary.index(constants.MASK)
        else:
            self.mask_idx = self.dictionary.add_symbol(constants.MASK)
        self.langs = args['task']['langs']
        self.args = args

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** self.args['task']['multilang_sampling_alpha']
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # paths = self.args.data.split(':')
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        # split_path = os.path.join(data_path, split)

        if self.langs is None:
            languages = sorted([
                name for name in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, name))
            ])
        else:
            languages = self.langs  # .split(',')
            # for name in languages:
            #     assert os.path.exists(os.path.join(data_path, name)), FileNotFoundError(os.path.join(data_path, name))

        LOGGER.info("| Training on {0} languages: {1}".format(len(languages), languages))
        LOGGER.info("| Language to id mapping: ", {lang: id for id, lang in enumerate(languages)})

        mask_whole_words = get_whole_word_mask(self.args, self.dictionary)
        lang_datasets = []
        for language in languages:
            # split_path = os.path.join(data_path, language, split)
            if language == 'docstring':
                split_path = os.path.join(data_path, language, f"{split}.docstring.spm")
            else:
                split_path = os.path.join(data_path, language, f"{split}.code.spm")
            # split_path = os.path.join(data_path, language, f"{split}.spm.{language}")
            # dataset = data_utils.load_indexed_dataset(
            #     split_path,
            #     self.source_dictionary,
            #     self.args['dataset']['dataset_impl'],
            #     combine=combine,
            # )
            dataset = load_lang_dataset_denoising(path=split_path,
                                                  impl=self.args['dataset']['dataset_impl'],
                                                  dict=self.source_dictionary)

            if dataset is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

            dataset = AppendTokenDataset(
                TruncateDataset(StripTokenDataset(dataset, self.source_dictionary.eos()),
                                self.args['task']['max_source_positions'] - 3),  # <lang>, <bos>, <eos>
                token=self.source_dictionary.eos(),
            )

            end_token = self.source_dictionary.index('[{}]'.format(language)) \
                if self.args['task']['add_lang_token'] else self.source_dictionary.eos()

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args['task']['tokens_per_sample'] - 2,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=end_token,
                break_mode=self.args['task']['sample_break_mode'],
            )
            LOGGER.info('| loaded {} blocks from: {}'.format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
            dataset = AppendTokenDataset(dataset, end_token)

            lang_dataset = DenoisingDataset(
                dataset,
                dataset.sizes,
                self.dictionary,
                self.mask_idx,
                mask_whole_words,
                shuffle=self.args['dataset']['shuffle_instance'],
                seed=self.seed,
                args=self.args,
                eos=None if not self.args['task']['add_lang_token'] else self.source_dictionary.index(
                    '[{}]'.format(language)),
            )
            lang_datasets.append(lang_dataset)

        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        LOGGER.info(
            '| loaded total {} blocks for all languages'.format(
                dataset_lengths.sum(),
            )
        )
        if split == self.args['dataset']['train_subset']:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths)
            LOGGER.info("| Sample probability by language: ", {
                lang: "{0:.4f}".format(sample_probs[id])
                for id, lang in enumerate(languages)
            }
                        )
            size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
            LOGGER.info("| Up/Down Sampling ratio by language: ", {
                lang: "{0:.2f}".format(size_ratio[id])
                for id, lang in enumerate(languages)
            }
                        )

            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args['common']['seed'],
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            dataset = ConcatDataset(
                resampled_lang_datasets,
            )
        else:
            dataset = ConcatDataset(lang_datasets)
            lang_splits = [split]
            # for lang_id, lang_dataset in enumerate(lang_datasets):
            #     split_name = split + '_' + languages[lang_id]
            #     lang_splits.append(split_name)
            #     self.datasets[split_name] = lang_dataset

            if split in self.args['dataset']['valid_subset']:
                self.args['dataset']['valid_subset'] = self.args['dataset']['valid_subset'].replace(
                    split, ','.join(lang_splits)
                )

        with data_utils.numpy_seed(self.args['common']['seed'] + epoch):
            shuffle = np.random.permutation(len(dataset))

        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )
