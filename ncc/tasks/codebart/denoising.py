# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from ncc.data.tools import data_utils
from ncc.data.tools.token_block_dataset import TokenBlockDataset
from ncc.tasks.ncc_task import NccTask
from ncc.tasks import register_task
from ncc.tokenizers.utils import get_whole_word_mask
from ncc.utils import utils
from ncc.data.wrappers.prepend_token_dataset import PrependTokenDataset
from ncc.data.dictionary import Dictionary
from ncc.data.codebart.denoising_dataset import DenoisingDataset
from ncc.data.wrappers.append_token_dataset import AppendTokenDataset
from ncc.data.wrappers.strip_token_dataset import StripTokenDataset
from ncc.data import constants
# logger = logging.getLogger(__name__)
from ncc import LOGGER


@register_task('denoising')
class DenoisingTask(NccTask):
    """
    Denoising task for applying sequence to sequence denoising. (ie. BART)
    """

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args['common']['seed']

        # add mask token
        self.mask_idx = self.dictionary.add_symbol(constants.MASK)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task.
        """
        dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
        LOGGER.info('dictionary: {} types'.format(len(dictionary)))
        if not hasattr(args, 'shuffle_instance'):
            args.shuffle_instance = False
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        dataset = StripTokenDataset(dataset, self.dictionary.eos())

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 2,  # one less for <s> and one for </s>
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            document_sep_len=0
        )

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
        dataset = AppendTokenDataset(dataset, self.source_dictionary.eos())

        mask_whole_words = get_whole_word_mask(self.args, self.source_dictionary) \
            if self.args.mask_length != 'subword' else None

        self.datasets[split] = DenoisingDataset(
            dataset, dataset.sizes, self.dictionary, self.mask_idx,
            mask_whole_words, shuffle=self.args.shuffle_instance,
            seed=self.seed, args=self.args
        )
        LOGGER.info(
            "Split: {0}, Loaded {1} samples of denoising_dataset".format(
                split,
                len(self.datasets[split]),
            )
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args['task']['max_source_positions'], self.args['task']['max_target_positions'])

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dictionary
