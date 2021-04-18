import os
from collections import OrderedDict

import dgl
import ujson

from ncc import LOGGER
from ncc.data import constants
from ncc.data import indexed_dataset
from ncc.data import iterators
from ncc.data.ncc_dataset import NccDataset
from ncc.data.tools import data_utils
from ncc.data.type_prediction.typilus.typilus_dictionary import TypilusDictionary as Dictionary
from ncc.data.type_prediction.typilus_prediction_pair_dataset import LanguagePairDataset
from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccComplTask
from ncc.utils import utils


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
    srcs, src_dicts,
    tgts, tgt_dicts,
    dataset_impl,
    **kwargs,
):
    # load source dataset
    src_datasets, src_sizes = OrderedDict(), OrderedDict()
    for src in srcs:
        src_path = os.path.join(data_path, '{}.{}'.format(split, src))
        if src == 'edges':
            src_datasets[src], _ = dgl.data.utils.load_graphs(src_path)
            src_sizes[src] = None
        else:
            src_datasets[src] = _load_dataset(path=src_path, impl=dataset_impl, dict=src_dicts[src])
            src_sizes[src] = src_datasets[src].sizes
    pad = src_dicts['nodes'].pad()
    # load target dataset
    tgt_datasets, tgt_sizes = OrderedDict(), OrderedDict()
    if 'supernodes.annotation' in tgts:
        tgts.pop(tgts.index('supernodes.annotation'))
        tgts.extend(['supernodes.annotation.node', 'supernodes.annotation.type'])
        tgt_dicts['supernodes.annotation.node'] = tgt_dicts['supernodes.annotation.type'] = \
            tgt_dicts.pop('supernodes.annotation')

    for tgt in tgts:
        tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
        tgt_datasets[tgt] = _load_dataset(path=tgt_path, impl=dataset_impl, dict=tgt_dicts[tgt])
        tgt_sizes[tgt] = tgt_datasets[tgt].sizes
    assert all(tgt_datasets['supernodes.annotation.node'].sizes == tgt_datasets['supernodes.annotation.type'].sizes)

    with open(os.path.join(data_path, f'{split}.supernodes.annotation.type.json'), 'r') as reader:
        tgt_datasets['supernodes.annotation.type.json'] = [ujson.loads(line) for line in reader]

    return LanguagePairDataset(
        src_datasets, src_sizes, src_dicts,
        tgt_datasets, tgt_sizes, tgt_dicts,
        pad=pad,
        shuffle=True,
        max_subtoken_len=kwargs.get('max_subtoken_len', 5)
    )


@register_task('typilus')
class TypilusTask(NccComplTask):

    def __init__(self, args, src_dicts, tgt_dicts):
        super().__init__(args)
        self.src_dicts = src_dicts
        self.tgt_dicts = tgt_dicts

    def source_dictionary(self, key):
        return self.src_dicts[key]

    def target_dictionary(self, key):
        return self.tgt_dicts[key]

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        # load dictionaries
        src_dicts = OrderedDict()
        for lang in args['task']['source_langs']:
            src_dicts[lang] = cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format(lang)))
            LOGGER.info('[{}] dictionary: {} types'.format(lang, len(src_dicts[lang]) if lang != 'edges' else 0))
        tgt_dicts = OrderedDict()
        for lang in args['task']['target_langs']:
            tgt_dicts[lang] = cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format(lang)))
            LOGGER.info('[{}] dictionary: {} types'.format(lang, len(tgt_dicts[lang])))
        return cls(args, src_dicts, tgt_dicts)

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
                filename, d, tokenize_func, workers
            )

        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

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

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        srcs, tgts = self.args['task']['source_langs'], self.args['task']['target_langs']

        self.datasets[split] = load_langpair_dataset(
            data_path, split, srcs, self.src_dicts, tgts, self.tgt_dicts,
            dataset_impl=self.args['dataset']['dataset_impl'],
            max_subtoken_len=self.args['model']['max_subtoken_len'],
        )

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

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch(indices, max_sentences)

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
