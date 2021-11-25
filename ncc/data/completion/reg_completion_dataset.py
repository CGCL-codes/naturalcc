import random
import torch
import numpy as np

from ncc import LOGGER

from ncc.data.constants import DEFAULT_MAX_TARGET_POSITIONS
from ncc.data.ncc_dataset import NccDataset
from ncc.data.tools import data_utils
from ncc.data.completion.completion_dataset import CompletionDataset
from ncc.data.completion.replay_completion_dataset import ReplayCompletionDataset
from ncc.utils.utils import move_to_cuda


class RegMemory(object):
    def __init__(self):
        self.prev_models = []
        self.prev_datasets = {'train': [], 'valid': []}
        self.curr_model = None
        self.curr_datasets = {'train': None, 'valid': None}

    def __len__(self):
        return len(self.prev_models)

    def init(self):
        del self.prev_models, self.prev_datasets, self.curr_model, self.curr_datasets
        self.__init__()

    def load(self, model, train_dataset, valid_dataset):
        self.prev_models.append(model.state_dict())
        self.prev_datasets['train'].append(train_dataset)
        self.prev_datasets['valid'].append(valid_dataset)

    def store_curr_env(self, model):
        self.prev_models.append(model.state_dict())
        self.prev_datasets['train'].append(self.curr_datasets['train'])
        self.prev_datasets['valid'].append(self.curr_datasets['valid'])
        self.curr_model = None
        self.curr_datasets = {'train': None, 'valid': None}

    def update_curr_env(self, curr_model, train_dataset, valid_dataset):
        self.curr_model = curr_model
        self.curr_datasets['train'] = train_dataset
        self.curr_datasets['valid'] = valid_dataset


def collate(samples, pad_idx, unk_idx, attrs=None, last_model_params=None):
    # no need for left padding
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
        )

    def merge_repr(key):
        reprs = torch.stack([s[key] for s in samples], dim=0)
        return reprs

    src_tokens = merge('source')
    tgt_tokens = merge('target')
    # hidden_reprs = merge_repr('hidden_reprs')[:, :src_tokens.size(1), :]

    attr_masks = {attr: [] for attr in attrs} if attrs is not None else None

    extends = []
    max_len = src_tokens.size(-1)
    for i, s in enumerate(samples):
        extends.append(s['extend'])
        if attr_masks is not None:
            for attr in attrs:
                attr_masks[attr].append(s['attr_masks'][attr] + max_len * i)
    if attrs:
        for attr in attrs:
            attr_masks[attr] = np.concatenate(attr_masks[attr], axis=0)

    ntokens = sum(sum(s['target'][s['extend']:] != pad_idx) for s in samples).item()

    batch = {
        'id': [s['id'] for s in samples],
        'net_input': {
            'src_tokens': src_tokens,
        },
        'target': tgt_tokens,
        # 'hidden_reprs': hidden_reprs,
        'attr_masks': attr_masks,
        'extends': extends,
        'ntokens': ntokens,
        'last_model_params': last_model_params,
    }
    return batch


class RegCompletionDataset(ReplayCompletionDataset):
    def __init__(
        self,
        tgt, tgt_sizes,
        tgt_dict, extends=None,
        attrs=None, attr_indices=None, attr_dict=None,
        attrs_mapping=None, reversed_attrs_mapping=None,
        left_pad_source=False, left_pad_target=False,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        shuffle=True,
    ):
        self.tgt = tgt
        self.tgt_sizes = np.array(tgt_sizes)
        self.tgt_dict = tgt_dict

        self.extends = extends
        self.attrs = attrs
        self.attr_indices = attr_indices
        self.attr_dict = attr_dict
        self.attrs_mapping = attrs_mapping
        self.reversed_attrs_mapping = reversed_attrs_mapping
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_target_positions = max_target_positions

        self.shuffle = shuffle

        self.pad = self.tgt_dict.pad()
        self.unk = self.tgt_dict.unk()

    def __getitem__(self, index):
        # Append EOS to end of tgt sentence if it does not have an EOS
        # and remove EOS from end of src sentence if it exists.
        # This is useful when we use existing datasets for opposite directions
        #   i.e., when we want to use tgt_dataset as src_dataset and vice versa
        src_item = self.tgt[index][:-1]
        tgt_item = self.tgt[index][1:]

        extend = 0 if self.extends is None else self.extends[index].item()
        if self.attrs_mapping:
            # do not move attr_masks into cuda
            attr_masks = {attr: [] for attr in self.attrs}
            for idx, attr_idx in enumerate(self.attr_indices[index].tolist()[1:][extend:], start=extend):
                if attr_idx in self.reversed_attrs_mapping:
                    attr_masks[self.reversed_attrs_mapping[attr_idx]].append(idx)
            for attr in self.attrs:
                attr_masks[attr] = np.array(attr_masks[attr])
        else:
            attr_masks = None

        # self.last_model.load(self.last_model_params)
        # with torch.no_grad():
        #     tmp_src = move_to_cuda(src_item[None, ...])
        #     hidden_reprs = self.last_model.extract_features(tmp_src).squeeze(dim=0)
        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,

            # 'hidden_reprs': hidden_reprs,
            'attr_masks': attr_masks,
            'extend': extend,
        }
        return example

    def collater(self, samples):
        return collate(samples, pad_idx=self.pad, unk_idx=self.unk, attrs=self.attrs,
                       last_model_params=getattr(self, 'last_model_params', None))

    def register_last_model(self, model):
        self.last_model = model
        self.last_model_params = model.state_dict()

    def register_prev_datasets(self, model_params, prev_datasets):
        self.model_params = model_params
        self.prev_tgts, self.prev_tgt_indices = [], []
        for dataset in prev_datasets:
            self.prev_tgts.append(dataset.tgt)
            indices = np.arange(len(dataset))
            indices = indices[dataset.tgt_sizes > 1]
            self.prev_tgt_indices.append(indices)

    def get_prev_task_batches(self, model, batch_size):
        MAX_SENTENCE_LENGTH = self.tgt_sizes.max() - 1

        def get_prev_item(prev_model, task_idx, index):
            src_item = self.prev_tgts[task_idx][index][:-1]
            tgt_item = self.prev_tgts[task_idx][index][1:]
            model.load_state_dict(prev_model)
            with torch.no_grad():
                tmp_src = torch.cat([src_item, torch.LongTensor([self.pad] * (MAX_SENTENCE_LENGTH - len(src_item)))],
                                    dim=-1)
                tmp_src = move_to_cuda(tmp_src[None, ...])
                hidden_reprs = model.extract_features(tmp_src).squeeze(dim=0)

            extend = 0 if self.extends is None else self.extends[index].item()
            if self.attrs_mapping:
                # do not move attr_masks into cuda
                attr_masks = {attr: [] for attr in self.attrs}
                for idx, attr_idx in enumerate(self.attr_indices[index].tolist()[1:][extend:], start=extend):
                    if attr_idx in self.reversed_attrs_mapping:
                        attr_masks[self.reversed_attrs_mapping[attr_idx]].append(idx)
                for attr in self.attrs:
                    attr_masks[attr] = np.array(attr_masks[attr])
            else:
                attr_masks = None

            example = {
                'id': index,
                'source': src_item,
                'target': tgt_item,

                'hidden_reprs': hidden_reprs,
                'attr_masks': attr_masks,
                'extend': extend,
            }
            return example

        curr_model = model.state_dict()
        prev_batches = []
        for task_idx, (prev_model, prev_tgt, prev_ids) in enumerate(
            zip(self.model_params, self.prev_tgts, self.prev_tgt_indices)):
            np.random.shuffle(prev_ids)
            prev_batches.append(
                self.collater([get_prev_item(prev_model, task_idx, idx) for idx in prev_ids[:batch_size]])
            )
        model.load_state_dict(curr_model)
        return prev_batches
