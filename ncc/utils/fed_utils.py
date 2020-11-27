import glob
import hashlib
import os
import torch
from tqdm import tqdm
from ncc.utils import utils, distributed_utils
import numpy as np
import ujson as json
from ncc import LOGGER
import struct

FED_VERSION_FN = 'fed_version.v3.idx'

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


def dist2topk(out_dist, k):
    topk_prob, topk_idx = torch.topk(out_dist, k, dim=-1)
    topk_prob = topk_prob.view(-1, k)  # (B x T) x k
    topk_prob = topk_prob / topk_prob.sum(1, keepdim=True)
    topk_idx = topk_idx.view(-1, k)  # (B x T) x k
    return topk_idx, topk_prob


def output2topk(output, k):
    topk_outp, topk_idx = torch.topk(output, k, dim=-1)
    topk_outp = topk_outp.view(-1, k)  # (B x T) x k
    topk_idx = topk_idx.view(-1, k)  # (B x T) x k
    return topk_idx, topk_outp


def get_sample_key(ids):
    if not hasattr(get_sample_key, 'sample_key_cache'):
        get_sample_key.sample_key_cache = {}
    ids_str = ','.join([str(id) for id in sorted(ids)])
    if ids_str not in get_sample_key.sample_key_cache:
        hash_object = hashlib.md5(ids_str.encode())
        get_sample_key.sample_key_cache[ids_str] = hash_object.hexdigest()
    return get_sample_key.sample_key_cache[ids_str]


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.mmap'


class IndexedDataset(torch.utils.data.Dataset):
    """Loader for TorchNet IndexedDataset"""

    def __init__(self, path, fix_lua_indexing=False, read_data=True):
        super().__init__()
        self.fix_lua_indexing = fix_lua_indexing
        self.read_index(path)
        self.data_file = None
        if read_data:
            self.read_data(path)

    def read_index(self, path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            assert magic == b'TNTIDX\x00\x00'
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1,)
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code]
            self.size, self.s = struct.unpack('<QQ', f.read(16))
            self.dim_offsets = read_longs(f, self.size + 1)
            self.data_offsets = read_longs(f, self.size + 1)
            self.sizes = read_longs(f, self.s)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        self.data_file.seek(self.data_offsets[i] * self.element_size)
        self.data_file.readinto(a)
        item = torch.from_numpy(a).long()
        if self.fix_lua_indexing:
            item -= 1  # subtract 1 for 0-based indexing
        return item

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and
            os.path.exists(data_file_path(path))
        )


class IndexedDatasetBuilder(object):
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float: 4,
        np.double: 8
    }

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, 'wb')
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]

    def add_item(self, tensor):
        # +1 for Lua compatibility
        bytes = self.out_file.write(np.array(tensor.numpy() + 1, dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def merge_file_(self, another_file):
        index = IndexedDataset(another_file, read_data=False)
        assert index.dtype == self.dtype

        begin = self.data_offsets[-1]
        for offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + offset)
        self.sizes.extend(index.sizes)
        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)

        with open(data_file_path(another_file), 'rb') as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, 'wb')
        index.write(b'TNTIDX\x00\x00')
        index.write(struct.pack('<Q', 1))
        index.write(struct.pack('<QQ', code(self.dtype), self.element_size))
        index.write(struct.pack('<QQ', len(self.data_offsets) - 1, len(self.sizes)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        index.close()


class TeacherOutputDatasetBuilder(IndexedDatasetBuilder):
    def add_item(self, data):
        # +1 for Lua compatibility
        data = np.array(data, dtype=self.dtype)
        bytes = self.out_file.write(data)
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in data.shape:
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(data.shape))


class IndexedCachedDataset(IndexedDataset):

    def __init__(self, path, fix_lua_indexing=False):
        super().__init__(path, fix_lua_indexing, True)
        self.cache = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        pass

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)

        if i in self.cache:
            np.copyto(a, self.cache[i])
        else:
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            self.cache[i] = a

        item = torch.from_numpy(a).long()
        if self.fix_lua_indexing:
            item -= 1  # subtract 1 for 0-based indexing
        return item


class TeacherOutputDataset(IndexedCachedDataset):
    dtype2size = {
        float: 8,
        int: 4,
    }

    def __init__(self, prefix):
        self.cache_index = {}
        super().__init__(prefix, fix_lua_indexing=False)

    @staticmethod
    def save_bin(prefix, data_list, dtype=np.float):
        bin_path = prefix + '.mmap'
        idx_path = prefix + '.idx'
        builder = TeacherOutputDatasetBuilder(bin_path, dtype)
        for d in data_list:
            builder.add_item(d)
        builder.finalize(idx_path)

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)

        if i in self.cache:
            np.copyto(a, self.cache[i])
        else:
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            self.cache[i] = a

        item = torch.from_numpy(a)
        if self.dtype == np.int32 or self.dtype == np.int or self.dtype == np.int64:
            item = item.long()
        else:
            item = item.float()
        return item


def gen_outputs(args, task, trainer):
    # cause some data mighe been filtered by max_source/target_position
    tmp_cache = [
        [8 * [0] for _ in range(6)],  # topk idx
        [8 * [0] for _ in range(6)],  # topk prob
    ]

    trainer.model.eval()
    itr = task.get_batch_iterator(
        dataset=task.dataset('train'),
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['dataset']['max_sentences_valid'],
        # max_sentences=16,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            trainer.get_model().max_positions(),
        ),
        ignore_invalid_inputs=args['dataset']['skip_invalid_size_inputs_valid_test'],
        # required_batch_size_multiple=8,
        seed=args['common']['seed'],
        num_shards=args['distributed_training']['distributed_world_size'],
        shard_id=args['distributed_training']['distributed_rank'],
    ).next_epoch_itr(shuffle=False)

    # outputs = [None for _ in range(len(task.dataset('train')))]
    outputs = {}
    for sample in tqdm(itr, mininterval=5):
        with torch.no_grad():
            if sample is None or len(sample) == 0:
                continue
            sample = utils.move_to_cuda(sample)

            bs, srclen = sample['net_input']['src_tokens'].shape
            output = trainer.model(**sample['net_input'])[0].detach()
            non_padding_mask = sample['target'].ne(task.target_dictionary.pad()).cpu()
            _, tgtlen = sample['target'].shape
            topk_idx, topk_v = output2topk(output, args['kd']['distill_topk'])
            topk_x_shape = (bs, tgtlen, args['kd']['distill_topk'])
            topk_idx, topk_v = topk_idx.view(*topk_x_shape).cpu().numpy(), topk_v.view(*topk_x_shape).cpu().numpy()
            non_padding_mask = non_padding_mask.view(*topk_x_shape[:2]).cpu().numpy().astype(bool)
            for b in range(bs):
                outputs[sample['id'][b].item()] = \
                    topk_idx[b, non_padding_mask[b]].tolist(), \
                    topk_v[b, non_padding_mask[b]].tolist()

    return [outputs[idx] if idx in outputs else tmp_cache for idx in list(range(len(task.dataset('train'))))]


def save_expert_outputs(args, task, trainer):
    print("| Start saving expert outputs..")
    expert_outputs = gen_outputs(args, task, trainer)
    output_path = os.path.join(args['checkpoint']['save_dir'],
                               'train_output.json.{}'.format(args['distributed_training']['distributed_rank']))
    print('Save topk output at {}'.format(output_path))
    json.dump(expert_outputs, open(output_path, 'w'))
    # distributed_utils.barrier(args, 'save_expert_outputs')
    if distributed_utils.is_master(args):
        expert_outputs_ = []
        # copy valid bleu result
        val_bleu_path1 = os.path.join(args['checkpoint']['save_dir'], 'val_bleu.json')
        val_bleu_path2 = os.path.join(
            args['task']['data'],
            'expert_bleu_{}_{}_{}.json'.format(
                '_'.join(args['task']['programming_langs']), args['task']['source_lang'], args['task']['target_lang']
            )
        )
        cmd = 'cp {} {}'.format(val_bleu_path1, val_bleu_path2)
        print(cmd)
        os.system(cmd)

        for i in range(args['distributed_training']['distributed_world_size']):
            output_path = os.path.join(args['checkpoint']['save_dir'], 'train_output.json.{}'.format(i))
            expert_outputs_.append(json.load(open(output_path, 'r')))
            try:
                os.remove(output_path)
            except:
                pass

        for j in range(len(expert_outputs_[0])):
            for i in range(args['distributed_training']['distributed_world_size']):
                if expert_outputs_[i][j] is not None:
                    expert_outputs[j] = expert_outputs_[i][j]
                    break
            assert expert_outputs[j] is not None

        path = os.path.join(args['task']['data'], '{}_{}_{}_topk_idx'.format(
            '_'.join(args['task']['programming_langs']), args['task']['source_lang'], args['task']['target_lang'])
                            )
        TeacherOutputDataset.save_bin(path, [o[0] for o in expert_outputs], np.int32)

        path = os.path.join(args['task']['data'], '{}_{}_{}_topk_prob'.format(
            '_'.join(args['task']['programming_langs']), args['task']['source_lang'], args['task']['target_lang'])
                            )
        TeacherOutputDataset.save_bin(path, [o[1] for o in expert_outputs], np.float)

        LOGGER.info(
            "| Save expert@{}_{}_{}. Bleu.Json: {}, TopK.Idx/Prob: {}.".format(
                '_'.join(args['task']['programming_langs']),
                args['task']['source_lang'], args['task']['target_lang'],
                val_bleu_path2, path,
            )
        )
