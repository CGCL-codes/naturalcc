# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from collections import namedtuple

import numpy as np
import torch
from ncc.tasks.type_prediction.type_prediction_typilus import load_codetype_dataset

from ncc import LOGGER
# import sentencepiece as spm
from ncc import tasks
# from ncc.trainer.ncc_trainer import Trainer
from ncc.trainers.ncc_trainers import Trainer
from ncc.utils import checkpoint_utils, distributed_utils
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.file_utils import remove_files


def train(args, task, model):
    # optimizer = torch.optim.Adam(model.parameters(), lr=args['optimization']['lr'][0], betas=(0.9, 0.98), eps=1e-6,
    #                              weight_decay=0)
    # scheduler = get_linear_schedule_with_warmup(optimizer, 5000, 200000)
    criterion = task.build_criterion(args)

    data_path = os.path.expanduser('~/.ncc/python_typilus/data-raw')
    split = 'train'
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    edge_types = set()
    with open(os.path.join(data_path, '__edge_types_mdata.txt'), 'r') as f:
        for l in f.readlines():
            edge_types.add(l.strip('\n'))
    print('edge_types: ', edge_types)
    # Build trainer
    trainer = Trainer(args, task, model, criterion)

    epoch = 1
    # dataset = load_langpair_dataset(args, epoch, data_path, split, src_dict, False)
    src, tgt = args['task']['source_lang'], args['task']['target_lang']
    combine = False

    # sp = spm.SentencePieceProcessor()
    # sp.load(args['dataset']['src_sp'])

    dataset = load_codetype_dataset(
        data_path, split, src, src_dict, tgt, tgt_dict, edge_types,
        combine=combine, dataset_impl=args['dataset']['dataset_impl'],
        # upsample_primary=self.args['task']['upsample_primary'],
        # left_pad_source=self.args['task']['left_pad_source'],
        # left_pad_target=self.args['task']['left_pad_target'],
        max_source_positions=args['task']['max_source_positions'],
        # max_target_positions=self.args['task']['max_target_positions'],
        # load_alignments=self.args['task']['load_alignments'],
        # truncate_source=self.args['task']['truncate_source'],
        # append_eos_to_target=self.args['task']['append_eos_to_target'],
    )

if __name__ == '__main__':
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('python_train.yml')  # train_sl
    LOGGER.info(args_)
    print('args: ', type(args_))
    yaml_file = os.path.join('../../../../naturalcc-dev/run/type_prediction/typilus/', 'config', args_.yaml)
    yaml_file = os.path.realpath(yaml_file)
    # yaml_file = os.path.join('/data/wanyao/Dropbox/ghproj-titan/naturalcodev3/run/summarization/seq2seq/', args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)

    # 0. Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args['common']['cpu']:
        torch.cuda.set_device(args['distributed_training']['device_id'])
    np.random.seed(args['common']['seed'])
    torch.manual_seed(args['common']['seed'])
    init_distributed = False
    if init_distributed:
        args['distributed_training']['distributed_rank'] = distributed_utils.distributed_init(args)

    # Verify checkpoint directory
    if distributed_utils.is_master(args):
        save_dir = args['checkpoint']['save_dir']
        checkpoint_utils.verify_checkpoint_directory(save_dir)
        remove_files(save_dir, 'pt')

    # Print args
    LOGGER.info(args)

    task = tasks.setup_task(args)  # task.tokenizer
    model = task.build_model(args)  # , config

    train(args, task, model)