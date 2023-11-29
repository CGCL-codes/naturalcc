# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from collections import namedtuple

import numpy as np
import sentencepiece as spm
import torch
from torch.optim.lr_scheduler import LambdaLR

from ncc import LOGGER
from ncc import tasks
from ncc.data import iterators
from ncc.tasks.type_prediction.type_prediction import load_codetype_dataset
from ncc.trainers.ncc_trainers import Trainer
from ncc.utils import checkpoint_utils, distributed_utils
from ncc.utils import utils
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.file_utils import remove_files
from ncc.utils.logging import progress_bar


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def accuracy(output, target, topk=(1,), ignore_idx=[]):
    with torch.no_grad():
        maxk = max(topk)

        # Get top predictions per position that are not in ignore_idx
        target_vocab_size = output.size(2)
        keep_idx = torch.tensor([i for i in range(target_vocab_size) if i not in ignore_idx], device=output.device).long()
        _, pred = output[:, :, keep_idx].topk(maxk, 2, True, True)  # BxLx5
        pred = keep_idx[pred]  # BxLx5

        # Compute statistics over positions not labeled with an ignored idx
        correct = pred.eq(target.unsqueeze(-1).expand_as(pred)).long()
        mask = torch.ones_like(target).long()
        for idx in ignore_idx:
            mask = mask.long() & (~target.eq(idx)).long()
        mask = mask.long()
        deno = mask.sum().item()
        correct = correct * mask.unsqueeze(-1)
        res = []
        for k in topk:
            correct_k = correct[..., :k].view(-1).float().sum(0)
            res.append(correct_k.item())

        return res, deno


def _evaluate(args, task, model, split):
    data_path = os.path.expanduser('/mnt/wanyao/.ncc/augmented_javascript/type_prediction/data-raw')
    # split = 'valid'

    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Build trainer
    # trainer = Trainer(args, task, model, criterion)

    epoch = 1
    src, tgt = args['task']['source_lang'], args['task']['target_lang']
    combine = False

    sp = spm.SentencePieceProcessor()
    sp.load(args['dataset']['src_sp'])

    dataset = load_codetype_dataset(
        data_path, split, src, src_dict, tgt, tgt_dict, sp,
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
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     collate_fn=dataset.collater,
    #     num_workers=args['dataset']['num_workers'],
    # )

    # item = dataset.__getitem__(52)
    # print('item: ', item)
    # exit()
    epoch_itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['dataset']['max_sentences'],
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args['dataset']['required_batch_size_multiple'],
        seed=args['common']['seed'],
        num_shards=1,
        shard_id=0,
        num_workers=0,  # args['dataset']['num_workers'],
        # epoch=0,
    )

    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args['distributed_training']['fix_batches_to_gpus'],
        shuffle=(epoch_itr.next_epoch_idx > args['dataset']['curriculum']),
    )

    update_freq = (
        args['optimization']['update_freq'][epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args['optimization']['update_freq'])
        else args['optimization']['update_freq'][-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args['common']['log_format'],
        log_interval=args['common']['log_interval'],
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
        ),
        default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
    )

    model.eval()

    # no_type_id = target_to_id["O"]
    # any_id = target_to_id["$any$"]
    global_step = 0
    with torch.no_grad():
        # Accumulate metrics across batches to compute label-wise accuracy
        num1, num5, num_labels_total = 0, 0, 0
        num1_any, num5_any, num_labels_any_total = 0, 0, 0

        # with Timer() as t:
        # Compute average loss
        # total_loss = 0
        # num_examples = 0
        # pbar = tqdm.tqdm(loader, desc=f"evalaute")
        for samples in progress:
        # for X, lengths, output_attn, labels in pbar:
        #     if use_cuda:
        #         X, lengths, output_attn, labels = X.cuda(), lengths.cuda(), output_attn.cuda(), labels.cuda()
        #     if no_output_attention:
        #         logits = model(X, lengths, None)  # BxLxVocab
        #     else:
        #         logits = model(X, lengths, output_attn)  # BxLxVocab
            global_step += 1
            sample = samples[0]
            sample = utils.move_to_cuda(sample)
            model = model.cuda()
            net_output = model(**sample['net_input'])
            # Compute loss
            # loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=no_type_id)
            #
            # total_loss += loss.item() * X.size(0)
            # num_examples += X.size(0)
            # avg_loss = total_loss / num_examples

            logits = net_output[0]
            labels = model.get_targets(sample, net_output)  # .view(-1)

            mask = torch.ones_like(labels).long()
            mask = mask.long() & (~labels.eq(0)).long()
            mask = mask.long()
            deno = mask.sum().item()
            # print('deno: ', deno)

            # if global_step == 53:
            #     print('labels: ', labels[0].tolist())
            #     # print('labels[:100]: ', labels[0][:100].tolist())
            #     # print('labels[100:200]: ', labels[0][100:200].tolist())
            #     exit()

            # Compute accuracy
            (corr1_any, corr5_any), num_labels_any = accuracy(
                logits.cpu(), labels.cpu(), topk=(1, 5), ignore_idx=(task.target_dictionary.index('O'),))
            num1_any += corr1_any
            num5_any += corr5_any
            num_labels_any_total += num_labels_any
            # print('global_step: {}, corr1_any: {}, corr5_any: {}, num_labels_any: {}'.format(global_step, corr1_any, corr5_any, num_labels_any))

            (corr1, corr5), num_labels = accuracy(
                logits.cpu(), labels.cpu(), topk=(1, 5), ignore_idx=(task.target_dictionary.index('O'), task.target_dictionary.index('$any$'),))
            num1 += corr1
            num5 += corr5
            num_labels_total += num_labels

            # pbar.set_description(f"evaluate average loss {avg_loss:.4f} num1 {num1_any} num_labels_any_total {num_labels_any_total} avg acc1_any {num1_any / (num_labels_any_total + 1e-6) * 100:.4f}")

        # Average accuracies
        acc1 = float(num1) / num_labels_total * 100
        acc5 = float(num5) / num_labels_total * 100
        acc1_any = float(num1_any) / num_labels_any_total * 100
        acc5_any = float(num5_any) / num_labels_any_total * 100

        result = {
            # "eval/loss": avg_loss,
            "eval/acc@1": acc1,
            "eval/acc@5": acc5,
            "eval/num_labels": num_labels_total,
            "eval/acc@1_any": acc1_any,
            "eval/acc@5_any": acc5_any,
            "eval/num_labels_any": num_labels_any_total
        }
        print('result: ', result)

        # logger.debug(f"Loss calculation took {t.interval:.3f}s")
        # return (
        #     -acc1_any,
        #     {
        #         "eval/loss": avg_loss,
        #         "eval/acc@1": acc1,
        #         "eval/acc@5": acc5,
        #         "eval/num_labels": num_labels_total,
        #         "eval/acc@1_any": acc1_any,
        #         "eval/acc@5_any": acc5_any,
        #         "eval/num_labels_any": num_labels_any_total
        #     },
        # )

def train(args, task, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args['optimization']['lr'][0], betas=(0.9, 0.98), eps=1e-6,
                                 weight_decay=0)
    scheduler = get_linear_schedule_with_warmup(optimizer, 5000, 200000)

    criterion = task.build_criterion(args)
    LOGGER.info(model)
    LOGGER.info('model {}, criterion {}'.format(args['model']['arch'], criterion.__class__.__name__))
    LOGGER.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    data_path = os.path.expanduser('/mnt/wanyao/.ncc/augmented_javascript/type_prediction/data-raw')
    split = 'train'
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Build trainer
    trainer = Trainer(args, task, model, criterion)

    epoch = 1
    # dataset = load_langpair_dataset(args, epoch, data_path, split, src_dict, False)
    src, tgt = args['task']['source_lang'], args['task']['target_lang']
    combine = False

    sp = spm.SentencePieceProcessor()
    sp.load(args['dataset']['src_sp'])

    dataset = load_codetype_dataset(
        data_path, split, src, src_dict, tgt, tgt_dict, sp,
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

    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collater,
        # batch_sampler=batches[offset:],
        num_workers=args['dataset']['num_workers'],
    )
    # samples = []
    # for i in range(100):
    #     data_item = dataset.__getitem__(i)
    #     samples.append(data_item)
    # print('samples: ', samples)
    # sys.exit()

    # batch = collate(
    #     samples, pad_idx=src_dict.pad(), eos_idx=dataset.eos,
    #     left_pad_source=dataset.left_pad_source, left_pad_target=dataset.left_pad_target,
    #     input_feeding=dataset.input_feeding,
    # )
    # batch = collate(
    #     samples, src_dict.pad_index, src_dict.eos_index,
    #     left_pad_source=dataset.left_pad_source, left_pad_target=dataset.left_pad_target,
    #     input_feeding=dataset.input_feeding,
    # )
    # print(batch)
    # sys.exit()

    # data_iter = iter(dataloader)
    # batch_data = data_iter.__next__()
    # # print(utils.resolve_max_positions(
    # #             task.max_positions(),
    # #             trainer.get_model().max_positions(),
    # #         ))
    # data_iter
    #
    # exit()
    epoch_itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['dataset']['max_sentences'],
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            trainer.get_model().max_positions(),
        ),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args['dataset']['required_batch_size_multiple'],
        seed=args['common']['seed'],
        num_shards=1,
        shard_id=0,
        num_workers=0,  # args['dataset']['num_workers'],
        # epoch=0,
    )
    # batch_data = epoch_itr.__next__()
    # print('batch_data: ', batch_data)
    # exit()
    # epoch = 1
    # task.load_dataset(
    #             args['dataset']['train_subset'],
    #             epoch=1,
    #             combine=combine,
    #             data_selector=None,
    #         )
    # epoch_itr = task.get_batch_iterator(
    #     dataset=dataset, #task.dataset(args['dataset']['train_subset']), #=self.task.dataset(self.args['dataset']['train_subset']),
    #     max_tokens=args['dataset']['max_tokens'],
    #     max_sentences=args['dataset']['max_sentences'],
    #     max_positions=512,
    #     ignore_invalid_inputs=True,
    #     required_batch_size_multiple=args['dataset']['required_batch_size_multiple'],
    #     seed=args['common']['seed'],
    #     num_shards=1, #args['distributed_training']['distributed_world_size'] if shard_batch_itr else 1,
    #     shard_id=0, #self.args['distributed_training']['distributed_rank'] if shard_batch_itr else 0,
    #     num_workers=args['dataset']['num_workers'],
    #     epoch=epoch,
    # )
    # epoch_itr = trainer.get_train_iterator(
    #     epoch=1, load_dataset=True
    # )
    # itr = epoch_itr.next_epoch_itr(
    #     fix_batches_to_gpus=args['distributed_training']['fix_batches_to_gpus'],
    #     shuffle=(epoch_itr.next_epoch_idx > args['dataset']['curriculum']),
    # )
    # print('itr: ', itr)
    # for i, obj in enumerate(itr):
    #     print('i: ', i)
    #     print('obj: ', obj)

    # itr = epoch_itr.next_epoch_itr(
    #     fix_batches_to_gpus=args['distributed_training']['fix_batches_to_gpus'],
    #     shuffle=(epoch_itr.next_epoch_idx > args['dataset']['curriculum']),
    # )
    # update_freq = (
    #     args['optimization']['update_freq'][epoch_itr.epoch - 1]
    #     if epoch_itr.epoch <= len(args['optimization']['update_freq'])
    #     else args['optimization']['update_freq'][-1]
    # )
    # itr = iterators.GroupedIterator(itr, update_freq)
    # progress = progress_bar.progress_bar(
    #     itr,
    #     log_format=args['common']['log_format'],
    #     log_interval=args['common']['log_interval'],
    #     epoch=epoch_itr.epoch,
    #     tensorboard_logdir=(
    #         args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
    #     ),
    #     default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
    # )

    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args['distributed_training']['fix_batches_to_gpus'],
        shuffle=(epoch_itr.next_epoch_idx > args['dataset']['curriculum']),
    )

    update_freq = (
        args['optimization']['update_freq'][epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args['optimization']['update_freq'])
        else args['optimization']['update_freq'][-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args['common']['log_format'],
        log_interval=args['common']['log_interval'],
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
        ),
        default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
    )

    # task specific setup per epoch
    task.begin_epoch(epoch_itr.epoch, trainer.get_model())
    global_step = 0
    model.train()
    total_loss = 0
    for samples in progress:
        # print('samples: ', samples)
        # exit()
        target = samples[0]['target']
        no_type_id = task.target_dictionary.index('O')
        # if torch.sum(target.ne(no_type_id)) > 0:
        # net_output = trainer.train_step(samples)
        # if net_output is None:  # OOM, overflow, ...
        #     continue
        # loss, sample_size, logging_output = net_output
        # loss = net_output['loss']
        # print('loss: ', loss.item())
        # print('sample_size: ', sample_size)
        # exit()
        # loss, sample_size_i, logging_output = task.train_step(
        #     sample=sample,
        #     model=self.model,
        #     criterion=self.criterion,
        #     optimizer=self.optimizer,
        #     update_num=self.get_num_updates(),
        #     ignore_grad=is_dummy_batch,
        # )

        optimizer.zero_grad()
        sample = samples[0]
        sample = utils.move_to_cuda(sample)
        loss, sample_size, logging_output = criterion(model, sample)
        # loss = loss/sample_size
        # print('loss: ', loss.item())
        loss.backward()
        optimizer.step()
        # scheduler.step()
        total_loss += loss.item()
        global_step += 1

        # if global_step > 100:
        #     exit()

        if global_step % 100 == 0:
            print('step: {}, loss: {}, avg_loss: {}'.format(global_step, loss.item(), total_loss / global_step))
        # if global_step > 100:
        #     exit()
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args['checkpoint']['save_dir'], 'WEIGHTS_NAME')
    torch.save(model.state_dict(), output_model_file)

# def test(args, task, model):
#     # model = task.build_model(args)  # , config
#     # model.load_state_dict(torch.load(args['eval']['path']))
#     model.eval()


if __name__ == '__main__':
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('javascript_train.yml')  # train_sl
    LOGGER.info(args_)
    # print('args: ', type(args_))
    # yaml_file = os.path.join('../../../naturalcodev3/run/summarization/lstm2lstm/', 'config', args_.yaml)
    yaml_file = os.path.join('../../../../naturalcc-dev/run/type_prediction/transformer/', 'config', args_.yaml)
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
    exit()
    # model.load_state_dict(torch.load(args['eval']['path']))
    checkpoint = torch.load(args['eval']['path'])
    model.load_state_dict(checkpoint)
    _evaluate(args, task, model, split=args['dataset']['test_subset'])
    # test(args, task, model)

