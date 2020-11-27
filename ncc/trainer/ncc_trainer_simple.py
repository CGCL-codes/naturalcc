# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""
from itertools import chain
import logging
import torch
from ncc.utils import checkpoint_utils, distributed_utils, utils
# from ncc import optim
from ncc.logging import meters, metrics
# from ncc.optim import lr_scheduler
import torch.optim as optim

logger = logging.getLogger(__name__)


class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion):
        self.args = args
        self.task = task

        self.cuda = torch.cuda.is_available() and not args['common']['cpu']
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # copy model and criterion to current device
        self.criterion = criterion.to(device=self.device)
        self.model = model.to(device=self.device)
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), self.criterion.parameters()),
            )
        )
        # self.optimizer = optim.build_optimizer(self.args, params)
        self.optimizer = optim.Adam(params, 0.002, weight_decay=0)
        self.num_updates = 0

        # self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
        # self.lr_scheduler.step_update(0)

    # def save_checkpoint(self, filename, extra_state):
    #     """Save all training state in a checkpoint file."""
    #     if distributed_utils.is_master(self.args):  # only save one checkpoint
    #         extra_state["metrics"] = metrics.state_dict()
    #         checkpoint_utils.save_state(
    #             filename,
    #             self.args,
    #             self.get_model().state_dict(),
    #             self.get_criterion(),
    #             self.optimizer,
    #             self.lr_scheduler,
    #             self.get_num_updates(),
    #             # self._optim_history,
    #             extra_state,
    #         )

    def get_train_iterator(
        self,
        epoch,
        combine=True,
        load_dataset=True,
        data_selector=None,
        shard_batch_itr=True,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        if load_dataset:
            logger.info("loading train data for epoch {}".format(epoch))
            self.task.load_dataset(
                self.args['dataset']['train_subset'],
                epoch=epoch,
                combine=combine,
                data_selector=data_selector,
            )
        max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            self.model.max_positions(),
            self.args['dataset']['max_tokens'],
        )
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args['dataset']['train_subset']),
            max_tokens=self.args['dataset']['max_tokens'],
            max_sentences=self.args['dataset']['max_sentences'],
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args['dataset']['required_batch_size_multiple'],
            seed=self.args['common']['seed'],
            num_shards=self.args['distributed_training']['distributed_world_size'] if shard_batch_itr else 1,
            shard_id=self.args['distributed_training']['distributed_rank'] if shard_batch_itr else 0,
            num_workers=self.args['dataset']['num_workers'],
            epoch=epoch,
        )

    @metrics.aggregate("train")
    def train_step(self, samples, raise_oom=False):
        """Do forward, backward and parameter update."""
        # self._set_seed()
        # seed = self.args['common']['seed'] + self.get_num_updates()
        # torch.manual_seed(seed)
        # if self.cuda:
        #     torch.cuda.manual_seed(seed)
        # self.model.train()
        # self.criterion.train()
        # self.zero_grad()
        # forward and backward pass
        # logging_outputs, sample_size, ooms = [], 0, 0
        for i, sample in enumerate(samples):
            if self.cuda:
                sample = utils.move_to_cuda(sample)
            loss, sample_size_i, logging_output = self.task.train_step(
                sample=sample,
                model=self.model,
                criterion=self.criterion,
                optimizer=self.optimizer,
                update_num=self.get_num_updates(),
                ignore_grad=False,
            )
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()
        # if torch.is_tensor(sample_size):
        #     sample_size = sample_size.float()
        # else:
        #     sample_size = float(sample_size)
        #
        # grad_norm = self.clip_grad_norm(self.args['optimization']['clip_norm'])

    def clip_grad_norm(self, clip_norm):
        return self.optimizer.clip_grad_norm(clip_norm)

    @metrics.aggregate("valid")
    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""

        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            if self.cuda:
                sample = utils.move_to_cuda(sample)

            # _loss, sample_size, logging_output
            predictions, loss, sample_size, logging_output = self.task.valid_step(
                sample, self.model, self.criterion
            )
        return predictions, loss, sample_size, logging_output
    # def zero_grad(self):
    #     self.optimizer.zero_grad()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.lr_scheduler.step_update(self.get_num_updates())
        metrics.log_scalar("lr", new_lr, weight=0, priority=300)
        return new_lr

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self.model

    def get_criterion(self):
        """Get the (non-wrapped) criterion instance."""
        return self.criterion

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self.num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self.num_updates = num_updates
        self.lr_step_update()
        metrics.log_scalar("num_updates", self.num_updates, weight=0, priority=200)

    # def test_bleu_step(self, generator, sample, bleu_scorers, print_to_file=None):
    #     with torch.no_grad():
    #         self.model.eval()
    #         # sample = self._prepare_sample(sample)
    #         if self.cuda:
    #             sample = utils.move_to_cuda(sample)
    #         if sample is not None:
    #             self.test_bleu(generator, sample, bleu_scorers, print_to_file)
    #
    # def test_bleu(self, generator, sample, bleu_scorers, print_to_file=None):
    #     if sample is None:
    #         return
    #
    #     beam = self.args['eval']['beam']
    #     with torch.no_grad():
    #         while True:
    #             try:
    #                 hypos = generator.generate(
    #                     {'src_tokens': sample['net_input']['src_tokens'],
    #                      'src_lengths': sample['net_input']['src_lengths'], },
    #                     beam
    #                 )
    #                 break
    #             except RuntimeError as e:
    #                 if 'out of memory' in str(e) and beam >= 3:
    #                     beam = beam - 1
    #                     print('| WARNING: ran out of memory, reduce beam size to %d' % beam)
    #                 else:
    #                     raise e
    #
    #         assert len(sample['target']) == len(hypos)
    #         for dataset_id, tgt, hypo in zip(
    #             list(sample['dataset_id']) if 'dataset_id' in sample else list(torch.LongTensor([0] * len(hypos))),
    #             list(sample['target']),
    #             hypos
    #         ):
    #             # remove BOS/EOS and keep UNK
    #             target_tokens = torch.IntTensor(
    #                 [
    #                     i for i in tgt.tolist()
    #                     if not (i in {self.task.tgt_dict.eos(), self.task.tgt_dict.pad(), self.task.tgt_dict.bos()})
    #                 ]
    #             )
    #             hypo_tokens = torch.IntTensor(
    #                 [
    #                     i for i in hypo[0]['tokens'].tolist()
    #                     if not (i in {self.task.tgt_dict.eos(), self.task.tgt_dict.pad(), self.task.tgt_dict.bos()})
    #                 ]
    #             )
    #             bleu_scorer_ = bleu_scorers[dataset_id.item()]
    #             bleu_scorer_.add(target_tokens, hypo_tokens)
    #
    #             if print_to_file:
    #                 target_str = self.task.tgt_dict.string(tgt.tolist(), escape_unk=True)
    #                 hypo_str = self.task.tgt_dict.string(hypo[0]['tokens'].tolist())
    #                 print_to_file(dataset_id.item(), target_str, hypo_str)
