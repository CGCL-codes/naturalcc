# -*- coding: utf-8 -*-

from ncc.utils import utils
from ..ncc_trainers import Trainer


class LifeLongKDTrainer(Trainer):

    def get_train_review_iterator(
        self,
        epoch,
        shard_batch_itr=True,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            self.model.max_positions(),
            self.args['dataset']['max_tokens'],
        )
        return self.task.get_batch_iterator(
            dataset=self.task.dataset('review'),
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
