# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F

from ncc.criterions import NccCriterion, register_criterion
from ncc.utils.logging import metrics


@register_criterion('cross_entropy_contracode')
class CrossEntropyContraCodeCriterion(NccCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # net_output = model(**sample['net_input'])
        if sample['loss_mode'] == 'hybrid':
            predicted_masked_tokens, moco_logits, moco_targets = \
                model(sample['net_input']['tokens_q'], sample['net_input']['tokens_k'],
                      sample['net_input']['lengths_q'], sample['net_input']['lengths_k'])
            moco_loss = F.cross_entropy(moco_logits, moco_targets, reduction='sum' if reduce else 'none')
            mlm_loss = F.cross_entropy(predicted_masked_tokens.flatten(end_dim=1), sample['mlm_targets'].flatten(),
                                       ignore_index=self.padding_idx, reduction='sum' if reduce else 'none')
            loss = 4 * moco_loss + mlm_loss

        elif sample['loss_mode'] == 'moco':
            moco_logits, moco_targets = model(sample['net_input']['tokens_q'], sample['net_input']['tokens_k'],
                               sample['net_input']['lengths_q'], sample['net_input']['lengths_k'])
            loss = F.cross_entropy(moco_logits, moco_targets, reduction='sum' if reduce else 'none')

        elif sample['loss_mode'] == 'mlm':
            predicted_masked_tokens = model(sample['net_input']['tokens'], sample['net_input']['lengths'])
            loss = F.cross_entropy(predicted_masked_tokens.flatten(end_dim=1), sample['mlm_targets'].flatten(),
                                   ignore_index=self.padding_idx, reduction='sum' if reduce else 'none')

        print('loss: ', loss)
        sample_size = sample['id'].size(0)
        logging_output = {
            'loss': loss.data,
            # 'ntokens': sample['ntokens'],
            # 'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        # ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum/sample_size)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
