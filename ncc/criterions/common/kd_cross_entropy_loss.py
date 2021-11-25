# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import torch
import torch.nn.functional as F

from ncc.criterions import register_criterion
from ncc.utils import utils
from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion('kd_cross_entropy')
class KDCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing=0.1, distill_temp=2, hard_weight=0.5):
        super().__init__(task, sentence_avg, label_smoothing)
        self.temperature = distill_temp
        self.hard_weight = hard_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        with torch.no_grad():
            teacher_out = self.task.teacher(**sample['net_input'])

        loss, nll_loss = self.compute_loss(model, net_output, teacher_out, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, teacher_out, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        target = model.get_targets(sample, net_output).view(-1)  # [B, L]

        # soft loss
        net_output_lprobs_t = F.log_softmax(net_output[0] / self.temperature, -1)  # [B, L, D]
        teacher_probs = F.softmax(teacher_out[0] / self.temperature, -1)
        distill_loss = -(teacher_probs * net_output_lprobs_t)  # [B, L, D]
        distill_loss = distill_loss.sum(-1).view(-1)
        padding_mask = target == self.padding_idx
        distill_loss = distill_loss.masked_fill_(padding_mask, .0).sum()
        soft_loss = distill_loss * self.temperature * self.temperature
        loss = soft_loss

        # hard loss
        if self.hard_weight > 0.0:
            lprobs = model.get_normalized_probs(net_output, log_probs=True)  # [B, L, D]
            lprobs = lprobs.view(-1, lprobs.size(-1))
            hard_loss = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction='sum' if reduce else 'none',
            )
            loss = loss + self.hard_weight * hard_loss

        return loss, (hard_loss if self.hard_weight > 0.0 else loss)
