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
from ..common.kd_cross_entropy_loss import KDCrossEntropyCriterion


@register_criterion('completion_lifelong_kd_cross_entropy_loss')
class LifeLongKDCrossEntropyCriterion(KDCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing=0.1, distill_temp=2, hard_weight=0.5):
        super().__init__(task, sentence_avg, label_smoothing, distill_temp, hard_weight)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        prev_out = [
            net_output[0][sample['prev_indices'], ...]
        ]
        curr_out = [
            net_output[0][sample['distill_indices'], ...]
        ]
        with torch.no_grad():
            src_tokens = sample['net_input']['src_tokens'][sample['distill_indices'], ...]
            teacher_out = self.task.teacher(src_tokens=src_tokens)

        loss, nll_loss = self.compute_loss(model, prev_out, curr_out, teacher_out, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, prev_out, curr_out, teacher_out, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss = 0.0

        ################### previous data ###################
        # cross entropy
        if prev_out is not None:
            prev_lprobs = model.get_normalized_probs(prev_out, log_probs=True)  # [B, L, D]
            prev_lprobs = prev_lprobs.view(-1, prev_lprobs.size(-1))
            prev_target = model.get_targets(sample, net_output=None)[sample['prev_indices'], ...]
            # CE loss
            prev_ce_loss = F.nll_loss(
                prev_lprobs,
                prev_target.view(-1),
                ignore_index=self.padding_idx,
                reduction='sum' if reduce else 'none',
            )
            loss = loss + prev_ce_loss

        ################### current data ###################
        # soft loss + lambda * hard loss
        if curr_out is not None:
            # soft loss
            curr_out_lprobs_t = F.log_softmax(curr_out[0] / self.temperature, -1)  # [B, L, D]
            teacher_probs = F.softmax(teacher_out[0] / self.temperature, -1)
            distill_loss = -(teacher_probs * curr_out_lprobs_t)  # [B, L, D]
            distill_loss = distill_loss.sum(-1).view(-1)
            curr_target = (model.get_targets(sample, net_output=None)[sample['distill_indices'], ...]).view(-1)
            padding_mask = curr_target == self.padding_idx
            distill_loss = distill_loss.masked_fill_(padding_mask, .0).sum()
            soft_loss = distill_loss * self.temperature * self.temperature
            loss = loss + soft_loss

            # hard loss
            if self.hard_weight > 0.0:
                curr_lprobs = model.get_normalized_probs(curr_out, log_probs=True)  # [B, L, D]
                curr_lprobs = curr_lprobs.view(-1, curr_lprobs.size(-1))
                hard_loss = F.nll_loss(
                    curr_lprobs,
                    curr_target,
                    ignore_index=self.padding_idx,
                    reduction='sum' if reduce else 'none',
                )
                loss = loss + self.hard_weight * hard_loss
        return loss, loss
