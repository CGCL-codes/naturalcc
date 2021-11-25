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


@register_criterion('completion_lifelong_topk_kd_cross_entropy_loss')
class LifeLongTopkKDCrossEntropyCriterion(KDCrossEntropyCriterion):

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
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        loss = 0.0

        ################### previous data ###################
        # cross entropy
        if 'prev_ids' in sample and len(sample['prev_ids']) > 0:
            prev_ids = sample['prev_ids']
            prev_lprobs = lprobs[prev_ids, ...]
            prev_lprobs = prev_lprobs.view(-1, prev_lprobs.size(-1))
            prev_target = model.get_targets(sample, net_output)[prev_ids, ...]

            # CE loss
            prev_ce_loss = F.nll_loss(
                prev_lprobs,
                prev_target.view(-1),
                ignore_index=self.padding_idx,
                reduction='sum' if reduce else 'none',
            )
            loss = loss + prev_ce_loss

        ################### current data ###################
        # kd loss + lambda * ce loss
        if 'teacher_out_ids' in sample:
            teacher_out_ids, out_ids, out_probs = sample['teacher_out_ids'], sample['out_ids'], sample['out_probs']
            out_ids = out_ids.view(-1, out_ids.shape[-1])  # [B, L, topk] => [B*L, topk]
            out_probs = out_probs.view(-1, out_probs.shape[-1])  # [B, L, topk] => [B*L, topk]
            out_probs = F.softmax(out_probs / self.temperature, -1)

            # kd loss
            curr_lprobs = lprobs[teacher_out_ids, ...]
            net_output_lprobs_t = F.log_softmax(curr_lprobs / self.temperature, -1)
            net_output_lprobs_t = net_output_lprobs_t.view(-1, net_output_lprobs_t.shape[-1])  # [B, L, D] => [B*L, D]

            distill_loss = -(net_output_lprobs_t.gather(dim=-1, index=out_ids) * out_probs)
            distill_loss = distill_loss.sum(dim=-1)

            curr_target = model.get_targets(sample, net_output)[teacher_out_ids, ...]
            teacher_padding_mask = (curr_target == self.padding_idx).view(-1)
            distill_loss = distill_loss.masked_fill_(teacher_padding_mask, .0).sum()
            soft_loss = distill_loss * self.temperature * self.temperature

            # lambda * ce loss
            # CE loss
            hard_loss = F.nll_loss(
                curr_lprobs.view(-1, curr_lprobs.size(-1)),
                curr_target.view(-1),
                ignore_index=self.padding_idx,
                reduction='sum' if reduce else 'none',
            )
            curr_loss = soft_loss + self.hard_weight * hard_loss
            loss = loss + curr_loss
        return loss, loss
