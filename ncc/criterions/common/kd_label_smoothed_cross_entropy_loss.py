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


@register_criterion('kd_label_smoothed_cross_entropy')
class KDLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing=0.1, distill_temp=0.6, kd_loss_weight=1.0):
        super().__init__(task, sentence_avg, label_smoothing)
        self.temperature = distill_temp
        self.kd_loss_weight = kd_loss_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)

        # CE loss
        nll_loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        # nll_prob = -lprobs.gather(dim=-1, index=target)
        # smoothed CE loss
        padding_mask = target == self.padding_idx
        smooth_loss = -lprobs.sum(dim=-1)
        smooth_loss = smooth_loss.masked_fill_(padding_mask, .0).sum()

        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        if ('out_ids' in sample) and ('out_probs' in sample):
            out_ids, out_probs = sample['out_ids'], sample['out_probs']
            net_output_lprobs_t = F.log_softmax(net_output[0] / self.temperature, -1)
            net_output_lprobs_t = net_output_lprobs_t.view(-1, net_output_lprobs_t.shape[-1])  # [B, L, D] => [B*L, D]
            out_ids = out_ids.view(-1, out_ids.shape[-1])  # [B, L, topk] => [B*L, topk]
            out_probs = out_probs.view(-1, out_probs.shape[-1])  # [B, L, topk] => [B*L, topk]
            out_probs = F.softmax(out_probs / self.temperature, -1)
            # kd loss
            distill_loss = -(net_output_lprobs_t.gather(dim=-1, index=out_ids) * out_probs)
            distill_loss = distill_loss.sum(dim=-1)
            distill_loss = distill_loss.masked_fill_(padding_mask, .0).sum()
            kd_loss = distill_loss * self.temperature * self.temperature

            loss = self.kd_loss_weight * kd_loss + (1 - self.kd_loss_weight) * loss
        return loss, nll_loss
