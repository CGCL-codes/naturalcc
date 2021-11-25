# -*- coding: utf-8 -*-

import math

import torch
import torch.nn.functional as F

from ncc.criterions import NccCriterion, register_criterion
from ncc.data.constants import EPS
from ncc.utils.logging import metrics


@register_criterion('code_disen_search_triplet')
class CodeDisenSearchTripletCriterion(NccCriterion):
    def __init__(self, task, sentence_avg, forward_func):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.margin = self.task.args['optimization']['margin']
        self.forward_func = forward_func

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        def _forward(key):
            output = getattr(model, self.forward_func)(**sample[key])
            return output

        anchor_output = _forward(key='anchor')
        pos_output = _forward(key='pos')
        neg_output = _forward(key='neg')
        net_output = anchor_output, pos_output, neg_output,

        loss, mrr = self.compute_loss(model, net_output, reduce=reduce)
        sample_size = sample['nsentences']
        logging_output = {
            'loss': loss.data,
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def cos_similarity(self, src_repr, tgt_repr):
        return F.cosine_similarity(src_repr, tgt_repr)

    def compute_loss(self, model, net_output, reduce=True):
        anchor_repr, pos_repr, neg_repr = net_output
        pos_dist = F.cosine_similarity(anchor_repr, pos_repr)  # B X 1
        neg_dist = F.cosine_similarity(anchor_repr, neg_repr)  # B X 1
        loss = (self.margin - pos_dist + neg_dist).clamp(EPS).sum()
        return loss, None

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
