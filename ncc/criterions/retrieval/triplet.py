# -*- coding: utf-8 -*-

import math

import torch
import torch.nn.functional as F

from ncc.criterions import NccCriterion, register_criterion
from ncc.data.constants import EPS
from ncc.utils import utils
from ncc.utils.logging import metrics


@register_criterion('triplet')
class TripletCriterion(NccCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.margin = self.task.args['optimization']['margin']

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, mrr = self.compute_loss(model, net_output, reduce=reduce)
        sample_size = sample['nsentences']
        logging_output = {
            'loss': loss.data,
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'mrr': mrr,
        }
        return loss, sample_size, logging_output

    def cos_similarity(self, src_repr, tgt_repr):
        return F.cosine_similarity(src_repr, tgt_repr)

    def compute_loss(self, model, net_output, reduce=True):
        src_repr, pos_repr, neg_repr = net_output
        pos_dist = self.cos_similarity(src_repr, pos_repr)  # B X 1
        neg_dist = self.cos_similarity(src_repr, neg_repr)  # B X 1
        loss = (self.margin - pos_dist + neg_dist).clamp(EPS).sum()
        with torch.no_grad():
            src_norm_repr = src_repr / torch.norm(src_repr, dim=-1)[..., None]
            pos_norm_repr = pos_repr / torch.norm(pos_repr, dim=-1)[..., None]
            logits = pos_norm_repr @ src_norm_repr.t()
            correct_scores = logits.diag()
            compared_scores = logits >= correct_scores.unsqueeze(dim=-1)
            mrr = round((1 / compared_scores.sum(dim=-1)).sum().item(), 6)
        return loss, mrr

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=3)

        mrr_sum = sum(log.get('mrr', 0) for log in logging_outputs)
        metrics.log_scalar('mrr', mrr_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
