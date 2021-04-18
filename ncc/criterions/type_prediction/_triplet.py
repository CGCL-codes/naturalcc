# -*- coding: utf-8 -*-

import math

import torch

from ncc.criterions import NccCriterion
from ncc.data.constants import EPS
from ncc.utils.logging import metrics


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
        loss, _ = self.compute_loss(model, net_output, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample_size,
            'nsentences': sample_size,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, repr, equal_ids):
        distance = torch.norm(repr.unsqueeze(dim=0) - repr.unsqueeze(dim=1), dim=-1, p=1)  # B x B
        max_pos_distance = (distance * equal_ids).max(dim=-1)[0]
        neg_filter = distance <= (max_pos_distance + self.margin).unsqueeze(dim=-1)
        pos_mask = equal_ids + torch.eye(*equal_ids.size()).type_as(distance)
        neg_filter = neg_filter * (1 - pos_mask)
        avg_neg_distance = (distance * neg_filter).sum(dim=-1) / (neg_filter.sum(dim=-1) + EPS)
        min_neg_distance = (distance + pos_mask * 99999).min(dim=-1)[0]
        pos_filter = (distance >= (min_neg_distance - self.margin).unsqueeze(dim=-1)).type_as(distance)
        pos_filter = pos_filter * equal_ids
        avg_pos_distance = (distance * pos_filter).sum(dim=-1) / (pos_filter.sum(dim=-1) + EPS)
        triplet_loss = 0.5 * torch.relu(avg_pos_distance - min_neg_distance + self.margin) + \
                       0.5 * torch.relu(max_pos_distance - avg_neg_distance + self.margin)
        triplet_loss = triplet_loss.sum()
        return triplet_loss, None

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        # ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
