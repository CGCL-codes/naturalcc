# -*- coding: utf-8 -*-

import math

import torch

from ncc.criterions import NccCriterion, register_criterion
from ncc.data.constants import EPS
from ncc.utils import utils
from ncc.utils.logging import metrics


@register_criterion('avg_triplet')
class AvgTripletCriterion(NccCriterion):
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

    def compute_loss(self, model, net_output, reduce=True):
        src_emb, tgt_emb = net_output  # B x T
        batch_size = src_emb.size(0)
        src_emb = src_emb.unsqueeze(dim=0).expand(batch_size, -1, -1)  # B* x B x T
        src_emb = src_emb.transpose(0, 1)  # B x B* x T
        tgt_emb = tgt_emb.unsqueeze(dim=0).expand(batch_size, -1, -1)  # B* x B x T
        distance = torch.norm(tgt_emb - src_emb, dim=-1)  # B x B
        correct_distances = torch.diag(distance).unsqueeze(dim=-1)  # B x 1
        pointwise_loss = torch.relu(correct_distances - distance + self.margin)
        pointwise_loss = pointwise_loss * (1 - torch.eye(*pointwise_loss.size()).to(pointwise_loss.device))  # B x B
        loss = pointwise_loss.sum(dim=-1) / ((pointwise_loss > 0).sum(dim=-1) + EPS)  # B x 1
        loss = loss.sum()
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
