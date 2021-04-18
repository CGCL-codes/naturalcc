# -*- coding: utf-8 -*-

import math

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
        loss, _ = self.compute_loss(model, net_output, reduce=reduce)
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
        src_repr, pos_repr, neg_repr = net_output
        pos_dist = self.cos_similarity(src_repr, pos_repr)  # B X 1
        neg_dist = self.cos_similarity(src_repr, neg_repr)  # B X 1
        loss = (self.margin - pos_dist + neg_dist).clamp(EPS).sum()
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=6)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=6)
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
