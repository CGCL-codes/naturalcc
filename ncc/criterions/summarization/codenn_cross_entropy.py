# -*- coding: utf-8 -*-

import math

import torch.nn.functional as F

from ncc.criterions import register_criterion
from ncc.utils import utils
from ncc.utils.logging import metrics
from .be_cross_entropy import BECrossEntropyCriterion


def mean(*args):
    return sum(*args) / len(args)


@register_criterion('codenn_cross_entropy')
class CodeNNCrossEntropyCriterion(BECrossEntropyCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task, sentence_avg)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])

        loss, match, total = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'match': match,
            'total': total,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        """
        """
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs[:, :-1].contiguous()

        # predicted ids
        _, pred_ids = lprobs.topk(k=1, dim=-1)
        pred_ids = pred_ids.view(-1)

        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        target = target[:, 1:].contiguous().view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )

        valid_ids = target != self.task.tgt_dict.pad()

        match = (pred_ids[valid_ids] == target[valid_ids]).sum().item()
        total = valid_ids.sum().item()
        return loss, match, total

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        match = sum(log.get('match', 0) for log in logging_outputs)
        total = sum(log.get('total', 0) for log in logging_outputs)

        metrics.log_scalar('match', value=match, round=0)
        metrics.log_scalar('total', value=total, round=0)
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))
