# -*- coding: utf-8 -*-

import math

import torch.nn.functional as F

from ncc.criterions import NccCriterion, register_criterion
from ncc.data.dictionary import Dictionary
from ncc.utils import utils
from ncc.utils.logging import metrics


@register_criterion('code_disen_criterion')
class CodeDisenCriterion(NccCriterion):
    def __init__(self, task, sentence_avg, forward_func):
        super().__init__(task)
        if hasattr(task, 'target_dictionary'):
            tgt_dict = task.target_dictionary
            self.padding_idx = tgt_dict.pad() if (tgt_dict is not None) and isinstance(tgt_dict, Dictionary) else -100
        self.sentence_avg = sentence_avg
        self.forward_func = forward_func

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss, reconstruction_loss, paraphrase_loss = getattr(model, self.forward_func)(**sample['net_input'])
        # sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        sample_size = 1
        logging_output = {
            'loss': loss.data,
            'recon_loss': reconstruction_loss.data,
            'para_loss': paraphrase_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        recon_loss_sum = sum(log.get('recon_loss', 0) for log in logging_outputs)
        para_loss_sum = sum(log.get('para_loss', 0) for log in logging_outputs)
        bleu_sum = sum(log.get('bleu', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('recon_loss', recon_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('para_loss', para_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('bleu', bleu_sum / sample_size, sample_size, round=3)
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
