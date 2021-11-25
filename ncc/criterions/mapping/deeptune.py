# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

from ncc.criterions import NccCriterion, register_criterion
from ncc.data.dictionary import Dictionary
from ncc.utils import utils
from ncc.utils.logging import metrics


@register_criterion('deeptune_loss')
class DeepTuneLoss(NccCriterion):

    def __init__(self, task, sentence_avg, hybrid_weight=1.0, src_weight=0.2):
        super(DeepTuneLoss, self).__init__(task)
        if hasattr(task, 'target_dictionary'):
            tgt_dict = task.target_dictionary
            self.padding_idx = tgt_dict.pad() if (tgt_dict is not None) and isinstance(tgt_dict, Dictionary) else -100
        self.sentence_avg = sentence_avg
        self.hybrid_weight = hybrid_weight
        self.src_weight = src_weight

    def forward(self, model, sample, reduce=True):
        hybrid_out, src_out = model(**sample['net_input'])
        loss, accuracy = self.compute_loss(model, hybrid_out, src_out, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'accuracy': accuracy,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, hybrid_out, src_out, sample, reduce=True):
        target = sample['target'].view(-1)
        target_onehot = F.one_hot(target, num_classes=2).float()
        # hybrid representation
        hybrid_loss = F.binary_cross_entropy(
            hybrid_out,
            target_onehot,
            reduction='sum' if reduce else 'none',
        )
        # source code representation
        src_loss = F.binary_cross_entropy(
            src_out,
            target_onehot,
            reduction='sum' if reduce else 'none',
        )
        loss = self.hybrid_weight * hybrid_loss + self.src_weight * src_loss
        with torch.no_grad():
            predictions = hybrid_out.argmax(dim=-1)
            accuracy = (predictions == target).sum()
        return loss, accuracy

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        accuracy_sum = sum(log.get('accuracy', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar('accuracy', accuracy_sum / sample_size, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
