# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from ncc.criterions import NccCriterion, register_criterion
from ncc.utils.logging import metrics


@register_criterion('type_predicition_cross_entropy')
class TypePredictionCrossEntropyCriterion(NccCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, sample_size = self.compute_loss(model, net_output, sample, reduce=reduce)
        # sample_size = 100 # TODO sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            # 'ntokens': sample['ntokens'],
            # 'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        logits = net_output[0]
        target = model.get_targets(sample, net_output)#.view(-1)
        no_type_id = self.task.target_dictionary.index('O')
        loss = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=no_type_id) # , reduction='sum'
        # loss2 = F.cross_entropy(logits.data.transpose(1, 2), target.data, ignore_index=no_type_id)
        # print('loss1: ', loss.item())
        sample_size = torch.sum(target.ne(no_type_id))
        # if sample_size == 0:
        #     sample_size += 1
        # print('sample_size: {}; loss1: {}; loss2: {}'.format(sample_size, loss.item(), loss2.item()))
        # print('target: ', target)
        return loss, sample_size
        # exit()
        #
        # lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        # target = model.get_targets(sample, net_output).view(-1)
        # # loss = F.nll_loss(
        # #     lprobs,
        # #     target,
        # #     ignore_index=self.task.target_dictionary.index('O'),#self.padding_idx,
        # #     reduction='sum' if reduce else 'none',
        # # )
        # no_type_id = self.task.target_dictionary.index('O')
        # ignore_any_loss = False
        # if ignore_any_loss:
        #     any_id = self.task.target_dictionary.index('$any$')
        #     labels_ignore_any = target.clone()
        #     labels_ignore_any[labels_ignore_any == any_id] = no_type_id
        #     loss = F.nll_loss(
        #         lprobs,
        #         labels_ignore_any,
        #         ignore_index=no_type_id,  # self.padding_idx,
        #         reduction='sum' if reduce else 'none',
        #     )
        #     sample_size = torch.sum(labels_ignore_any.ne(no_type_id))
        # else:
        #     loss = F.nll_loss(
        #         lprobs,
        #         target,
        #         ignore_index=no_type_id,  # self.padding_idx,
        #         reduction='sum' if reduce else 'none',
        #     )
        #     sample_size = torch.sum(target.ne(no_type_id))
        #
        #
        # return loss, sample_size

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        # ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=3) # / math.log(2)
        # if sample_size != ntokens:
        #     metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
        #     metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        # else:
        #     metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
