import torch.nn.functional as F

from ncc.criterions import register_criterion
from ncc.criterions.common.cross_entropy import CrossEntropyCriterion


@register_criterion('be_cross_entropy')
class BECrossEntropyCriterion(CrossEntropyCriterion):
    """
    BECrossEntropyCriterion only handles:
        input:  <bos> a b c <eos>
        target: a b c <eos>
    """

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

        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        """
        """
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs[:, :-1, :].contiguous()
        target = model.get_targets(sample, net_output)
        target = target[:, 1:].contiguous()
        loss = F.nll_loss(
            lprobs.view(-1, lprobs.size(-1)),
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss
