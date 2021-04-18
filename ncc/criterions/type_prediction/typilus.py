from ncc.criterions import NccCriterion, register_criterion
from ncc.criterions.common.cross_entropy import CrossEntropyCriterion
from ncc.utils.logging import metrics
from ._triplet import TripletCriterion


@register_criterion('typilus')
class TypilusCriterion(NccCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.cross_entropy = CrossEntropyCriterion(task, sentence_avg)
        self.triplet_loss = TripletCriterion(task, sentence_avg)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'], tgt_ids=sample['target'])
        loss, sample_size = self.compute_loss(model, net_output, sample, reduce=reduce)
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        ce_loss, _ = self.cross_entropy.compute_loss(model, [net_output], sample, reduce)
        triple_loss, _ = self.triplet_loss.compute_loss(net_output, sample['target_equal_ids'])
        sample_size = sample['target'].size(0)
        return ce_loss + triple_loss, sample_size

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=3)  # / math.log(2)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
