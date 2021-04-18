# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Any, Dict, List

from torch.nn.modules.loss import _Loss

from ncc.utils import utils
# from src import metrics
from ncc.utils.logging import metrics


class NccCriterion(_Loss):

    def __init__(self, task):
        super().__init__()
        self.task = task
        # pad in target_dictionary is not necessary, e.g. type inference task
        # if hasattr(task, 'target_dictionary'):
        #     # for fairseq
        #     tgt_dict = task.target_dictionary
        #     self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100
        # # for BERT tokenizer
        # self.padding_idx = task.tokenizer.pad_token_id

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        # Criterions can override this, but for convenience we also try
        # to automatically map argparse.Namespace keys to corresponding
        # arguments in the __init__.
        init_args = {}
        for p in inspect.signature(cls).parameters.values():
            if (
                p.kind == p.POSITIONAL_ONLY
                or p.kind == p.VAR_POSITIONAL
                or p.kind == p.VAR_KEYWORD
            ):
                # we haven't implemented inference for these argument types,
                # but PRs welcome :)
                raise NotImplementedError('{} not supported'.format(p.kind))

            assert p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}

            if p.name == 'task':
                init_args['task'] = task
            elif p.name in args['optimization']:
                init_args[p.name] = args['optimization'][p.name]
            # elif hasattr(args, p.name):
            #     init_args[p.name] = getattr(args, p.name)
            elif p.default != p.empty:
                pass  # we'll use the default value
            else:
                raise NotImplementedError(
                    'Unable to infer Criterion arguments, please implement '
                    '{}.build_criterion'.format(cls.__name__)
                )
        return cls(**init_args)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(
        logging_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            'The aggregate_logging_outputs API is deprecated. '
            'Please use the reduce_metrics API instead.'
        )
        raise NotImplementedError

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        """Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            'Criterions should implement the reduce_metrics API. '
            'Falling back to deprecated aggregate_logging_outputs API.'
        )
        agg_logging_outputs = cls.aggregate_logging_outputs(logging_outputs)
        for k, v in agg_logging_outputs.items():
            if k in {'nsentences', 'ntokens', 'sample_size'}:
                continue
            metrics.log_scalar(k, v)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


class LegacyNccCriterion(NccCriterion):

    def __init__(self, args, task):
        super().__init__(task=task)
        self.args = args

        utils.deprecation_warning(
            'Criterions should take explicit arguments instead of an '
            'argparse.Namespace object, please update your criterion by '
            'extending NccCriterion instead of LegacyNccCriterion.'
        )

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)
