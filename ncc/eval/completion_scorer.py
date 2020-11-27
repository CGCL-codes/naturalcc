# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys

from ncc.utils import utils
import numpy as np


class CompletionScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None, compute_alignment=False, eos=None):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment

    @torch.no_grad()
    def complete(self, models, sample, predict_type, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']
        # node_id = sample['node_ids']
        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        avg_probs = None
        avg_curr_probs = None

        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            curr_prob = model.get_normalized_probs(decoder_out, log_probs=len(models) == 1, sample=sample).data

            probs = gather_target_probs(curr_prob, sample['target'])
            probs = probs.view(sample['target'].shape)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)

            if avg_curr_probs is None:
                avg_curr_probs = curr_prob
            else:
                avg_curr_probs.add_(curr_prob)

        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            avg_curr_probs.div_(len(models))
            avg_curr_probs.log_()

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        # mask = sample['target'] != self.pad
        selected = sample['node_ids'][predict_type]

        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len

            lprob = avg_curr_probs[i]

            if selected[i]:
                selected_prob = lprob[selected[i]].contiguous()
                rank = torch.argmax(selected_prob, 1)
                mrr = np.mean([1. / (r.item() + 1) for r in rank.view(-1)])

                ncorrect = torch.sum(rank == sample['target'][i][selected[i]].contiguous())
                accuracy = ncorrect / sum(selected[i])

                hypos.append([{
                    'tokens': ref,
                    'score': score_i,
                    'positional_scores': avg_probs_i,
                    'accuracy': accuracy,
                    'mrr': mrr,
                }])
            else:
                hypos.append([{
                    'tokens': ref,
                    'score': score_i,
                    'positional_scores': avg_probs_i,
                    'accuracy': 0.0,
                    'mrr': 0.0,
                }])
        return hypos
