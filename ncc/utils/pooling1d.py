# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

from ncc.data.constants import (
    EPS,
    INF,
)


def pooling1d(pooling1d):
    """
    Args:
        pooling1d: mean/max/weighted_mean
    References:
        https://github.com/github/CodeSearchNet/blob/47636ea72e9d7ea1ad3a5c977076cb54797bfa2f/src/utils/tfutils.py#L107-L149
    """

    def _mean_pooling1d(input_emb, **kwargs):
        """
        input_emb: [B x T x D]
        input_len: [B x 1]
        input_mask: [B x T]
        """
        input_emb_masked = input_emb * kwargs['input_mask'].unsqueeze(-1)  # B x T x D
        input_emb_sum = input_emb_masked.sum(dim=1)  # B x D
        input_emb = input_emb_sum / kwargs['input_len']
        return input_emb

    def _max_pooling1d(input_emb, **kwargs):
        """
        input_emb: [B x T x D]
        input_len: [B x 1]
        input_mask: [B x T]
        """
        input_mask = -INF * (1 - kwargs['input_mask'])  # B x T
        input_mask = input_mask.unsqueeze(dim=-1)  # B x T x 1
        input_emb, _ = (input_emb + input_mask).max(dim=1)  # B x D
        return input_emb

    def _weighted_mean_pooling1d(input_emb, **kwargs):
        """
        input_emb: [B x T x D]
        input_len: [B x 1]
        input_mask: [B x T]
        input_weights_layer: D => 1
        """
        input_weights = torch.sigmoid(kwargs['weight_layer'](input_emb))  # B x T x 1
        # weighted_input_emb = (input_emb * input_weights).masked_fill(kwargs['input_mask'].unsqueeze(dim=-1) == 0, 0)
        # weighted_input_emb = weighted_input_emb.transpose(-2, -1)
        # weighted_input_emb = F.avg_pool1d(weighted_input_emb, kernel_size=weighted_input_emb.size(-1)).squeeze(-1)
        # return weighted_input_emb
        input_weights = input_weights * kwargs['input_mask'].unsqueeze(dim=-1)  # B x T x 1
        input_emb_weighted_sum = (input_emb * input_weights).sum(dim=1)  # B x D
        return input_emb_weighted_sum / (input_weights.sum(dim=1) + EPS)  # B x D

    if pooling1d == 'mean':
        return _mean_pooling1d
    elif pooling1d == 'max':
        return _max_pooling1d
    elif pooling1d == 'weighted_mean':
        return _weighted_mean_pooling1d
    else:
        # raise NotImplementedError('No such pooling method, only [mean/max] pooling are available')
        return None
