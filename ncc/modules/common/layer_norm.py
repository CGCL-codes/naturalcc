# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True


    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            return super().forward(x)

except ImportError:
    has_fused_layernorm = False

from .layers import Parameter
from .initializers import constant
from ncc.data.constants import EPS


class NCCLayerNorm(nn.Module):
    """
        Layer Normalization class
    """

    def __init__(self, normalized_shape, eps=EPS):
        super(NCCLayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.weight = Parameter(normalized_shape, initializer=constant(1.0))
        self.bias = Parameter(normalized_shape, initializer=constant(1.0))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


TorchLayerNorm = torch.nn.LayerNorm


def LayerNorm(normalized_shape, eps=EPS, elementwise_affine=True):
    if torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
