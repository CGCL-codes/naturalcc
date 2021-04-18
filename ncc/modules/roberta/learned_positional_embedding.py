# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch.nn as nn

from ncc.utils import utils


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int],
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings + self.padding_idx + 1
        else:
            self.max_positions = self.num_embeddings

    def forward(self, input, incremental_state=None, positions=None):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (
            (positions is None) or (self.padding_idx is None)
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                if self.padding_idx is None:
                    positions = input.data.new(input.size(0), 1).fill_(int(input.size(1)))
                else:
                    positions = input.data.new(input.size(0), 1).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = utils.make_positions(input, self.padding_idx)
        return super().forward(positions)
