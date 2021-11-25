# -*- coding: utf-8 -*-


from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


class TransformerDecoderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TransformerDecoderLayer, self).__init__()

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Scriptable reorder incremental state in transformer layers."""
        self.self_attn.reorder_incremental_state(incremental_state, new_order)

        if self.encoder_attn is not None:
            self.encoder_attn.reorder_incremental_state(incremental_state, new_order)
