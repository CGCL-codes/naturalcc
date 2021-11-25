from typing import Optional

import torch.nn as nn
from torch import Tensor


class TransformerEncoderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayer, self).__init__()

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        raise NotImplementedError

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]
