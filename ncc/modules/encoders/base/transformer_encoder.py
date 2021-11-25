# -*- coding: utf-8 -*-


from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor

from ncc.utils import utils
from ..ncc_encoder import (
    NccEncoder,
    EncoderOut,
)
from ...positional_embedding import (
    SinusoidalPositionalEmbedding,
)


class TransformerEncoder(NccEncoder):
    def __init__(self, dictionary):
        super(TransformerEncoder, self).__init__(dictionary)

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def load_state_dict(self, state_dict, strict=True, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        state_dict = self.upgrade_state_dict(state_dict)

        return super().load_state_dict(state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        # Upgrade Roberta state dict for new versions of fairseq.
        if 'decoder.sentence_encoder.emb_layer_norm.weight' in state_dict:
            state_dict = self.upgrade_state_dict_from_roberta(state_dict)
        # """Upgrade old state dicts to work with newer code."""
        state_dict = self.upgrade_state_dict_named(state_dict, "")
        return state_dict

    def upgrade_state_dict_named_(self, state_dict, name):
        # "Upgrade a (possibly old) state dict for new versions of fairseq."
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}embed_positions.weight".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}layers.{}".format(name, i)
            )

        version_key = "{}version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

    def upgrade_state_dict_named(self, state_dict, name):
        # "Upgrade a (possibly old) state dict for new versions of fairseq."
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

    def upgrade_state_dict_from_roberta(self, state_dict):

        keys_to_delete = [
            'decoder.sentence_encoder.emb_layer_norm.weight', 'decoder.sentence_encoder.emb_layer_norm.bias',
            'decoder.lm_head.weight', 'decoder.lm_head.bias',
            'decoder.lm_head.dense.weight', 'decoder.lm_head.dense.bias',
            'decoder.lm_head.layer_norm.weight', 'decoder.lm_head.layer_norm.bias',
        ]

        for k in keys_to_delete:
            del state_dict[k]

        component_type = 'decoder.sentence_encoder'
        component_state_dict = OrderedDict()
        for key in state_dict.keys():
            if key.startswith(component_type):
                # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
                component_subkey = key[len(component_type) + 1:]
                component_state_dict[component_subkey] = state_dict[key]

        state_dict = component_state_dict
        return state_dict

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(
                    utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
                )
        return self._future_mask[:dim, :dim]
