import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ncc.modules.code2vec.ncc_encoder import EncoderOut
from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.code2vec.neural_transformer.neural_transformer_encoder_layer import NeuralTransformerEncoderLayer
from ncc.modules.roberta.learned_positional_embedding import LearnedPositionalEmbedding
from ncc.modules.roberta.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from ncc.utils import utils

DEFAULT_MAX_SOURCE_POSITIONS = 1e5


class NeuralTransformerEncoder(NccEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout = args['model']['dropout']
        self.encoder_layerdrop = args['model']['encoder_layerdrop']

        self.embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args['model']['max_source_positions']

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args['model']['no_scale_embedding'] else math.sqrt(self.embed_dim)

        offset_positions_by_padding = args['model'].get('offset_positions_by_padding', False)
        if args['model']['encoder_positional_embeddings']:
            # Option 1
            if args['model']['encoder_position_encoding_version'] == 'ncc_sinusoidal':
                self.embed_positions = SinusoidalPositionalEmbedding(
                    self.embed_dim,
                    padding_idx=self.padding_idx if offset_positions_by_padding else None,
                    init_size=args['model']['max_source_positions'] + self.padding_idx + 1 \
                        if offset_positions_by_padding else args['model']['max_source_positions'],
                )
            # Option 2
            elif args['model']['encoder_position_encoding_version'] == 'ncc_learned':
                num_embeddings = args['model']['max_source_positions']
                if offset_positions_by_padding:
                    num_embeddings += self.padding_idx + 1
                m = LearnedPositionalEmbedding(num_embeddings, self.embed_dim, padding_idx=None)
                nn.init.normal_(m.weight, mean=0, std=self.embed_dim ** -0.5)
                self.embed_positions = m
        else:
            self.embed_positions = None

        self.layer_wise_attention = args['model']['layer_wise_attention']
        self.layers = nn.ModuleList(
            [NeuralTransformerEncoderLayer(args) for _ in range(args['model']['encoder_layers'])]
        )
        self.num_layers = len(self.layers)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        encoder_states = [] if return_all_hiddens else None
        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out: Dict[str, Tensor] = {}

        new_encoder_out["encoder_out"] = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_out["encoder_padding_mask"] = (
            encoder_out.encoder_padding_mask
            if encoder_out.encoder_padding_mask is None
            else encoder_out.encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_out["encoder_embedding"] = (
            encoder_out.encoder_embedding
            if encoder_out.encoder_embedding is None
            else encoder_out.encoder_embedding.index_select(0, new_order)
        )

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out["encoder_out"],  # T x B x C
            encoder_padding_mask=new_encoder_out["encoder_padding_mask"],  # B x T
            encoder_embedding=new_encoder_out["encoder_embedding"],  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=encoder_out.src_tokens,
            src_lengths=encoder_out.src_lengths,
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
