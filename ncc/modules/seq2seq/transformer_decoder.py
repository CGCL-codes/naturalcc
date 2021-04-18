import math
from typing import Optional, Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ncc.modules.code2vec.ncc_encoder import EncoderOut
from ncc.modules.common.layers import Linear
from ncc.modules.roberta.learned_positional_embedding import LearnedPositionalEmbedding
from ncc.modules.roberta.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from ncc.modules.seq2seq.ncc_incremental_decoder import NccIncrementalDecoder
from ncc.modules.seq2seq.transformer_decoder_layer import TransformerDecoderLayer
from ncc.utils import utils


class TransformerDecoder(NccIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout = args['model']['dropout']
        self.decoder_layerdrop = args['model']['decoder_layerdrop']
        self.share_input_output_embed = args['model']['share_decoder_input_output_embed']

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args['model']['decoder_embed_dim']
        self.embed_dim = embed_dim
        self.output_embed_dim = args['model']['decoder_output_dim']

        self.padding_idx = dictionary.pad()  # embed_tokens.padding_idx TODO
        self.max_target_positions = args['task']['max_target_positions']

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args['model']['no_scale_embedding'] else math.sqrt(embed_dim)

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        offset_positions_by_padding = args['model'].get('offset_positions_by_padding', True)
        if args['model']['decoder_positional_embeddings']:
            self.embed_positions = None
        else:
            # Option 1
            if args['model']['decoder_position_encoding_version'] == 'ncc_sinusoidal':
                self.embed_positions = SinusoidalPositionalEmbedding(
                    self.embed_dim,
                    padding_idx=self.padding_idx if offset_positions_by_padding else None,
                    init_size=args['model']['max_target_positions'] + self.padding_idx + 1 \
                        if offset_positions_by_padding else args['model']['max_target_positions'],
                )
            # Option 2
            elif args['model']['decoder_position_encoding_version'] == 'ncc_learned':
                num_embeddings = args['model']['max_target_positions']
                if offset_positions_by_padding:
                    num_embeddings += self.padding_idx + 1
                m = LearnedPositionalEmbedding(num_embeddings, self.embed_dim,
                                               padding_idx=self.padding_idx if offset_positions_by_padding else None)
                nn.init.normal_(m.weight, mean=0, std=self.embed_dim ** -0.5)
                if self.padding_idx is not None:
                    nn.init.constant_(m.weight[self.padding_idx], 0)
                self.embed_positions = m

        self.cross_self_attention = args['model']['cross_self_attention']
        self.layer_wise_attention = args['model']['layer_wise_attention']

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args['model']['decoder_layers'])
        ])
        self.num_layers = len(self.layers)

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args['model']['tie_adaptive_weights']
            else None
        )

        self.out_generator = Linear(embed_dim, len(dictionary), bias=args['model']['decoder_out_embed_bias'])
        if self.share_input_output_embed:
            self.out_generator.weight = self.embed_tokens.weight

        if args['model']['decoder_normalize_before'] and not args['model']['no_decoder_final_norm']:
            self.layer_norm = nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if args['model']['layernorm_embedding']:
            self.layernorm_embedding = nn.LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state: Optional[Tensor] = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_states = encoder_out.encoder_states
                    assert encoder_states is not None
                    encoder_state = encoder_states[idx]
                else:
                    encoder_state = encoder_out.encoder_out

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn, _ = layer(
                    x,
                    encoder_state,
                    encoder_out.encoder_padding_mask
                    if encoder_out is not None
                    else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        return self.out_generator(features)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    # Overwirte the method to temporaily soppurt jit scriptable in Transformer
    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Scriptable reorder incremental state in the transformer."""
        for layer in self.layers:
            layer.reorder_incremental_state(incremental_state, new_order)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict
