import math
from typing import Optional, Dict, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ncc.modules.layer_norm import LayerNorm
from ncc.modules.seq2seq.ncc_incremental_decoder import NccIncrementalDecoder
from ncc.modules.code2vec.ncc_encoder import EncoderOut
from ncc.modules.roberta.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from ncc.modules.seq2seq.neural_transformer.neural_transformer_decoder_layer import NueralTransformerDecoderLayer
from ncc.modules.adaptive_softmax import AdaptiveSoftmax
from ncc.utils import utils


class NeuralTransformerDecoder(NccIncrementalDecoder):
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

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args['task']['max_target_positions']

        self.embed_tokens = embed_tokens
        self.out_proj_bias = nn.Parameter(torch.Tensor(len(dictionary)))
        nn.init.constant_(self.out_proj_bias, 0.0)

        self.embed_scale = 1.0 if args['model']['no_scale_embedding'] else math.sqrt(embed_dim)

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        if args['model']['decoder_positional_embeddings']:
            self.embed_positions = None
        else:
            # Option 1
            if args['model']['decoder_position_encoding_version'] == 'ncc_sinusoidal':
                from ncc.modules.roberta.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
                offset_positions_by_padding = False
                self.embed_positions = SinusoidalPositionalEmbedding(
                    self.embed_dim,
                    padding_idx=self.padding_idx,  # (self.padding_idx if offset_positions_by_padding else None),
                    init_size=args['model'][
                                  'max_source_positions'] + self.padding_idx + 1 if offset_positions_by_padding else
                    args['model']['max_source_positions'],  # + 1 why?
                )
            # Option 2
            elif args['model']['decoder_position_encoding_version'] == 'ncc_learned':
                from ncc.modules.roberta.learned_positional_embedding import LearnedPositionalEmbedding
                if self.padding_idx is not None:
                    num_embeddings = args['model']['max_target_positions'] + self.padding_idx + 1
                m = LearnedPositionalEmbedding(num_embeddings, self.embed_dim, self.padding_idx)
                nn.init.normal_(m.weight, mean=0, std=self.embed_dim ** -0.5)
                if self.padding_idx is not None:
                    nn.init.constant_(m.weight[self.padding_idx], 0)
                self.embed_positions = m

        self.cross_self_attention = args['model']['cross_self_attention']
        self.layer_wise_attention = args['model']['layer_wise_attention']

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                NueralTransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args['model']['decoder_layers'])
            ]
        )
        self.num_layers = len(self.layers)

        self.adaptive_softmax = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args['model']['tie_adaptive_weights']
            else None
        )

        if args['model']['adaptive_softmax_cutoff'] is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                args['model']['adaptive_softmax_cutoff'],
                # options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args['model']['adaptive_softmax_dropout'],
                adaptive_inputs=embed_tokens if args['model']['tie_adaptive_weights'] else None,
                factor=args['model']['adaptive_softmax_factor'],
                tie_proj=args['model']['tie_adaptive_proj'],
            )
            # self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
            #                                         dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args['model']['decoder_normalize_before'] and not args['model'][
            'no_decoder_final_norm']:  # getattr(args, "no_decoder_final_norm", False)
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if args['model']['layernorm_embedding']:  # getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
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
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    def extract_features_scriptable(
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
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
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

        return x, {"attn": attn, "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight, bias=self.out_proj_bias)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

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


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
