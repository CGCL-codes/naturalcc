import math
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ncc.modules.code2vec.transformer_encoder_layer import TransformerEncoderLayer
from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.code2vec.ncc_encoder import EncoderOut
from ncc.utils import utils
from collections import OrderedDict

DEFAULT_MAX_SOURCE_POSITIONS = 1e5


class TransformerEncoder(NccEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        num_segments: int = 2,
        offset_positions_by_padding: bool = False,  # True,
        # apply_bert_init: bool = False,
        # freeze_embeddings: bool = False,
        # n_trans_layers_to_freeze: int = 0,
        # export: bool = False,
        traceable: bool = False,
    ):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self.args = args
        self.dropout = args['model']['dropout']
        self.encoder_layerdrop = args['model']['encoder_layerdrop']

        self.embed_dim = embed_tokens.embedding_dim
        self.padding_idx = dictionary.pad()  # embed_tokens.padding_idx TODO
        # self.vocab_size = vocab_size
        self.max_source_positions = args['model']['max_source_positions']

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args['model']['no_scale_embedding'] else math.sqrt(self.embed_dim)

        # Option 1
        if args['model']['position_encoding_version'] == 'ncc_sinusoidal':
            from ncc.modules.roberta.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
            self.embed_positions = SinusoidalPositionalEmbedding(
                self.embed_dim,
                padding_idx=self.padding_idx,  # (self.padding_idx if offset_positions_by_padding else None),
                init_size=args['model'][
                              'max_source_positions'] + self.padding_idx + 1 if offset_positions_by_padding else
                args['model']['max_source_positions'],  # + 1 why?
            )
        # Option 2
        elif args['model']['position_encoding_version'] == 'ncc_learned':
            from ncc.modules.roberta.learned_positional_embedding import LearnedPositionalEmbedding
            if self.padding_idx is not None:
                num_embeddings = args['model']['max_source_positions'] + self.padding_idx + 1
            m = LearnedPositionalEmbedding(num_embeddings, self.embed_dim, self.padding_idx)
            nn.init.normal_(m.weight, mean=0, std=self.embed_dim ** -0.5)
            if self.padding_idx is not None:
                nn.init.constant_(m.weight[self.padding_idx], 0)
            self.embed_positions = m
        # Option 3
        elif args['model']['position_encoding_version'] == 'contracode':
            from ncc.modules.roberta.sinusoidal_positional_embedding_simple import SinusoidalPositionalEmbedding_Simple
            self.embed_positions = SinusoidalPositionalEmbedding_Simple(self.embed_dim, args['model']['dropout'],
                                                                        max_len=args['model']['max_source_positions'])

        self.num_segments = num_segments
        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embed_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )
        torch.manual_seed(1)
        # Option 1
        # encoder_layer = nn.TransformerEncoderLayer(self.embed_dim, args['model']['encoder_attention_heads'],
        #                                            args['model']['encoder_ffn_embed_dim'], 0, args['model']['activation_fn']) # args['model']['dropout']
        # self.layers = nn.ModuleList([encoder_layer for i in range(args['model']['encoder_layers'])])
        # Option 2
        encoder_layer = TransformerEncoderLayer(args)
        self.layers = nn.ModuleList([encoder_layer for i in range(args['model']['encoder_layers'])])
        # Option 3: This will create different TransformerEncoderLayer for different i
        # self.layers = nn.ModuleList([TransformerEncoderLayer(args) for i in range(args['model']['encoder_layers'])])

        self.num_layers = len(self.layers)
        # if args['model']['encoder_normalize_before']:
        self.layer_norm = nn.LayerNorm(self.embed_dim)  # LayerNorm(self.embed_dim) TODO
        # else:
        #     self.layer_norm = None
        if args['model']['layernorm_embedding']:
            self.layernorm_embedding = nn.LayerNorm(self.embed_dim)  # LayerNorm(self.embed_dim, export=export) TODO
        else:
            self.layernorm_embedding = None

        self.traceable = traceable

    def forward_embedding(self, src_tokens, positions, segment_labels, padding_mask):
        x = self.embed_tokens(src_tokens)

        if self.embed_scale is not None:
            x = embed = x * self.embed_scale

        if self.embed_positions is not None:
            if self.args['model']['position_encoding_version'] == 'contracode':
                x += self.embed_positions(src_tokens)
            elif self.args['model']['position_encoding_version'] == 'ncc_sinusoidal':
                x += self.embed_positions(src_tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        # x = F.dropout(x, p=self.dropout, training=self.training) #TODO, position里面如果已经dropout了，这里就没必要了
        # # account for padding while computing the representation
        # if padding_mask is not None:
        #     x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        return x, embed

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor = None,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not self.traceable and not encoder_padding_mask.any():
            encoder_padding_mask = None

        x, encoder_embedding = self.forward_embedding(src_tokens, positions, segment_labels, encoder_padding_mask)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        # if not last_state_only:
        #     encoder_states.append(x)

        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # dropout_probability = random.uniform(0, 1)
            # if not self.training or (dropout_probability > self.encoder_layerdrop):
            x = layer(x, encoder_padding_mask)  # TODO
            # x = layer(x, src_mask=None, src_key_padding_mask=encoder_padding_mask)
            if not last_state_only:
                encoder_states.append(x)

        # if self.layer_norm is not None:
        x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

        # sentence_rep = x[0, :, :]  # <CLS> presentation
        #
        # if last_state_only:
        #     encoder_states = [x]
        #
        # if self.traceable:
        #     return torch.stack(encoder_states), sentence_rep
        # else:
        #     return encoder_states, sentence_rep

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
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

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
