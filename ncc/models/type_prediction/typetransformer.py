import math

import torch
import torch.nn as nn

from ncc.models.type_prediction.encoder import CodeEncoder, CodeEncoderLSTM
from ncc.models import NccLanguageModel, register_model
from ncc.modules.seq2seq.ncc_decoder import NccDecoder


@register_model('typetransformer')
class TypeTransformer(NccLanguageModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        # self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""

        # make sure all arguments are present
        # base_architecture(args)

        # if not hasattr(args, 'max_positions'):
        if 'max_positions' not in args['model']:
            args['model']['max_positions'] = args['task']['tokens_per_sample']

        encoder = RobertaEncoder(args, task.source_dictionary, task.target_dictionary, encoder_type=args['model']['encoder_type'])
        return cls(args, encoder)

    def forward(self, src_tokens, **kwargs): #, features_only=False, return_all_hiddens=False, classification_head_name=None,
        # if classification_head_name is not None:
        #     features_only = True
        #
        # x, extra = self.decoder(src_tokens, features_only, return_all_hiddens, **kwargs)
        #
        # if classification_head_name is not None:
        #     x = self.classification_heads[classification_head_name](x)
        # return x, extra

        x = self.decoder(src_tokens, **kwargs)

        return x, None


class RobertaEncoder(NccDecoder):
    def __init__(
        self,
        args,
        source_dictionary,
        target_dictionary,
        # n_tokens,
        # n_output_tokens,
        d_model=512,
        d_rep=128,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.0, # 0.1
        activation="relu",
        norm=True,
        # pad_id=None,
        encoder_type="transformer"
    ):
        # super(TypeTransformer, self).__init__()
        super().__init__(source_dictionary)
        self.args = args
        assert norm
        # assert pad_id is not None
        padding_idx = source_dictionary.pad()
        self.config = {k: v for k, v in locals().items() if k != "self"}

        # Encoder and output for type prediction
        assert (encoder_type in ["transformer", "lstm"])
        if encoder_type == "transformer":
            self.encoder = CodeEncoder(
                len(source_dictionary), d_model, d_rep, n_head, n_encoder_layers, d_ff, dropout, activation, norm, padding_idx, project=False
            )
            # TODO: Try LeakyReLU
            self.output = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, len(target_dictionary)))
        elif encoder_type == "lstm":
            self.encoder = CodeEncoderLSTM(
                n_tokens=len(source_dictionary),
                d_model=d_model,
                d_rep=d_rep,
                n_encoder_layers=n_encoder_layers,
                dropout=dropout,
                pad_id=padding_idx,
                project=False
            )
            self.output = nn.Sequential(nn.Linear(d_model*2, d_model), nn.ReLU(), nn.Linear(d_model, len(target_dictionary)))

    def forward(self, src_tokens, src_length=None, output_attention=None):
        r"""
        Arguments:
            src_tok_ids: [B, L] long tensor
            output_attention: [B, L, L] float tensor
        """
        if output_attention is not None and src_tokens.size(0) != output_attention.size(0):
            raise RuntimeError("the batch number of src_tok_ids and output_attention must be equal")

        # Encode
        memory = self.encoder(src_tokens, src_length)  # LxBxD
        memory = memory.transpose(0, 1)  # BxLxD

        if output_attention is not None:
            # Aggregate features to the starting token in each labeled identifier
            memory = torch.matmul(output_attention, memory)  # BxLxD

        # Predict logits over types
        return self.output(memory)  # BxLxV
