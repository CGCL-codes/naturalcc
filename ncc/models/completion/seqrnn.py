# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ncc.models import register_model
from ncc.models.ncc_model import NccLanguageModel
from ncc.modules.seq2seq.lstm_decoder import LSTMDecoder
DEFAULT_MAX_TARGET_POSITIONS = 1e5


@register_model('seqrnn')
class SeqRNNModel(NccLanguageModel):
    def __init__(self, args, decoder):
        super().__init__(decoder)
        self.args = args

    @classmethod
    def build_model(cls, args, config, task):
        max_target_positions = args['model']['max_target_positions'] if args['model']['max_target_positions'] \
            else DEFAULT_MAX_TARGET_POSITIONS

        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args['model']['decoder_embed_dim'],
            hidden_size=args['model']['decoder_hidden_size'],
            out_embed_dim=args['model']['decoder_out_embed_dim'],
            num_layers=args['model']['decoder_layers'],
            dropout_in=args['model']['decoder_dropout_in'],
            dropout_out=args['model']['decoder_dropout_out'],
            attention=args['model']['decoder_attention'],
            encoder_output_units=0,
            pretrained_embed=None,  # pretrained_decoder_embed
            share_input_output_embed=args['model']['share_decoder_input_output_embed'],
            adaptive_softmax_cutoff=(
                args['model']['adaptive_softmax_cutoff']
                if args['criterion'] == 'adaptive_loss' else None
            ),
            max_target_positions=max_target_positions
        )
        return cls(args, decoder)
