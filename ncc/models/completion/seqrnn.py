from ncc.models import register_model
from ncc.models.ncc_model import NccLanguageModel
from ncc.modules.common.layers import (
    Embedding, Linear, LSTM
)
from ncc.modules.seq2seq.ncc_decoder import NccDecoder
import torch.nn.functional as F


class LSTMDecoder(NccDecoder):
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512,
        num_layers=1, bidirectional=False, dropout=0.5,
        pretrained_embed=None,
        shared_embedding=False,
    ):
        super(LSTMDecoder, self).__init__(dictionary)
        if pretrained_embed is None:
            self.embed_tokens = Embedding(len(dictionary), embed_dim, padding_idx=dictionary.pad())
        else:
            self.embed_tokens = pretrained_embed
        self.rnn = LSTM(
            embed_dim, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True,
            bidirectional=False,  # in prediction task, cannot set bidirectional True
        )
        # self.dropout = dropout
        # self.bidirectional = bidirectional
        # if bidirectional:
        #     self.proj = Linear(hidden_size * 2, hidden_size)
        self.fc_out = Linear(hidden_size, len(dictionary))
        if shared_embedding:
            self.fc_out.weight = self.embed_tokens.weight

    def forward(self, src_tokens, **kwargs):
        x = self.embed_tokens(src_tokens)  # B, L-1, E
        x, _ = self.rnn(x)
        # if self.bidirectional:
        #     x = self.proj(x)
        #     x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc_out(x)
        return [x]


@register_model('completion_seqrnn')
class SeqRNNModel(NccLanguageModel):
    def __init__(self, args, decoder):
        super().__init__(decoder)
        self.args = args

    @classmethod
    def build_model(cls, args, config, task):
        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args['model']['decoder_embed_dim'],
            hidden_size=args['model']['decoder_hidden_size'],
            num_layers=args['model']['decoder_layers'],
            # bidirectional=args['model']['decoder_bidirectional'],
            dropout=args['model']['dropout'],
            pretrained_embed=None,
            shared_embedding=args['model'].get('shared_embedding', False),
        )
        return cls(args, decoder)

    def forward(self, src_tokens, **kwargs):
        return self.decoder(src_tokens, **kwargs)
