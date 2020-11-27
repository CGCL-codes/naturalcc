import torch
import torch.nn as nn
import torch.nn.functional as F
from ncc.modules.seq2seq.ncc_incremental_decoder import NccIncrementalDecoder
from ncc.modules.embedding import Embedding
from ncc.utils import utils
from ncc.modules.adaptive_softmax import AdaptiveSoftmax

DEFAULT_MAX_TARGET_POSITIONS = 1e5


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class LSTMDecoder(NccIncrementalDecoder):
    """LSTM decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=input_feed_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        if attention:
            # TODO make bias configurable
            self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        return self.output_layer(x), attn_scores

    def extract_features(self, prev_output_tokens, encoder_out, incremental_state=None):
        """
        Similar to *forward* but only return features.
        """
        if encoder_out is not None:
            encoder_padding_mask = encoder_out['encoder_padding_mask']
            encoder_out = encoder_out['encoder_out']
        else:
            encoder_padding_mask = None
            encoder_out = None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
            srclen = encoder_outs.size(0)
        else:
            srclen = None

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        elif encoder_out is not None:
            # setup recurrent cells
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)
        else:
            # setup zero cells, since there is no encoder
            num_layers = len(self.layers)
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(num_layers)]
            prev_cells = [zero_state for i in range(num_layers)]
            input_feed = None

        assert srclen is not None or self.attention is None, \
            "attention is not supported if there are no encoder outputs"
        attn_scores = x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training) # 16x512

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.attention is not None:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return x, attn_scores

    # def __tens2sent(self,
    #                 t,
    #                 tgt_dict):
    #     words = []
    #     for idx, w in enumerate(t):
    #         widx = w.item()
    #         # if widx < len(tgt_dict):
    #         words.append(tgt_dict[widx])
    #         # else:
    #         #     widx = widx - len(tgt_dict)
    #         #     words.append(src_vocabs[idx][widx])
    #     return words

    # def generate_sentence(self, prev_output_tokens, encoder_out, incremental_state=None):
    #     """
    #             Similar to *forward* but only return features.
    #             """
    #     if encoder_out is not None:
    #         encoder_padding_mask = encoder_out['encoder_padding_mask']
    #         encoder_out = encoder_out['encoder_out']
    #     else:
    #         encoder_padding_mask = None
    #         encoder_out = None
    #
    #     device = encoder_out[0].device
    #     bsz = encoder_out[0].size(1)
    #     seqlen = 50
    #     prev_output_tokens = torch.zeros(bsz, 1).long().fill_(2).to(device)
    #
    #     # if incremental_state is not None:
    #     #     prev_output_tokens = prev_output_tokens[:, -1:]
    #     # bsz, seqlen = prev_output_tokens.size()
    #
    #     # get outputs from encoder
    #     if encoder_out is not None:
    #         encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
    #         srclen = encoder_outs.size(0)
    #     else:
    #         srclen = None
    #
    #     # initialize previous states (or get from cache during incremental generation)
    #     cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
    #     if cached_state is not None:
    #         prev_hiddens, prev_cells, input_feed = cached_state
    #     elif encoder_out is not None:
    #         # setup recurrent cells
    #         num_layers = len(self.layers)
    #         prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
    #         prev_cells = [encoder_cells[i] for i in range(num_layers)]
    #         if self.encoder_hidden_proj is not None:
    #             prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
    #             prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
    #         input_feed = None #torch.zeros(bsz, self.hidden_size).to(device)
    #     else:
    #         # setup zero cells, since there is no encoder
    #         num_layers = len(self.layers)
    #         zero_state = torch.zeros(bsz, self.hidden_size).to(device)
    #         prev_hiddens = [zero_state for i in range(num_layers)]
    #         prev_cells = [zero_state for i in range(num_layers)]
    #         input_feed = None
    #
    #     assert srclen is not None or self.attention is None, \
    #         "attention is not supported if there are no encoder outputs"
    #     attn_scores = torch.zeros(srclen, seqlen, bsz).to(device) if self.attention is not None else None
    #     # outs = []
    #     dec_preds = []
    #     # seq, seq_logp_gathered, seq_lprob_sum = torch.zeros(batch_size, seq_length).long().to(device), \
    #     #                                         torch.zeros(batch_size, seq_length).to(device), \
    #     #                                         torch.zeros(batch_size, seq_length).to(device)
    #     # seq = torch.zeros()
    #     for j in range(seqlen):
    #         # embed tokens
    #         prev_output_tokens_emb = self.embed_tokens(prev_output_tokens)
    #         prev_output_tokens_emb = F.dropout(prev_output_tokens_emb, p=self.dropout_in, training=self.training)
    #
    #         # B x T x C -> T x B x C
    #         prev_output_tokens_emb = prev_output_tokens_emb.squeeze(1) #transpose(0, 1)
    #
    #         # input feeding: concatenate context vector from previous time step
    #         if input_feed is not None:
    #             input = torch.cat((prev_output_tokens_emb, input_feed), dim=1)
    #         else:
    #             input = prev_output_tokens_emb
    #
    #         for i, rnn in enumerate(self.layers):
    #             # recurrent cell
    #             hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
    #
    #             # hidden state becomes the input to the next layer
    #             # hidden = F.dropout(hidden, p=self.dropout_out, training=self.training)
    #
    #             # save state for next time step
    #             prev_hiddens[i] = hidden
    #             prev_cells[i] = cell
    #
    #         # apply attention using the last layer's hidden state
    #         if self.attention is not None:
    #             out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
    #         else:
    #             out = hidden
    #         # out = F.dropout(out, p=self.dropout_out, training=self.training)
    #         # input feeding
    #         if input_feed is not None:
    #             input_feed = out
    #
    #         decoded = self.output_layer(out)  # (batch_size*comment_dict_size)
    #         logprobs = F.log_softmax(decoded, dim=-1)  # (batch_size*comment_dict_size)
    #         prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)
    #
    #         # input feeding
    #         if input_feed is not None:
    #             input_feed = out
    #
    #         # save final output
    #         # outs.append(out)
    #
    #         # if choice == 'greedy':
    #         #     tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
    #         #     log_prob = torch.log(tgt_prob + 1e-20)
    #         # elif choice == 'sample':
    #         #     tgt, log_prob = self.reinforce.sample(prediction.unsqueeze(1))
    #         # else:
    #         #     assert False
    #
    #         sample_max = True
    #         if sample_max:
    #             sample_logprobs, predicted = torch.max(prob_prev, 1)
    #             dec_preds.append(predicted.clone())
    #             # seq[:, j] = predicted.reshape(-1)
    #             # seq_logp_gathered[:, j] = sample_logprobs
    #             # seq_logprobs[:, j, :] = logprobs
    #         else:
    #             predicted = torch.multinomial(prob_prev, 1)  # .to(device)
    #             # seq[:, j] = predicted.reshape(-1)
    #             # seq_logp_gathered[:, j] = logprobs.gather(1, predicted).reshape(-1)
    #             # seq_logprobs[:, j, :] = logprobs
    #             dec_preds.append(predicted.clone())
    #
    #         # dec_log_probs.append(log_prob.squeeze(1))
    #         # dec_preds.append(tgt.squeeze(1).clone())
    #
    #         # if "std" in attns:
    #         #     std_attn = f.softmax(attns["std"], dim=-1)
    #         #     attentions.append(std_attn.squeeze(1))
    #         # if self.copy_attn:
    #         #     mask = tgt.gt(len(params['tgt_dict']) - 1)
    #         #     copy_info.append(mask.float().squeeze(1))
    #
    #         words = self.__tens2sent(predicted, self.dictionary)
    #
    #
    #         # words = [self.dictionary[w] for w in words]
    #         # words = torch.Tensor(words).type_as(predicted)
    #         # predicted_words = words.unsqueeze(1)
    #
    #         # cache previous states (no-op except during incremental generation)
    #         utils.set_incremental_state(
    #             self, incremental_state, 'cached_state',
    #             (prev_hiddens, prev_cells, input_feed),
    #         )
    #         prev_output_tokens = predicted.reshape(-1, 1)
    #
    #     dec_preds = torch.stack(dec_preds, dim=1)
    #
    #     predssss = []
    #     for pred in dec_preds.tolist():
    #
    #         predssss.append([{'tokens': torch.Tensor(pred).type_as(dec_preds)}])
    #     return predssss
    #     # # collect outputs across time steps
    #     # x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)
    #     #
    #     # # T x B x C -> B x T x C
    #     # x = x.transpose(1, 0)
    #     #
    #     # if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
    #     #     x = self.additional_fc(x)
    #     #     x = F.dropout(x, p=self.dropout_out, training=self.training)
    #     #
    #     # # # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
    #     # # if not self.training and self.need_attn and self.attention is not None:
    #     # #     attn_scores = attn_scores.transpose(0, 2)
    #     # # else:
    #     # #     attn_scores = None
    #     # return x, attn_scores

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            elif state is not None:
                return state.index_select(0, new_order)
            else:
                return None

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn