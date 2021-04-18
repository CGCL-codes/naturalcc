import math

import torch
from torch import nn


class PositionalEncoding_bak(nn.Module):
    """From https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model, dropout=0.1, max_len=9000):
        super(PositionalEncoding_bak, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

    def _load_from_state_dict(self, *args):
        print("PositionalEncoding: doing nothing on call to _load_from_state_dict")


class PositionalEncoding(nn.Module):
    """From https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model, dropout=0.0, max_len=9000):
        super(PositionalEncoding, self).__init__()
        torch.manual_seed(1)
        # self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(1, max_len+1, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        half_dim = d_model // 2
        emb = math.log(10000) / (half_dim - 1) # TODO -1 according to fairseq
        div_term = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x = x + self.pe[: x.size(1), :]
        # torch.manual_seed(1)
        # return self.dropout(x) # TODO 这里不需要dropout，因为fairseq都没有
        # return x
        return self.pe[: x.size(1), :]

    def _load_from_state_dict(self, *args):
        print("PositionalEncoding: doing nothing on call to _load_from_state_dict")


class CodeEncoder(nn.Module):
    def __init__(
        self,
        n_tokens,
        d_model=512,
        d_rep=256,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.0,
        activation="relu",
        norm=True,
        pad_id=None,
        project=False,
    ):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=9000)
        norm_fn = nn.LayerNorm(d_model) if norm else None
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers, norm=norm_fn)
        if project:
            self.project_layer = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_rep))
        # NOTE: We use the default PyTorch intialization, so no need to reset parameters.

    def forward(self, x, lengths=None, no_project_override=False):
        # src_emb = self.embedding(x).transpose(0, 1) * math.sqrt(self.config["d_model"])
        # src_emb = self.pos_encoder(src_emb)
        src_emb = self.embedding(x).transpose(0, 1)
        src_emb = src_emb * math.sqrt(self.config["d_model"])
        # src_emb = self.pos_encoder(src_emb)
        pe = self.pos_encoder(x)
        src_emb += pe
        if self.config["pad_id"] is not None:
            src_key_padding_mask = x == self.config["pad_id"]
        else:
            src_key_padding_mask = None
        out = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)  # TxBxD
        if not no_project_override and self.config["project"]:
            return self.project_layer(out.mean(dim=0))
        else:
            return out


class CodeEncoderLSTM(nn.Module):
    def __init__(
        self,
        n_tokens,
        # Deeptyper set d_model to 200
        d_model=512,
        d_rep=256,
        n_encoder_layers=2,
        dropout=0.1,
        pad_id=None,
        project=False,
    ):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=9000)

        #Currently using 2 layers of LSTM
        print(f"CodeEncoderLSTM: Creating BiLSTM with {n_encoder_layers} layers, {d_model} hidden and input size")
        # TODO: Apply dropout to LSTM
        self.encoder = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_encoder_layers, bidirectional=True)

        if project:
            if project == "sequence_mean" or project == "sequence_mean_nonpad":
                project_in = 2 * d_model
                self.project_layer = nn.Sequential(nn.Linear(project_in, d_model), nn.ReLU(), nn.Linear(d_model, d_rep))
            elif project == "hidden":
                project_in = n_encoder_layers * 2 * d_model
                self.project_layer = nn.Sequential(nn.Linear(project_in, d_model), nn.ReLU(), nn.Linear(d_model, d_rep))
            # elif project == "hidden_identity":
            #     pass
            else:
                raise ValueError(f"Unknown value '{project}' for CodeEncoderLSTM project argument")
        # NOTE: We use the default PyTorch intialization, so no need to reset parameters.

    def forward(self, x, lengths, no_project_override=False):
        self.encoder.flatten_parameters()
        B, T = x.size(0), x.size(1)
        src_emb = self.embedding(x).transpose(0, 1) * math.sqrt(self.config["d_model"])
        # src_emb = self.pos_encoder(src_emb)
        pe = self.pos_encoder(x)
        src_emb += pe
        # Compute sequence lengths and pack src_emb
        # src_emb_packed = torch.nn.utils.rnn.pack_padded_sequence(src_emb, lengths, enforce_sorted=False)
        out, (h_n, c_n) = self.encoder(src_emb)  # TxBxD
        # out, _ = torch.nn.utils.rnn.pad_packed_sequence(out)
        # if out.size(0) != T:
        #     print("WARNING: unexpected size of encoder output: out.shape=", out.shape, "x.shape=", x.shape, "src_emb.shape=", src_emb.shape)
        #     print("lengths.max()=", lengths.max())
        #     print("lengths.min()=", lengths.min())
        #
        # if not no_project_override and self.config["project"]:
        #     if self.config["project"] == "sequence_mean":
        #         # out is T x B x n_directions*d_model
        #         rep = out.mean(dim=0)  # B x n_directions*d_model
        #     elif self.config["project"] == "sequence_mean_nonpad":
        #         out_ = out.transpose(0, 1)  # B x T x n_directions*d_model
        #         mask = torch.arange(out_.size(1), device=out_.device).unsqueeze(0).unsqueeze(-1).expand_as(out_) < lengths.unsqueeze(1).unsqueeze(2)
        #         rep = (out_ * mask.float()).sum(dim=1)  # B x n_directions*d_model
        #         rep = rep / lengths.unsqueeze(1).float()
        #     elif self.config["project"] == "hidden":
        #         # h_n is n_layers*n_directions x B x d_model
        #         rep = torch.flatten(h_n.transpose(0, 1), start_dim=1)
        #     # elif self.config["project"] == "hidden_identity"
        #     #     return torch.flatten(h_n.transpose(0, 1), start_dim=1)
        #     else:
        #         raise ValueError
        #     return self.project_layer(rep)

        return out


# class CodeEncoderGGNN(nn.Module):
#     def __init__(
#         self,
#         n_tokens,
#         # Deeptyper set d_model to 200
#         d_model=512,
#         d_rep=256,
#         n_encoder_layers=2,
#         dropout=0.1,
#         pad_id=None,
#         project=False,
#     ):
#         super().__init__()
#         self.config = {k: v for k, v in locals().items() if k != "self"}
#         self.embedding = nn.Embedding(n_tokens, d_model)
#
#         message_passing_layers = xxx
#         node_embedder = xxx
#
#     def forward(self, x, lengths, no_project_override=False):
#         all_states = [node_representations]
#         for mp_layer_idx, mp_layer in enumerate(self.__message_passing_layers):
#             node_representations = mp_layer(
#                 node_states=node_representations,
#                 adjacency_lists=adjacency_lists,
#                 node_to_graph_idx=node_to_graph_idx,
#                 reference_node_ids=reference_node_ids,
#                 reference_node_graph_idx=reference_node_graph_idx,
#             )
#             all_states.append(node_representations)
#         if return_all_states:
#             node_representations = torch.cat(all_states, dim=-1)
#         return node_representations
