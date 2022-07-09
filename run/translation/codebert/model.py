# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncc import LOGGER

from ncc.data import constants
from ncc.data.dictionary import TransformersDictionary
from ncc.modules.base.layers import (Linear, )
from ncc.utils.file_ops import file_io
from ncc.utils.path_manager import PathManager


class CodeBERT(nn.Module):

    @classmethod
    def build_model(cls, model="microsoft/codebert-base", max_length=200):
        from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
        config = RobertaConfig.from_pretrained(model)
        # vocab
        vocab = TransformersDictionary.from_pretrained(model, do_lower_case=False)
        # model
        encoder = RobertaModel.from_pretrained(model, config=config)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = cls(encoder=encoder, decoder=decoder,
                    vocab_size=len(vocab), hidden_size=config.hidden_size,
                    max_length=max_length, bos=vocab.bos(), eos=vocab.eos(), pad=vocab.pad(), )
        return vocab, model

    def __init__(self, encoder, decoder, vocab_size, hidden_size, max_length=None, bos=None, eos=None, pad=None):
        super(CodeBERT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # shared weight
        # self.lm_head.weight.data.copy_(self.encoder.embeddings.word_embeddings.weight.data)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)

        self.max_length = max_length
        self.bos = bos
        self.pad = pad
        self.eos = eos

        self.beam_size = 5

    def encoder_forward(self, src_tokens, src_mask):
        encoder_outputs = self.encoder(src_tokens, src_mask)
        encoder_output = encoder_outputs[0].permute([1, 0, 2]).contiguous()
        return encoder_output

    def decoder_forward(self, encoder_output, src_mask, tgt_tokens, tgt_mask=None):
        """
        target input: <bos> A B C <eos>
                      |    Decoder    |
                     A B C <eos> X
        loss(
        label: A B C <eos>
        logits: A B C <eos>
        )
        """
        # decoder
        attn_mask = -1e4 * (1 - self.bias[:tgt_tokens.size(1), :tgt_tokens.size(1)])
        tgt_embeddings = self.encoder.embeddings(tgt_tokens).permute([1, 0, 2]).contiguous()
        out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                           memory_key_padding_mask=(1 - src_mask).bool())
        hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
        logits = self.lm_head(hidden_states)
        return logits

    def forward(self, src_tokens, src_mask, tgt_tokens=None, tgt_mask=None):
        encoder_output = self.encoder_forward(src_tokens, src_mask)
        logits = self.decoder_forward(encoder_output, src_mask, tgt_tokens, tgt_mask)
        return logits

    def greedy_decode(self, encoder_output, src_mask):
        batch_size = encoder_output.size(1)
        context = encoder_output
        context_mask = src_mask

        input = encoder_output.new(batch_size, 1).long().fill_(self.bos)
        # greedy search
        for step in range(self.max_length):
            attn_mask = -1e4 * (1 - self.bias[:input.size(1), :input.size(1)])
            tgt_embeddings = self.encoder.embeddings(input).permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                               memory_key_padding_mask=(1 - context_mask).bool())
            out = torch.tanh(self.dense(out))
            hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
            out = self.lsm(self.lm_head(hidden_states)).detach()
            best_idx = torch.max(out, dim=-1)[1].view(-1, 1)
            input = torch.cat([input, best_idx], dim=-1)

        return input[:, 1:]

    def beam_decode(self, encoder_output, src_mask, max_length=None):
        batch_size = encoder_output.size(1)
        max_length = self.max_length if max_length is None else max_length

        # Predict
        preds = []
        zero = torch.cuda.LongTensor(1).fill_(self.bos)
        for i in range(batch_size):
            context = encoder_output[:, i:i + 1]
            context_mask = src_mask[i:i + 1, :]
            beam = Beam(self.beam_size, self.bos, self.eos)
            input_ids = beam.getCurrentState()
            context = context.repeat(1, self.beam_size, 1)
            context_mask = context_mask.repeat(self.beam_size, 1)
            for _ in range(max_length):
                if beam.done():
                    break
                attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                   memory_key_padding_mask=(1 - context_mask).bool())
                out = torch.tanh(self.dense(out))
                hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p] + [zero] * (max_length - len(p))).view(1, -1) for p in
                    pred]
            preds.append(torch.cat(pred, 0).unsqueeze(0))

        preds = torch.cat(preds, 0)
        return preds


def save_checkpoint(filename, model, optimizer, lr_scheduler=None, epoch=None, bleu4=None):
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": None if lr_scheduler is None else lr_scheduler.state_dict(),
        "epoch": 1 if epoch is None else epoch,
        "bleu4": 0 if bleu4 is None else bleu4,
    }
    PathManager.mkdir(os.path.dirname(filename))
    with file_io.open(filename, "wb") as f:
        torch.save(state_dict, f)


def load_checkpoint(filename, model, optimizer=None, lr_scheduler=None):
    with file_io.open(filename, "rb") as f:
        state_dict = torch.load(f, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
        # state_dict = torch.load(f, map_location=lambda storage, loc: storage)
    LOGGER.info(f"initialize model with {filename}")

    is_parallel = all(
        'module.' in name
        for name in state_dict["model"].keys()
    )
    if is_parallel:
        params = {
            name[len('module.'):]: param
            for name, param in state_dict["model"].items()
        }
        model.load_state_dict(params)
    else:
        model.load_state_dict(state_dict["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(torch.cuda.current_device())
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
    return state_dict["epoch"], state_dict["bleu4"]


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
