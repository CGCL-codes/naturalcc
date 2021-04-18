# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from ncc.utils import utils
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict


class TransformerGenerator(object):
    def __init__(
        self,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        temperature=1.,
        match_source_len=False,
        no_repeat_ngram_size=0,
        eos=None
    ):
        """Generates translations of a given source sentence.

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.bos = tgt_dict.bos()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert temperature > 0, '--temperature must be greater than 0'

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.NccModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        model = models[0]  # for ensemble expansion

        if not self.retain_dropout:
            model.eval()

        src_tokens = sample['net_input']['src_tokens']
        src_lengths = (src_tokens != self.pad).int().sum(-1)
        bsz, src_len = src_tokens.size()
        device = src_tokens.device

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )
        assert self.min_len <= max_len, 'min_len cannot be larger than max_len, please adjust these!'

        encoder_out = model.encoder(sample['net_input']['src_tokens'], src_lengths=sample['net_input']['src_lengths'])

        prev_output_tokens = torch.zeros(bsz, 1).long().fill_(self.bos).to(device)
        # prev_output_tokens = torch.zeros(bsz, 1).long().fill_(self.eos).to(device)

        dec_preds = []
        # 2. generate
        from collections import OrderedDict
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = OrderedDict()
        full_context_alignment: bool = False
        alignment_layer: Optional[int] = None
        alignment_heads: Optional[int] = None
        for j in range(max_len + 1):

            # incremental_state['step'] = j
            decoder_outputs, attns = model.decoder(prev_output_tokens, encoder_out=encoder_out, \
                                                   incremental_state=incremental_state)

            prediction = decoder_outputs.squeeze(1)
            prediction = prediction.log_softmax(dim=1)

            sample_max = True
            if sample_max:
                sample_logprobs, predicted = torch.max(prediction, dim=-1, keepdim=True)
            else:
                predicted = torch.multinomial(prediction, 1)  # .to(device)
            dec_preds.append(predicted.squeeze(1).clone())
            prev_output_tokens = torch.cat((prev_output_tokens, predicted), dim=-1)

        dec_preds = torch.stack(dec_preds, dim=1)

        predictions = []
        for pred in dec_preds.tolist():
            predictions.append([{'tokens': torch.Tensor(pred).type_as(dec_preds)}])

        return predictions
