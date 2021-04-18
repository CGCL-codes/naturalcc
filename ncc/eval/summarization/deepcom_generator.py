import torch
from ncc.utils import utils
import torch.nn.functional as F


class DeepComGenerator(object):
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

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )
        assert self.min_len <= max_len, 'min_len cannot be larger than max_len, please adjust these!'

        # 1. encoder
        encoder_out = model.encoder(sample['net_input']['src_tokens'], src_lengths=sample['net_input']['src_lengths'],
                                    **kwargs)
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out = encoder_out['encoder_out']

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]

        # setup recurrent cells
        num_layers = len(model.decoder.layers)
        state_size = bsz, model.decoder.hidden_size
        # for support float16
        prev_hiddens = [encoder_outs.new_zeros(*state_size) for _ in range(num_layers)]
        prev_cells = [encoder_outs.new_zeros(*state_size) for _ in range(num_layers)]
        prev_output_tokens = encoder_outs.new(bsz).long().fill_(self.eos)  # [B]

        dec_preds = []
        # 2. generate
        for j in range(max_len):
            # embed tokens
            input = model.decoder.embed_tokens(prev_output_tokens)  # [B] => [B, C]

            for i, rnn in enumerate(model.decoder.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out = model.decoder.attention(prev_hiddens[-1], encoder_outs, encoder_padding_mask)
            decoded = model.decoder.output_layer(out)  # (batch_size*comment_dict_size)

            prob_prev = torch.softmax(decoded, dim=-1)  # (batch_size*comment_dict_size)
            sample_logprobs, predicted = torch.max(prob_prev, 1)
            dec_preds.append(predicted.clone())
            prev_output_tokens = predicted

        dec_preds = torch.stack(dec_preds, dim=1)

        predictions = []
        for pred in dec_preds.tolist():
            predictions.append([{'tokens': torch.Tensor(pred).type_as(dec_preds)}])

        return predictions
