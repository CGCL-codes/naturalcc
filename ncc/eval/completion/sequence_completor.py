import torch

from ncc.modules.seq2seq.ncc_incremental_decoder import NccIncrementalDecoder


class SequenceCompletor(object):
    def __init__(
        self,
        retain_dropout=False,
    ):
        """Generates translations of a given source sentence.

        Args:
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
        """
        self.retain_dropout = retain_dropout

    @torch.no_grad()
    def complete(self, models, sample, **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.NccModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        model = EnsembleModel(models)
        return self._complete(model, sample, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.complete(*args, **kwargs)

    @torch.no_grad()
    def _complete(
        self,
        model,
        sample,
        **kwargs
    ):
        if not self.retain_dropout:
            model.eval()

        net_output = model(**sample['net_input'])
        return net_output


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        if all(hasattr(m, 'decoder') and isinstance(m.decoder, NccIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}

    @torch.no_grad()
    def forward(self, src_tokens, **kwargs):
        """
        Run the forward pass for a decoder-only model.

        Feeds a batch of tokens through the decoder to predict the next tokens.

        Args:
            src_tokens (LongTensor): tokens on which to condition the decoder,
                of shape `(batch, tgt_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, seq_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        if len(self.models) == 1:
            return self.models[0](src_tokens, **kwargs)
        for model in zip(self.models):
            pass
