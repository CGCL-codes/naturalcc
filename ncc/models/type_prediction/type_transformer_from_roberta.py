from ncc.modules.code2vec.transformer_encoder import TransformerEncoder
from ncc.modules.seq2seq.transformer_decoder import TransformerDecoder
from ncc.models import register_model
from ncc.models.type_prediction.typetransformer import TypeTransformer
DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


@register_model('type_prediction_transformer_from_roberta')
class TypePredictionTransformerFromRobertaModel(TypeTransformer):
    """
        Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
        <https://arxiv.org/abs/1706.03762>`_.

        Args:
            encoder (TransformerEncoder): the encoder
            decoder (TransformerDecoder): the decoder

        The Transformer model provides the following named architectures and
        command-line arguments:

        .. argparse::
            :ref: fairseq.models.transformer_parser
            :prog:
    """

    def __init__(self, args, decoder):
        super().__init__(args, decoder)
        # self.args = args
        # self.supports_align_args = True

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, "")

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code.

        Args:
            state_dict (dict): state dictionary to upgrade, in place, state_dict() of Roberta model
            name (str): the state dict key corresponding to the current module
        """
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += "."

            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, "upgrade_state_dict_named"):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, "upgrade_state_dict"):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)
