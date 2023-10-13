# # -*- coding: utf-8 -*-
#
# import sys
#
# # sys.path.append('.')
#

# from ncc.modules.layer_norm import LayerNorm
# from ncc.modules.transformer_sentence_encoder import TransformerSentenceEncoder
from .multihead_attention import MultiheadAttention
from ncc.modules.positional_embedding import PositionalEmbedding
# from ncc.modules.transformer_sentence_encoder_layer import TransformerSentenceEncoderLayer
from .ncc_dropout import NccDropout
from .adaptive_softmax import AdaptiveSoftmax
from .base_layer import BaseLayer
from .layer_drop import LayerDropModuleList
from .layer_norm import LayerNorm
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from .transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer
from .adaptive_input import AdaptiveInput
from .character_token_embedder import CharacterTokenEmbedder
from .transformer_sentence_encoder_layer import TransformerSentenceEncoderLayer
from .gelu import gelu, gelu_accurate
    
__all__ = [
    'LayerNorm',
    # 'TransformerSentenceEncoder',
    'MultiheadAttention',
    # 'TransformerSentenceEncoderLayer',
    'PositionalEmbedding',
    "NccDropout",
    "AdaptiveSoftmax",
    "BaseLayer",
    "LayerDropModuleList",
    "SinusoidalPositionalEmbedding",
    "TransformerDecoderLayer",
    "TransformerEncoderLayer",
    "AdaptiveInput",
    "CharacterTokenEmbedder",
    "TransformerSentenceEncoderLayer",
    "gelu",
    "gelu_accurate",
]