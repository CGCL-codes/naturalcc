# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from ncc.models import register_model
from ncc.models.ncc_model import NccRetrievalModel
from ncc.modules.retrieval.nbow_encoder import NBOWEncoder

import logging

logger = logging.getLogger(__name__)


@register_model('nbow')
class NBOW(NccRetrievalModel):
    def __init__(self, args, src_encoder, tgt_encoder):
        super().__init__(src_encoder, tgt_encoder)
        self.args = args

    @classmethod
    def build_model(cls, args, config, task):
        """dictionary, embed_dim: Int_t, dropout: Float_t = 0.1, pooling: String_t = None"""
        src_encoder = NBOWEncoder(
            dictionary=task.source_dictionary, embed_dim=args['model']['code_embed_dim'],
            dropout=args['model']['dropout'], pooling=args['model']['pooling'],
        )
        tgt_encoder = NBOWEncoder(
            dictionary=task.target_dictionary, embed_dim=args['model']['query_embed_dim'],
            dropout=args['model']['dropout'], pooling=args['model']['pooling'],
        )
        return cls(args, src_encoder, tgt_encoder)
