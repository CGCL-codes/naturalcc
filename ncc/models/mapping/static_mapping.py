# -*- coding: utf-8 -*-


from collections import Counter

import numpy as np

from ncc.models import register_model
from ncc.models.ncc_model import (
    NccEncoder,
    NccEncoderModel,
)


class StaticMappingEncoder(NccEncoder):
    def __init__(self, devices):
        """
        optimal_device: 0 for cpu; 1 for gpu
        """
        super(StaticMappingEncoder, self).__init__(dictionary=None)
        most_common_device = Counter(devices).most_common(1)[0][0]
        if isinstance(most_common_device, str):
            self.optimal_device = 1 if str.lower(most_common_device) == 'gpu' else 0
        elif isinstance(most_common_device, np.int_):
            self.optimal_device = int(most_common_device)

    def forward(self, src_tokens, **kwargs):
        return src_tokens.new(
            src_tokens.size(0)
        ).int().fill_(self.optimal_device)


@register_model('static_mapping')
class StaticMapping(NccEncoderModel):
    def __init__(self, encoder):
        super(StaticMapping, self).__init__(encoder)

    @classmethod
    def build_model(cls, devices):
        encoder = StaticMappingEncoder(devices)
        return cls(encoder)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)
