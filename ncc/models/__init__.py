# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import importlib
import os
import time

from .distributed_ncc_model import DistributedNccModel
from .ncc_model import (
    BaseNccModel,
    NccEncoderModel,
    NccEncoderDecoderModel,
    NccLanguageModel,
    NccMultiModel,
    NccMoCoModel,
)

MODEL_REGISTRY = {}

__all__ = [
    'datetime', 'time',
    'BaseNccModel',
    # 'CompositeEncoder',
    'DistributedNccModel',
    # 'NccDecoder',
    'NccEncoderDecoderModel',
    'NccEncoderModel',
    # 'NccIncrementalDecoder',
    'NccLanguageModel',
    'NccMultiModel',
]


def build_model(args, config, task):
    return MODEL_REGISTRY[args['model']['arch']].build_model(args, config, task)


def register_model(name):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(NccEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseNccModel` interface.
        Typically you will extend :class:`NccEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`NccLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        if not issubclass(cls, BaseNccModel):
            raise ValueError('Model ({}: {}) must extend BaseNccModel'.format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('ncc.models.' + model_name)
