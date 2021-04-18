# -*- coding: utf-8 -*-

import importlib
import os

from ncc.optimizers.ncc_optimizer import NccOptimizer
from ncc.optimizers.bmuf import NccBMUF
from ncc.optimizers.fp16_optimizer import (
    FP16Optimizer,
    MemoryEfficientFP16Optimizer,
)

OPTIMIZER_REGISTRY = {}
OPTIMIZER_CLASS_NAMES = set()


def setup_optimizer(args, params, **kwargs):
    return OPTIMIZER_REGISTRY[args['optimizer']].setup_optimizer(args, params, **kwargs)


def register_optimizer(name):
    def register_optimizer_cls(cls):
        if name in OPTIMIZER_REGISTRY:
            raise ValueError('Cannot register duplicate optimizer ({})'.format(name))
        if not issubclass(cls, NccOptimizer):
            raise ValueError('Task ({}: {}) must extend NccOptimizer'.format(name, cls.__name__))
        if cls.__name__ in OPTIMIZER_CLASS_NAMES:
            raise ValueError('Cannot register optimizer with duplicate class name ({})'.format(cls.__name__))
        OPTIMIZER_REGISTRY[name] = cls
        OPTIMIZER_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_optimizer_cls


def get_optimizer(name):
    return OPTIMIZER_REGISTRY[name]


optimizers_dir = os.path.dirname(__file__)
for file in os.listdir(optimizers_dir):
    path = os.path.join(optimizers_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
        and file != 'lr_schedulers'
    ):
        optimizer_name = file[:file.find('.py')] if file.endswith('.py') else file
        importlib.import_module('ncc.optimizers.' + optimizer_name)
