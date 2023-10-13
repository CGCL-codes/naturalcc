# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import importlib
import os

from ncc import registry
from ncc.optim.bmuf import NccBMUF  # noqa
from ncc.optim.ncc_optimizer import (  # noqa
    NccOptimizer,
    LegacyNccOptimizer,
)
from ncc.optim.amp_optimizer import AMPOptimizer
from ncc.optim.fp16_optimizer import FP16Optimizer, MemoryEfficientFP16Optimizer
from ncc.optim.shard import shard_
from omegaconf import DictConfig

__all__ = [
    "AMPOptimizer",
    "FairseqOptimizer",
    "FP16Optimizer",
    "MemoryEfficientFP16Optimizer",
    "shard_",
]

(
    _build_optimizer,
    register_optimizer,
    OPTIMIZER_REGISTRY,
    OPTIMIZER_DATACLASS_REGISTRY,
) = registry.setup_registry("--optimizer", base_class=NccOptimizer, required=True)


def build_optimizer(cfg: DictConfig, params, *extra_args, **extra_kwargs):
    if all(isinstance(p, dict) for p in params):
        params = [t for p in params for t in p.values()]
    params = list(filter(lambda p: p.requires_grad, params))
    return _build_optimizer(cfg, params, *extra_args, **extra_kwargs)


# automatically import any Python files in the optim/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("ncc.optim." + file_name)

# # -*- coding: utf-8 -*-

# import importlib
# import os

# from ncc.optim.ncc_optimizer import NccOptimizer
# from ncc.optim.bmuf import NccBMUF
# from ncc.optim.amp_optimizer import (
#     AMPOptimizer,
# )
# from ncc.optim.fp16_optimizer import (
#     FP16Optimizer,
#     MemoryEfficientFP16Optimizer,
# )

# OPTIMIZER_REGISTRY = {}
# OPTIMIZER_CLASS_NAMES = set()


# def setup_optimizer(args, params, **kwargs):
#     return OPTIMIZER_REGISTRY[args['optimizer']].setup_optimizer(args, params, **kwargs)


# def register_optimizer(name):
#     def register_optimizer_cls(cls):
#         if name in OPTIMIZER_REGISTRY:
#             raise ValueError('Cannot register duplicate optimizer ({})'.format(name))
#         if not issubclass(cls, NccOptimizer):
#             raise ValueError('Task ({}: {}) must extend NccOptimizer'.format(name, cls.__name__))
#         if cls.__name__ in OPTIMIZER_CLASS_NAMES:
#             raise ValueError('Cannot register optimizer with duplicate class name ({})'.format(cls.__name__))
#         OPTIMIZER_REGISTRY[name] = cls
#         OPTIMIZER_CLASS_NAMES.add(cls.__name__)
#         return cls

#     return register_optimizer_cls


# def get_optimizer(name):
#     return OPTIMIZER_REGISTRY[name]


# optimizers_dir = os.path.dirname(__file__)
# for file in os.listdir(optimizers_dir):
#     path = os.path.join(optimizers_dir, file)
#     if (
#         not file.startswith('_')
#         and not file.startswith('.')
#         and (file.endswith('.py') or os.path.isdir(path))
#         and file != 'lr_schedulers'
#     ):
#         optimizer_name = file[:file.find('.py')] if file.endswith('.py') else file
#         importlib.import_module('ncc.optim.' + optimizer_name)
