# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from ncc import registry
from ncc.optim.ncc_optimizer import NccOptimizer
from ncc.optim.fp16_optimizer import FP16Optimizer, MemoryEfficientFP16Optimizer
from ncc.optim.bmuf import NccBMUF  # noqa


__all__ = [
    'NccOptimizer',
    'FP16Optimizer',
    'MemoryEfficientFP16Optimizer',
]


build_optimizer, register_optimizer, OPTIMIZER_REGISTRY = registry.setup_registry(
    'optimizer',
    base_class=NccOptimizer,
    default='nag',
)


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('ncc.optim.' + module)
