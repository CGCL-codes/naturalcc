# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from ncc import registry
from ncc.optimizers.lr_schedulers.ncc_lr_scheduler import NccLRScheduler

build_lr_scheduler, register_lr_scheduler, LR_SCHEDULER_REGISTRY = registry.setup_registry(
    'lr_scheduler',
    base_class=NccLRScheduler,
    default='fixed',
)

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('ncc.optimizers.lr_schedulers.' + module)
