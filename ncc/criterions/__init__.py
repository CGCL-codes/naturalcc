# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from ncc import registry
from ncc.criterions.ncc_criterion import NccCriterion, LegacyNccCriterion


build_criterion, register_criterion, CRITERION_REGISTRY = registry.setup_registry(
    'criterion',
    base_class=NccCriterion,
    default='cross_entropy',
)

# # automatically import any Python files in the criterions/ directory
# for file in os.listdir(os.path.dirname(__file__)):
#     if file.endswith('.py') and not file.startswith('_'):
#         module = file[:file.find('.py')]
#         importlib.import_module('ncc.criterions.' + module)

criterions_dir = os.path.dirname(__file__)
for file in os.listdir(criterions_dir):
    path = os.path.join(criterions_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        criterion_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('ncc.criterions.' + criterion_name)
