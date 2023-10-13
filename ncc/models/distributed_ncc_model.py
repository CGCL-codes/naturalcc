# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import signal
import threading

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from ncc.distributed import (
    DistributedTimeoutWrapper,
    LegacyDistributedDataParallel,
    ModuleProxyWrapper,
    TPUDistributedDataParallel,
)

logger = logging.getLogger(__name__)


_SLOWMO_DDP_DISABLED = False
try:
    from fairscale.experimental.nn.data_parallel import (
        SlowMoBaseAlgorithm,
        SlowMoDistributedDataParallel,
    )
except ImportError:
    _SLOWMO_DDP_DISABLED = True


def DistributedNccModel(args, model, process_group, device):
    """
    Wrap a *model* to support distributed data parallel training.

    This is similar to the built-in DistributedDataParallel, but allows
    additional configuration of the DistributedDataParallel class to
    use, and also provides easier access to the wrapped model by
    forwarding requests for missing attributes to the wrapped model.

    Args:
        args (argparse.Namespace): fairseq args
        model (BaseFairseqModel): model to wrap
        process_group: the c10d process group to be used for distributed data
            parallel all-reduction.
        device: device to move model to
    """
    assert isinstance(model, nn.Module)
    if args.tpu:
        wrapped_model = TPUDistributedDataParallel(
            module=model.to(device),
            process_group=process_group,
        )
        # forward missing getattr and state_dict/load_state_dict to orig model
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend in {"c10d", "pytorch_ddp"}:
        wrapped_model = DistributedDataParallel(
            module=model.to(device),
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=args.broadcast_buffers,
            bucket_cap_mb=args.bucket_cap_mb,
            process_group=process_group,
            find_unused_parameters=args.find_unused_parameters,
            gradient_as_bucket_view=args.gradient_as_bucket_view,
        )
        if args.ddp_comm_hook == "fp16":
            logger.info("enable fp16 communication hook in DDP")
            try:
                from torch.distributed.algorithms.ddp_comm_hooks import (
                    DDPCommHookType,
                    register_ddp_comm_hook,
                )
            except:
                logger.error(
                    "Could not import from torch.distributed.algorithms.ddp_comm_hooks; you may need to update your pytorch version"
                )
                raise

            register_ddp_comm_hook(DDPCommHookType.FP16_COMPRESS, wrapped_model)
        # forward missing getattr and state_dict/load_state_dict to orig model
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend in {"no_c10d", "legacy_ddp"}:
        wrapped_model = LegacyDistributedDataParallel(
            module=model.to(device),
            buffer_size=2**28,
            process_group=process_group,
        )
        # forward missing getattr and state_dict/load_state_dict to orig model
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend == "slowmo":
        if _SLOWMO_DDP_DISABLED:
            raise ImportError(
                "Cannot find SlowMoDistributedDataParallel. "
                "Please install fairscale with: pip install fairscale"
            )

        # The values of slowmo_momentum below were obtained by tuning on the
        # En-De 16 dataset by training the transformer_wmt_en_de_large model
        if args.slowmo_momentum is None:
            if args.distributed_world_size <= 16:
                args.slowmo_momentum = 0.0
            elif args.distributed_world_size <= 32:
                args.slowmo_momentum = 0.2
            elif args.distributed_world_size <= 64:
                args.slowmo_momentum = 0.5
            else:
                args.slowmo_momentum = 0.6
        slowmo_base_algorithm = SlowMoBaseAlgorithm[args.slowmo_base_algorithm.upper()]

        wrapped_model = SlowMoDistributedDataParallel(
            module=model.to(device),
            broadcast_buffers=args.broadcast_buffers,
            nprocs_per_node=args.nprocs_per_node,
            slowmo_momentum=args.slowmo_momentum,
            slowmo_base_algorithm=slowmo_base_algorithm,
            localsgd_frequency=args.localsgd_frequency,
        )
        # forward missing getattr and state_dict/load_state_dict to orig model
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend == "fully_sharded":
        try:
            from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
        except ImportError:
            raise ImportError(
                "Cannot find FullyShardedDataParallel. "
                "Please install fairscale with: pip install fairscale"
            )
        assert isinstance(model, FSDP), "expected model to already be wrapped in FSDP"
        wrapped_model = model
        if args.memory_efficient_fp16:
            wrapped_model = wrapped_model.half()
        if not args.cpu_offload:
            wrapped_model = wrapped_model.to(device=device)
    else:
        raise ValueError("Unknown --ddp-backend: " + args.ddp_backend)

    # kill hung distributed jobs after a timeout
    if getattr(args, "heartbeat_timeout", -1) > 0:
        wrapped_model = DistributedTimeoutWrapper(
            wrapped_model, timeout=getattr(args, "heartbeat_timeout", -1)
        )

    return wrapped_model


# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# import inspect

# import torch.nn as nn

# from ncc.models.legacy_distributed_data_parallel import LegacyDistributedDataParallel


# def DistributedNccModel(args, model):
#     """
#     Wrap a *model* to support distributed data parallel training.

#     This is similar to the built-in DistributedDataParallel, but allows
#     additional configuration of the DistributedDataParallel class to
#     use, and also provides easier access to the wrapped model by
#     forwarding requests for missing attributes to the wrapped model.

#     Args:
#         args (argparse.Namespace): fairseq args
#         model (BaseNccModel): model to wrap
#     """
#     # determine which DDP class to extend
#     assert isinstance(model, nn.Module)
#     if args['distributed_training']['ddp_backend'] == 'c10d':
#         ddp_class = nn.parallel.DistributedDataParallel
#         init_kwargs = dict(
#             module=model,
#             device_ids=[args['distributed_training']['device_id']],
#             output_device=args['distributed_training']['device_id'],
#             broadcast_buffers=args['distributed_training']['broadcast_buffers'],
#             bucket_cap_mb=args['distributed_training']['bucket_cap_mb'],
#         )
#         # Maintain backward compatibility
#         if 'check_reduction' in inspect.getargspec(ddp_class)[0]:
#             init_kwargs['check_reduction'] = True
#         if 'find_unused_parameters' in inspect.getargspec(ddp_class)[0]:
#             init_kwargs['find_unused_parameters'] = args['distributed_training']['find_unused_parameters']
#     elif args['distributed_training']['ddp_backend'] == 'no_c10d':
#         ddp_class = LegacyDistributedDataParallel
#         init_kwargs = dict(
#             module=model,
#             world_size=args['distributed_training']['distributed_world_size'],
#             buffer_size=2 ** 28,
#         )
#     else:
#         raise ValueError('Unknown --ddp-backend: ' + args['distributed_training']['ddp_backend'])

#     class _DistributedNccModel(ddp_class):
#         """Extend DistributedDataParallel to check for missing
#         attributes in the wrapped module."""

#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)

#         def __getattr__(self, name):
#             wrapped_module = super().__getattr__('module')
#             if hasattr(wrapped_module, name):
#                 return getattr(wrapped_module, name)
#             return super().__getattr__(name)

#     return _DistributedNccModel(**init_kwargs)
