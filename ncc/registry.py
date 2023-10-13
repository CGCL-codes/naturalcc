# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace

from typing import Union
from ncc.dataclass import NccDataclass
from ncc.dataclass.utils import merge_with_parent
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

REGISTRIES = {}


def setup_registry(registry_name: str, base_class=None, default=None, required=False):
    assert registry_name.startswith("--")
    registry_name = registry_name[2:].replace("-", "_")

    REGISTRY = {}
    REGISTRY_CLASS_NAMES = set()
    DATACLASS_REGISTRY = {}

    # maintain a registry of all registries
    if registry_name in REGISTRIES:
        return  # registry already exists
    REGISTRIES[registry_name] = {
        "registry": REGISTRY,
        "default": default,
        "dataclass_registry": DATACLASS_REGISTRY,
    }

    def build_x(cfg: Union[DictConfig, str, Namespace], *extra_args, **extra_kwargs):
        if isinstance(cfg, DictConfig):
            choice = cfg._name

            if choice and choice in DATACLASS_REGISTRY:
                from_checkpoint = extra_kwargs.get("from_checkpoint", False)
                dc = DATACLASS_REGISTRY[choice]
                cfg = merge_with_parent(dc(), cfg, remove_missing=from_checkpoint)
        elif isinstance(cfg, str):
            choice = cfg
            if choice in DATACLASS_REGISTRY:
                cfg = DATACLASS_REGISTRY[choice]()
        else:
            choice = getattr(cfg, registry_name, None)
            if choice in DATACLASS_REGISTRY:
                cfg = DATACLASS_REGISTRY[choice].from_namespace(cfg)

        if choice is None:
            if required:
                raise ValueError("{} is required!".format(registry_name))
            return None

        cls = REGISTRY[choice]
        if hasattr(cls, "build_" + registry_name):
            builder = getattr(cls, "build_" + registry_name)
        else:
            builder = cls

        if "from_checkpoint" in extra_kwargs:
            del extra_kwargs["from_checkpoint"]

        return builder(cfg, *extra_args, **extra_kwargs)

    def register_x(name, dataclass=None):
        def register_x_cls(cls):
            if name in REGISTRY:
                raise ValueError(
                    "Cannot register duplicate {} ({})".format(registry_name, name)
                )
            if cls.__name__ in REGISTRY_CLASS_NAMES:
                raise ValueError(
                    "Cannot register {} with duplicate class name ({})".format(
                        registry_name, cls.__name__
                    )
                )
            if base_class is not None and not issubclass(cls, base_class):
                raise ValueError(
                    "{} must extend {}".format(cls.__name__, base_class.__name__)
                )

            if dataclass is not None and not issubclass(dataclass, NccDataclass):
                raise ValueError(
                    "Dataclass {} must extend NccDataclass".format(dataclass)
                )

            cls.__dataclass = dataclass
            if cls.__dataclass is not None:
                DATACLASS_REGISTRY[name] = cls.__dataclass

                cs = ConfigStore.instance()
                node = dataclass()
                node._name = name
                cs.store(name=name, group=registry_name, node=node, provider="ncc")

            REGISTRY[name] = cls

            return cls

        return register_x_cls

    return build_x, register_x, REGISTRY, DATACLASS_REGISTRY

# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# REGISTRIES = {}


# def setup_registry(
#     registry_name: str,
#     base_class=None,
#     default=None,
# ):
#     registry_name = registry_name.replace('-', '_')

#     REGISTRY = {}
#     REGISTRY_CLASS_NAMES = set()

#     # maintain a registry of all registries
#     if registry_name in REGISTRIES:
#         return  # registry already exists
#     REGISTRIES[registry_name] = {
#         'registry': REGISTRY,
#         'default': default,
#     }

#     def build_x(args, *extra_args, **extra_kwargs):
#         choice = args[registry_name] if registry_name in args else None
#         if choice is None:
#             return None
#         cls = REGISTRY[choice]
#         if hasattr(cls, 'build_' + registry_name):
#             builder = getattr(cls, 'build_' + registry_name)
#         else:
#             builder = cls
#         set_defaults(args, cls)
#         return builder(args, *extra_args, **extra_kwargs)

#     def register_x(name):

#         def register_x_cls(cls):
#             if name in REGISTRY:
#                 raise ValueError('Cannot register duplicate {} ({})'.format(registry_name, name))
#             if cls.__name__ in REGISTRY_CLASS_NAMES:
#                 raise ValueError(
#                     'Cannot register {} with duplicate class name ({})'.format(
#                         registry_name, cls.__name__,
#                     )
#                 )
#             if base_class is not None and not issubclass(cls, base_class):
#                 raise ValueError('{} must extend {}'.format(cls.__name__, base_class.__name__))
#             REGISTRY[name] = cls
#             REGISTRY_CLASS_NAMES.add(cls.__name__)
#             return cls

#         return register_x_cls

#     return build_x, register_x, REGISTRY


# def set_defaults(*args, **kwargs):
#     """Helper to set default arguments based on *add_args*."""
#     pass
