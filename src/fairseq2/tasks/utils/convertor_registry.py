# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import AbstractSet, Callable, Dict, Generic, Protocol, TypeVar

ConvertorConfigT = TypeVar("ConvertorConfigT", covariant=True)

REGISTER_ARCH = {}


class ConvertorConfigFactory(Protocol[ConvertorConfigT]):
    """Constructs instances of ``ConvertorConfigT``."""

    def __call__(self) -> ConvertorConfigT:
        ...


class ConvertorRegistry(Generic[ConvertorConfigT]):
    """Represents a registry of task and it's converter."""

    convertor_type: str
    configs: Dict[str, ConvertorConfigFactory[ConvertorConfigT]]

    def __init__(self, convertor_type: str) -> None:
        """
        :param convertor_type:
            The type of the model for which architectures will be registered.
        """
        self.convertor_type = convertor_type
        
        self.configs = {}

    def register(
        self, task_name: str, config_factory: ConvertorConfigFactory[ConvertorConfigT]
    ) -> None:
        """Register a new architecture.

        :param task_name:
            The name of the architecture.
        :param config_factory:
            The callable responsible for constructing the model configuration of
            the architecture.
        """
        if task_name in self.configs:
            raise ValueError(
                f"The architecture name '{task_name}' is already registered for '{self.convertor_type}'."
            )

        self.configs[task_name] = config_factory

    def get_config(self, task_name: str) -> ConvertorConfigT:
        """Return the model configuration of the specified architecture.

        :param task_name:
            The name of the architecture.
        """
        try:
            return self.configs[task_name]()
        except KeyError:
            raise ValueError(
                f"The registry of '{self.convertor_type}' does not contain an architecture named '{task_name}'."
            )

    def names(self) -> AbstractSet[str]:
        """Return the names of all supported architectures."""
        return self.configs.keys()

    def marker(
        self, task_name: str
    ) -> Callable[[ConvertorConfigFactory[ConvertorConfigT]], ConvertorConfigFactory[ConvertorConfigT]]:
        """Return a decorator that registers the specified architecture with the
        decorated callable as its model configuration factory.

        :param task_name:
            The name of the architecture.
        """

        def register(
            config_factory: ConvertorConfigFactory[ConvertorConfigT],
        ) -> ConvertorConfigFactory[ConvertorConfigT]:
            self.register(task_name, config_factory)

            return config_factory

        return register
