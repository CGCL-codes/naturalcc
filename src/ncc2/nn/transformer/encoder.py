# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Iterable, Optional, Protocol, Tuple, final

from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from ncc2.nn.module_list import ModuleList
from ncc2.nn.normalization import LayerNorm
from ncc2.nn.padding import PaddingMask
from ncc2.nn.transformer.attention_mask import AttentionMaskFactory
from ncc2.nn.transformer.encoder_layer import TransformerEncoderLayer
from ncc2.nn.transformer.layer_norm import (
    LayerNormFactory,
    create_standard_layer_norm,
)
from ncc2.nn.transformer.norm_order import TransformerNormOrder
from ncc2.typing import DataType, Device, finaloverride


class TransformerEncoder(Module, ABC):
    """Represents a Transformer encoder."""

    model_dim: int
    layers: ModuleList

    _layer_output_hooks: Dict[int, EncoderLayerOutputHook]

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

        self._layer_output_hooks = OrderedDict()

    @abstractmethod
    def forward(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """
        :param seqs:
            The sequences to encode. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            - The encoder output. *Shape:* Same as ``seqs``.
            - The padding mask of the encoder output. *Shape:* Same as
              ``padding_mask``.
        """

    def register_layer_output_hook(
        self, hook: EncoderLayerOutputHook
    ) -> RemovableHandle:
        """Register a layer output hook on the module.

        The hook will be called every time after a layer in the encoder stack
        has computed an output.

        :param hook:
            The hook to register.

        :returns:
            A handle that can be used to remove the added hook by calling
            ``handle.remove()``.
        """
        handle = RemovableHandle(self._layer_output_hooks)

        self._layer_output_hooks[handle.id] = hook

        return handle

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


class EncoderLayerOutputHook(Protocol):
    """Represents a hook to pass to
    :meth:`~TransformerEncoder.register_layer_output_hook`."""

    def __call__(
        self,
        layer_idx: int,
        layer_output: Tensor,
        layer_padding_mask: Optional[PaddingMask],
        num_layers: int,
    ) -> bool:
        """
        :param layer_idx:
            The index of the layer in the encoder stack.
        :param layer_output:
            The encoded output of the layer.
        :param layer_padding_mask:
            The padding mask of ``layer_output``.
        :param num_layers:
            The number of layers in the encoder stack.

        :returns:
            ``True`` if the encoder should continue executing the remaining
            layers in the stack; ``False`` if the encoder should treat this
            layer as the final layer in the stack.
        """


@final
class StandardTransformerEncoder(TransformerEncoder):
    """Represents a Transformer encoder as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    self_attn_mask_factory: Optional[AttentionMaskFactory]
    layer_norm: Optional[LayerNorm]
    norm_order: TransformerNormOrder

    def __init__(
        self,
        layers: Iterable[TransformerEncoderLayer],
        *,
        self_attn_mask_factory: Optional[AttentionMaskFactory] = None,
        layer_drop_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param layers:
            The encoder layers.
        :param self_attn_mask_factory:
            The self attention mask factory.
        :param layer_drop_p:
            If greater than zero, applies LayerDrop to the encoder layers as
            described in :cite:t:`https://doi.org/10.48550/arxiv.1909.11556`.
        :param norm_order:
            The Layer Normalization order to use.
        :param layer_norm_factory:
            The factory to use to construct the Layer Normalization module.
        """
        layer_list = ModuleList(layers, drop_p=layer_drop_p)
        if not layer_list:
            raise ValueError("`layers` must be non-empty.")

        model_dim = layer_list[0].model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.self_attn_mask_factory = self_attn_mask_factory

        self.layers = layer_list

        if norm_order != TransformerNormOrder.POST:
            self.layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        self.norm_order = norm_order

    @finaloverride
    def forward(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if self._layer_output_hooks and self.layers.drop_p > 0.0:
            raise RuntimeError(
                "The layer output hooks cannot be run when LayerDrop is enabled."
            )

        num_layers = len(self.layers)

        if self.self_attn_mask_factory is None:
            self_attn_mask = None
        else:
            self_attn_mask = self.self_attn_mask_factory(
                seqs, keys=seqs, training=self.training
            )

        for layer_idx, layer in enumerate(self.layers.drop_iter()):
            seqs, padding_mask = layer(seqs, padding_mask, self_attn_mask)

            for hook in self._layer_output_hooks.values():
                if not hook(layer_idx, seqs, padding_mask, num_layers):
                    break

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        return seqs, padding_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.self_attn_mask_factory is not None:
            self_attn_mask_factory = getattr(
                self.self_attn_mask_factory, "__name__", self.self_attn_mask_factory
            )

            s = f"{s}, self_attn_mask_factory={self_attn_mask_factory}"

        return f"{s}, norm_order={self.norm_order}"