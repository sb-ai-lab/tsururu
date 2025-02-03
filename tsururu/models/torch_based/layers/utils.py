"""Module containing utility functions and classes."""

from typing import Callable, Union

try:
    import torch
    import torch.nn as nn
    from torch.nn import Module
except ImportError:
    from abc import ABC
    torch = None
    nn = None
    Module = ABC


def get_activation_fn(activation: Union[str, Callable[[], Module]]) -> Module:
    """Get the activation function based on the provided name or callable.

    Args:
        activation: activation function name ('relu', 'gelu') or a callable.

    Returns:
        activation function instance.

    Raises:
        ValueError: if the activation function is not available.

    """
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


class Transpose(Module):
    """Module for transposing tensors with optional contiguous memory layout.

    Args:
        dims: dimensions to transpose.
        contiguous: whether to make the tensor contiguous in memory after transposing.

    """

    def __init__(self, *dims: int, contiguous: bool = False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass for transposing the tensor.

        Args:
            x: input tensor.

        Returns:
            transposed tensor.

        """
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)
