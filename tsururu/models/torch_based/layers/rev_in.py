"""Reversible Instance Normalization (RevIN) module."""

try:
    import torch
    import torch.nn as nn
    from torch.nn import Module
except ImportError:
    from abc import ABC
    torch = None
    nn = None
    Module = ABC


class RevIN(Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """Reversible Instance Normalization (RevIN) module.

        Args:
            num_features: number of features or channels.
            eps: a value added for numerical stability. Default is 1e-5.
            affine: if True, RevIN has learnable affine parameters. Default is True.
            subtract_last: if True, subtracts the last value for normalization. Default is False.

        Notes:
            code from https://github.com/ts-kim/RevIN, with minor modifications

        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """Forward pass for RevIN.

        Args:
            x: input tensor of shape (batch_size, seq_len, num_features).
            mode: mode of operation, either "norm" for normalization or "denorm" for denormalization.

        Returns:
            normalized or denormalized tensor.

        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        """Initialize learnable affine parameters."""
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: "torch.Tensor"):
        """Compute mean and standard deviation for normalization.

        Args:
            x: input tensor of shape (batch_size, seq_len, num_features).

        """
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: "torch.Tensor") -> "torch.Tensor":
        """Normalize the input tensor.

        Args:
            x: input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            normalized tensor.

        """
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x: "torch.Tensor") -> "torch.Tensor":
        """Denormalize the input tensor.

        Args:
            x: input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            denormalized tensor.

        """
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean

        return x
