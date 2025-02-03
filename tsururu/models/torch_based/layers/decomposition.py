"""Module for decomposition layers."""

try:
    import torch
    import torch.nn as nn
    from torch.nn import Module
except ImportError:
    from abc import ABC
    torch = None
    nn = None
    Module = ABC


class moving_avg(Module):
    """Moving average block to highlight the trend of time series.

    Args:
        kernel_size: size of the kernel.
        stride: stride for the moving average.

    """

    def __init__(self, kernel_size: int, stride: int):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass for computing the moving average.

        Args:
            x: input tensor.

        Returns:
            tensor with the moving average applied.

        """
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x


class series_decomp(Module):
    """Series decomposition block.

    Args:
        kernel_size: size of the kernel for the moving average.

    """

    def __init__(self, kernel_size: int):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Forward pass for decomposing the series into trend and remainder.

        Args:
            x: input tensor.

        Returns:
            tuple of tensors (remainder, trend).

        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean

        return res, moving_mean
