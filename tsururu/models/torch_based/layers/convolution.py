"""Module for convolution layers."""

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class Inception_Block_V1(nn.Module):
    """Inception Block Version 1.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_kernels: number of convolutional kernels.
        init_weight: whether to initialize weights.

    """

    def __init__(
        self, in_channels: int, out_channels: int, num_kernels: int = 6, init_weight: bool = True
    ):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []

        # Create a list of convolutional layers with varying kernel sizes
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)

        # Initialize weights if specified
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights of the convolutional layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Inception block.

        Args:
            x: input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            output tensor after applying Inception block.

        """
        # Apply each convolutional kernel to the input and collect results
        res_list = [kernel(x) for kernel in self.kernels]

        # Stack the results along a new dimension and take the mean along that dimension
        res = torch.stack(res_list, dim=-1).mean(-1)

        return res
