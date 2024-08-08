"""Module for creating custom metrics for neural networks."""

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class NegativeMSEMetric(nn.Module):
    """Custom metric that returns the negative of the Mean Squared Error (MSE)."""

    def __init__(self):
        super(NegativeMSEMetric, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the negative mean squared error between input and target.

        Args:
            input: predicted tensor.
            target: ground truth tensor.

        Returns:
            the negative mean squared error.

        """
        loss = self.mse_loss(input, target)
        negative_loss = -1 * loss
        return negative_loss
