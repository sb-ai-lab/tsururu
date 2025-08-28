"""Module for creating custom metrics for neural networks."""

from tsururu.utils.optional_imports import OptionalImport

torch = OptionalImport("torch")
Module = OptionalImport("torch.nn.Module")
MSELoss = OptionalImport("torch.nn.MSELoss")


class NegativeMSEMetric(Module):
    """Custom metric that returns the negative of the Mean Squared Error (MSE)."""

    def __init__(self):
        super(NegativeMSEMetric, self).__init__()
        self.mse_loss = MSELoss()

    def forward(self, input: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
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
