from torch import nn


class NegativeMSEMetric(nn.Module):
    def __init__(self):
        super(NegativeMSEMetric, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        loss = self.mse_loss(input, target)
        negative_loss = -1 * loss
        return negative_loss
