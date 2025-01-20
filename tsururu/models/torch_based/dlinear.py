"""DLinear model for time series forecasting."""

from .layers.decomposition import series_decomp

try:
    import torch
    import torch.nn as nn
    from torch.nn import Module
except ImportError:
    from abc import ABC
    torch = None
    nn = None
    Module = ABC


class DLinear_NN(Module):
    """DLInear model from the paper https://arxiv.org/pdf/2205.13504.pdf.

    Args:
        - seq_len: int, the length of the input sequence.
        - pred_len: int, the length of the output sequence.
        - moving_avg: int, the size of the moving average window.
        - individual: bool, whether shared model among different variates.
            true may be better for time series with different trend and seasonal patterns.
        - enc_in: int, the number of input time series.
            Needs for individual=True.

    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        moving_avg: int = 25,
        individual: bool = False,
        enc_in: int = 1,
    ):
        super(DLinear_NN, self).__init__()
        # Params from model_params
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decomposition
        self.decompsition = series_decomp(moving_avg)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )

        self.trainer_type = "DLTrainer"

    def encoder(self, x: "torch.Tensor") -> "torch.Tensor":
        """Encode the input sequence by decomposing it into seasonal and trend components.

        Args:
            x: input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            Encoded tensor of shape (batch_size, pred_len, num_features).

        """
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype
            ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output

        return x.permute(0, 2, 1)

    def forecast(self, x_enc: "torch.Tensor") -> "torch.Tensor":
        """Forecast the output sequence.

        Args:
            x_enc: input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            Forecasted tensor of shape (batch_size, pred_len, num_features).

        """
        return self.encoder(x_enc)

    def forward(self, x_enc: "torch.Tensor") -> "torch.Tensor":
        """Forward pass of the model.

        Args:
            x_enc: input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            Output tensor of shape (batch_size, pred_len, num_features).

        """
        dec_out = self.forecast(x_enc)

        return dec_out[:, -self.pred_len :, :]
