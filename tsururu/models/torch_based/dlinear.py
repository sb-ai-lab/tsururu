import torch
import torch.nn as nn


class DLinear_NN(nn.Module):
    """DLInear model from the paper https://arxiv.org/pdf/2205.13504.pdf.

    Args:
        model_params: dict with parameters for the model:
            - seq_len: int, the length of the input sequence.
            - pred_len: int, the length of the output sequence.
            - moving_avg: int, the size of the moving average window.
            - individual: bool, whether shared model among different variates.
                true may be better for time series with different trend and seasonal patterns.
            - enc_in: int, the number of input time series.
                Needs for individual=True.

    """

    def __init__(self, seq_len, pred_len, moving_avg=7, individual=False, enc_in=1):
        super(DLinear_NN, self).__init__()
        # Params from model_params
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decomposition
        self.decompsition = self.series_decomp(moving_avg)
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

    def encoder(self, x):
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

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]

    @staticmethod
    def my_Layernorm(channels):
        layernorm = nn.LayerNorm(channels)

        def forward(x):
            x_hat = layernorm(x)
            bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
            return x_hat - bias

        return forward

    @staticmethod
    def moving_avg(kernel_size, stride):
        avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

        def forward(x):
            front = x[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=1)
            x = avg(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
            return x

        return forward

    @staticmethod
    def series_decomp(kernel_size):
        moving_avg = DLinear_NN.moving_avg(kernel_size, stride=1)

        def forward(x):
            moving_mean = moving_avg(x)
            res = x - moving_mean
            return res, moving_mean

        return forward
