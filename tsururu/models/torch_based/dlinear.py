"""DLinear model for time series forecasting."""

from tsururu.models.torch_based.dl_base import DLEstimator
from tsururu.models.torch_based.layers.decomposition import series_decomp
from tsururu.models.torch_based.utils import slice_features, slice_features_4d
from tsururu.utils.optional_imports import OptionalImport

torch = OptionalImport("torch")
nn = OptionalImport("torch.nn")
Module = OptionalImport("torch.nn.Module")
einops = OptionalImport("einops")


class DLinear_NN(DLEstimator):
    """DLInear model from the paper https://arxiv.org/abs/2205.13504.

    Args:
        - features_groups: dict, dictionary with the number of features for each group.
        - pred_len: int, the length of the output sequence.
        - seq_len: input sequence length.
        - moving_avg: int, the size of the moving average window.
        - individual: bool, whether shared model among different variates.
            true may be better for time series with different trend and seasonal patterns.
        - channel_independent: channel independence.

    """

    def __init__(
        self,
        features_groups: dict,
        pred_len: int,
        seq_len: int,
        moving_avg: int = 25,
        individual: bool = False,
        channel_independent: bool = False,
    ):
        super().__init__(features_groups, pred_len, seq_len)

        assert not (
            individual and not channel_independent
        ), "individual must be False when channel_independent is False."

        self.decompsition = series_decomp(moving_avg)
        self.individual = individual
        self.channel_independent = channel_independent

        num_channels = sum(self.features_groups_corrected.values())
        num_exog_features = num_channels - self.num_series

        if self.channel_independent:
            if self.individual:
                self.Linear_Seasonal = nn.ModuleList()
                self.Linear_Trend = nn.ModuleList()

                for i in range(self.num_series):
                    self.Linear_Seasonal.append(
                        nn.Linear(self.seq_len * (num_exog_features + 1), self.pred_len)
                    )
                    self.Linear_Trend.append(
                        nn.Linear(self.seq_len * (num_exog_features + 1), self.pred_len)
                    )

                    self.Linear_Seasonal[i].weight = nn.Parameter(
                        (1 / (self.seq_len * (num_exog_features + 1)))
                        * torch.ones([self.pred_len, self.seq_len * (num_exog_features + 1)])
                    )
                    self.Linear_Trend[i].weight = nn.Parameter(
                        (1 / (self.seq_len * (num_exog_features + 1)))
                        * torch.ones([self.pred_len, self.seq_len * (num_exog_features + 1)])
                    )
            else:
                self.Linear_Seasonal = nn.Linear(
                    self.seq_len * (num_exog_features + 1), self.pred_len
                )
                self.Linear_Trend = nn.Linear(
                    self.seq_len * (num_exog_features + 1), self.pred_len
                )

                self.Linear_Seasonal.weight = nn.Parameter(
                    (1 / (self.seq_len * (num_exog_features + 1)))
                    * torch.ones([self.pred_len, self.seq_len * (num_exog_features + 1)])
                )
                self.Linear_Trend.weight = nn.Parameter(
                    (1 / (self.seq_len * (num_exog_features + 1)))
                    * torch.ones([self.pred_len, self.seq_len * (num_exog_features + 1)])
                )
        else:
            self.Linear_Seasonal = nn.Linear(
                self.seq_len * num_channels, self.pred_len * self.num_series
            )
            self.Linear_Trend = nn.Linear(
                self.seq_len * num_channels, self.pred_len * self.num_series
            )

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / (self.seq_len * num_channels))
                * torch.ones([self.pred_len * self.num_series, self.seq_len * num_channels])
            )
            self.Linear_Trend.weight = nn.Parameter(
                (1 / (self.seq_len * num_channels))
                * torch.ones([self.pred_len * self.num_series, self.seq_len * num_channels])
            )

    def encoder(self, x: "torch.Tensor") -> "torch.Tensor":
        """Encode the input sequence by decomposing it into seasonal and trend components.

        Args:
            x: input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            Encoded tensor of shape (batch_size, pred_len, num_series).

        """
        series = slice_features(
            x, ["series"], self.features_groups_corrected
        )  # (batch_size, seq_len, num_series)
        seasonal_init, trend_init = self.decompsition(series)  # (batch_size, seq_len, num_series)

        if self.channel_independent:
            # 4d tensors
            exog_features_4d = slice_features_4d(
                x,
                ["id", "fh", "datetime_features", "series_features", "other_features"],
                self.features_groups_corrected,
                self.num_series,
            )  # (batch_size, seq_len, num_series, num_exog_features)

            seasonal_init, trend_init = (
                seasonal_init.unsqueeze(-1),
                trend_init.unsqueeze(-1),
            )  # (batch_size, seq_len, num_series, 1)
            seasonal_init = torch.concat(
                [seasonal_init, exog_features_4d], dim=-1
            )  # (batch_size, seq_len, num_series, num_exog_features + 1)
            trend_init = torch.concat(
                [trend_init, exog_features_4d], dim=-1
            )  # (batch_size, seq_len, num_series, num_exog_features + 1)

            seasonal_reshaped = einops.rearrange(
                seasonal_init, "b s c f -> b c (f s)"
            )  # (batch_size, num_series, (num_exog_features + 1) * seq_len)
            trend_reshaped = einops.rearrange(
                trend_init, "b s c f -> b c (f s)"
            )  # (batch_size, num_series, (num_exog_features + 1) * seq_len)

            if self.individual:
                seasonal_output = torch.zeros(
                    [seasonal_init.size(0), self.num_series, self.pred_len],
                    dtype=seasonal_init.dtype,
                ).to(
                    seasonal_init.device
                )  # (batch_size, num_series, pred_len)
                trend_output = torch.zeros(
                    [trend_init.size(0), self.num_series, self.pred_len], dtype=trend_init.dtype
                ).to(
                    trend_init.device
                )  # (batch_size, num_series, pred_len)
                for i in range(self.num_series):
                    seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_reshaped[:, i, :])
                    trend_output[:, i, :] = self.Linear_Trend[i](trend_reshaped[:, i, :])
            else:
                seasonal_output = self.Linear_Seasonal(
                    seasonal_reshaped
                )  # (batch_size, num_series, pred_len)
                trend_output = self.Linear_Trend(
                    trend_reshaped
                )  # (batch_size, num_series, pred_len)

        else:
            # 3d tensors
            exog_features = slice_features(
                x,
                ["id", "fh", "datetime_features", "series_features", "other_features"],
                self.features_groups_corrected,
            )  # (batch_size, seq_len, num_exog_features)
            seasonal_init = torch.concat(
                [seasonal_init, exog_features], dim=-1
            )  # (batch_size, seq_len, num_series + num_exog_features)
            trend_init = torch.concat(
                [trend_init, exog_features], dim=-1
            )  # (batch_size, seq_len, num_series + num_exog_features)

            seasonal_reshaped = einops.rearrange(
                seasonal_init, "b s c -> b (c s)"
            )  # (batch_size, (num_series + num_exog_features) * seq_len)
            trend_reshaped = einops.rearrange(
                trend_init, "b s c -> b (c s)"
            )  # (batch_size, (num_series + num_exog_features) * seq_len)

            seasonal_output = self.Linear_Seasonal(
                seasonal_reshaped
            )  # (batch_size, num_series * pred_len)
            trend_output = self.Linear_Trend(trend_reshaped)  # (batch_size, num_series * pred_len)

            seasonal_output = einops.rearrange(
                seasonal_output, "b (c s) -> b c s", s=self.pred_len
            )  # (batch_size, num_series, pred_len)
            trend_output = einops.rearrange(
                trend_output, "b (c s) -> b c s", s=self.pred_len
            )  # (batch_size, num_series, pred_len)

        x = seasonal_output + trend_output  # (batch_size, num_series, pred_len)

        return x.permute(0, 2, 1)  # (batch_size, pred_len, num_series)

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
