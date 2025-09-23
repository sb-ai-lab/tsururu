"""TimesNet neural network for time series forecasting."""

from typing import Tuple

import numpy as np

from tsururu.models.torch_based.dl_base import DLEstimator
from tsururu.models.torch_based.layers.convolution import Inception_Block_V1
from tsururu.models.torch_based.layers.embedding import Embedding
from tsururu.models.torch_based.utils import slice_features, slice_features_4d
from tsururu.utils.optional_imports import OptionalImport

torch = OptionalImport("torch")
nn = OptionalImport("torch.nn")
F = OptionalImport("torch.nn.functional")
rearrange = OptionalImport("einops.rearrange")
Module = OptionalImport("torch.nn.Module")


def FFT_for_Period(x: "torch.Tensor", k: int = 2) -> Tuple[np.ndarray, "torch.Tensor"]:
    """Compute the FFT for the input tensor and find the top-k periods.

    Args:
        x: input tensor of shape [B, T, C].
        k: number of top periods to return. Default is 2.

    Returns:
        Tuple containing:
            - array of top-k periods
            - tensor of top-k period weights

    """
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list

    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(Module):
    """TimesBlock module for time series forecasting.

    Args:
        seq_len: length of the input sequence.
        pred_len: length of the prediction sequence.
        top_k: number of top periods to consider.
        d_model: dimension of the model.
        d_ff: dimension of the feed-forward network.
        num_kernels: number of kernels for convolutional layers.

    """

    def __init__(
        self, seq_len: int, pred_len: int, top_k: int, d_model: int, d_ff: int, num_kernels: int
    ):
        super(TimesBlock, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k

        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass for the TimesBlock module.

        Args:
            x: input tensor of shape [B, T, N].

        Returns:
            Output tensor of shape [B, T, N].

        """
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([B, (length - (self.seq_len + self.pred_len)), N]).to(
                    x.device
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x

        return res


class TimesNet_NN(DLEstimator):
    """TimesNet model from the paper https://arxiv.org/abs/2210.02186.

    Args:
        - features_groups: dict, dictionary with the number of features for each group.
        - pred_len: length of the prediction sequence.
        - seq_len: input sequence length.
        - e_layers: number of encoder layers. Default is 2.
        - d_model: dimension of the model. Default is 512.
        - d_ff: dimension of the feed-forward network. Default is 2048.
        - num_kernels: number of kernels for convolutional layers. Default is 6.
        - top_k: number of top periods to consider. Default is 5.
        - c_out: number of output channels. Default is 7.
        - dropout: dropout rate. Default is 0.1.
        - embed: type of embedding. Default is "timeF".
        - freq: frequency of the data. Default is "h".
        - channel_independent: channel independence.

    """

    def __init__(
        self,
        features_groups: dict,
        pred_len: int,
        seq_len: int,
        e_layers: int = 2,
        d_model: int = 512,
        d_ff: int = 2048,
        num_kernels: int = 6,
        top_k: int = 5,
        dropout: float = 0.1,
        time_embed: str = "timeF",
        freq: str = "h",
        channel_independent: bool = False,
    ):
        super().__init__(features_groups, pred_len, seq_len)
        self.e_layers = e_layers
        self.channel_independent = channel_independent

        num_features = sum(self.features_groups_corrected.values()) - self.num_series
        num_datetime_features = self.features_groups_corrected["datetime_features"]

        self.model = nn.ModuleList(
            [
                TimesBlock(self.seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
                for _ in range(self.e_layers)
            ]
        )

        # Embedding
        self.time_embed = time_embed
        if self.channel_independent:
            emb_channels = num_features + 1
        else:
            emb_channels = self.num_series + num_features
        if self.time_embed:
            self.enc_embedding = Embedding(
                emb_channels - num_datetime_features,
                d_model,
                use_time=True,
                num_datetime_features=num_datetime_features,
                embed_type=time_embed,
                freq=freq,
                dropout=dropout,
            )
        else:
            self.enc_embedding = Embedding(emb_channels, d_model, use_time=False, dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model)
        self.predict_linear = nn.Linear(self.seq_len, pred_len + self.seq_len)
        if self.channel_independent:
            self.projection = nn.Linear(d_model, 1, bias=True)
        else:
            self.projection = nn.Linear(d_model, self.num_series, bias=True)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass for the TimesNet neural network.

        Args:
            x: input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            Output tensor of shape (batch_size, pred_len, num_series).

        """
        series = slice_features(
            x, ["series"], self.features_groups_corrected
        )  # (batch_size, seq_len, num_series)

        # RevIN on series
        series_means = series.mean(1, keepdim=True).detach()  # (batch_size, 1, num_series)
        series = series - series_means  # (batch_size, seq_len, num_series)
        series_stdev = torch.sqrt(
            torch.var(series, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # (batch_size, 1, num_series)
        series = series / series_stdev  # (batch_size, seq_len, num_series)

        if self.channel_independent:
            # 4d tensors
            if self.time_embed:
                time_features_4d = slice_features_4d(
                    x, ["datetime_features"], self.features_groups_corrected, self.num_series
                )  # (batch_size, seq_len, num_series, num_datetime_features)
                exog_features_4d = slice_features_4d(
                    x,
                    ["id", "fh", "series_features", "other_features"],
                    self.features_groups_corrected,
                    self.num_series,
                )  # (batch_size, seq_len, num_series, num_exog_features)
            else:
                time_features_4d = torch.empty(
                    series.shape[0], series.shape[1], self.num_series, 0
                ).to(
                    series.device
                )  # (batch_size, seq_len, num_series, 0)
                exog_features_4d = slice_features_4d(
                    x,
                    ["id", "fh", "datetime_features", "series_features", "other_features"],
                    self.features_groups_corrected,
                    self.num_series,
                )  # (batch_size, seq_len, num_series, num_exog_features)

            series = series.unsqueeze(-1)  # (batch_size, seq_len, num_series, 1)

            series = torch.concat(
                [series, exog_features_4d], dim=-1
            )  # (batch_size, seq_len, num_series, num_exog_features + 1)

            series = rearrange(
                series, "b s c f -> (b c) s f"
            )  # (batch_size * num_series, seq_len, num_exog_features + 1)
            time_features = rearrange(
                time_features_4d, "b s c f -> (b c) s f"
            )  # (batch_size * num_series, seq_len, num_datetime_features)

            # embedding
            enc_out = self.enc_embedding(
                series, time_features
            )  # (batch_size * num_series, seq_len, d_model)
            enc_out = rearrange(
                enc_out, "bc s d -> bc d s"
            )  # (batch_size * num_series, d_model, seq_len)
            enc_out = self.predict_linear(
                enc_out
            )  # (batch_size * num_series, d_model, seq_len + pred_len)
            enc_out = rearrange(
                enc_out, "bc d s -> bc s d"
            )  # (batch_size * num_series, seq_len + pred_len, d_model)

            # TimesNet
            for i in range(self.e_layers):
                enc_out = self.layer_norm(
                    self.model[i](enc_out)
                )  # (batch_size * num_series, seq_len + pred_len, d_model)

            # project back
            dec_out = self.projection(enc_out)  # (batch_size * num_series, seq_len + pred_len, 1)

            dec_out = rearrange(
                dec_out, "(b c) s d -> b s c d", c=self.num_series
            )  # (batch_size, seq_len + pred_len, num_series, 1)

            dec_out = dec_out.squeeze(-1)  # (batch_size, seq_len + pred_len, num_series)

        else:
            # 3d tensors
            if self.time_embed:
                time_features = slice_features(
                    x, ["datetime_features"], self.features_groups_corrected
                )  # (batch_size, seq_len, num_datetime_features)
                exog_features = slice_features(
                    x,
                    ["id", "fh", "series_features", "other_features"],
                    self.features_groups_corrected,
                )  # (batch_size, seq_len, num_exog_features)
            else:
                time_features = torch.empty(series.shape[0], series.shape[1], 0).to(
                    series.device
                )  # (batch_size, seq_len, 0)
                exog_features = slice_features(
                    x,
                    ["id", "fh", "datetime_features", "series_features", "other_features"],
                    self.features_groups_corrected,
                )  # (batch_size, seq_len, num_exog_features)

            series = torch.concat(
                [series, exog_features], dim=-1
            )  # (batch_size, seq_len, num_series + num_exog_features)

            # embedding
            enc_out = self.enc_embedding(series, time_features)  # (batch_size, seq_len, d_model)
            enc_out = rearrange(enc_out, "b s c -> b c s")  # (batch_size, d_model, seq_len)
            enc_out = self.predict_linear(enc_out)  # (batch_size, d_model, seq_len + pred_len)
            enc_out = rearrange(
                enc_out, "b c s -> b s c"
            )  # (batch_size, seq_len + pred_len, d_model)

            # TimesNet
            for i in range(self.e_layers):
                enc_out = self.layer_norm(
                    self.model[i](enc_out)
                )  # (batch_size, seq_len + pred_len, d_model)

            # project back
            dec_out = self.projection(enc_out)  # (batch_size, seq_len + pred_len, num_series)

        # RevIN back
        dec_out = dec_out * (
            series_stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            series_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )

        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
