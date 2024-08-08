"""TimesNet neural network for time series forecasting."""

from typing import Tuple

import numpy as np


from .layers.convolution import Inception_Block_V1
from .layers.embedding import DataEmbedding

try:
    import torch
    import torch.fft
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    torch.fft = None
    F = None


def FFT_for_Period(x: torch.Tensor, k: int = 2) -> Tuple[np.ndarray, torch.Tensor]:
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


class TimesBlock(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class TimesNet_NN(nn.Module):
    """TimesNet neural network for time series forecasting.

    Args:
        seq_len: length of the input sequence.
        pred_len: length of the prediction sequence.
        e_layers: number of encoder layers. Default is 2.
        enc_in: number of input channels. Default is 7.
        d_model: dimension of the model. Default is 512.
        d_ff: dimension of the feed-forward network. Default is 2048.
        num_kernels: number of kernels for convolutional layers. Default is 6.
        top_k: number of top periods to consider. Default is 5.
        c_out: number of output channels. Default is 7.
        dropout: dropout rate. Default is 0.1.
        embed: type of embedding. Default is "timeF".
        freq: frequency of the data. Default is "h".

    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        e_layers: int = 2,
        enc_in: int = 7,
        d_model: int = 512,
        d_ff: int = 2048,
        num_kernels: int = 6,
        top_k: int = 5,
        c_out: int = 7,
        dropout: float = 0.1,
        embed: str = "timeF",
        freq: str = "h",
    ):
        super(TimesNet_NN, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.e_layers = e_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.top_k = top_k
        self.c_out = c_out
        self.dropout = dropout
        self.embed = embed
        self.freq = freq

        self.model = nn.ModuleList(
            [
                TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
                for _ in range(self.e_layers)
            ]
        )

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.predict_linear = nn.Linear(seq_len, pred_len + seq_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        """Forward pass for the TimesNet neural network.

        Args:
            x_enc: input tensor of shape [B, T, C].
            x_mark_enc: time features tensor of shape [B, T, F].

        Returns:
            Output tensor of shape [B, pred_len, C].

        """
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        for i in range(self.e_layers):
            enc_out = self.layer_norm(self.model)

        dec_out = self.projection(enc_out)

        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )

        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
