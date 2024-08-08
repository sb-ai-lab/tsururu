"""Module for embedding layers."""

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class TokenEmbedding(nn.Module):
    """Token embedding layer using 1D convolution.

    Args:
        c_in: number of input channels.
        d_model: dimension of the model.

    """

    def __init__(self, c_in: int, d_model: int):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the token embedding.

        Args:
            x: input tensor of shape (batch_size, c_in, seq_len).

        Returns:
            tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)

        return x


class PositionalEmbedding(nn.Module):
    """Positional encoding using sine and cosine functions.

    Args:
        d_model: dimension of the model.
        max_len: maximum length of the sequence.

    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the positional embedding.

        Args:
            x: input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            positional encoding tensor of shape (1, seq_len, d_model).

        """
        return self.pe[:, : x.size(1)]


class FixedEmbedding(nn.Module):
    """Fixed embedding layer using precomputed sine and cosine values.

    Args:
        c_in: number of input channels.
        d_model: dimension of the model.

    """

    def __init__(self, c_in: int, d_model: int):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the fixed embedding.

        Args:
            x: input tensor of shape (batch_size, seq_len).

        Returns:
            embedding tensor of shape (batch_size, seq_len, d_model).

        """
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Temporal embedding layer for time-related features.

    Args:
        d_model: dimension of the model.
        embed_type: type of embedding ('fixed' or 'learned').
        freq: frequency of the time features ('h', 't', etc.).

    """

    def __init__(self, d_model: int, embed_type: str = "fixed", freq: str = "h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the temporal embedding.

        Args:
            x: input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            embedding tensor of shape (batch_size, seq_len, d_model).

        """
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """Time feature embedding layer using linear transformation.

    Args:
        d_model: dimension of the model.
        embed_type: type of embedding (default is 'timeF').
        freq: frequency of the time features ('h', 't', etc.).

    """

    def __init__(self, d_model: int, embed_type: str = "timeF", freq: str = "h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the time feature embedding.

        Args:
            x: input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            embedding tensor of shape (batch_size, seq_len, d_model).

        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """Data embedding layer combining token, positional, and temporal embeddings.

    Args:
        c_in: number of input channels.
        d_model: dimension of the model.
        embed_type: type of temporal embedding ('fixed', 'learned', or 'timeF').
        freq: frequency of the time features ('h', 't', etc.).
        dropout: dropout rate.

    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the data embedding.

        Args:
            x: input tensor of shape (batch_size, seq_len, c_in).
            x_mark: optional tensor for temporal features.

        Returns:
            embedding tensor of shape (batch_size, seq_len, d_model).

        """
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )

        return self.dropout(x)
