"""Positional encoding for transformers."""

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


def PositionalEncoding(q_len: int, d_model: int, normalize: bool = True) -> "torch.Tensor":
    """Generate positional encoding.

    Args:
        q_len: length of the query.
        d_model: dimension of the model.
        normalize: whether to normalize the positional encoding.

    Returns:
        positional encoding tensor.

    """
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)

    return pe


def Coord2dPosEncoding(
    q_len: int,
    d_model: int,
    exponential: bool = False,
    normalize: bool = True,
    eps: float = 1e-3,
    verbose: bool = False,
) -> "torch.Tensor":
    """Generate 2D coordinate positional encoding.

    Args:
        q_len: length of the query.
        d_model: dimension of the model.
        exponential: whether to use exponential scaling.
        normalize: whether to normalize the positional encoding.
        eps: tolerance for the mean value.
        verbose: whether to print intermediate values.

    Returns:
        positional encoding tensor.

    """
    x = 0.5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = (
            2
            * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x)
            * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x)
            - 1
        )
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)

    return cpe


def Coord1dPosEncoding(
    q_len: int, exponential: bool = False, normalize: bool = True
) -> "torch.Tensor":
    """Generate 1D coordinate positional encoding.

    Args:
        q_len: length of the query.
        exponential: whether to use exponential scaling.
        normalize: whether to normalize the positional encoding.

    Returns:
        positional encoding tensor.

    """
    cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (0.5 if exponential else 1)) - 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)

    return cpe


def positional_encoding(
    pe: Optional[str], learn_pe: bool, q_len: int, d_model: int
) -> "nn.Parameter":
    """Initialize positional encoding.

    Args:
        pe: type of positional encoding.
        learn_pe: whether the positional encoding is learnable.
        q_len: length of the query.
        d_model: dimension of the model.

    Returns:
        positional encoding parameter.

    """
    if pe is None:
        W_pos = torch.empty(
            (q_len, d_model)
        )  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "zeros":
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "normal" or pe == "gauss":
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == "lin1d":
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == "exp1d":
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == "lin2d":
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == "exp2d":
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == "sincos":
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)"
        )
    return nn.Parameter(W_pos, requires_grad=learn_pe)
