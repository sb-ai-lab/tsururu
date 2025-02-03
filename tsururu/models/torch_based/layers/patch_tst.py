"""Module for PatchTST layers."""

from typing import Optional, Tuple, Union

import numpy as np

from .positional_encoding import positional_encoding
from .rev_in import RevIN
from .utils import Transpose, get_activation_fn

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    from torch.nn import Module
except ImportError:
    from abc import ABC
    torch = None
    nn = None
    Tensor = None
    F = None
    Module = ABC


# Cell
class PatchTST_backbone(Module):
    """Backbone for Patch Time Series Transformer.

    Args:
        c_in: number of input channels.
        context_window: length of the context window.
        target_window: length of the target window.
        patch_len: length of each patch.
        stride: stride between patches.
        max_seq_len: maximum sequence length.
        n_layers: number of layers in the encoder.
        d_model: dimension of the model.
        n_heads: number of attention heads.
        d_k: dimension of the key vectors.
        d_v: dimension of the value vectors.
        d_ff: dimension of the feed-forward network.
        norm: type of normalization.
        attn_dropout: dropout rate for attention.
        dropout: dropout rate.
        act: activation function.
        key_padding_mask: use key padding mask.
        padding_var: padding variable.
        attn_mask: attention mask.
        res_attention: use residual attention.
        pre_norm: use pre-normalization.
        store_attn: store attention weights.
        pe: type of positional encoding.
        learn_pe: learn positional encoding.
        fc_dropout: dropout rate for fully connected layer.
        head_dropout: dropout rate for head.
        padding_patch: padding for patches.
        pretrain_head: use pretrain head.
        head_type: type of head.
        individual: use individual head.
        revin: use RevIN layer.
        affine: use affine transformation in RevIN.
        subtract_last: subtract last value in RevIN.
        verbose: print additional information.
        **kwargs: additional arguments.
    """

    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        max_seq_len: Optional[int] = 1024,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: Union[bool, str] = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout: float = 0,
        padding_patch: Optional[str] = None,
        pretrain_head: bool = False,
        head_type: str = "flatten",
        individual: bool = False,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        verbose: bool = False,
        **kwargs,
    ):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(
            c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            verbose=verbose,
            **kwargs,
        )

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(
                self.head_nf, c_in, fc_dropout
            )  # custom head passed as a partial func with all its kwargs
        elif head_type == "flatten":
            self.head = Flatten_Head(
                self.individual,
                self.n_vars,
                self.head_nf,
                target_window,
                head_dropout=head_dropout,
            )

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass of the PatchTST backbone.

        Args:
            z: input tensor of shape (batch_size, nvars, seq_len).

        Returns:
            output tensor of shape (batch_size, nvars, target_window).

        """
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

        # model
        z = self.backbone(z)  # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)  # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)

        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        """Create pretrain head.

        Args:
            head_nf: number of features in the head.
            vars: number of variables.
            dropout: dropout rate.

        Returns:
            pretrain head module.

        """
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))


class Flatten_Head(Module):
    """Flatten head for PatchTST.

    Args:
        individual: whether to use individual heads for each variable.
        n_vars: number of variables.
        nf: number of features.
        target_window: length of the target window.
        head_dropout: dropout rate for the head. Default is 0.

    """

    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """Forward pass of the flatten head.

        Args:
            x: input tensor of shape (batch_size, nvars, d_model, patch_num).

        Returns:
            output tensor of shape (batch_size, nvars, target_window).

        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TSTiEncoder(Module):
    """Time Series Transformer independent encoder.

    Args:
        c_in: number of input channels.
        patch_num: number of patches.
        patch_len: length of each patch.
        max_seq_len: maximum sequence length. Default is 1024.
        n_layers: number of layers in the encoder. Default is 3.
        d_model: dimension of the model. Default is 128.
        n_heads: number of attention heads. Default is 16.
        d_k: dimension of the key vectors. Default is None.
        d_v: dimension of the value vectors. Default is None.
        d_ff: dimension of the feed-forward network. Default is 256.
        norm: type of normalization. Default is "BatchNorm".
        attn_dropout: dropout rate for attention. Default is 0.0.
        dropout: dropout rate. Default is 0.0.
        act: activation function. Default is "gelu".
        store_attn: store attention weights. Default is False.
        key_padding_mask: use key padding mask. Default is "auto".
        padding_var: padding variable. Default is None.
        attn_mask: attention mask. Default is None.
        res_attention: use residual attention. Default is True.
        pre_norm: use pre-normalization. Default is False.
        pe: type of positional encoding. Default is "zeros".
        learn_pe: learn positional encoding. Default is True.
        verbose: print additional information. Default is False.
        **kwargs: additional arguments.

    """

    def __init__(
        self,
        c_in: int,
        patch_num: int,
        patch_len: int,
        max_seq_len: int = 1024,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        store_attn: bool = False,
        key_padding_mask: Union[bool, str] = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        verbose: bool = False,
        **kwargs,
    ):

        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(
            patch_len, d_model
        )  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            q_len,
            d_model,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the TSTi encoder.

        Args:
            x: input tensor of shape (batch_size, nvars, patch_len, patch_num).

        Returns:
            output tensor of shape (batch_size, nvars, d_model, patch_num).

        """

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )  # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(
            z, (-1, n_vars, z.shape[-2], z.shape[-1])
        )  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z


# Cell
class TSTEncoder(Module):
    """Time Series Transformer encoder.

    Args:
        q_len: length of the query sequence.
        d_model: dimension of the model.
        n_heads: number of attention heads.
        d_k: dimension of the key vectors. Default is None.
        d_v: dimension of the value vectors. Default is None.
        d_ff: dimension of the feed-forward network. Default is None.
        norm: type of normalization. Default is "BatchNorm".
        attn_dropout: dropout rate for attention. Default is 0.0.
        dropout: dropout rate. Default is 0.0.
        activation: activation function. Default is "gelu".
        res_attention: use residual attention. Default is False.
        n_layers: number of layers in the encoder. Default is 1.
        pre_norm: use pre-normalization. Default is False.
        store_attn: store attention weights. Default is False.

    """

    def __init__(
        self,
        q_len: int,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: Optional[int] = None,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        res_attention: bool = False,
        n_layers: int = 1,
        pre_norm: bool = False,
        store_attn: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    q_len,
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for i in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the TST encoder.

        Args:
            src: input tensor of shape (batch_size, seq_len, d_model).
            key_padding_mask: optional tensor for key padding mask.
            attn_mask: optional tensor for attention mask.

        Returns:
            output tensor of shape (batch_size, seq_len, d_model).

        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(
                    output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )

            return output
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

            return output


class TSTEncoderLayer(Module):
    """Encoder layer for the Time Series Transformer.

    Args:
        q_len: length of the query sequence.
        d_model: dimension of the model.
        n_heads: number of attention heads.
        d_k: dimension of the key vectors. Default is None.
        d_v: dimension of the value vectors. Default is None.
        d_ff: dimension of the feed-forward network. Default is 256.
        store_attn: store attention weights. Default is False.
        norm: type of normalization. Default is "BatchNorm".
        attn_dropout: dropout rate for attention. Default is 0.0.
        dropout: dropout rate. Default is 0.0.
        bias: use bias in linear layers. Default is True.
        activation: activation function. Default is "gelu".
        res_attention: use residual attention. Default is False.
        pre_norm: use pre-normalization. Default is False.

    """

    def __init__(
        self,
        q_len: int,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        store_attn: bool = False,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        bias: bool = True,
        activation: str = "gelu",
        res_attention: bool = False,
        pre_norm: bool = False,
    ):
        super().__init__()
        assert (
            not d_model % n_heads
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            d_model,
            n_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(
        self,
        src: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass of the TST encoder layer.

        Args:
            src: input tensor of shape (batch_size, seq_len, d_model).
            prev: previous attention scores for residual attention.
            key_padding_mask: optional tensor for key padding mask.
            attn_mask: optional tensor for attention mask.

        Returns:
            output tensor of shape (batch_size, seq_len, d_model).
            If res_attention is True, also returns attention scores.

        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        # Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(
                src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        else:
            src2, attn = self.self_attn(
                src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        if self.store_attn:
            self.attn = attn
        # Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        # Position-wise Feed-Forward
        src2 = self.ff(src)
        # Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(Module):
    """Multi-Head Attention Layer.

    Args:
        d_model: dimension of the model.
        n_heads: number of attention heads.
        d_k: dimension of the key vectors. Default is None.
        d_v: dimension of the value vectors. Default is None.
        res_attention: use residual attention. Default is False.
        attn_dropout: dropout rate for attention. Default is 0.0.
        proj_dropout: dropout rate for projection. Default is 0.0.
        qkv_bias: use bias in linear layers. Default is True.
        lsa: use locality-sensitive attention. Default is False.

    Notes:
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]

    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        res_attention: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True,
        lsa: bool = False,
    ):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa
        )

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(
        self,
        Q: Tensor,
        K: Optional[Tensor] = None,
        V: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass of the multi-head attention layer.

        Args:
            Q: query tensor of shape (batch_size, max_q_len, d_model).
            K: key tensor of shape (batch_size, seq_len, d_model). Default is None.
            V: value tensor of shape (batch_size, seq_len, d_model). Default is None.
            prev: previous attention scores for residual attention. Default is None.
            key_padding_mask: optional tensor for key padding mask.
            attn_mask: optional tensor for attention mask.

        Returns:
            output tensor of shape (batch_size, q_len, d_model).
            If res_attention is True, also returns attention scores.

        """

        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        # Linear (+ split in multiple heads)
        q_s = (
            self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        )  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = (
            self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        )  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = (
            self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        )  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        )  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(Module):
    """Scaled Dot-Product Attention module.

    Args:
        d_model: dimension of the model.
        n_heads: number of attention heads.
        attn_dropout: dropout rate for attention. Default is 0.0.
        res_attention: use residual attention. Default is False.
        lsa: use locality-sensitive attention. Default is False.

    Notes:
        Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017)
        with optional residual attention from previous layer (Realformer: Transformer likes
        residual attention by He et al, 2020) and locality self sttention (Vision Transformer for
        Small-Size Datasets by Lee et al, 2021)

    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        res_attention: bool = False,
        lsa: bool = False,
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim**-0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Forward pass of the scaled dot-product attention.

        Args:
            q: query tensor of shape (batch_size, n_heads, max_q_len, d_k).
            k: key tensor of shape (batch_size, n_heads, d_k, seq_len).
            v: value tensor of shape (batch_size, n_heads, seq_len, d_v).
            prev: previous attention scores for residual attention. Default is None.
            key_padding_mask: optional tensor for key padding mask.
            attn_mask: optional tensor for attention mask.

        Returns:
            output tensor of shape (batch_size, n_heads, max_q_len, d_v).
            If res_attention is True, also returns attention weights and scores.

        Notes:
            Input shape:
                q               : [bs x n_heads x max_q_len x d_k]
                k               : [bs x n_heads x d_k x seq_len]
                v               : [bs x n_heads x seq_len x d_v]
                prev            : [bs x n_heads x q_len x seq_len]
                key_padding_mask: [bs x seq_len]
                attn_mask       : [1 x seq_len x seq_len]
            Output shape:
                output:  [bs x n_heads x q_len x d_v]
                attn   : [bs x n_heads x q_len x seq_len]
                scores : [bs x n_heads x q_len x seq_len]
        """

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = (
            torch.matmul(q, k) * self.scale
        )  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev

        # Attention mask (optional)
        if (
            attn_mask is not None
        ):  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if (
            key_padding_mask is not None
        ):  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(
            attn_scores, dim=-1
        )  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
