"""PatchTST model for time series forecasting."""

from typing import Optional, Union

from .layers.decomposition import series_decomp
from .layers.patch_tst import PatchTST_backbone

try:
    import torch
    import torch.nn as nn
    from torch.nn import Module
except ImportError:
    from abc import ABC
    torch = None
    nn = None
    Module = ABC


class PatchTST_NN(Module):
    """PatchTST_NN model from the paper https://arxiv.org/abs/2211.14730.

    Args:
        seq_len: input sequence length
        pred_len: prediction sequence length
        enc_in: encoder input size
        e_layers: number of encoder layers
        n_heads: number of attention heads
        d_model: dimension of model
        d_ff: dimension of feedforward network
        dropout: dropout rate
        fc_dropout: fully connected dropout rate
        head_dropout: head dropout rate
        individual: individual head flag
        patch_len: patch length
        stride: stride length
        padding_patch: padding type (None or "end")
        revin: RevIN flag
        affine: RevIN-affine flag
        subtract_last: subtract last flag (0: subtract mean; 1: subtract last)
        decomposition: decomposition flag
        kernel_size: decomposition kernel size
        max_seq_len: maximum sequence length
        d_k: dimension of key
        d_v: dimension of value
        norm: normalization type
        attn_dropout: attention dropout rate
        act: activation function
        key_padding_mask: key padding mask
        padding_var: padding variable
        attn_mask: attention mask
        res_attention: residual attention flag
        pre_norm: pre-norm flag
        store_attn: store attention flag
        pe: positional encoding type
        learn_pe: learn positional encoding flag
        pretrain_head: pretrain head flag
        head_type: head type
        verbose: verbose flag

    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        e_layers: int = 3,
        n_heads: int = 4,
        d_model: int = 16,
        d_ff: int = 128,
        dropout: float = 0.05,
        fc_dropout: float = 0.05,
        head_dropout: float = 0.0,
        individual: int = 0,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: str = "end",
        revin: int = 1,
        affine: int = 0,
        subtract_last: int = 0,
        decomposition: int = 0,
        kernel_size: int = 25,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: Union[str, bool] = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional["torch.Tensor"] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = "flatten",
        verbose: bool = False,
        **kwargs
    ):

        super().__init__()

        # load parameters
        c_in = enc_in
        context_window = seq_len
        target_window = pred_len

        n_layers = e_layers
        n_heads = n_heads
        d_model = d_model
        d_ff = d_ff
        dropout = dropout
        fc_dropout = fc_dropout
        head_dropout = head_dropout

        individual = individual

        patch_len = patch_len
        stride = stride
        padding_patch = padding_patch

        revin = revin
        affine = affine
        subtract_last = subtract_last

        decomposition = decomposition
        kernel_size = kernel_size

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
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
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                **kwargs
            )
            self.model_res = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
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
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                **kwargs
            )
        else:
            self.model = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
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
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                **kwargs
            )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass for the PatchTST_NN model.

        Args:
            x: input tensor of shape [Batch, Input length, Channel].

        Returns:
            Output tensor of shape [Batch, Input length, Channel].

        """
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            x = self.model(x)
            x = x.permute(0, 2, 1)

        return x
