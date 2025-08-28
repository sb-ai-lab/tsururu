"""PatchTST model for time series forecasting."""

from typing import Optional, Union

from tsururu.models.torch_based.dl_base import DLEstimator
from tsururu.models.torch_based.layers.decomposition import series_decomp
from tsururu.models.torch_based.layers.patch_tst import PatchTST_backbone
from tsururu.models.torch_based.utils import slice_features
from tsururu.utils.optional_imports import OptionalImport

torch = OptionalImport("torch")
nn = OptionalImport("torch.nn")
PatchTST_backbone = OptionalImport("tsururu.models.torch_based.layers.patch_tst.PatchTST_backbone")


class PatchTST_NN(DLEstimator):
    """PatchTST_NN model from the paper https://arxiv.org/abs/2211.14730.

    Args:
        - features_groups: dict, dictionary with the number of features for each group.
        - pred_len: int, the length of the output sequence.
        - seq_len: input sequence length.
        - e_layers: number of encoder layers
        - n_heads: number of attention heads
        - d_model: dimension of model
        - d_ff: dimension of feedforward network
        - dropout: dropout rate
        - fc_dropout: fully connected dropout rate
        - head_dropout: head dropout rate
        - individual: individual head flag
        - patch_len: patch length
        - stride: stride length
        - padding_patch: padding type (None or "end")
        - revin: RevIN flag
        - affine: RevIN-affine flag
        - subtract_last: subtract last flag (0: subtract mean; 1: subtract last)
        - decomposition: decomposition flag
        - kernel_size: decomposition kernel size
        - max_seq_len: maximum sequence length
        - d_k: dimension of key
        - d_v: dimension of value
        - norm: normalization type
        - attn_dropout: attention dropout rate
        - act: activation function
        - key_padding_mask: key padding mask
        - padding_var: padding variable
        - attn_mask: attention mask
        - res_attention: residual attention flag
        - pre_norm: pre-norm flag
        - store_attn: store attention flag
        - pe: positional encoding type
        - learn_pe: learn positional encoding flag
        - pretrain_head: pretrain head flag
        - head_type: head type
        - verbose: verbose flag
        - channel_independent: channel independence.

    """

    def __init__(
        self,
        features_groups: dict,
        pred_len: int,
        seq_len: int,
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
        channel_independent: float = False,
        **kwargs,
    ):
        super().__init__(features_groups, pred_len, seq_len)
        self.decomposition = decomposition
        self.channel_independent = channel_independent

        c_in = sum(self.features_groups_corrected.values())

        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(
                n_vars=self.num_series,
                c_in=c_in,
                context_window=self.seq_len,
                target_window=self.pred_len,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                channel_independent=channel_independent,
                n_layers=e_layers,
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
                **kwargs,
            )
            self.model_res = PatchTST_backbone(
                n_vars=self.num_series,
                c_in=c_in,
                context_window=self.seq_len,
                target_window=self.pred_len,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                channel_independent=channel_independent,
                n_layers=e_layers,
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
                **kwargs,
            )
        else:
            self.model = PatchTST_backbone(
                n_vars=self.num_series,
                c_in=c_in,
                context_window=self.seq_len,
                target_window=self.pred_len,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                channel_independent=channel_independent,
                n_layers=e_layers,
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
                **kwargs,
            )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass for the PatchTST_NN model.

        Args:
            x: input tensor of shape [Batch, Input length, Channel].

        Returns:
            Output tensor of shape [Batch, Input length, Channel].

        """
        series = slice_features(x, ["series"], self.features_groups_corrected)
        exog_features = slice_features(
            x,
            ["id", "fh", "datetime_features", "series_features", "other_features"],
            self.features_groups_corrected,
        )

        if self.decomposition:
            res_init, trend_init = self.decomp_module(series)

            res_init = torch.concat([res_init, exog_features], dim=2)
            trend_init = torch.concat([trend_init, exog_features], dim=2)

            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)
        else:
            x = torch.concat([series, exog_features], dim=2)
            x = x.permute(0, 2, 1)
            x = self.model(x)
            x = x.permute(0, 2, 1)

        return x
