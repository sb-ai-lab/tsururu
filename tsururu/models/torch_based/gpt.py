"""GPT4TS neural network for time series forecasting."""

from tsururu.models.torch_based.dl_base import DLEstimator
from tsururu.models.torch_based.utils import slice_features, slice_features_4d
from tsururu.utils.optional_imports import OptionalImport

torch = OptionalImport("torch")
nn = OptionalImport("torch.nn")
GPT2Model = OptionalImport("transformers.models.gpt2.modeling_gpt2.GPT2Model")
GPT2Config = OptionalImport("transformers.models.gpt2.configuration_gpt2.GPT2Config")
rearrange = OptionalImport("einops.rearrange")


class GPT4TS_NN(DLEstimator):
    """GPT4TS model from the paper https://arxiv.org/abs/2302.11939.

    Args:
        - features_groups: dict, dictionary with the number of features for each group.
        - pred_len: length of the prediction sequence.
        - seq_len: input sequence length.
        - pretrain: use pretrain GPT2 backbone.
        - gpt_layers: number of gpt layers.
        - d_model: dimension of model.
        - patch_len: patch length.
        - stride: stride length.
        - freeze: freeze GPT2 backbone.
        - channel_independent: channel independence.

    """

    def __init__(
        self,
        features_groups: dict,
        pred_len: int,
        seq_len: int,
        pretrain: bool = True,
        gpt_layers: int = 6,
        d_model: int = 768,
        patch_len: int = 16,
        stride: int = 8,
        freeze: bool = False,
        channel_independent: bool = False,
    ):
        super().__init__(features_groups, pred_len, seq_len)

        self.patch_size = patch_len
        self.stride = stride

        num_channels = sum(self.features_groups_corrected.values())

        if channel_independent:
            patch_num = (
                seq_len * (sum(self.features_groups_corrected.values()) - self.num_series + 1)
                - patch_len
            ) // stride + 2
        else:
            patch_num = (self.seq_len - self.patch_size) // self.stride + 2

        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        self.channel_independent = channel_independent

        if pretrain:
            self.gpt2 = GPT2Model.from_pretrained(
                "gpt2", output_attentions=True, output_hidden_states=True
            )  # loads a pretrained GPT-2 base model
        else:
            print("------------------no pretrain------------------")
            self.gpt2 = GPT2Model(GPT2Config())

        self.gpt2.h = self.gpt2.h[:gpt_layers]
        print("gpt2 = {}".format(self.gpt2))

        if self.channel_independent:
            self.in_layer = nn.Linear(patch_len, d_model)
        else:
            self.in_layer = nn.Linear(num_channels * patch_len, d_model)

        self.out_layer = nn.Linear(d_model * patch_num, pred_len)

        if freeze and pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if "ln" in name or "wpe" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass for the GPT4TS neural network.

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
            exog_features_4d = slice_features_4d(
                x,
                ["id", "fh", "datetime_features", "series_features", "other_features"],
                self.features_groups_corrected,
                self.num_series,
            )  # (batch_size, seq_len, num_series, num_exog_features)

            series = series.unsqueeze(-1)  # (batch_size, seq_len, num_series, 1)

            x = torch.concat(
                [series, exog_features_4d], dim=-1
            )  # (batch_size, seq_len, num_series, num_exog_features + 1)
            x = rearrange(
                x, "b s c f -> b c (f s)"
            )  # (batch_size, num_series, (num_exog_features + 1) * seq_len)

            x = self.padding_patch_layer(
                x
            )  # (batch_size, num_series, (num_exog_features + 1) * (seq_len + stride))
            x = x.unfold(
                dimension=-1, size=self.patch_size, step=self.stride
            )  # (batch_size, num_series, patch_num, patch_size)

            x = rearrange(
                x, "b c pn ps -> (b c) pn ps"
            )  # (batch_size * num_series, patch_num, patch_size)

            outputs = self.in_layer(x)  # (batch_size * num_series, patch_num, d_model)
            outputs = self.gpt2(
                inputs_embeds=outputs
            ).last_hidden_state  # (batch_size * num_series, patch_num, d_model)
            outputs = rearrange(
                outputs, "bc pn d -> bc (pn d)"
            )  # (batch_size * num_series, patch_num * d_model)
            outputs = self.out_layer(outputs)  # (batch_size * num_series, pred_len)
            outputs = rearrange(
                outputs, "(b c) s -> b s c", s=self.pred_len, c=self.num_series
            )  # (batch_size, pred_len, num_series)

            outputs = outputs * series_stdev
            outputs = outputs + series_means

        else:
            # 3d tensors
            exog_features = slice_features(
                x,
                ["id", "fh", "datetime_features", "series_features", "other_features"],
                self.features_groups_corrected,
            )  # (batch_size, seq_len, num_exog_features)

            x = torch.concat(
                [series, exog_features], dim=-1
            )  # (batch_size, seq_len, num_series + num_exog_features)

            x = rearrange(
                x, "b s c -> b c s"
            )  # (batch_size, num_series + num_exog_features, seq_len)

            x = self.padding_patch_layer(
                x
            )  # (batch_size, num_series + num_exog_features, seq_len + stride)
            x = x.unfold(
                dimension=-1, size=self.patch_size, step=self.stride
            )  # (batch_size, num_series + num_exog_features, patch_num, patch_size)

            x = rearrange(
                x, "b c pn ps -> b pn (c ps)"
            )  # (batch_size, patch_num, (num_series + num_exog_features) * patch_size)

            outputs = self.in_layer(x)  # (batch_size, patch_num, d_model)
            outputs = self.gpt2(
                inputs_embeds=outputs
            ).last_hidden_state  # (batch_size, patch_num, d_model)
            outputs = rearrange(outputs, "b pn d -> b (pn d)")  # (batch_size, patch_num * d_model)
            outputs = self.out_layer(outputs)  # (batch_size, pred_len)
            outputs = outputs.unsqueeze(-1)  # (batch_size, pred_len, num_series)

            outputs = outputs * series_stdev
            outputs = outputs + series_means

        return outputs
