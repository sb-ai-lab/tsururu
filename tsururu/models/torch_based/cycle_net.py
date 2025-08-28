"""CycleNet_NN model for time series forecasting."""

from tsururu.models.torch_based.dl_base import DLEstimator
from tsururu.models.torch_based.layers.rev_in import RevIN
from tsururu.models.torch_based.utils import slice_features, slice_features_4d
from tsururu.utils.optional_imports import OptionalImport

torch = OptionalImport("torch")
nn = OptionalImport("torch.nn")
Module = OptionalImport("torch.nn.Module")


class RecurrentCycle(Module):
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (
            index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)
        ) % self.cycle_len
        return self.data[gather_index]


class CycleNet_NN(DLEstimator):
    """CycleNet model from the paper https://arxiv.org/abs/2409.18479.

    Args:
        features_groups: dictionary of feature groups.
        pred_len: prediction length.
        seq_len: sequence length.
        cycle_len: length of the cycle.
        model_type: type of the model, either "linear" or "mlp".
        d_model: dimension of the model.
        revin: whether to use RevIN layer.
        affine: whether to use affine parameters in RevIN.
        subtract_last: whether to subtract the last value in RevIN.
        channel_independent: whether to process each channel independently.

    """

    def __init__(
        self,
        features_groups: dict,
        pred_len: int,
        seq_len: int,
        cycle_len: int = 24,
        model_type: str = "mlp",
        d_model: int = 512,
        revin: bool = True,
        affine: bool = False,
        subtract_last: bool = False,
        channel_independent: bool = True,
    ):
        super(CycleNet_NN, self).__init__(features_groups, pred_len, seq_len)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.cycle_len = cycle_len
        self.model_type = model_type
        self.d_model = d_model
        self.revin = revin

        self.channel_independent = channel_independent

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.num_series)

        assert self.model_type in ["linear", "mlp"]

        exog_channels = (
            sum(self.features_groups_corrected.values())
            - self.features_groups_corrected["series"]
            - self.features_groups_corrected["cycle_features"]
        )

        if self.channel_independent:
            input_size = self.seq_len * (1 + exog_channels)
            output_size = self.pred_len
        else:
            input_size = self.seq_len * (self.num_series + exog_channels)
            output_size = self.pred_len * self.num_series

        if self.model_type == "linear":
            self.model = nn.Linear(input_size, output_size)
        elif self.model_type == "mlp":
            self.model = nn.Sequential(
                nn.Linear(input_size, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, output_size),
            )

        if self.revin:
            self.revin_layer = RevIN(self.num_series, affine=affine, subtract_last=subtract_last)

    def forward(self, x):
        # x: (batch_size, seq_len, enc_in)
        bs = x.shape[0]

        series = slice_features(
            x, ["series"], self.features_groups_corrected
        )  # [batch_size, seq_len, num_series]

        if self.revin:
            series = self.revin_layer(series, "norm")

        cycle_index = slice_features(x, ["cycle_features"], self.features_groups_corrected)[
            :, -1, 0
        ].long()
        series = series - self.cycleQueue(
            cycle_index, self.seq_len
        )  # [batch_size, seq_len, num_series]

        if self.channel_independent:
            exog_features = slice_features_4d(
                x,
                ["id", "fh", "datetime_features", "series_features", "other_features"],
                self.features_groups_corrected,
                self.num_series,
            )  # [batch_size, seq_len, num_series, features_per_series]

            z = torch.cat(
                [series.unsqueeze(-1), exog_features], dim=3
            )  # [batch_size, seq_len, num_series, features_per_series]
            z = z.transpose(1, 2)  # [batch_size, num_series, seq_len, features_per_series]
            z = z.reshape(bs, self.num_series, -1)

            y = self.model(z)  # [batch_size, num_series, pred_len]
            y = y.transpose(1, 2)  # [batch_size, pred_len, num_series]

        else:
            exog_features = slice_features(
                x,
                ["id", "fh", "datetime_features", "series_features", "other_features"],
                self.features_groups_corrected,
            )  # [batch_size, seq_len, features]

            z = torch.cat([series, exog_features], dim=2)
            z = z.reshape(bs, -1)

            y = self.model(z)  # [batch_size, num_series*pred_len]
            y = y.reshape(bs, self.pred_len, self.num_series)  # [batch_size, pred_len, num_series]

        # add back the cycle of the output data
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # instance denorm
        if self.revin:
            y = self.revin_layer(y, "denorm")

        return y
