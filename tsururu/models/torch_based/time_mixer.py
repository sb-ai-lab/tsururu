import torch
import torch.nn as nn
import torch.nn.functional as F

from .dl_base import DLEstimator
from .layers.decomposition import series_decomp
from .layers.embedding import Embedding
from .layers.rev_in import RevIN
from .utils import slice_features


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleSeasonMixing, self).__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers

        self.down_sampling_layers_module = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window**i),
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(self.down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers_module[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleTrendMixing, self).__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                        self.seq_len // (self.down_sampling_window**i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window**i),
                        self.seq_len // (self.down_sampling_window**i),
                    ),
                )
                for i in reversed(range(self.down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        down_sampling_window,
        d_model,
        dropout,
        channel_independence,
        decomp_method,
        top_k,
        d_ff,
        moving_avg,
        down_sampling_layers,
    ):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window
        self.d_model = d_model
        self.dropout_val = dropout
        self.channel_independence = channel_independence
        self.decomp_method = decomp_method
        self.top_k = top_k
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.down_sampling_layers = down_sampling_layers

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout_val)

        if self.decomp_method == "moving_avg":
            self.decompsition = series_decomp(self.moving_avg)
        elif self.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(self.top_k)
        else:
            raise ValueError("decompsition is error")

        if self.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=self.d_model, out_features=self.d_ff),
                nn.GELU(),
                nn.Linear(in_features=self.d_ff, out_features=self.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            seq_len=self.seq_len,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=self.down_sampling_layers,
        )

        # Mixing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            seq_len=self.seq_len,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=self.down_sampling_layers,
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_ff),
            nn.GELU(),
            nn.Linear(in_features=self.d_ff, out_features=self.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer_NN(DLEstimator):
    def __init__(
        self,
        features_groups: dict,
        pred_len: int,
        seq_len: int,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method="avg",
        d_ff=32,
        e_layers: int = 2,
        d_model: int = 512,
        moving_avg=25,
        use_norm=1,
        dropout: float = 0.1,
        embed: str = "timeF",
        freq: str = "h",
        channel_independence=True,
        decomp_method='moving_avg',
        c_out=7,
        top_k=5,
        use_future_temporal_feature=False
    ):
        super().__init__(features_groups, pred_len, seq_len)

        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.e_layers = e_layers
        self.d_model = d_model
        self.moving_avg = moving_avg
        self.use_norm = use_norm
        self.dropout = dropout
        self.embed = embed
        self.freq = freq
        self.channel_independence = channel_independence
        self.c_out = c_out
        self.down_sampling_method = down_sampling_method
        self.decomp_method = decomp_method
        self.top_k = top_k
        self.d_ff = d_ff
        self.use_future_temporal_feature = use_future_temporal_feature

        self.channel_independence = channel_independence
        self.pdm_blocks = nn.ModuleList(
            [
                PastDecomposableMixing(
                    seq_len=self.seq_len,
                    pred_len=self.pred_len,
                    down_sampling_window=self.down_sampling_window,
                    d_model=self.d_model,
                    dropout=self.dropout,
                    channel_independence=self.channel_independence,
                    decomp_method=self.decomp_method,
                    top_k=self.top_k,
                    d_ff=self.d_ff,
                    moving_avg=self.moving_avg,
                    down_sampling_layers=self.down_sampling_layers,
                )
                for _ in range(self.e_layers)
            ]
        )

        self.preprocess = series_decomp(moving_avg)

        self.num_features_wo_datetime = (
            sum(self.features_groups_corrected.values())
            - self.features_groups_corrected["datetime_features"]
        )

        num_datetime_features = self.features_groups_corrected["datetime_features"]
        num_features_wo_datetime = (
            sum(self.features_groups_corrected.values()) - self.num_series - num_datetime_features
        )

        if self.channel_independence == 1:
            self.enc_embedding = Embedding(
                1,
                d_model,
                use_pos=False,
                num_datetime_features=num_datetime_features,
                embed_type=embed,
                freq=freq,
                dropout=dropout,
            )
        else:
            self.enc_embedding = Embedding(
                num_features_wo_datetime,
                d_model,
                use_pos=False,
                num_datetime_features=num_datetime_features,
                embed_type=embed,
                freq=freq,
                dropout=dropout,
            )

        self.layer = e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                RevIN(
                    self.num_features_wo_datetime,
                    affine=True,
                    non_norm=True if use_norm == 0 else False,
                )
                for i in range(down_sampling_layers + 1)
            ]
        )
        
        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.seq_len // (down_sampling_window**i),
                    self.pred_len,
                )
                for i in range(down_sampling_layers + 1)
            ]
        )

        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(d_model, c_out, bias=True)

            self.out_res_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.seq_len // (down_sampling_window**i),
                        self.seq_len // (down_sampling_window**i),
                    )
                    for i in range(down_sampling_layers + 1)
                ]
            )

            self.regression_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.seq_len // (down_sampling_window**i),
                        self.pred_len,
                    )
                    for i in range(down_sampling_layers + 1)
                ]
            )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.down_sampling_method == "max":
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == "avg":
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            down_pool = nn.Conv1d(
                in_channels=self.num_features_wo_datetime,
                out_channels=self.num_series,
                kernel_size=3,
                padding=padding,
                stride=self.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(
                    x_mark_enc_mark_ori[:, :: self.down_sampling_window, :]
                )
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, :: self.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc):
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(
                range(len(x_enc)),
                x_enc,
            ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x):
        series = slice_features(x, ["series"], self.features_groups_corrected)
        date_features = slice_features(x, ["datetime_features"], self.features_groups_corrected)
        exog_features = slice_features(x, ["id", "fh", "series_features", "other_features"], self.features_groups_corrected)

        series = torch.concat([series, exog_features], dim=2)

        dec_out = self.forecast(series, date_features)
        return dec_out
