from typing import Dict, Union

import numpy as np

from .base import BaselineEstimator
from ..dataset.slice import IndexSlicer

index_slicer = IndexSlicer()

class MeanMethod(BaselineEstimator):
    """
    Mean Method entails predicting that all future values will be the same,
    specifically equal to the historical data's average, often referred to as the "mean".
    This approach simplifies forecasts by assuming a constant value.
    """

    def __init__(self):
        super().__init__()

    def predict(self, data: dict, pipeline) -> np.ndarray:
        columns_name = self._inspect_raw_ts_X_columns(data, pipeline)
        y_pred = (
            data["raw_ts_X"]
            .groupby(columns_name["id_column"])[columns_name["target_column"]]
            .agg("mean")
        ).values

        return y_pred


class NaiveMethod:
    """
    For naive forecasts, we simply set all forecasts to be the value of the last observation
    """

    def __init__(self):
        super().__init__()

    def predict(self, data: dict, pipeline) -> np.ndarray:
        columns_name = self._inspect_raw_ts_X_columns(data, pipeline)
        y_pred = (
            data["raw_ts_X"]
            .groupby(columns_name["id_column"])[columns_name["target_column"]]
            .fillna(method="ffill")
        )

        return y_pred


class SeasonalNaiveMethod:
    """
    This class implements the seasonal naive forecast method. It predicts future values
    based on the values from the corresponding season in the previous cycle.
    """

    def __init__(self, season_length: int):
        super().__init__()
        self.season_length = season_length

    def predict(self, data: dict, pipeline) -> np.ndarray:
        columns_name = self._inspect_raw_ts_X_columns(data, pipeline)

        assert (
            data["raw_ts_X"].shape[0] >= self.season_length
        ), "Dataframe does not have enough data for the given season length."

        season_indices = data["raw_ts_X"].index % self.season_length

        y_pred = np.empty(data["raw_ts_X"].shape[0])
        for i in range(self.season_length):
            season_data = data["raw_ts_X"][season_indices == i]
            y_pred[season_indices == i] = season_data[[columns_name["target_column"]]].fillna(
                method="ffill"
            )

        return y_pred


class DriftMethod:
    """
    Drift Forecasting Method fills missing values based on the drift method,
    where the drift is the average change seen in the historical data.
    """

    def __init__(self):
        super().__init__()

    def predict(self, data: dict, pipeline) -> np.ndarray:
        columns_name = self._inspect_raw_ts_X_columns(data, pipeline)

        for id in data["raw_ts_X"][columns_name["id_column"]].unique():
            series_data = data["raw_ts_X"].loc[data["raw_ts_X"][columns_name["id_column"]] == id]
            non_nan_indices = series_data[series_data[columns_name["target_column"]].notna()].index

            if len(non_nan_indices) < 2:
                raise ValueError(f"Not enough data to compute drift for series {id}.")

            y_t = series_data.loc[non_nan_indices[-1], columns_name["target_column"]]
            y_1 = series_data.loc[non_nan_indices[0], columns_name["target_column"]]
            T = non_nan_indices[-1] - non_nan_indices[0]

            drift = (y_t - y_1) / T

            y_pred = np.empty(data["raw_ts_X"].shape[0])
            last_known_index = non_nan_indices[-1]
            for i in range(last_known_index + 1, series_data.index[-1] + 1):
                h = i - last_known_index
                y_pred.loc[i, columns_name["target_column"]] = y_t + h * drift

        return y_pred
