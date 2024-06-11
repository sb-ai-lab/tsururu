"""Module for transformers for numeric features."""

import re
from typing import Sequence

import numpy as np
import pandas as pd

from ..dataset.slice import IndexSlicer
from .base import FeaturesToFeaturesTransformer, SeriesToSeriesTransformer

index_slicer = IndexSlicer()


class StandardScalerTransformer(SeriesToSeriesTransformer):
    """Transformer that standardizes features by removing the mean and scaling.

    Args:
        transform_features: whether to transform features.
        transform_target: whether to transform target.

    Notes:
        1. Transformer has a parameter self.fitted_params = {
            id_1: {
                (colname_1, 'mean'): mean, (colname_2, 'std'): std, ...
            },
            id_2: {
                (colname_1, 'mean'): mean, (colname_2, 'std'): std, ...
            },
            ...
        }.
        2. self.params: np.ndarray len(idx_y) x 2.

    """

    def __init__(self, transform_features: bool, transform_target: bool):
        super().__init__(
            transform_features=transform_features,
            transform_target=transform_target,
        )

        self.fitted_params = {}

    def fit(self, data: dict, input_features: Sequence[str]) -> SeriesToSeriesTransformer:
        """Fit transformer on "elongated series" and return it's instance.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            input_features: array with names of columns to transform.

        Returns:
            self.

        """
        super().fit(data, input_features)
        stat_df = (
            data["raw_ts_X"]
            .groupby(data["id_column_name"])[self.input_features]
            .agg(["mean", "std"])
        )
        self.fitted_params = stat_df.to_dict(orient="index")
        self.output_features = [f"{column}__standard_scaler" for column in self.input_features]

        return self

    def _get_mask_mean_std(self, segment, column_name, current_id):
        """
        Calculate the column_mask, mean and standard deviation of a segment
            for a given column name and current id.

        Args:
            segment: segment of "elongated series" to transform to calculate
                the mean and standard deviation for.
            column_name: the name of the column to calculate the mean and
                standard deviation for.
            current_id: the name of id column of the segment.

        Returns:
            column mask, mean, and standard deviation.

        """
        column_mask = [segment.columns.str.contains(column_name)][0]
        mean = self.fitted_params[current_id][(column_name, "mean")]
        std = self.fitted_params[current_id][(column_name, "std")]

        return column_mask, mean, std

    def _transform_segment(self, segment: pd.Series, id_column_name: str) -> pd.Series:
        """Transform segment (points with similar id) of "elongated series"
            for feautures' and targets' further generation.

        Args:
            segment: segment of "elongated series" to transform.
            id_column_name: name of id column.

        Returns:
            transformed segment of "elongated series".

        """
        current_id = segment[id_column_name].values[0]

        for i, column_name in enumerate(self.input_features):
            column_mask, mean, std = self._get_mask_mean_std(
                segment=segment,
                column_name=column_name,
                current_id=current_id,
            )
            segment.loc[:, self.output_features[i]] = (segment.loc[:, column_mask] - mean) / std

        return segment

    def transform(self, data: dict) -> dict:
        """Transform "elongated series" for feautures' and targets' further
            generation and update self.params.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        data = super().transform(data)

        # Update the params if self.transform_target is True
        if self.transform_target:
            id_from_target_idx = index_slicer.get_slice(
                data["raw_ts_y"][data["id_column_name"]], (data["idx_y"][:, 0], None)
            )
            self.params = [
                list(self.fitted_params[current_id].values()) for current_id in id_from_target_idx
            ]
            self.params = np.array(self.params)

        return data

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Inverse transforms on the target variable y.

        Args:
            y: the target variable to be inversed.

        Returns:
            the inversed target variable.

        """
        if self.transform_target:
            if len(y.shape) == 1 or y.shape[0] == 1:
                y = y * self.params[:, 1] + self.params[:, 0]
            else:
                y = y * self.params[:, np.newaxis, 1] + self.params[:, np.newaxis, 0]

        return y


class DifferenceNormalizer(SeriesToSeriesTransformer):
    """Transformer that normalizes values by the previous value.

    Args:
        regime: "delta" to take the difference or "ratio" to take the ratio
            between the current and the previous value.
        transform_features: whether to transform features.
        transform_target: whether to transform target.

    Notes:
        1. self.params: np.ndarray len(idx_y) x 1.

    """

    def __init__(self, regime: str, transform_features: bool, transform_target: bool):
        super().__init__(
            transform_features=transform_features,
            transform_target=transform_target,
        )
        self.regime = regime

    def fit(self, data: dict, input_features: Sequence[str]) -> SeriesToSeriesTransformer:
        """Fit transformer on "elongated series" and return it's instance.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            input_features: array with names of columns to transform.

        Returns:
            self.

        """
        super().fit(data, input_features)
        last_values_df = (
            data["raw_ts_X"].groupby(data["id_column_name"])[self.input_features].last()
        )
        self.params = last_values_df.to_dict(orient="index")
        self.output_features = [f"{column}__diff_norm" for column in self.input_features]

        return self

    def _transform_segment(self, segment: pd.Series, *_):
        """Transform segment (points with similar id) of "elongated series"
            for feautures' and targets' further generation.

        Args:
            segment: segment of "elongated series" to transform.

        Returns:
            transformed segment of "elongated series".

        """
        for i, column_name in enumerate(self.input_features):
            if self.regime == "delta":
                segment.loc[:, self.output_features[i]] = segment.loc[
                    :, column_name
                ] - segment.loc[:, column_name].shift(1)
            elif self.regime == "ratio":
                segment.loc[:, self.output_features[i]] = segment.loc[
                    :, column_name
                ] / segment.loc[:, column_name].shift(1)

        return segment

    def transform(self, data: dict) -> dict:
        """Transform "elongated series" for feautures' and targets' further
            generation and update self.params.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id column and
                arrays with indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        data = super().transform(data)

        # Update the params if self.transform_target is True
        if self.transform_target:
            self.params = index_slicer.get_slice(
                data["raw_ts_y"][self.input_features], (data["idx_y"][:, 0] - 1, None)
            )

        return data

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Inverse transforms on the target variable y.

        Args:
            y: the target variable to be inversed.

        Returns:
            the inversed target variable.

        """
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if len(self.params.shape) == 3:
            self.params = self.params[0]

        if self.transform_target:
            if self.regime == "delta":
                y = np.cumsum(np.hstack((self.params, y)), axis=1)[:, 1:]
            elif self.regime == "ratio":
                y = np.cumprod(np.hstack((self.params, y)), axis=1)[:, 1:]

        return y


class LastKnownNormalizer(FeaturesToFeaturesTransformer):
    """Transformer that normalizes values by the last known value.

    Args:
        regime: "delta" to take the difference or "ratio" -- the ratio
            between the current and the last known value in the future.
        last_lag_substring: a substring that is included in the name
            of any columns in the feature table and denotes
            the last known (nearest) lag features.

    Notes:
        1. self.params: np.ndarray len(idx_y) x 1.

    """

    def __init__(
        self,
        regime: str,
        transform_features: bool,
        transform_target: bool,
        last_lag_substring: str = "lag_0",
    ):
        super().__init__(
            transform_features=transform_features,
            transform_target=transform_target,
        )
        self.regime = regime
        self.last_lag_substring = last_lag_substring

    def fit(self, data: dict, input_features: Sequence[str]) -> FeaturesToFeaturesTransformer:
        """Fit transformer on "elongated series" and return it's instance.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            input_features: array with names of columns to transform.

        Returns:
            self.

        """
        super().fit(data, input_features)
        self.output_features = [f"{column}__last_known_norm" for column in self.input_features]

        return self

    def transform(self, data: dict) -> dict:
        """Update self.params.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        # Update the params if self.transform_target is True
        if self.transform_target:
            try:
                feature = re.compile("^(.*)__(lag_\d+)$").findall(self.input_features[0])[0][0]
            except IndexError:
                raise ValueError(
                    "There is no lags in data['raw_ts_X']! Make sure that you initialize LastKnownNormalizer AFTER LagTransformer!"
                )
            self.params = index_slicer.get_slice(
                data["raw_ts_X"][feature], (data["idx_X"][:, -1], None)
            )

        return data

    def generate(self, data: dict) -> dict:
        """Generate or transform features and targets in X, y arrays.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        assert (
            len(data["X"]) != 0
        ), "X is empty! Make sure that you initialize LastKnownNormalizer AFTER LagTransformer!"

        last_lag_idx_by_feature = {}
        feature_by_idx = {}
        for i, column in enumerate(self.input_features):
            feature, lag_suffix = re.compile("^(.*)__(lag_\d+)$").findall(column)[0]
            feature_by_idx[i] = feature
            if lag_suffix == self.last_lag_substring:
                last_lag_idx_by_feature[feature] = i

        if self.transform_target:
            feature = feature_by_idx[0]
            last_lag_idx = last_lag_idx_by_feature[feature]

            if self.regime == "delta":
                data["y"] = data["y"] - data["X"][:, last_lag_idx].reshape(-1, 1)
            elif self.regime == "ratio":
                data["y"] = data["y"] / data["X"][:, last_lag_idx].reshape(-1, 1)

        for i, column in enumerate(self.input_features):
            feature = feature_by_idx[i]
            last_lag_idx = last_lag_idx_by_feature[feature]

            if self.regime == "delta":
                if self.transform_features:
                    data["X"][:, i] = data["X"][:, i] - data["X"][:, last_lag_idx]
            elif self.regime == "ratio":
                if self.transform_features:
                    data["X"][:, i] = data["X"][:, i] / data["X"][:, last_lag_idx]

        return data

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Inverse transforms on the target variable y.

        Args:
            y: the target variable to be inversed.

        Returns:
            the inversed target variable.

        """
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(self.params.shape) == 1:
            self.params = self.params.reshape(-1, 1)
        elif len(self.params.shape) == 3:
            self.params = self.params[0]

        if self.transform_target:
            if self.regime == "delta":
                y = y + self.params
            elif self.regime == "ratio":
                y = y * self.params

        return y
