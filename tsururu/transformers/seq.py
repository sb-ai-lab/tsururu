"""Module for transformers which generate features and targets from "elongated" series."""

from itertools import product
from typing import Sequence, Union

import numpy as np
import pandas as pd

from ..dataset.slice import IndexSlicer
from .base import SeriesToFeaturesTransformer
from .utils import _seq_mult_ts

index_slicer = IndexSlicer()


class LagTransformer(SeriesToFeaturesTransformer):
    """A transformer that generates lag features.

    Args:
        lags: lags features to build.

    Notes:
        1. Lags can be represented either as an integer value or as a sequnece
            with specific values.

        2. Maximum lag must be less than history, otherwise it is impossible
            to generate features.

    """

    def __init__(self, lags: Union[int, Sequence[int]]):
        super().__init__()
        if isinstance(lags, Sequence):
            self.lags = np.array(lags)
        if isinstance(lags, int):
            self.lags = np.arange(lags)

    def fit(self, data: dict, input_features: Sequence[str]) -> "LagTransformer":
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

        self.output_features = [
            f"{column}__lag_{lag}" for column, lag in product(self.input_features, self.lags[::-1])
        ]

        return self

    def _check_lags_less_than_history(
        self,
        data: pd.DataFrame,
        idx: np.ndarray,
        input_features_idx: np.ndarray,
    ) -> None:
        """Check if the maximum value of the lags is less than the history.

        Args:
            data: the source "elongated series" raw_ts_X.
            idx: the indices to take one observation.
            input_features_idx: the indices of the input features.

        Raises:
            AssertionError: If the maximum value of the lags is not less
                than the number of columns in the sample data (history).
        """
        sample_data = index_slicer.get_slice(data, (idx, input_features_idx))
        assert self.lags.max() < sample_data.shape[1], "lags must be less than history"

    def generate(self, data: dict) -> dict:
        """Generate lag features in X array.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        Notes:
            1. Either both idx_X or idx_y must be specified,
                LagTransformer uses only idx_X.

        """
        input_features_idx = index_slicer.get_cols_idx(data["raw_ts_X"], self.input_features)

        if len(data["idx_X"].shape) == 3:
            self._check_lags_less_than_history(
                data["raw_ts_X"], data["idx_X"][0], input_features_idx
            )
            X = _seq_mult_ts(data["raw_ts_X"], data["idx_X"], input_features_idx)
        else:
            self._check_lags_less_than_history(data["raw_ts_X"], data["idx_X"], input_features_idx)
            X = index_slicer.get_slice(data["raw_ts_X"], (data["idx_X"], input_features_idx))

        X = X[:, (X.shape[1] - 1) - self.lags[::-1], :]
        X = np.moveaxis(X, 1, 2).reshape(len(X), -1)

        if data["X"].shape == (0,):
            data["X"] = X
        else:
            data["X"] = np.hstack((data["X"], X))

        return data


class TargetGenerator(SeriesToFeaturesTransformer):
    """A transformer that selects specific indices from "elongated" raw_ts_y
    and generates values for y array.
    """

    def fit(self, data: dict, input_features: Sequence[str]) -> "LagTransformer":
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

        self.output_features = None

    def generate(self, data: dict) -> dict:
        """Generate features in y array.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        Notes:
            1. Either both idx_X or idx_y must be specified,
                TargetGenerator uses only idx_y.

        """
        input_features_idx = index_slicer.get_cols_idx(data["raw_ts_y"], self.input_features)
        data["y"] = index_slicer.get_slice(
            data["raw_ts_y"], (data["idx_y"], input_features_idx)
        ).squeeze(-1)

        return data
