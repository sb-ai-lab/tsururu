"""Module for creating a custom dataset for neural networks."""

import numpy as np
from pandas import to_datetime

from ...dataset import Pipeline

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    from abc import ABC
    torch = None
    Dataset = ABC

from typing import List, Optional, Sequence, Union, Tuple, Dict


import re
import pandas as pd


class Dataset_NN(Dataset):
    """Custom Dataset for neural networks.

    Args:
        data: dictionary with current states of "elongated series",
            arrays with features and targets, name of id, date and target
            columns and indices for features and targets.
        pipeline: pipeline object for creating and applying a pipeline of transformers.

    """

    @staticmethod
    def sort_features_names(
        features_names: Union[List[str], np.ndarray],
        target_column_name: str,
        id_column_name: str,
        date_column_name: str,
        is_fwm: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Sorts the features names in the following order:
            1) Target column features,
            2) Id column features,
            3) FWM feature (if is_fwm is True),
            4) Date column features,
            5) Series-specific features,
            6) Common features.

        Args:
            features_names: array of features names.
            target_column_name: name of the target column.
            id_column_name: name of the id column.
            date_column_name: name of the date column.
            is_fwm: flag to indicate if the strategy is FlatWideMIMO.

        Returns
            an array of indices that can be used to sort features in the required order.
            a dict with number of features of each type.

        """
        # target -> "{target_column_name}__" in the beginning of the string
        target_mask = np.array(
            [bool(re.match(f"{target_column_name}__", feature)) for feature in features_names]
        )

        # id -> "{id_column_name}__" in the beginning of the string
        id_mask = np.array(
            [bool(re.match(f"{id_column_name}__", feature)) for feature in features_names]
        )

        if is_fwm:
            fh_mask = np.array([element == "FH" for element in features_names])
        else:
            fh_mask = np.array([False for element in features_names])

        # date -> "{date_column_name}__" in the beginning of the string
        date_mask = np.array(
            [bool(re.match(f"{date_column_name}__", feature)) for feature in features_names]
        )

        # features per series -> "__{int}" in the end of the string shows the series except target features
        # we want to sort features by series (all for first, all for second, etc.)
        series_mask = np.array(
            [bool(re.search(r"(?:__)(\d+)$", feature)) for feature in features_names]
        )
        series_mask = np.logical_and(series_mask, ~target_mask)

        other_mask = ~(target_mask | id_mask | fh_mask | date_mask | series_mask)

        new_order_idx = np.concatenate(
            [
                np.where(target_mask)[0],
                np.where(id_mask)[0],
                np.where(fh_mask)[0],
                np.where(date_mask)[0],
                np.where(series_mask)[0],
                np.where(other_mask)[0],
            ]
        )

        counts = {
            "target": np.sum(target_mask),
            "id": np.sum(id_mask),
            "fh": np.sum(fh_mask),
            "date": np.sum(date_mask),
            "series": np.sum(series_mask),
            "other": np.sum(other_mask),
        }

        assert len(new_order_idx) == len(
            features_names
        ), "Number of features should not change after sorting"

        return new_order_idx, counts

    def __init__(self, data: dict, pipeline: Pipeline):
        self.data = data
        self.pipeline = pipeline
        self.idx_X = self.data["idx_X"]
        self.idx_y = self.data["idx_y"]

        self.indices = self._create_indices()

        self.num_lags = None

    def _create_date_indices(self):
        """Creates indices for each unique date for multivariate data."""
        unique_dates, inverse_indices = np.unique(
            self.data["raw_ts_X"][self.data["date_column_name"]], return_inverse=True
        )
        unique_dates = to_datetime(unique_dates)
        self.date_indices = {
            date: np.where(inverse_indices == idx)[0] for idx, date in enumerate(unique_dates)
        }

    def _create_indices(self) -> np.ndarray:
        """Creates indices for the dataset based on the pipeline configuration.

        Returns:
            array of indices.

        """
        # Create indices for the dataset
        # If multivariate, we need to create indices for each unique date
        # (we use a bunch of rows in the idx_X for each date)
        if self.pipeline.multivariate:
            arange_value = len(self.data["idx_X"]) // self.data["raw_ts_X"]["id"].nunique()
            self._create_date_indices()
        else:
            # If global, we need to create indices for each row in the idx_X
            arange_value = len(self.data["idx_X"])

        if self.pipeline.strategy_name == "FlatWideMIMOStrategy":
            horizon = self.idx_y.shape[1]
            arange_value *= horizon

        return np.arange(arange_value)

    def _adjust_fvm_indices(self, index: int) -> tuple:
        """Adjusts indices for the FlatWideMIMO strategy.

        Args:
            index: index to adjust.

        Returns:
            adjusted index and sample index for the horizon.

        Notes: If the strategy is FlatWideMIMO, we need to adjust the index to get firstly get the
            correct index for the MIMO sample and then the correct index for the sample in the horizon.

        """
        horizon = self.idx_y.shape[1]
        index_of_sample = index % horizon
        index = index // horizon
        return index, index_of_sample

    def _adjust_multivariate_indices(self, index: int) -> tuple:
        """Adjusts indices for the current date in multivariate data.

        Args:
            index: index to get the date indices for.

        Returns:
            indices for features and targets.

        Notes: If the data is multivariate, we need to get the indices for the current date and
            then get the correct indices for the sample.

        """
        current_date = self.data["raw_ts_X"][self.data["date_column_name"]].iloc[
            self.idx_X[index][0]
        ]
        current_date = to_datetime(current_date)
        first_idx = self.date_indices[current_date]
        idx_X = self.idx_X[np.isin(self.idx_X[:, 0], first_idx)]
        idx_y = self.idx_y[np.isin(self.idx_X[:, 0], first_idx)]
        return idx_X, idx_y

    def _get_adjusted_data(self, idx_X: np.ndarray, idx_y: np.ndarray) -> dict:
        """Adjusts raw time series data based on indices.

        Args:
            idx_X: indices for features.
            idx_y: indices for targets.

        Returns:
            adjusted data and indices.

        Notes: We want to get only the time series points that are necessary for the current sample

        """
        raw_ts_X_adjusted = self.data["raw_ts_X"].iloc[idx_X.flatten()].reset_index(drop=True)
        raw_ts_y_adjusted = self.data["raw_ts_y"].iloc[idx_y.flatten()].reset_index(drop=True)

        idx_X_adjusted = np.arange(np.size(idx_X)).reshape(idx_X.shape)
        idx_y_adjusted = np.arange(np.size(idx_y)).reshape(idx_y.shape)

        if self.pipeline.strategy_name == "FlatWideMIMOStrategy":
            idx_X_adjusted = idx_X_adjusted.reshape(-1, idx_X_adjusted.shape[-1])
            idx_y_adjusted = idx_y_adjusted.reshape(-1, idx_y_adjusted.shape[-1])

        return {
            "raw_ts_X": raw_ts_X_adjusted,
            "raw_ts_y": raw_ts_y_adjusted,
            "X": np.array([]),
            "y": np.array([]),
            "idx_X": idx_X_adjusted,
            "idx_y": idx_y_adjusted,
            "target_column_name": self.data["target_column_name"],
            "date_column_name": self.data["date_column_name"],
            "id_column_name": self.data["id_column_name"],
        }

    def __getitem__(self, index: int) -> tuple:
        """Gets a data sample for the given index.

        Args:
            index: index of the data sample to retrieve.

        Returns:
            a tuple containing the feature tensor and target tensor.

        """
        if self.pipeline.strategy_name == "FlatWideMIMOStrategy":
            index, index_of_sample = self._adjust_fvm_indices(index)
        else:
            index_of_sample = None

        if self.pipeline.multivariate:
            idx_X, idx_y = self._adjust_multivariate_indices(index)
        else:
            idx_X = self.idx_X[index]
            idx_y = self.idx_y[index]

        data = self._get_adjusted_data(idx_X, idx_y)

        X, y = self.pipeline.generate(data)

        if self.num_lags is None:
            self.num_lags = self.find_lags(self.pipeline.output_features)

        if self.pipeline.strategy_name == "FlatWideMIMOStrategy":
            X = X[index_of_sample, :]
            y = y[index_of_sample, :].reshape(1, -1)

            # Find "FH" feature idx
            FH_idx_start = np.where(self.pipeline.output_features == "FH")[0][0]
            FH_idx_end = np.where(self.pipeline.output_features == "FH")[0][-1]

            # Breed "FH" feature N = self.num_lags times
            X = np.concatenate(
                (X[:FH_idx_start], np.tile(X[FH_idx_start], self.num_lags), X[FH_idx_end + 1 :]),
                axis=0,
            )

            # Correct pipeline output features
            self.pipeline.output_features = pd.concat(
                [
                    pd.Series(self.pipeline.output_features[:FH_idx_start]),
                    pd.Series(["FH"] * self.num_lags),
                    pd.Series(self.pipeline.output_features[FH_idx_end + 1 :]),
                ]
            )

            X = X.reshape(1, -1)

        idx_sorted, counts = self.sort_features_names(
            self.pipeline.output_features,
            self.data["target_column_name"],
            self.data["id_column_name"],
            self.data["date_column_name"],
            is_fwm=True,
        )

        X = X[:, idx_sorted]

        if self.pipeline.strategy_name == "FlatWideMIMOStrategy":
            # Breed datetime features N = self.num_lags times
            datetime_features_start = counts["target"] + counts["id"] + counts["fh"]
            datetime_features_end = datetime_features_start + counts["date"]

            X = np.concatenate(
                (
                    X[:, :datetime_features_start],
                    np.repeat(
                        X[:, datetime_features_start:datetime_features_end], self.num_lags
                    ).reshape(1, -1),
                    X[:, datetime_features_end:],
                ),
                axis=1,
            )

        try:
            X = X.reshape(self.num_lags, -1, order="F")
        except:
            raise ValueError(
                "Failed to reshape data. Check feature lags and data shape compatibility."
            )

        # try:
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        # except:

        print("X", X)
        print("X shape", X.shape)
        print("X type", X.dtype)
        print("X float", np.float32(X))


        print("Y", y)
        print("Y shape", y.shape)
        print("Y type", y.dtype)
        print("Y float", np.float32(y))

        raise ValueError

        if self.pipeline.multivariate:
            y_tensor = y_tensor.reshape(self.data["num_series"], -1)

        y_tensor = y_tensor.T # (horizon, num_series)

        return X_tensor, y_tensor

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            number of samples in the dataset.

        """
        return len(self.indices)

    @staticmethod
    def find_lags(output_features):
        max_lag = 0
        pattern = re.compile(r"^(.*?)__lag_(\d+)(.*)$")
        for feature in output_features:
            match = pattern.match(feature)
            if match:
                feature_name, lag, series_id = match.groups()
                if int(lag) > max_lag:
                    max_lag = int(lag)

        num_lags = max_lag + 1
        return num_lags
