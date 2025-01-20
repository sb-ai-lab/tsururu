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


class Dataset_NN(Dataset):
    """Custom Dataset for neural networks.

    Args:
        data: dictionary with current states of "elongated series",
            arrays with features and targets, name of id, date and target
            columns and indices for features and targets.
        pipeline: pipeline object for creating and applying a pipeline of transformers.

    """

    def __init__(self, data: dict, pipeline: Pipeline):
        self.data = data
        self.pipeline = pipeline
        self.idx_X = self.data["idx_X"]
        self.idx_y = self.data["idx_y"]

        self.indices = self._create_indices()

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

    def _reshape_tensors(
        self, X_tensor: "torch.Tensor", y_tensor: "torch.Tensor", index_of_sample: int
    ) -> tuple:
        """Reshapes and transposes tensors to fit the neural network input.

        Args:
            X_tensor: feature tensor.
            y_tensor: target tensor.
            index_of_sample: index of the sample in the horizon.

        Returns:
            reshaped feature and target tensors.
        """
        if self.pipeline.strategy_name == "FlatWideMIMOStrategy":
            X_tensor = X_tensor[index_of_sample, :].reshape(-1, X_tensor.shape[1])
            y_tensor = y_tensor[index_of_sample, :].reshape(-1, y_tensor.shape[1])

        if self.pipeline.multivariate:
            other_columns_idx_list = []
            target_column_idx_list = []
            for i, feature in enumerate(self.pipeline.output_features):
                if feature.split("__")[0] != self.data["target_column_name"]:
                    other_columns_idx_list.append(i)
                else:
                    target_column_idx_list.append(i)

            num_series = self.data["num_series"]

            X_tensor_target_column = X_tensor[:, target_column_idx_list]
            X_tensor_other_column = X_tensor[:, other_columns_idx_list]

            X_tensor_target_column = X_tensor_target_column.view(num_series, -1).T
            X_tensor_other_column = torch.repeat_interleave(
                X_tensor_other_column,
                X_tensor_target_column.size(0) // X_tensor_other_column.size(0),
                dim=0,
            )
            X_tensor = torch.cat([X_tensor_target_column, X_tensor_other_column], dim=1)

            y_tensor = y_tensor.view(num_series, -1).T

        else:
            X_tensor = X_tensor.reshape(-1, 1)
            y_tensor = y_tensor.reshape(-1, 1)

        return X_tensor, y_tensor

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

        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        # Reshape and transpose tensors to fit the neaural network input
        return self._reshape_tensors(X_tensor, y_tensor, index_of_sample)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            number of samples in the dataset.

        """
        return len(self.indices)
