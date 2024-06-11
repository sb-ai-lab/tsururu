import numpy as np
import torch
from pandas import to_datetime
from torch.utils.data import Dataset


class nnDataset(Dataset):
    def __init__(self, data, pipeline):
        self.data = data
        self.pipeline = pipeline
        self.idx_X = self.data["idx_X"]
        self.idx_y = self.data["idx_y"]

        if pipeline.multivariate:
            self.indices = np.arange(
                len(self.data["idx_X"]) // self.data["raw_ts_X"]["id"].nunique()
            )
            unique_dates, inverse_indices = np.unique(
                self.data["raw_ts_X"]["date"], return_inverse=True
            )
            unique_dates = to_datetime(unique_dates)
            self.date_indices = {
                date: np.where(inverse_indices == idx)[0] for idx, date in enumerate(unique_dates)
            }

        else:
            self.indices = np.arange(len(self.data["idx_X"]))

    def __getitem__(self, index):
        if self.pipeline.multivariate:
            current_date = self.data["raw_ts_X"]["date"].iloc[self.idx_X[index][0]]
            current_date = to_datetime(current_date)
            first_idx = self.date_indices[current_date]
            idx_X = self.idx_X[np.isin(self.idx_X[:, 0], first_idx)]
            idx_y = self.idx_y[np.isin(self.idx_X[:, 0], first_idx)]
        else:
            idx_X = self.idx_X[index]
            idx_y = self.idx_y[index]

        raw_ts_X_adjusted = self.data["raw_ts_X"].iloc[idx_X.flatten()].reset_index(drop=True)
        raw_ts_y_adjusted = self.data["raw_ts_y"].iloc[idx_y.flatten()].reset_index(drop=True)

        idx_X_adjusted = np.arange(np.size(idx_X)).reshape(idx_X.shape)
        idx_y_adjusted = np.arange(np.size(idx_y)).reshape(idx_y.shape)

        data = {
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

        X, y = self.pipeline.generate(data)

        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        if self.pipeline.multivariate:
            num_series = raw_ts_X_adjusted["id"].nunique()
            X_tensor = X_tensor.view(num_series, -1).T.flip(0)
            y_tensor = y_tensor.view(num_series, -1).T

        return X_tensor, y_tensor

    def __len__(self):
        return len(self.indices)
