import numpy as np
import torch
from torch.utils.data import Dataset


class nnDataset(Dataset):
    def __init__(self, data, pipeline):
        self.data = data
        self.pipeline = pipeline
        self.idx_X = self.data["idx_X"]
        self.idx_y = self.data["idx_y"]
        self.indices = np.arange(len(self.data["idx_X"]))

    def __getitem__(self, index):
        data = {
            "raw_ts_X": self.data["raw_ts_X"],
            "raw_ts_y": self.data["raw_ts_y"],
            "X": np.array([]),
            "y": np.array([]),
            "idx_X": self.idx_X[index],
            "idx_y": self.idx_y[index],
            "target_column": self.data["target_column"],
            "date_column": self.data["date_column"],
            "id_column": self.data["id_column"],
        }

        # Копия всего дикта data, всего но для индекса (копия, но маленькой штуки)
        X, y = self.pipeline.generate(data)

        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        return X_tensor, y_tensor

    def __len__(self):
        return len(self.indices)
