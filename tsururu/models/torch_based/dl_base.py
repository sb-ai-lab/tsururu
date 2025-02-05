from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class DLEstimator(nn.Module):
    """Base class for all DL models."""

    def __init__(
        self,
        features_groups: dict,
        pred_len: int,
        seq_len: int,
    ):

        super(DLEstimator, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.features_groups_corrected = self.adjust_features_groups(features_groups, self.seq_len)
        self.num_series = self.features_groups_corrected["series"]

    @staticmethod
    def adjust_features_groups(features_groups: Dict[str, int], num_lags: int) -> Dict[str, int]:
        """Adjust the feature group counts based on normalization rules.

        Args:
            features_groups: a dictionary where keys are feature group names and values represent the total number.
            num_lags: number of lag features.

        Returns:
            A new dictionary with the adjusted feature counts.

        """
        print(features_groups, num_lags)
        result_features_groups = deepcopy(features_groups)
        result_features_groups = {
            "series": features_groups["series"] // num_lags,
            "id": features_groups["id"] // num_lags,
            "fh": features_groups["fh"] // num_lags,
            "datetime_features": features_groups["datetime_features"] // num_lags,
            "series_features": features_groups["series_features"] // num_lags,
            # "cycle_features": features_groups["cycle_features"] // num_lags,
            "other_features": features_groups["other_features"] // num_lags,
        }
        print(result_features_groups)
        return result_features_groups

    def slice_features(
        self,
        X: torch.Tensor,
        feature_list: List[str],
    ) -> torch.Tensor:
        """Slice the input tensor X based on the corrected feature groups.

        Args:
            X: input tensor containing features arranged sequentially by groups.
            feature_list: feature group keys to select. The order in this list determines
                the order of features in the output tensor.

        Returns:
            Tensor containing the concatenated columns corresponding to the selected feature groups.

        Notes:
            1. The function computes the starting and ending indices for each group based on the
            corrected feature counts and extracts the slices for each group specified in feature_list.
            Finally, the selected slices are concatenated along the feature dimension.

        """
        # Fixed order of feature groups in tensor X
        groups_order: List[str] = [
            "series",
            "id",
            "fh",
            "datetime_features",
            "series_features",
            # "cycle_features",
            "other_features",
        ]

        # Compute starting and ending indices for each group in tensor X
        indices = {}
        start_idx = 0
        for group in groups_order:
            group_size = self.features_groups_corrected.get(group, 0)
            indices[group] = (start_idx, start_idx + group_size)
            start_idx += group_size

        # Extract slices for the groups specified in feature_list and concatenate them
        slices = []
        for group in feature_list:
            if group not in indices:
                raise ValueError(
                    f"Feature group '{group}' not found in indices. Available groups: {list(indices.keys())}."
                )
            s, e = indices[group]
            slices.append(X[:, :, s:e])

        # Concatenate the slices along the feature dimension (axis=2)
        result = torch.cat(slices, dim=2)
        return result
