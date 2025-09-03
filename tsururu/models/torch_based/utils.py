from copy import deepcopy
from typing import Dict, List

from tsururu.utils.optional_imports import OptionalImport

torch = OptionalImport("torch")
nn = OptionalImport("torch.nn")


def adjust_features_groups(features_groups: Dict[str, int], num_lags: int) -> Dict[str, int]:
    """Adjust the feature group counts based on normalization rules.

    Args:
        features_groups: a dictionary where keys are feature group names and values represent the total number.
        num_lags: number of lag features.

    Returns:
        A new dictionary with the adjusted feature counts.

    """
    result_features_groups = deepcopy(features_groups)
    result_features_groups = {
        "series": features_groups["series"] // num_lags,
        "id": features_groups["id"] // num_lags,
        "fh": features_groups["fh"] // num_lags,
        "datetime_features": features_groups["datetime_features"] // num_lags,
        "series_features": features_groups["series_features"] // num_lags,
        "cycle_features": features_groups["cycle_features"] // num_lags,
        "other_features": features_groups["other_features"] // num_lags,
    }

    return result_features_groups


def slice_features(
    X: "torch.Tensor",
    feature_list: List[str],
    features_groups_corrected,
) -> "torch.Tensor":
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
        "cycle_features",
        "other_features",
    ]

    # Compute starting and ending indices for each group in tensor X
    indices = {}
    start_idx = 0
    for group in groups_order:
        group_size = features_groups_corrected.get(group, 0)
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


def slice_features_4d(
    X: "torch.Tensor",
    features_list: List[str],
    features_groups_corrected,
    num_series,
) -> "torch.Tensor":
    """Slice the input tensor X based on the corrected feature groups and reshape it to 4D."""
    groups_order: List[str] = [
        "series",
        "id",
        "fh",
        "datetime_features",
        "series_features",
        "cycle_features",
        "other_features",
    ]
    unique_group_names = {"series", "id", "series_features"}
    common_group_names = {"fh", "datetime_features", "cycle_features", "other_features"}

    # Compute starting and ending indices for each group in tensor X
    indices = {}
    start_idx = 0
    for group in groups_order:
        group_size = features_groups_corrected.get(group, 0)
        indices[group] = (start_idx, start_idx + group_size)
        start_idx += group_size

    # Split features_list into unique and common groups based on their order in features_list
    unique_groups = []
    common_groups = []
    for group in features_list:
        if group in unique_group_names:
            unique_groups.append(group)
        elif group in common_group_names:
            common_groups.append(group)
        else:
            raise ValueError(f"Неизвестная группа признаков: {group}")

    # Extract slices for unique groups
    unique_tensors = []
    for group in unique_groups:
        s, e = indices[group]
        # Extract slice: shape [batch_size, seq_len, group_feature_count]
        slice_tensor = X[:, :, s:e]
        unique_tensors.append(slice_tensor)

    # Extract slices for common groups
    common_tensors = []
    for group in common_groups:
        s, e = indices[group]
        slice_tensor = X[:, :, s:e]
        common_tensors.append(slice_tensor)

    batch_size, seq_len, _ = X.shape

    # Reshape unique groups by splitting features into series:
    # Transform from [batch_size, seq_len, total_features]
    # to [batch_size, seq_len, num_series, per_series_features]
    reshaped_uniques = []
    for tensor, group in zip(unique_tensors, unique_groups):
        total_features = tensor.shape[-1]
        per_series_features = total_features // num_series
        tensor_reshaped = tensor.view(batch_size, seq_len, num_series, per_series_features)
        reshaped_uniques.append(tensor_reshaped)

    # Expand common features to repeat across all series:
    # Original shape [batch_size, seq_len, common_features]
    # -> unsqueeze -> [batch_size, seq_len, 1, common_features]
    # -> expand to [batch_size, seq_len, num_series, common_features]
    expanded_commons = []
    for tensor, group in zip(common_tensors, common_groups):
        tensor_expanded = tensor.unsqueeze(2).expand(-1, -1, num_series, -1)
        expanded_commons.append(tensor_expanded)

    final_tensor = torch.cat(reshaped_uniques + expanded_commons, dim=-1)
    return final_tensor  # [batch_size, seq_len, num_series, features_per_series]
