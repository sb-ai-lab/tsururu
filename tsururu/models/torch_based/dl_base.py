try:
    from torch.nn import Module
except ImportError:
    from abc import ABC

    Module = ABC

from .utils import adjust_features_groups


class DLEstimator(Module):
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
        self.features_groups_corrected = adjust_features_groups(features_groups, self.seq_len)
        self.num_series = self.features_groups_corrected["series"]
