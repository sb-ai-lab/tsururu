"""Module for validation strategy."""

from typing import Dict, Iterator, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit

from ..dataset import TSDataset


class Validator:
    """Base class for validation strategy.

    Args:
        validation_params: optional dictionary with validation parameters.
            unique for each validation strategy.

    """

    def __init__(self, validation_params: Optional[Dict[str, Union[str, int]]] = None):
        self.validation_params = validation_params

    def get_split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Iterator[
        Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
    ]:
        """Get splits for training and validation data.

        Args:
            X: feature array.
            y: target array.
            X_val: validation feature array.
            y_val: validation target array.

        Returns:
            iterator over tuples of training and validation data splits.

        Notes:
            For some purposes, the validation data is not needed, so it is optional.

        """
        raise NotImplementedError()


class KFoldCrossValidator(Validator):
    """K-Fold Cross-Validation strategy.

    Args:
        n_splits: number of folds. Must be at least 2.
        shuffle: whether to shuffle the data before splitting into batches.
        random_state: when shuffle is True, random_state affects the ordering of the indices.

    """

    def __init__(self, n_splits: int = 3, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Iterator[
        Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
    ]:
        cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for X_train_idx, X_val_idx in cv.split(X):
            if y is None:
                yield X[X_train_idx], None, X[X_val_idx], None
            else:
                yield X[X_train_idx], y[X_train_idx], X[X_val_idx], y[X_val_idx]


class TimeSeriesValidator(Validator):
    """Time Series Cross-Validation strategy.

    Args:
        n_splits: number of folds. Must be at least 2.

    """

    def __init__(self, n_splits: int = 3):
        self.n_splits = n_splits

    def get_split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Iterator[
        Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
    ]:
        cv = TimeSeriesSplit(n_splits=self.n_splits)
        for X_train_idx, X_val_idx in cv.split(X):
            if y is None:
                yield X[X_train_idx], None, X[X_val_idx], None
            else:
                yield X[X_train_idx], y[X_train_idx], X[X_val_idx], y[X_val_idx]


class HoldOutValidator(Validator):
    """Hold-Out Validation strategy.

    Args:
        validation_data: validation dataset.

    """

    def __init__(self, validation_data: TSDataset):
        self.validation_data = validation_data

    def get_split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Iterator[
        Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
    ]:
        train_idx = np.arange(len(X))
        val_idx = np.arange(len(X_val))
        if y is None:
            yield (X[train_idx], None, X_val[val_idx], None)
        else:
            yield (X[train_idx], y[train_idx], X_val[val_idx], y_val[val_idx])
