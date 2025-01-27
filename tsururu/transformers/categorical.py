"""Module for transformers for categorical features."""

from typing import Optional, Sequence

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .base import FeaturesGenerator


class LabelEncodingTransformer(FeaturesGenerator):
    """A transformer that encodes categorical features into integer values."""

    def __init__(self):
        super().__init__()
        self.les = {}

    def fit(
        self, data: dict, input_features: Optional[Sequence[str]] = None
    ) -> "LabelEncodingTransformer":
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
        for column_name in self.input_features:
            self.les[column_name] = LabelEncoder().fit(data["raw_ts_X"][column_name])

        self.output_features = [f"{column_name}__label" for column_name in self.input_features]

        return self

    def transform(self, data: dict) -> dict:
        """Generate features in `raw_ts_X`.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        new_arr = np.empty((len(data["raw_ts_X"]), len(self.output_features)), np.int32)
        for i, column_name in enumerate(self.input_features):
            new_arr[:, i] = self.les[column_name].transform(data["raw_ts_X"][column_name])
        data["raw_ts_X"][self.output_features] = new_arr

        return data


class OneHotEncodingTransformer(FeaturesGenerator):
    """A transformer that encodes categorical features as a one-hot
        numeric array.

    Args:
        drop: one from ['first', 'if_binary', None] or
            array-list of shape (n_features, ):
            1. if None: retain all features.
            2. if "first": drop the first category in each feature.
            3. if "if_binary": drop the first category in each feature with
                two categories.
            4. if `array`: drop[i] is the category in feature X[:, i] that
                should be dropped.

    """

    def __init__(self, drop: str = None):
        super().__init__()
        self.drop = drop
        self.ohes = {}

    def fit(
        self, data: dict, input_features: Optional[Sequence[str]] = None
    ) -> "LabelEncodingTransformer":
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
        self.output_features = []

        for column_name in self.input_features:
            self.ohes[column_name] = OneHotEncoder(drop=self.drop).fit(
                data["raw_ts_X"][column_name].values.reshape(-1, 1)
            )

        if self.drop == "first":
            for column_name in self.input_features:
                for id_name in data["raw_ts_X"][column_name].unique()[1:]:
                    self.output_features.append(f"{column_name}__{id_name}_ohe")

        elif self.drop == "is_binary":
            for column_name in self.input_features:
                if data["raw_ts_X"][column_name].nunique() == 2:
                    for id_name in data["raw_ts_X"][column_name].unique()[1:]:
                        self.output_features.append(f"{column_name}__{id_name}_ohe")
                else:
                    for id_name in data["raw_ts_X"][column_name].unique():
                        self.output_features.append(f"{column_name}__{id_name}_ohe")

        elif isinstance(self.drop, np.ndarray):
            for column_i, column_name in enumerate(self.input_features):
                for id_name in np.delete(
                    data["raw_ts_X"][column_name].unique(),
                    np.where(data["raw_ts_X"][column_name].unique() == self.drop[column_i]),
                ):
                    self.output_features.append(f"{column_name}__{id_name}_ohe")

        else:
            for column_i, column_name in enumerate(self.input_features):
                for id_name in data["raw_ts_X"][column_name].unique():
                    self.output_features.append(f"{column_name}__{id_name}_ohe")

        return self

    def transform(self, data: dict) -> dict:
        """Generate features in `raw_ts_X`.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        result_data = [
            self.ohes[column_name]
            .transform(data["raw_ts_X"][column_name].values.reshape(-1, 1))
            .todense()
            for column_name in self.input_features
        ]
        data["raw_ts_X"][self.output_features] = np.hstack(result_data)

        return data
