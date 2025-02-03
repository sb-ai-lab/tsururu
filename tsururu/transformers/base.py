"""Base classes for transformers, that are needed for feature generating."""

from typing import Optional, Sequence

import numpy as np


class Transformer:
    """Base class for transformers, that are needed for feature generating.

    Args:
        input_features: array with names of columns to transform.

    Note: there are two categories of transformers:

        1. Transformers that are used to collect pipelines:
            - "Union" transformers;
            - "Sequential" transformers.

        2. Transformers that are used to transform raw rows:
            and generate features and targets:
            - "FeaturesGenerator" transformers;
            - "SeriesToSeries" transformers;
            - "SeriesToFeatures" transformers;
            - "FeaturesToFeatures" transformers.

        3. In methods `fit`, `transform`, `fit_transform` and `generate`,
            all transformers take as input and pass as output the dictionary
            named `data`, which contains 7 objects:
            - `raw_ts_X` and `raw_ts_y`: pd.DataFrame - "elongated series";
            - `X` and `y`: np.ndarray - arrays with features and targets;
            - `id_column_name`: str - name of id column;
            - `idx_X` and `idx_y`: np.ndarray - arrays with indices of
                time series' points for features and targets generating.

            Though each method uses and modifies only part of them:
            1. `fit` is trained on `raw_ts_X`;
            2. `transform` changes `raw_ts_X` and `raw_ts_y` (depending on the
                flags `transform_features`, `transform_targets`);
            3. `generate` uses `raw_ts_X` and `raw_ts_y` and
                modifies `X` and `y`.
            4. `transform` and `generate` can use `idx_X` and `idx_y` to update
                transformer params and generate features and targets.
            5. all methods can use `id_column_name`.

    """

    def __init__(self, input_features: Optional[Sequence[str]] = None):
        self.input_features = input_features
        self.output_features = None  # array with names of resulting columns

    def fit(self, data: dict, input_features: Optional[Sequence[str]] = None) -> "Transformer":
        """Fit transformer on "elongated series" and return it's instance.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            input_features: array with names of columns to transform.

        Returns:
            self.

        """
        if input_features is not None:
            self.input_features = input_features

        return self

    def transform(self, data: dict) -> dict:
        """Transform "elongated series" for feautures' and targets' further
            generation and update self.params if needed.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        raise NotImplementedError()

    def fit_transform(self, data: dict, input_features: Optional[Sequence[str]] = None) -> dict:
        """Default implementation of fit_transform - fit and then transform.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            input_features: array with names of columns to transform.

        Returns:
            current states of `data` dictionary.

        """
        self.fit(data, input_features)

        return self.transform(data)

    def generate(self, data: dict) -> dict:
        """Generate or transform features and targets in X, y arrays.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        raise NotImplementedError()


class SequentialTransformer(Transformer):
    """Transformer that contains the sequence of transformers
        and apply them one by one sequentially.

    Args:
        transformers_list: Sequence of transformers.
        input_features: array with names of columns to transform.

    Notes:
        1. In this transformer, the names of the input columns should be
            provided at initialisation rather than at fitting.

    """

    def __init__(self, transformers_list: Sequence[Transformer], input_features: Sequence[str]):
        super().__init__(input_features=input_features)
        self.transformers_list = transformers_list
        self.inverse_transformers_list = []

    def fit(
        self, data: dict, input_features: Optional[Sequence[str]] = None
    ) -> "SequentialTransformer":
        """Fit not supported. Needs output to fit next transformer.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            input_features: array with names of columns to transform.

        Raises:
            NotImplementedError: raised if called.

        """
        raise NotImplementedError(
            "Sequential supports only fit_transform since needs output" "to fit next transformer."
        )

    def transform(self, data: dict) -> dict:
        """Apply the sequence of transformers to data containers
            one after the other and transform "elongated series".

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        self.inverse_transformers_list = []

        for trf in self.transformers_list:
            data = trf.transform(data)
            if hasattr(trf, "transform_target") and trf.transform_target:
                self.inverse_transformers_list.append(trf)

        self.inverse_transformers_list = self.inverse_transformers_list[::-1]

        return data

    def fit_transform(self, data, input_features: Optional[Sequence[str]] = None) -> dict:
        """Fit and apply the sequence of transformers to data containers
            one after the other and transform "elongated series".

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            input_features: array with names of columns to transform.

        Returns:
            current states of `data` dictionary.

        """
        self.inverse_transformers_list = []

        if input_features is not None:
            self.input_features = input_features

        current_input_features = self.input_features

        for trf in self.transformers_list:
            data = trf.fit_transform(data, current_input_features)
            current_input_features = trf.output_features
            if hasattr(trf, "transform_target") and trf.transform_target:
                # Check that transform_target corresponding to transformer for target column
                assert self.input_features == [
                    data["target_column_name"]
                ], f"`transform_target` can't be used with exogenous features. You try use it on {self.input_features}, while target column is `{data['target_column_name']}`"
                self.inverse_transformers_list.append(trf)
            elif hasattr(trf, "inverse_transformers_list") and trf.inverse_transformers_list:
                self.inverse_transformers_list.extend(trf.inverse_transformers_list)

        self.inverse_transformers_list = self.inverse_transformers_list[::-1]
        self.output_features = current_input_features

        return data

    def generate(self, data: dict) -> dict:
        """Apply the sequence of transformers to containers with data
            one after the other and generate or transform features and
            targets in X, y arrays.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        for trf in self.transformers_list:
            data = trf.generate(data)

        return data

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Inverse transforms on the target variable y.

        Args:
            y: the target variable to be inversed.

        Returns:
            the inversed target variable.

        """
        for trf in self.inverse_transformers_list:
            y = trf.inverse_transform_y(y)

        return y


class UnionTransformer(Transformer):
    """Transformer that contains the sequence of transformers
        and apply them `in parallel` and concatenate the result.

    Args:
        transformer_list: Sequence of transformers.

    Notes:
        1. There is no true parallelism, but the idea is to apply all
            transformers to the same dataset and concatenate the results.

    """

    def __init__(self, transformers_list: Sequence[Transformer]):
        super().__init__()
        self.transformers_list = transformers_list
        self.inverse_transformers_list = []

    def fit(
        self, data: dict, input_features: Optional[Sequence[str]] = None
    ) -> "UnionTransformer":
        """Fit transformers on "elongated series" in parallel and return
            their instances.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            input_features: array with names of columns to transform.

        Returns:
            self.

        """
        for trf in self.transformers_list:
            trf.fit(data, input_features)

        return self

    def transform(self, data: dict) -> dict:
        """Apply the sequence of transformers to data containers in parallel
           and transform "elongated series".

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        self.inverse_transformers_list = []

        for trf in self.transformers_list:
            data = trf.transform(data)
            if hasattr(trf, "transform_target") and trf.transform_target:
                self.inverse_transformers_list.append(trf)
            elif hasattr(trf, "inverse_transformers_list") and trf.inverse_transformers_list:
                self.inverse_transformers_list.extend(trf.inverse_transformers_list)

        return data

    def fit_transform(self, data: dict, input_features: Optional[Sequence[str]] = None) -> dict:
        """Fit and apply the sequence of transformers to data containers
            in parallel and transform "elongated series".

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            input_features: array with names of columns to transform.

        Returns:
            current states of `data` dictionary.

        """
        self.inverse_transformers_list = []
        output_features_list = []

        for trf in self.transformers_list:
            data = trf.fit_transform(data, input_features)
            if trf.output_features is not None:
                output_features_list.append(trf.output_features)
            if hasattr(trf, "transform_target") and trf.transform_target:
                # Check that transform_target corresponding to transformer for target column
                assert self.input_features == [
                    data["target_column_name"]
                ], "`transform_target` can't be used with exogenous features. You try use it on"
                f"{self.input_features}, while target column is `{data['target_column_name']}`"
                self.inverse_transformers_list.append(trf)
            elif hasattr(trf, "inverse_transformers_list") and trf.inverse_transformers_list:
                self.inverse_transformers_list.extend(trf.inverse_transformers_list)

        self.output_features = np.concatenate(output_features_list)

        return data

    def generate(self, data: dict) -> dict:
        """Apply the sequence of transformers to containers with data
            in parallel and generate or transform features and targets
            in X, y arrays

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        for trf in self.transformers_list:
            copy_X = data["X"]
            data["X"] = np.array([])

            data = trf.generate(data)

            if copy_X.shape != (0,) and data["X"].shape != (0,):
                data["X"] = np.hstack((copy_X, data["X"]))
            elif data["X"].shape == (0,):
                data["X"] = copy_X

        return data

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Inverse transforms on the target variable y.

        Args:
            y: the target variable to be inversed.

        Returns:
            the inversed target variable.

        """
        for trf in self.inverse_transformers_list:
            y = trf.inverse_transform_y(y)

        return y


class FeaturesGenerator(Transformer):
    """A transformer that is trained on the "elongated series"
        and uses them to generate new columns.

    Notes:
        1. For this transformer, the active method is `transform`, which
            changes the state of raw_ts_X, raw_ts_y; `generate` does nothing
            and just passes data through it.

    """

    def generate(self, data: dict) -> dict:
        """For FeaturesGenerator `generate` does nothing and just
            passes data through it.
        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        return data


class SeriesToSeriesTransformer(Transformer):
    """A transformer that is trained on the "elongated series"
        and applied to them.

    Args:
        transform_features: whether to transform features.
        transform_target: whether to transform targets.

    Notes:
        1. For this transformer, the active method is `transform`, which
            changes the state of raw_ts_X, raw_ts_y; `generate` does nothing
            and just passes data through it.

        2. This transformer has flags `transform_features`, `transform_target`.

        3. This transformer has inverse_transform_y method.

    """

    def __init__(self, transform_features: bool, transform_target: bool):
        super().__init__()
        self.transform_features = transform_features
        self.transform_target = transform_target
        self.params = {}

    def _transform(self, data: dict, data_key: str) -> dict:
        """A method to transform the data based on the given data key
            (`raw_ts_X` or `raw_ts_y`).

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            data_key: A string representing the key in the data dictionary:
                either `raw_ts_X` or `raw_ts_y`.

        Returns:
            A dictionary with the transformed data.

        """
        data[data_key] = (
            data[data_key]
            .groupby(data["id_column_name"], sort=False)
            .apply(lambda group: self._transform_segment(group, group.name), include_groups=False)
            .reset_index(level=data["id_column_name"], drop=False)
        )

        return data

    def transform(self, data: dict) -> dict:
        """Transform "elongated series" for feautures' and targets' further
            generation and update self.params if needed.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        if self.transform_features:
            self._transform(data, "raw_ts_X")
        else:
            for i, column_name in enumerate(self.input_features):
                data["raw_ts_X"].loc[:, self.output_features[i]] = data["raw_ts_X"].loc[
                    :, column_name
                ]
        if self.transform_target:
            self._transform(data, "raw_ts_y")
        else:
            for i, column_name in enumerate(self.input_features):
                if column_name.split("__")[0] == data["target_column_name"]:
                    data["raw_ts_y"].loc[:, self.output_features[i]] = data["raw_ts_y"].loc[
                        :, column_name
                    ]

        return data

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Inverse transforms on the target variable y.

        Args:
            y: the target variable to be inversed.

        Returns:
            the inversed target variable.

        """
        assert NotImplementedError()

    def generate(self, data: dict) -> dict:
        """For SeriesToSeriesTransformer `generate` does nothing and just
            passes data through it.
        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        return data


class SeriesToFeaturesTransformer(Transformer):
    """Transformer that is trained on the "elongated series" and applied
        to them to generate or transform features and targets in X, y arrays.

    Notes:
        1. For this transformer, the active method is `generate`, which
            changes the state of X, y arrays; `transform` does nothing and
            just passes data through it.

    """

    def transform(self, data: dict) -> dict:
        """For SeriesToFeaturesTransformer transform does nothing and just
            passes data through it.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        return data


class FeaturesToFeaturesTransformer(Transformer):
    """A transformer that is trained on generated features in X, y arrays
        and applied to them to transform features and targets.

    Notes:
        1. For this transformer, the active method is `generate`, which
            changes the state of X, y arrays; `transform` does nothing and
            just passes data through it.

        2. This transformer has flags `transform_features`, `transform_target`.

        3. This transformer has inverse_transform_y method.

    """

    def __init__(self, transform_features: bool, transform_target: bool):
        super().__init__()
        self.transform_features = transform_features
        self.transform_target = transform_target

    def transform(self, data: dict) -> dict:
        """For FeaturesToFeaturesTransformer `transform` update
            self.params if needed.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        return data

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Inverse transforms on the target variable y.

        Args:
            y: the target variable to be inversed.

        Returns:
            the inversed target variable.

        """
        assert NotImplementedError()
