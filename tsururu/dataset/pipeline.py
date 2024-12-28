"""Module for Pipeline class, which is a wrapper for the transformers."""

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from ..transformers import (
    SequentialTransformer,
    TargetGenerator,
    Transformer,
    TransformersFactory,
    UnionTransformer,
)
from .dataset import TSDataset
from .slice import IndexSlicer

transormers_factory = TransformersFactory()
index_slicer = IndexSlicer()


class Pipeline:
    """Class for creating and applying a pipeline of transformers.

    Args:
        transformers: an outer transformer to be applied.
        multivariate: whether to apply the pipeline to get
            multivariate data.

    """

    def __init__(self, transformers: Transformer, multivariate: bool = False):
        self.transformers = transformers
        self.multivariate = multivariate

        self.is_fitted = False
        self.strategy_name = None
        self.output_features = None
        self.y_original_shape = None

    @classmethod
    def from_dict(cls, columns_params: dict, multivariate: bool) -> "Pipeline":
        """Create a pipeline from a dict of column parameters.

        Args:
            columns_params: a dictionary containing the parameters
                for each column.
            multivariate: whether the pipeline is multivariate.

        Returns:
            the created pipeline.

        """
        # Resulting pipeline is a Union transformer with Sequential transformers
        result_union_transformers_list = []

        # For each column create a list of transformers for resulting Sequential transformer
        for role, columns_params in columns_params.items():
            current_sequential_transformers_list = []

            transformers_dict = columns_params["features"]
            for transformer_name, transformer_params in transformers_dict.items():
                assert (
                    role != "target" and transformer_params.get("transform_target", False)
                ) is False, "It is not possible to use transform_target=True with transformers for exogenous variables"

                if transformer_name == "LagTransformer" and role == "target":
                    features_transformer = transormers_factory.create_transformer(
                        transformer_name, transformer_params
                    )
                    target_transformer = TargetGenerator()
                    transformer = UnionTransformer(
                        transformers_list=[features_transformer, target_transformer]
                    )
                else:
                    transformer = transormers_factory.create_transformer(
                        transformer_name, transformer_params
                    )

                current_sequential_transformers_list.append(transformer)

            result_union_transformers_list.append(
                SequentialTransformer(
                    transformers_list=current_sequential_transformers_list,
                    input_features=columns_params["columns"],
                )
            )

        union = UnionTransformer(transformers_list=result_union_transformers_list)

        return cls(union, multivariate)

    @classmethod
    def easy_setup(cls, roles: dict, pipeline_params: dict, multivariate: bool) -> "Pipeline":
        """Create a pipeline semi-automatically from a dict of columns roles
            and a dict of small description of pipeline.

        Args:
            roles: columns' roles and types.
            pipeline_params: a dictionary containing the parameters
                for pipeline.
            multivariate: whether the pipeline is multivariate.

        Returns:
            the created pipeline.

        Notes: pipeline_params is a dictionary with the following keys:
            - target_lags (necessary): list of lags for target
            - date_lags (optional, default False): list of lags for date
            - exog_lags (optional, deafult False): list of lags for exogenous features
            - target_normalizer (optional, default standard_scaler): type of target normalizer
                (none, standard_scaler, difference_normalizer, last_known_normalizer)
            - target_normalizer_regime (optional, default none): regime of target normalizer
                (none, delta, ratio)

        """
        # Check if all necessary keys are in pipeline_params
        assert "target_lags" in pipeline_params, "target_lags MUST BE in pipeline_params!"

        # Add default values for pipeline_params if they are not provided
        if "date_lags" not in pipeline_params:
            pipeline_params["date_lags"] = False
        if "exog_lags" not in pipeline_params:
            pipeline_params["exog_lags"] = False
        if "target_normalizer" not in pipeline_params:
            pipeline_params["target_normalizer"] = "standard_scaler"
        if "target_normalizer_regime" not in pipeline_params:
            pipeline_params["target_normalizer_regime"] = "none"

        # Check some params' correctness
        assert pipeline_params["target_normalizer"] in [
            "none",
            "standard_scaler",
            "difference_normalizer",
            "last_known_normalizer",
        ], "there is no such target_normalizer!"

        assert pipeline_params["target_normalizer_regime"] in [
            "none",
            "delta",
            "ratio",
        ], "there is no such target_normalizer_regime!"

        if pipeline_params["target_normalizer"] in ["standard_scaler", "none"]:
            assert (
                pipeline_params["target_normalizer_regime"] == "none"
            ), "target_normalizer_regime MUST BE `none` for this normalizer"
        else:
            assert (
                pipeline_params["target_normalizer_regime"] != "none"
            ), "target_normalizer_regime MUST BE NOT `none` for this normalizer"

        # Resulting pipeline is a Union transformer with Sequential transformers
        result_union_transformers_list = []

        # For each column create a list of transformers for resulting Sequential transformer
        for role, columns_params in roles.items():
            current_sequential_transformers_list = []
            if role == "target":
                target_lag = transormers_factory.create_transformer(
                    "LagTransformer", {"lags": pipeline_params["target_lags"]}
                )
                target_generator = TargetGenerator()
                target_union = UnionTransformer(transformers_list=[target_lag, target_generator])

                if pipeline_params["target_normalizer"] == "standard_scaler":
                    target_normalizer = transormers_factory.create_transformer(
                        "StandardScalerTransformer",
                        {
                            "transform_features": True,
                            "transform_target": True,
                        },
                    )
                    current_sequential_transformers_list.append(target_normalizer)
                    current_sequential_transformers_list.append(target_union)

                elif pipeline_params["target_normalizer"] == "difference_normalizer":
                    target_normalizer = transormers_factory.create_transformer(
                        "DifferenceNormalizer",
                        {
                            "transform_features": True,
                            "transform_target": True,
                            "regime": pipeline_params["normalizer_regime"],
                        },
                    )
                    current_sequential_transformers_list.append(target_normalizer)
                    current_sequential_transformers_list.append(target_union)

                elif pipeline_params["target_normalizer"] == "last_known_normalizer":
                    target_normalizer = transormers_factory.create_transformer(
                        "LastKnownNormalizer",
                        {
                            "transform_features": True,
                            "transform_target": True,
                            "regime": pipeline_params["normalizer_regime"],
                        },
                    )
                    current_sequential_transformers_list.append(target_union)
                    current_sequential_transformers_list.append(target_normalizer)

                elif pipeline_params["target_normalizer"] == "none":
                    current_sequential_transformers_list.append(target_union)
                    current_sequential_transformers_list.append(target_lag)

                else:
                    assert (
                        pipeline_params["target_normalizer"] == "none"
                    ), "there is no such target_normalizer!"

            elif role == "date":
                date_season = transormers_factory.create_transformer(
                    "DateSeasonsGenerator",
                    {
                        "seasonalities": ["y", "m", "d", "doy", "wd"],
                        "from_target_date": True,
                    },
                )
                date_scaler = transormers_factory.create_transformer(
                    "StandardScalerTransformer",
                    {
                        "transform_features": True,
                        "transform_target": False,
                        "agg_by_id": False,
                    },
                )
                date_lag = transormers_factory.create_transformer(
                    "LagTransformer", {"lags": pipeline_params["date_lags"]}
                )

                current_sequential_transformers_list.append(date_season)
                current_sequential_transformers_list.append(date_scaler)
                current_sequential_transformers_list.append(date_lag)

            elif role == "id":
                id_encoder = transormers_factory.create_transformer("LabelEncodingTransformer", {})
                id_scaler = transormers_factory.create_transformer(
                    "StandardScalerTransformer",
                    {
                        "transform_features": True,
                        "transform_target": False,
                        "agg_by_id": False,
                    },
                )
                id_lag = transormers_factory.create_transformer("LagTransformer", {"lags": 1})
                current_sequential_transformers_list.append(id_encoder)
                current_sequential_transformers_list.append(id_scaler)
                current_sequential_transformers_list.append(id_lag)

            else:
                exog_scaler = transormers_factory.create_transformer(
                    "StandardScalerTransformer",
                    {
                        "transform_features": True,
                        "transform_target": False,
                    },
                )
                exog_lag = transormers_factory.create_transformer(
                    "LagTransformer", {pipeline_params["exog_lags"]}
                )
                current_sequential_transformers_list.append(exog_scaler)
                current_sequential_transformers_list.append(exog_lag)

            result_union_transformers_list.append(
                SequentialTransformer(
                    transformers_list=current_sequential_transformers_list,
                    input_features=columns_params["columns"],
                )
            )

        union = UnionTransformer(transformers_list=result_union_transformers_list)

        return cls(union, multivariate)

    @staticmethod
    def create_data_dict_for_pipeline(
        dataset: TSDataset, features_idx: np.ndarray, target_idx: np.ndarray
    ) -> dict:
        """Create a data dictionary for the pipeline.

        Args:
            dataset: the input time series dataset.
            features_idx: the indices of the features in the dataset.
            target_idx: the indices of the target in the dataset.

        Returns:
            the created data dictionary.

        """
        data = {}
        data["raw_ts_X"] = dataset.data.copy()
        data["raw_ts_y"] = dataset.data.copy()
        data["X"] = np.array([])
        data["y"] = np.array([])
        data["id_column_name"] = dataset.id_column
        data["date_column_name"] = dataset.date_column
        data["target_column_name"] = dataset.target_column
        data["num_series"] = dataset.data[dataset.id_column].nunique()
        data["idx_X"] = features_idx
        data["idx_y"] = target_idx

        return data

    def fit_transform(self, data: dict, strategy_name: str) -> dict:
        """Fit the transformers to the data and transform

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            strategy_name: the name of the strategy to use.

        Returns:
            the transformed data dictionary.

        """
        self.strategy_name = strategy_name

        data = self.transformers.fit_transform(data)
        self.is_fitted = True

        return data

    def transform(self, data: dict) -> dict:
        """Transforms the input data using the transformers.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            the transformed data dictionary.

        """
        data = self.transformers.transform(data)

        return data

    def from_mimo_to_flatwidemimo(self, data: dict) -> dict:
        """
        Converts the input data from MIMO to FlatWideMIMO format.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            the dictionary containing the converted data and the
            column names of the converted data.

        """
        X = pd.DataFrame(data["X"], columns=self.transformers.output_features)

        date_features_mask = X.columns.str.contains(data["date_column_name"])
        id_features_mask = X.columns.str.contains(data["id_column_name"])

        horizon = data["idx_y"].shape[-1]
        fh_array = np.arange(1, horizon + 1)

        direct_lag_index_dict = {}

        # TODO: Can we use only else?
        if sum(id_features_mask) > 0:
            id_count = len(X.loc[:, id_features_mask].value_counts())
        else:
            id_count = len(
                data["raw_ts_X"].iloc[data["idx_X"][:, 0]][data["id_column_name"]].value_counts()
            )

        if self.multivariate:
            direct_lag_index_dict["ID"] = np.repeat(
                np.arange(id_count),
                repeats=(len(X) / id_count * len(fh_array)),
            )
        else:
            unique_id = np.unique(
                [tuple(x) for x in X.loc[:, id_features_mask].values], axis=0, return_index=1
            )
            sort_unique_id = unique_id[0][np.argsort(unique_id[1])]

            for id_idx, id_feature in enumerate(X.loc[:, id_features_mask].columns):
                direct_lag_index_dict[id_feature] = np.repeat(
                    sort_unique_id[:, id_idx],
                    repeats=(len(X) / id_count * len(fh_array)),
                )

        direct_lag_index_dict["FH"] = np.tile(fh_array, len(X.index))
        direct_lag_index_df = pd.DataFrame(direct_lag_index_dict)

        # get date features for each horizon (unfolding MIMO lags over time)
        new_date_features = np.empty(
            (X.shape[0] * horizon, X.loc[:, date_features_mask].shape[1] // horizon)
        )
        try:
            for i in range(horizon):
                new_date_features[i::horizon, :] = X.loc[:, date_features_mask].values[
                    :, i::horizon
                ]

            # get unique date feature names without lag suffix
            date_feature_names = (
                X.columns[date_features_mask].str.replace("__lag_\d+$", "", regex=True).unique()
            )

            features_df = pd.DataFrame(
                np.repeat(
                    X.loc[:, ~id_features_mask & ~date_features_mask].values, horizon, axis=0
                ),
                columns=X.loc[:, ~id_features_mask & ~date_features_mask].columns,
            )

            X = pd.concat(
                [
                    direct_lag_index_df,
                    pd.DataFrame(new_date_features, columns=date_feature_names),
                    features_df,
                ],
                axis=1,
            )

        except ValueError:
            raise ValueError(
                "Something is wrong while making FlatWideMIMO strategy's X. Check that you use number of lags equal to horizon for datetime features!"
            )

        data["X"] = X.values

        data["y"] = data["y"].reshape(-1, 1)

        return data, X.columns

    def _make_multivariate_X_y(self, data: dict) -> dict:
        """Converts the input data dictionary into a multivariate
            X and y arrays.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            the updated data dictionary and the columns of the X.

        """
        X = pd.DataFrame(data["X"], columns=self.output_features)

        date_features_colname = X.columns[X.columns.str.contains(data["date_column_name"])].values
        id_features_colname = X.columns[X.columns.str.contains(data["id_column_name"])].values

        if id_features_colname.size == 0:
            # add temporary column with id for make multivariate merging
            id_idx = index_slicer.get_cols_idx(data["raw_ts_X"], data["id_column_name"])
            X["temp_ID"] = index_slicer.get_slice(data["raw_ts_X"], (data["idx_X"][:, 0], id_idx))
            id_features_colname = np.array(["temp_ID"])

        other_features_colname = X.columns.difference(
            np.hstack((id_features_colname, date_features_colname)), sort=False
        )

        date_features_idx = index_slicer.get_cols_idx(X, date_features_colname)
        other_features_idx = index_slicer.get_cols_idx(X, other_features_colname)

        segments_ids = np.append(
            np.unique([tuple(x) for x in X[id_features_colname].values], axis=0, return_index=1)[
                1
            ],
            len(X),
        )
        segments_ids = np.sort(segments_ids)
        segments_ids_array = np.array(
            [
                np.arange(segments_ids[segment_id - 1], segments_ids[segment_id])
                for segment_id in range(1, len(segments_ids))
            ]
        ).T

        date_features_array = index_slicer.get_slice(
            X, (segments_ids_array[:, 0], date_features_idx)
        ).reshape(len(segments_ids_array), len(date_features_colname))
        other_features_array = index_slicer.get_slice(
            X, (segments_ids_array, other_features_idx)
        ).reshape(
            len(segments_ids_array),
            len(other_features_colname) * (len(segments_ids) - 1),
        )

        final_other_features_colname = [
            f"{feat}__{i}"
            for i, feat in product(range(len(segments_ids) - 1), other_features_colname)
        ]

        data["X"] = np.hstack((date_features_array, other_features_array))
        new_columns = np.hstack((date_features_colname, final_other_features_colname))

        if data["y"] is not None:
            data["y"] = index_slicer.get_slice(data["y"], (segments_ids_array, None)).reshape(
                len(segments_ids_array), -1
            )

        return data, new_columns

    def _make_multivariate_X_y_flatwidemimo(self, data: dict) -> dict:
        """Converts the input data dictionary into a multivariate
            X and y arrays for FlatWideMIMO strategy.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            the updated data dictionary and the columns of the X.

        """
        X = pd.DataFrame(data["X"], columns=self.output_features)

        id_feature_colname = np.array(["ID"])
        fh_feature_colname = np.array(["FH"])
        date_features_colname = X.columns[X.columns.str.contains(data["date_column_name"])].values
        other_features_colname = [
            col
            for col in X.columns.values
            if col
            not in np.hstack([id_feature_colname, date_features_colname, fh_feature_colname])
        ]

        date_features_idx = index_slicer.get_cols_idx(X, date_features_colname)
        other_features_idx = index_slicer.get_cols_idx(X, other_features_colname)
        fh_feature_idx = index_slicer.get_cols_idx(X, fh_feature_colname)

        segments_ids = np.append(np.unique(X[id_feature_colname], return_index=1)[1], len(X))
        segments_ids = np.sort(segments_ids)
        segments_ids_array = np.array(
            [
                np.arange(segments_ids[segment_id - 1], segments_ids[segment_id])
                for segment_id in range(1, len(segments_ids))
            ]
        ).T

        date_features_array = index_slicer.get_slice(
            X, (segments_ids_array[:, 0], date_features_idx)
        ).reshape(len(segments_ids_array), len(date_features_colname))
        fh_feature_array = index_slicer.get_slice(
            X, (segments_ids_array[:, 0], fh_feature_idx)
        ).reshape(len(segments_ids_array), -1)
        other_features_array = index_slicer.get_slice(
            X, (segments_ids_array, other_features_idx)
        ).reshape(
            len(segments_ids_array),
            len(other_features_colname) * (len(segments_ids) - 1),
        )

        final_other_features_colname = [
            f"{feat}__{i}"
            for i, feat in product(range(len(segments_ids) - 1), other_features_colname)
        ]

        data["X"] = np.hstack((fh_feature_array, date_features_array, other_features_array))
        new_columns = np.hstack(
            (
                fh_feature_colname,
                date_features_colname,
                final_other_features_colname,
            )
        )

        if data["y"] is not None:
            data["y"] = index_slicer.get_slice(data["y"], (segments_ids_array, None)).reshape(
                len(segments_ids_array), -1
            )

        return data, new_columns

    def generate(self, data: dict) -> Tuple[np.ndarray]:
        """Generate the X and y arrays based on the provided data.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            tuple containing the generated dataset's input features (X)
                and targets (y).
        """
        data = self.transformers.generate(data)
        self.y_original_shape = data["y"].shape

        if self.strategy_name == "FlatWideMIMOStrategy":
            data, new_output_features = self.from_mimo_to_flatwidemimo(data)
            self.output_features = new_output_features
        else:
            self.output_features = self.transformers.output_features

        if self.multivariate:
            if self.strategy_name == "FlatWideMIMOStrategy":
                data, new_output_features = self._make_multivariate_X_y_flatwidemimo(data)
            else:
                data, new_output_features = self._make_multivariate_X_y(data)
            self.output_features = new_output_features

        return data["X"], data["y"]

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Applies the inverse transformation to the target variable.

        Args:
            y: the target variable to be transformed.

        Returns:
            the inverse transformed target variable.

        """
        y = self.transformers.inverse_transform_y(y)

        return y.reshape(-1)
