"""Module for Pipeline class, which is a wrapper for the transformers."""

from typing import Tuple

import numpy as np
import pandas as pd

from ..dataset import TSDataset
from ..transformers import TransformersFactory
from ..transformers.base import SequentialTransformer, Transformer, UnionTransformer

transormers_factory = TransformersFactory()


class Pipeline:
    def __init__(self, transfomers: Transformer):
        self.transformers = transfomers

        self.is_fitted = False
        self.strategy_name = None
        self.output_features = None
        self.y_original_shape = None

    @classmethod
    def from_dict(cls, columns_params: dict) -> "Pipeline":
        # Resulting pipeline is a Union transformer with Sequential transformers for each column
        result_union_transformers_list = []

        # For each column create a list of transformers for resulting Sequential transformer
        for columns_params in columns_params.values():
            current_sequential_transformers_list = []

            transformers_dict = columns_params["features"]
            for transformer_name, transformer_params in transformers_dict.items():
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

        return cls(union)

    def create_data_dict_for_pipeline(
        self, dataset: TSDataset, features_idx: np.ndarray, target_idx: np.ndarray
    ) -> dict:
        data = {}
        data["raw_ts_X"] = dataset.data.copy()
        data["raw_ts_y"] = dataset.data.copy()
        data["X"] = np.array([])
        data["y"] = np.array([])
        data["id_column_name"] = dataset.id_column
        data["date_column_name"] = dataset.date_column
        data["idx_X"] = features_idx
        data["idx_y"] = target_idx

        return data

    def fit_transform(self, data: dict, strategy_name: str) -> dict:
        self.strategy_name = strategy_name

        data = self.transformers.fit_transform(data)
        self.is_fitted = True

        return data

    def transform(self, data: dict) -> dict:
        data = self.transformers.transform(data)

        return data

    def from_mimo_to_flatwidemimo(self, data: dict) -> dict:
        X = pd.DataFrame(data["X"], columns=self.transformers.output_features)

        date_features_mask = X.columns.str.contains(data["date_column_name"])
        id_features_mask = X.columns.str.contains(data["id_column_name"])

        horizon = data["idx_y"].shape[1]
        fh_array = np.arange(1, horizon + 1)

        direct_lag_index_dict = {}

        if sum(id_features_mask) > 0:
            id_count = len(X.loc[:, id_features_mask].value_counts())
        else:
            id_count = 1
        direct_lag_index_dict["ID"] = np.repeat(
            np.arange(id_count),
            repeats=(len(X) / id_count * len(fh_array)),
        )
        direct_lag_index_dict["FH"] = np.tile(fh_array, len(X.index))
        direct_lag_index_df = pd.DataFrame(direct_lag_index_dict)

        # get date features for each horizon (unfolding MIMO lags over time)
        new_date_features = np.empty(
            (X.shape[0] * horizon, X.loc[:, date_features_mask].shape[1] // horizon)
        )
        for i in range(horizon):
            new_date_features[i::horizon, :] = X.loc[:, date_features_mask].values[:, i::horizon]

        # get unique date feature names without lag suffix
        date_feature_names = (
            X.columns[date_features_mask].str.replace("__lag_\d+$", "", regex=True).unique()
        )

        features_df = pd.DataFrame(
            np.repeat(X.loc[:, ~id_features_mask & ~date_features_mask].values, horizon, axis=0),
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

        data["X"] = X.values

        self.y_original_shape = data["y"].shape
        data["y"] = data["y"].reshape(-1, 1)

        return data, X.columns

    def generate(self, data: dict) -> Tuple[np.ndarray]:
        data = self.transformers.generate(data)

        if self.strategy_name == "FlatWideMIMOStrategy":
            data, new_output_features = self.from_mimo_to_flatwidemimo(data)
            self.output_features = new_output_features
        else:
            self.output_features = self.transformers.output_features

        return data["X"], data["y"]

    # def _make_multivariate_X_y(
    #     self,
    #     X: pd.DataFrame,
    #     y: Optional[np.ndarray] = None,
    #     date_column: Optional[str] = None,
    # ):
    #     date_features_colname = X.columns[X.columns.str.contains(date_column)].values
    #     other_features_colname = np.setdiff1d(
    #         X.columns.values,
    #         np.concatenate([self.id_feature_column.columns.values, date_features_colname]),
    #     )

    #     date_features_idx = index_slicer.get_cols_idx(X, date_features_colname)
    #     other_features_idx = index_slicer.get_cols_idx(X, other_features_colname)

    #     segments_ids = np.append(np.unique(self.id_feature_column, return_index=1)[1], len(X))
    #     segments_ids_array = np.array(
    #         [
    #             np.arange(segments_ids[segment_id - 1], segments_ids[segment_id])
    #             for segment_id in range(1, len(segments_ids))
    #         ]
    #     ).T

    #     date_features_array = index_slicer.get_slice(
    #         X, (segments_ids_array[:, 0], date_features_idx)
    #     ).reshape(len(segments_ids_array), len(date_features_colname))
    #     other_features_array = index_slicer.get_slice(
    #         X, (segments_ids_array, other_features_idx)
    #     ).reshape(
    #         len(segments_ids_array),
    #         len(other_features_colname) * (len(segments_ids) - 1),
    #     )

    #     final_other_features_colname = [
    #         f"{feat}__{i}"
    #         for i, feat in product(range(len(segments_ids) - 1), other_features_colname)
    #     ]
    #     final_X = pd.DataFrame(
    #         np.hstack((date_features_array, other_features_array)),
    #         columns=np.hstack((date_features_colname, final_other_features_colname)),
    #     )

    #     if y is not None:
    #         final_y = index_slicer.get_slice(y, (segments_ids_array, None)).reshape(
    #             len(segments_ids_array), -1
    #         )
    #         return final_X, final_y
    #     return final_X

    # @staticmethod
    # def _make_multivariate_X_y(X, y=None, date_column=None):
    #     idx_slicer = IndexSlicer()

    #     id_feature_colname = np.array(["ID"])
    #     fh_feature_colname = np.array(["FH"])
    #     date_features_colname = X.columns[X.columns.str.contains(date_column)].values
    #     other_features_colname = np.setdiff1d(
    #         X.columns.values,
    #         np.concatenate([id_feature_colname, date_features_colname, fh_feature_colname]),
    #     )

    #     date_features_idx = idx_slicer.get_cols_idx(X, date_features_colname)
    #     other_features_idx = idx_slicer.get_cols_idx(X, other_features_colname)
    #     fh_feature_idx = idx_slicer.get_cols_idx(X, fh_feature_colname)

    #     segments_ids = np.append(np.unique(X[id_feature_colname], return_index=1)[1], len(X))

    #     segments_ids_array = np.array(
    #         [
    #             np.arange(segments_ids[segment_id - 1], segments_ids[segment_id])
    #             for segment_id in range(1, len(segments_ids))
    #         ]
    #     ).T
    #     date_features_array = idx_slicer.get_slice(
    #         X, (segments_ids_array[:, 0], date_features_idx)
    #     ).reshape(len(segments_ids_array), len(date_features_colname))
    #     fh_feature_array = idx_slicer.get_slice(
    #         X, (segments_ids_array[:, 0], fh_feature_idx)
    #     ).reshape(len(segments_ids_array), -1)
    #     other_features_array = idx_slicer.get_slice(
    #         X, (segments_ids_array, other_features_idx)
    #     ).reshape(
    #         len(segments_ids_array),
    #         len(other_features_colname) * (len(segments_ids) - 1),
    #     )

    #     final_other_features_colname = [
    #         f"{feat}__{i}"
    #         for i, feat in product(range(len(segments_ids) - 1), other_features_colname)
    #     ]
    #     final_X = pd.DataFrame(
    #         np.hstack((fh_feature_array, date_features_array, other_features_array)),
    #         columns=np.hstack(
    #             (
    #                 fh_feature_colname,
    #                 date_features_colname,
    #                 final_other_features_colname,
    #             )
    #         ),
    #     )

    #     if y is not None:
    #         final_y = idx_slicer.get_slice(y, (segments_ids_array, None)).reshape(
    #             len(segments_ids_array), -1
    #         )
    #         return final_X, final_y
    #     return final_X

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.strategy_name == "FlatWideMIMOStrategy":
            y = y.reshape(self.y_original_shape)
            y = self.transformers.inverse_transform_y(y)
            y = y.reshape(-1)
        else:
            y = self.transformers.inverse_transform_y(y)

        return y
