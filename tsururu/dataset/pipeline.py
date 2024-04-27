"""Module for Pipeline class, which is a wrapper for the transformers."""

from typing import Tuple

import numpy as np

from ..dataset import TSDataset
from ..transformers import TransformersFactory
from ..transformers.base import SequentialTransformer, Transformer, UnionTransformer

transormers_factory = TransformersFactory()


class Pipeline:
    def __init__(self, transfomers: Transformer):
        self.transformers = transfomers

        self.is_fitted = False
        self.strategy_name = None

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

    def generate(self, data: dict) -> Tuple[np.ndarray]:
        data = self.transformers.generate(data)

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

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        y = self.transformers.inverse_transform_y(y)

        return y
