from typing import Dict, Union, Optional
from itertools import product
from copy import deepcopy

import numpy as np
import pandas as pd

from .utils import timing_decorator
from .mimo import MIMOStrategy
from ..dataset import IndexSlicer, Pipeline, TSDataset
from ..models import Estimator


class FlatWideMIMOStrategy(MIMOStrategy):
    # занести имя стартегии в pipeline, и генерирровать horizon лагов для даты и перевернуть
    # make-mult сделать функцией пайплайна
    """A strategy that uses a single model for all points
        in the prediction horizon.

    Fit: mixture of DirectStrategy and MIMOStrategy, fit one
        model, but uses deployed over horizon DirectStrategy's features.

    Inference: similarly.

    Arguments:
        horizon: forecast horizon.
        history: number of previous for feature generating (i.e., features for observation y_t are counted from observations (y_{t-history}, ..., y_{t-1}).
        step :  in how many points to take the next observation in the training sample (the higher the step value --> the fewer observations fall into the training sample).
        is_multivariate: whether the fitting and prediction mode is multivariate.
        model_name: model name from the model factory.
        model_params: base model's params,
            for example: {
                "loss_function": "MultiRMSE",
                "early_stopping_rounds": 100,
            }.
        validation_params: execution params for base model,
            for example: {
                "type": "KFold",
                "n_splits": 3,
                "loss_function": "MAE",
            }.
    """

    def __init__(
        self,
        horizon: int,
        history: int,
        step: int,
        model: Estimator,
        pipeline: Pipeline,
        is_multivariate: bool = False,
    ):
        super().__init__(horizon, history, step, model, pipeline, is_multivariate)
        self.strategy_name = "FlatWideMIMOStrategy"

    @staticmethod
    def _make_multivariate_X_y(X, y=None, date_column=None):
        idx_slicer = IndexSlicer()

        id_feature_colname = np.array(["ID"])
        fh_feature_colname = np.array(["FH"])
        date_features_colname = X.columns[X.columns.str.contains(date_column)].values
        other_features_colname = np.setdiff1d(
            X.columns.values,
            np.concatenate([id_feature_colname, date_features_colname, fh_feature_colname]),
        )

        date_features_idx = idx_slicer.get_cols_idx(X, date_features_colname)
        other_features_idx = idx_slicer.get_cols_idx(X, other_features_colname)
        fh_feature_idx = idx_slicer.get_cols_idx(X, fh_feature_colname)

        segments_ids = np.append(np.unique(X[id_feature_colname], return_index=1)[1], len(X))

        segments_ids_array = np.array(
            [
                np.arange(segments_ids[segment_id - 1], segments_ids[segment_id])
                for segment_id in range(1, len(segments_ids))
            ]
        ).T
        date_features_array = idx_slicer.get_slice(
            X, (segments_ids_array[:, 0], date_features_idx)
        ).reshape(len(segments_ids_array), len(date_features_colname))
        fh_feature_array = idx_slicer.get_slice(
            X, (segments_ids_array[:, 0], fh_feature_idx)
        ).reshape(len(segments_ids_array), -1)
        other_features_array = idx_slicer.get_slice(
            X, (segments_ids_array, other_features_idx)
        ).reshape(
            len(segments_ids_array),
            len(other_features_colname) * (len(segments_ids) - 1),
        )

        final_other_features_colname = [
            f"{feat}__{i}"
            for i, feat in product(range(len(segments_ids) - 1), other_features_colname)
        ]
        final_X = pd.DataFrame(
            np.hstack((fh_feature_array, date_features_array, other_features_array)),
            columns=np.hstack(
                (
                    fh_feature_colname,
                    date_features_colname,
                    final_other_features_colname,
                )
            ),
        )

        if y is not None:
            final_y = idx_slicer.get_slice(y, (segments_ids_array, None)).reshape(
                len(segments_ids_array), -1
            )
            return final_X, final_y
        return final_X

    def _generate_X_y(
        self,
        dataset: TSDataset,
        train_horizon: int,
        target_horizon: int,
        is_train: bool,
        history: str = None,
        idx: Optional[np.ndarray] = None,
        n_last_horizon: Optional[int] = None,
        X_only: bool = False,
    ):
        direct_df, direct_y = super()._generate_X_y(
            dataset,
            train_horizon,
            target_horizon,
            is_train,
            history,
            idx,
            n_last_horizon,
            X_only,
        )
        direct_lag_index_dict = {}

        fh_array = np.arange(1, self.horizon + 1)

        id_mask = direct_df.columns.str.contains(dataset.id_column)
        if sum(id_mask) > 0:
            id_count = len(direct_df.loc[:, id_mask].value_counts())
        else:
            id_count = 1
        direct_lag_index_dict["ID"] = np.repeat(
            np.arange(id_count),
            repeats=(len(direct_df) / id_count * len(fh_array)),
        )
        direct_lag_index_dict["FH"] = np.tile(fh_array, len(direct_df.index))
        direct_lag_index_df = pd.DataFrame(direct_lag_index_dict)

        direct_df = direct_df.loc[:, ~id_mask]
        features_df = pd.DataFrame(
            np.repeat(direct_df.values, self.horizon, axis=0),
            columns=direct_df.columns,
        )
        final_df = direct_lag_index_df.merge(features_df, left_index=True, right_index=True)

        final_y = direct_y.reshape(-1, 1)
        return final_df, final_y

    @timing_decorator
    def predict(self, dataset):
        target_column_name = dataset.columns_params["target"]["column"][0]

        new_data = dataset.make_padded_test(self.horizon, self.history)
        new_dataset = deepcopy(dataset)
        new_dataset.data = new_data

        # Make predictions and fill NaNs
        index_slicer = IndexSlicer()
        test_ids = index_slicer.create_idx_test(
            new_dataset.data,
            self.horizon,
            dataset.history,
            dataset.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )
        X, _ = self._generate_X_y(
            new_dataset,
            train_horizon=1,
            target_horizon=1,
            is_train=False,
            idx=test_ids,
            X_only=True,
        )
        if self.is_multivariate:
            X = self._make_multivariate_X_y(X, date_column=dataset.date_column)
        pred = self.models[0].predict(X)

        nan_mask = np.isnan(new_dataset.data[dataset.target_column].astype("float"))
        new_dataset.data.loc[nan_mask, target_column_name] = np.hstack(pred)
        new_dataset.data.loc[nan_mask] = self._inverse_transform_y(
            new_dataset.data.loc[nan_mask]
        )

        # Get dataframe with predictions only
        pred_df = self._make_preds_df(new_dataset, self.horizon)
        return pred_df
