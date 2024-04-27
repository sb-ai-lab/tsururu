from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..dataset import IndexSlicer, Pipeline, TSDataset
from ..models import Estimator
from .utils import timing_decorator


class Strategy:
    """Base class for strategies, that are needed for fitting and inference
        of base models.

    Args:
        horizon: forecast horizon.
        history: number of previous points for feature generating.
            (i.e., features for observation y_t are counted from observations
            (y_{t-history}, ..., y_{t-1}).
        step:  in how many points to take the next observation while making
            samples' matrix.
        model: base model.
        pipeline: pipeline for feature and target generation.
        is_multivariate: whether the prediction mode is multivariate.

    Notes:
        1. A type of strategy defines what features and targets will be built
        for subsequent training and inference of the base model.
        2. Now the `step` param should be 1. It will be changed in the future.

    """

    @staticmethod
    def check_step_param(step: int):
        assert step == 1, "Step should be 1. It will be changed in the future."

    def __init__(
        self,
        horizon: int,
        history: int,
        step: int,
        model: Estimator,
        pipeline: Pipeline,
        is_multivariate: bool = False,
    ):
        self.check_step_param(step)

        self.horizon = horizon
        self.history = history
        self.step = step
        self.model = model
        self.pipeline = pipeline
        self.is_multivariate = is_multivariate

        self.models = []

    @staticmethod
    def _make_preds_df(
        dataset: TSDataset, horizon: int, history: int, id_column_name: Optional[str] = None
    ) -> pd.DataFrame:
        if id_column_name is None:
            id_column_name = dataset.id_column

        columns_list = [id_column_name, dataset.date_column, dataset.target_column]

        index_slicer = IndexSlicer()
        # Get dataframe with predictions only
        target_ids = index_slicer.create_idx_target(
            data=dataset.data,
            horizon=horizon,
            history=history,
            step=horizon + history,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )

        columns_ids = index_slicer.get_cols_idx(dataset.data, columns_list)
        data = index_slicer.get_slice(dataset.data, (target_ids, columns_ids))
        pred_df = pd.DataFrame(np.vstack(data), columns=columns_list)
        return pred_df

    @staticmethod
    def _backtest_generator(dataset: TSDataset, cv: int, horizon: int):
        index_slicer = IndexSlicer()
        segments_ids = index_slicer.ids_from_date(
            dataset.data, dataset.date_column, delta=dataset.delta
        )
        segments_ids = [0] + segments_ids + [len(dataset.data)]

        for val_idx in range(cv):
            full_train = np.array([])
            full_test = np.array([])

            for i in range(len(segments_ids) - 1):
                if len(full_train) > 0:
                    full_train = np.vstack(
                        (
                            full_train,
                            np.arange(
                                segments_ids[i],
                                segments_ids[i + 1] - horizon * (val_idx + 1),
                            ),
                        )
                    )
                    full_test = np.vstack(
                        (
                            full_test,
                            np.arange(
                                segments_ids[i + 1] - horizon * (val_idx + 1),
                                segments_ids[i + 1] - horizon * (val_idx),
                            ),
                        )
                    )
                else:
                    full_train = np.arange(
                        segments_ids[i], segments_ids[i + 1] - horizon * (val_idx + 1)
                    )
                    full_test = np.arange(
                        segments_ids[i + 1] - horizon * (val_idx + 1),
                        segments_ids[i + 1] - horizon * (val_idx),
                    )

            yield (full_train, full_test)

    def make_step(self, dataset: TSDataset):
        raise NotImplementedError()

    @timing_decorator
    def fit(self, dataset: TSDataset):
        raise NotImplementedError()

    def back_test(
        self, dataset: TSDataset, cv: int
    ) -> Union[List, np.ndarray]:
        ids_list = []
        test_list = []
        preds_list = []
        fit_time_list = []
        forecast_time_list = []
        num_iterations_list = []

        for train_idx, test_idx in self._backtest_generator(dataset, cv, self.horizon):
            current_train = dataset.data.iloc[train_idx.reshape(-1)]
            current_test = dataset.data.iloc[test_idx.reshape(-1)]
            current_dataset = TSDataset(
                current_train,
                dataset.columns_params,
                dataset.history,
                dataset.step,
            )

            fit_time, _ = self.fit(current_dataset)
            forecast_time, current_pred = self.predict(current_dataset)

            test_list.append(np.asarray(current_test[dataset.target_column].values))
            preds_list.append(np.asarray(current_pred[dataset.target_column].values))
            fit_time_list.append(fit_time)
            forecast_time_list.append(forecast_time)

            if dataset.data[dataset.id_column].nunique() > 1:
                ids_list.append(np.asarray(current_pred[dataset.id_column].values))
        return (
            ids_list,
            test_list,
            preds_list,
            fit_time_list,
            forecast_time_list,
            num_iterations_list,
        )

    @timing_decorator
    def predict(self, dataset: TSDataset) -> np.ndarray:
        raise NotImplementedError()
