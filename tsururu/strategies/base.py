import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ..dataset.dataset import TSDataset
from ..dataset.pipeline import Pipeline
from ..dataset.slice import IndexSlicer
from ..model_training.trainer import DLTrainer, MLTrainer
from ..utils.logging import set_stdout_level, verbosity_to_loglevel
from .utils import timing_decorator

logger = logging.getLogger(__name__)


class Strategy:
    """Base class for strategies, that are needed for fitting and
        inference of base models.

    Args:
        horizon: forecast horizon.
        history: number of previous for feature generating
            (i.e., features for observation y_t are counted from
            observations (y_{t-history}, ..., y_{t-1}).
        trainer: trainer with model params and validation params.
        pipeline: pipeline for feature and target generation.
        step:  in how many points to take the next observation while
            making samples' matrix.

    Notes:
        1. A type of strategy defines what features and targets will be
        built for subsequent training and inference of the base model.
        2. Now the `step` param should be 1. It will be changed in the
        future.

    """

    @staticmethod
    def set_verbosity_level(verbose):
        """Verbosity level setter.

        Args:
            verbose: Controls the verbosity: the higher, the more messages.
                <1  : messages are not displayed;
                >=1 : the common information about training and testing is displayed;
                >=2 : the information about folds processing is also displayed;
                >=3 : the training process for every algorithm is displayed;
                >=4 : the debug information is displayed;
        """
        level = verbosity_to_loglevel(verbose)
        set_stdout_level(level)

        logger.info(f"Stdout logging level is {logging._levelToName[level]}.")

    @staticmethod
    def check_step_param(step: int):
        """Check if the given step parameter is valid.

        Args:
            step: the step parameter to be checked.

        Raises:
            AssertionError: if the step parameter is not equal to 1.

        """
        assert step == 1, "Step should be 1. It will be changed in the future."

    def __init__(
        self,
        horizon: int,
        history: int,
        trainer: Union[MLTrainer, DLTrainer],
        pipeline: Pipeline,
        step: int = 1,
    ):
        self.check_step_param(step)

        self.horizon = horizon
        self.history = history
        self.step = step
        self.trainer = trainer
        self.pipeline = pipeline

        self.trainers = []
        self.is_fitted = False

    @staticmethod
    def _make_preds_df(
        dataset: TSDataset, horizon: int, history: int, id_column_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Create a DataFrame with predictions based on the given
            dataset.

        Args:
            dataset: the input time series dataset.
            horizon: forecast horizon.
            history: number of previous for feature generating
            (i.e., features for observation y_t are counted from
            observations (y_{t-history}, ..., y_{t-1}).
                historical data.
            id_column_name: the name of the column containing the IDs.
                If not provided, the ID column name from the dataset
                will be used. Defaults to None.

        Returns:
            A DataFrame with the predicted values,
            including the ID, date, and target columns.

        """
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
        """Generate train-test splits for cross-validation.

        Args:
            dataset: the time series dataset.
            cv: the number of cross-validation folds.
            horizon: the forecast horizon.

        Yields:
            a tuple containing the train and test indices for each fold.

        """
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
        """Make a step in the strategy.

        Args:
            step: the step number.
            dataset: the dataset to make the step on.

        Returns:
            the updated dataset.

        """
        raise NotImplementedError()

    @timing_decorator
    def fit(
        self,
        dataset: TSDataset,
    ):
        """Fits the strategy to the given dataset.

        Args:
            dataset: The dataset to fit the strategy on.

        Returns:
            self.

        """
        raise NotImplementedError()

    def back_test(self, dataset: TSDataset, cv: int) -> Union[List, np.ndarray]:
        """Perform backtesting on the given dataset using
        cross-validation.

        Args:
            dataset: the dataset to perform backtesting on.
            cv: the number of cross-validation folds.

        Returns:
            a tuple containing the following lists:
                ids_list: IDs of the predictions.
                test_list: actual test values.
                preds_list: predicted values.
                fit_time_list: fit times for each fold.
                forecast_time_list: forecast times for each fold.

        """
        ids_list = []
        test_list = []
        preds_list = []
        fit_time_list = []
        forecast_time_list = []

        for train_idx, test_idx in self._backtest_generator(dataset, cv, self.horizon):
            current_train = dataset.data.iloc[train_idx.reshape(-1)]
            current_test = dataset.data.iloc[test_idx.reshape(-1)]
            current_dataset = TSDataset(current_train, dataset.columns_params, dataset.delta)

            fit_time, _ = self.fit(current_dataset)
            forecast_time, current_pred = self.predict(current_dataset)

            test_list.append(np.asarray(current_test[dataset.target_column].values))
            preds_list.append(np.asarray(current_pred[dataset.target_column].values))
            fit_time_list.append(fit_time)
            forecast_time_list.append(forecast_time)

            if dataset.data[dataset.id_column].nunique() > 1:
                ids_list.append(np.asarray(current_pred[dataset.id_column].values))

        return (ids_list, test_list, preds_list, fit_time_list, forecast_time_list)

    @timing_decorator
    def predict(self, dataset: TSDataset, test_all: bool = False) -> np.ndarray:
        """Predicts the target values for the given dataset.

        Args:
            dataset: the dataset to make predictions on.
            test_all: whether to predict all the target values
                (like rolling forecast) or only the last one.

        Returns:
            a pandas DataFrame containing the predicted target values.

        """
        raise NotImplementedError()
