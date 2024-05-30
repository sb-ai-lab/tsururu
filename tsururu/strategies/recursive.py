from typing import Optional

import pandas as pd

from ..dataset import IndexSlicer, Pipeline, TSDataset
from ..model_training.trainer import MLTrainer, DLTrainer
from ..model_training.validator import Validator
from ..models.base import Estimator
from .base import Strategy
from .utils import timing_decorator

index_slicer = IndexSlicer()


class RecursiveStrategy(Strategy):
    """Strategy that uses a single model to predict all points in the
        forecast horizon.

    Arguments:
        horizon: forecast horizon.
        history: number of previous for feature generating
            (i.e., features for observation y_t are counted from
            observations (y_{t-history}, ..., y_{t-1}).
        step:  in how many points to take the next observation while
            making samples' matrix.
        model: base model.
        validator: validator for model training.
        pipeline: pipeline for feature and target generation.
        model_horizon: how many points to predict at a time,
            if model_horizon > 1, then it's an intermediate strategy
            between RecursiveStrategy and MIMOStrategy.
        reduced: whether to form features for all test observations at
            once, in this case, unavailable values are replaced by NaN.

    Notes:
        1. Fit: the model is fitted to predict one point ahead.
        2. Inference: the model iteratively predicts the next point and
            - use this prediction to build further features
                (`reduced` == False);
            - use NaN instead of prediction (`reduced` == True).

    """

    def __init__(
        self,
        horizon: int,
        history: int,
        step: int,
        model: Estimator,
        validator: Validator,
        pipeline: Pipeline,
        model_horizon: int = 1,
        reduced: bool = False,
    ):
        super().__init__(horizon, history, step, model, validator, pipeline)
        self.model_horizon = model_horizon
        self.reduced = reduced
        self.strategy_name = "recursive"

    @timing_decorator
    def fit(
        self,
        dataset: TSDataset,
        val_dataset: Optional[TSDataset] = None,
        trainer_params: dict = {},
    ) -> "RecursiveStrategy":
        """Fits the recursive strategy to the given dataset.

        Args:
            dataset: The dataset to fit the strategy on.
            val_dataset: The validation dataset.
            trainer_params: Additional parameters for the trainer.

        Returns:
            self.

        """
        features_idx = index_slicer.create_idx_data(
            dataset.data,
            self.model_horizon,
            self.history,
            self.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )

        target_idx = index_slicer.create_idx_target(
            dataset.data,
            self.model_horizon,
            self.history,
            self.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )

        data = self.pipeline.create_data_dict_for_pipeline(dataset, features_idx, target_idx)
        data = self.pipeline.fit_transform(data, self.strategy_name)

        if val_dataset:
            val_features_idx = index_slicer.create_idx_data(
                val_dataset.data,
                self.model_horizon,
                self.history,
                self.step,
                date_column=dataset.date_column,
                delta=dataset.delta,
            )

            val_target_idx = index_slicer.create_idx_target(
                val_dataset.data,
                self.model_horizon,
                self.history,
                self.step,
                date_column=dataset.date_column,
                delta=dataset.delta,
            )

            val_data = self.pipeline.create_data_dict_for_pipeline(
                val_dataset, val_features_idx, val_target_idx
            )
            val_data = self.pipeline.transform(val_data)
        else:
            val_data = None

        if self.model.trainer_type == "MLTrainer":
            trainer = MLTrainer(self.model, self.validator, **trainer_params)
            trainer.fit(data, self.pipeline, val_data)

        elif self.model.trainer_type == "DLTrainer":
            trainer = DLTrainer(self.model, self.validator, **trainer_params)
            trainer.fit(data, self.pipeline, val_data)

        self.trainers.append(trainer)
        return self

    def make_step(self, step: int, dataset: TSDataset) -> TSDataset:
        """Make a step in the recursive strategy.

        Args:
            step: the step number.
            dataset: the dataset to make the step on.

        Returns:
            the updated dataset.

        """
        test_idx = index_slicer.create_idx_test(
            dataset.data,
            self.horizon - step * self.model_horizon,
            self.history,
            self.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )

        target_idx = index_slicer.create_idx_target(
            dataset.data,
            self.horizon,
            self.history,
            self.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )[:, self.model_horizon * step : self.model_horizon * (step + 1)]

        data = self.pipeline.create_data_dict_for_pipeline(dataset, test_idx, target_idx)
        data = self.pipeline.transform(data)

        pred = self.trainers[0].predict(data, self.pipeline)
        pred = self.pipeline.inverse_transform_y(pred)

        dataset.data.loc[target_idx.reshape(-1), dataset.target_column] = pred.reshape(-1)

        return dataset

    @timing_decorator
    def predict(self, dataset: TSDataset, test_all: bool = False) -> pd.DataFrame:
        """Predicts the target values for the given dataset.

        Args:
            dataset: the dataset to make predictions on.

        Returns:
            a pandas DataFrame containing the predicted target values.

        """
        new_data = dataset.make_padded_test(
            self.horizon, self.history, test_all, self.model_horizon
        )
        new_dataset = TSDataset(new_data, dataset.columns_params, dataset.delta)

        if self.reduced:
            current_test_ids = index_slicer.create_idx_data(
                new_dataset.data,
                self.model_horizon,
                self.history,
                step=self.model_horizon,
                date_column=dataset.date_column,
                delta=dataset.delta,
            )

            target_ids = index_slicer.create_idx_target(
                new_dataset.data,
                self.horizon,
                self.history,
                step=self.model_horizon,
                date_column=dataset.date_column,
                delta=dataset.delta,
            )

            data = self.pipeline.create_data_dict_for_pipeline(
                new_dataset, current_test_ids, target_ids
            )
            data = self.pipeline.transform(data)

            pred = self.trainers[0].predict(data, self.pipeline)
            pred = self.pipeline.inverse_transform_y(pred)

            new_dataset.data.loc[target_ids.reshape(-1), dataset.target_column] = pred.reshape(-1)

        else:
            for step in range(self.horizon // self.model_horizon):
                new_dataset = self.make_step(step, new_dataset)

        # Get dataframe with predictions only
        pred_df = self._make_preds_df(new_dataset, self.horizon, self.history)
        return pred_df
