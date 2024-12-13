from copy import deepcopy
from typing import Union

import pandas as pd

from ..dataset import IndexSlicer, Pipeline, TSDataset
from ..model_training.trainer import DLTrainer, MLTrainer
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
        trainer: trainer with model params and validation params.
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
        trainer: Union[MLTrainer, DLTrainer],
        pipeline: Pipeline,
        step: int = 1,
        model_horizon: int = 1,
        reduced: bool = False,
    ):
        super().__init__(horizon, history, trainer, pipeline, step)
        self.model_horizon = model_horizon
        self.reduced = reduced
        self.strategy_name = "recursive"

    @timing_decorator
    def fit(
        self,
        dataset: TSDataset,
    ) -> "RecursiveStrategy":
        """Fits the recursive strategy to the given dataset.

        Args:
            dataset: The dataset to fit the strategy on.

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

        val_dataset = self.trainer.validation_params.get("validation_data")

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

        if isinstance(self.trainer, DLTrainer):
            if self.strategy_name == "FlatWideMIMOStrategy":
                self.trainer.horizon = 1
            else:
                self.trainer.horizon = self.model_horizon
            self.trainer.history = self.history

        current_trainer = deepcopy(self.trainer)

        # In Recursive strategy, we train the individual model
        if isinstance(current_trainer, DLTrainer):
            checkpoint_path = current_trainer.checkpoint_path
            pretrained_path = current_trainer.pretrained_path

            current_trainer.checkpoint_path /= "trainer_0"
            if pretrained_path:
                current_trainer.pretrained_path /= "trainer_0"

        current_trainer.fit(data, self.pipeline, val_data)

        if isinstance(current_trainer, DLTrainer):
            current_trainer.checkpoint_path = checkpoint_path
            current_trainer.pretrained_path = pretrained_path

        self.trainers.append(current_trainer)
        
        self.is_fitted = True
        
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
        if not self.is_fitted:
            raise ValueError("The strategy is not fitted yet.")
        
        new_data = dataset.make_padded_test(
            self.horizon, self.history, test_all=test_all, step=self.step
        )
        new_dataset = TSDataset(new_data, dataset.columns_params, dataset.delta)

        if test_all:
            new_dataset.data = new_dataset.data.sort_values(
                [dataset.id_column, "segment_col", dataset.date_column]
            )

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
