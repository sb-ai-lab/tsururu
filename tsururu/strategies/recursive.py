from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd

from tsururu.dataset.dataset import TSDataset
from tsururu.dataset.pipeline import Pipeline
from tsururu.dataset.slice import IndexSlicer
from tsururu.model_training.trainer import DLTrainer, MLTrainer
from tsururu.strategies.base import Strategy
from tsururu.strategies.utils import timing_decorator

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
        subsampling_rate: float = 1.0,
        subsampling_seed: int = 42,
    ) -> "RecursiveStrategy":
        """Fits the recursive strategy to the given dataset.

        Args:
            dataset: The dataset to fit the strategy on.
            subsampling_rate: The rate at which to subsample the data for training.
                A value of 1.0 means no subsampling.

        Returns:
            self.

        """
        if self.is_fitted:
            raise RuntimeError("The strategy is already fitted!")
        
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

        if subsampling_rate < 1.0:
            all_idx = np.arange(features_idx.shape[0])
            np.random.seed(subsampling_seed)
            sampled_idx = np.random.choice(
                all_idx, size=int(subsampling_rate * len(all_idx)), replace=False
            )
            features_idx = features_idx[sampled_idx]
            target_idx = target_idx[sampled_idx]

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

    def make_step(
        self, step: int, horizon: int, dataset: TSDataset, inverse_transform: bool
    ) -> TSDataset:
        """Make a step in the recursive strategy.

        Args:
            step: the step number.
            horizon: the horizon length.
            dataset: the dataset to make the step on.

        Returns:
            the updated dataset.

        """
        assert horizon % self.model_horizon == 0

        test_idx = index_slicer.create_idx_test(
            dataset.data,
            horizon - step * self.model_horizon,
            self.history,
            self.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )

        target_idx = index_slicer.create_idx_target(
            dataset.data,
            horizon,
            self.history,
            self.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )[:, self.model_horizon * step : self.model_horizon * (step + 1)]

        data = self.pipeline.create_data_dict_for_pipeline(dataset, test_idx, target_idx)
        data = self.pipeline.transform(data)

        pred = self.trainers[0].predict(data, self.pipeline)
        if inverse_transform:
            pred = self.pipeline.inverse_transform_y(pred)

        num_series = data["num_series"] if self.pipeline.multivariate else 1

        target_idx = target_idx.reshape(num_series, -1, self.model_horizon)
        pred = pred.reshape(num_series, -1, self.model_horizon)

        target_idx = target_idx[:, : pred.shape[1]]

        dataset.data.loc[target_idx.reshape(-1), dataset.target_column] = pred.reshape(-1)

        return dataset

    @timing_decorator
    def predict(
        self,
        dataset: TSDataset,
        horizon: int | None = None,
        test_all: bool = False,
        inverse_transform: bool = True,
    ) -> pd.DataFrame:
        """Predicts the target values for the given dataset.

        Args:
            dataset (TSDataset): the dataset to make predictions on.
            horizon (int, optional): number of steps ahead to predict. If None, defaults to the model's training horizon.
            test_all (bool, default=False): if True, performs rolling window prediction over the entire dataset.
                Otherwise, predicts only the last window.
            inverse_transform (bool, default=True): if True, applies inverse transformations to the predictions
                (e.g., reversing normalization/scaling).

        Returns:
            a pandas DataFrame containing the predicted target values.

        """
        if not self.is_fitted:
            raise ValueError("The strategy is not fitted yet.")

        if horizon is None:
            horizon = self.horizon

        # intrinsic_horizon is a multiple of model_horizon
        intrinsic_horizon = self.model_horizon * (
            (horizon + self.model_horizon - 1) // self.model_horizon
        )

        new_data = dataset.make_padded_test(
            intrinsic_horizon, self.history, test_all=test_all, step=self.step
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
                intrinsic_horizon,
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
            if inverse_transform:
                pred = self.pipeline.inverse_transform_y(pred)

            new_dataset.data.loc[target_ids.reshape(-1), dataset.target_column] = pred.reshape(-1)

        else:
            for step in range(intrinsic_horizon // self.model_horizon):
                new_dataset = self.make_step(
                    step, intrinsic_horizon, new_dataset, inverse_transform
                )

        # Get dataframe with predictions only
        pred_df = self._make_preds_df(new_dataset, intrinsic_horizon, self.history)
        return pred_df
