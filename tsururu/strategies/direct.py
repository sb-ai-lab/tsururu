from copy import deepcopy
from typing import Union

from ..dataset.dataset import TSDataset
from ..dataset.pipeline import Pipeline
from ..dataset.slice import IndexSlicer
from ..model_training.trainer import DLTrainer, MLTrainer
from .recursive import RecursiveStrategy
from .utils import timing_decorator

index_slicer = IndexSlicer()


class DirectStrategy(RecursiveStrategy):
    """A strategy that uses an individual model for each point in the
        forecast horizon.

    Args:
        horizon: forecast horizon.
        history: number of previous for feature generating
            (i.e., features for observation y_t are counted from observations
            (y_{t-history}, ..., y_{t-1}).
        trainer: trainer with model params and validation params.
        pipeline: pipeline for feature and target generation.
        step:  in how many points to take the next observation while making
            samples' matrix.
        model_horizon: how many points to predict at a time,
            if model_horizon > 1, then it's an intermediate strategy between
            RecursiveStrategy and MIMOStrategy.
        equal_train_size: if true, all models are trained with the same
            training sample (which is equal to the training sample
            of the last model if equal_train_size=false).

    Notes:
        1. Fit: the models is fitted to predict certain point in the
            forecasting horizon (number of models = horizon).
        2. Inference: each model predict one point.

    """

    def __init__(
        self,
        horizon: int,
        history: int,
        trainer: Union[MLTrainer, DLTrainer],
        pipeline: Pipeline,
        step: int = 1,
        model_horizon: int = 1,
        equal_train_size: bool = False,
    ):
        super().__init__(horizon, history, trainer, pipeline, step, model_horizon)
        self.equal_train_size = equal_train_size
        self.strategy_name = "direct"

    @timing_decorator
    def fit(
        self,
        dataset: TSDataset,
    ) -> "DirectStrategy":
        """Fits the direct strategy to the given dataset.

        Args:
            dataset: The dataset to fit the strategy on.

        Returns:
            self.

        """
        self.trainers = []

        if self.equal_train_size:
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
                    date_column=val_dataset.date_column,
                    delta=val_dataset.delta,
                )

                val_target_idx = index_slicer.create_idx_target(
                    val_dataset.data,
                    self.model_horizon,
                    self.history,
                    self.step,
                    date_column=val_dataset.date_column,
                    delta=val_dataset.delta,
                )

                val_data = self.pipeline.create_data_dict_for_pipeline(
                    val_dataset, val_features_idx, val_target_idx
                )
                val_data = self.pipeline.transform(val_data)
            else:
                val_data = None

            for model_i, horizon in enumerate(range(1, self.horizon // self.model_horizon + 1)):
                target_idx = index_slicer.create_idx_target(
                    dataset.data,
                    self.horizon,
                    self.history,
                    self.step,
                    date_column=dataset.date_column,
                    delta=dataset.delta,
                )[:, (horizon - 1) * self.model_horizon : horizon * self.model_horizon]

                data["target_idx"] = target_idx

                if val_dataset:
                    val_target_idx = index_slicer.create_idx_target(
                        val_dataset.data,
                        self.horizon,
                        self.history,
                        self.step,
                        date_column=val_dataset.date_column,
                        delta=val_dataset.delta,
                    )[:, (horizon - 1) * self.model_horizon : horizon * self.model_horizon]

                    val_data["target_idx"] = val_target_idx

                if isinstance(self.trainer, DLTrainer):
                    self.trainer.horizon = self.model_horizon
                    self.trainer.history = self.history

                current_trainer = deepcopy(self.trainer)

                # In Direct strategy, we train several models, one for each model_horizon
                if isinstance(current_trainer, DLTrainer):
                    checkpoint_path = current_trainer.checkpoint_path
                    pretrained_path = current_trainer.pretrained_path

                    current_trainer.checkpoint_path /= f"trainer_{model_i}"
                    if pretrained_path:
                        current_trainer.pretrained_path /= f"trainer_{model_i}"

                current_trainer.fit(data, self.pipeline, val_data)

                if isinstance(current_trainer, DLTrainer):
                    current_trainer.checkpoint_path = checkpoint_path
                    current_trainer.pretrained_path = pretrained_path

                self.trainers.append(current_trainer)

        else:
            for model_i, horizon in enumerate(range(1, self.horizon // self.model_horizon + 1)):
                features_idx = index_slicer.create_idx_data(
                    dataset.data,
                    self.model_horizon * horizon,
                    self.history,
                    self.step,
                    date_column=dataset.date_column,
                    delta=dataset.delta,
                )

                target_idx = index_slicer.create_idx_target(
                    dataset.data,
                    self.model_horizon * horizon,
                    self.history,
                    self.step,
                    date_column=dataset.date_column,
                    delta=dataset.delta,
                    n_last_horizon=self.model_horizon,
                )

                data = self.pipeline.create_data_dict_for_pipeline(
                    dataset, features_idx, target_idx
                )
                data = self.pipeline.fit_transform(data, self.strategy_name)

                val_dataset = self.trainer.validation_params.get("validation_data")

                if val_dataset:
                    val_features_idx = index_slicer.create_idx_data(
                        val_dataset.data,
                        self.model_horizon * horizon,
                        self.history,
                        self.step,
                        date_column=val_dataset.date_column,
                        delta=val_dataset.delta,
                    )

                    val_target_idx = index_slicer.create_idx_target(
                        val_dataset.data,
                        self.model_horizon * horizon,
                        self.history,
                        self.step,
                        date_column=val_dataset.date_column,
                        delta=val_dataset.delta,
                        n_last_horizon=self.model_horizon,
                    )

                    val_data = self.pipeline.create_data_dict_for_pipeline(
                        val_dataset, val_features_idx, val_target_idx
                    )
                    val_data = self.pipeline.transform(val_data)
                else:
                    val_data = None

                if isinstance(self.trainer, DLTrainer):
                    self.trainer.horizon = self.model_horizon
                    self.trainer.history = self.history

                current_trainer = deepcopy(self.trainer)

                # In Direct strategy, we train several models, one for each model_horizon
                if isinstance(current_trainer, DLTrainer):
                    checkpoint_path = current_trainer.checkpoint_path
                    pretrained_path = current_trainer.pretrained_path

                    current_trainer.checkpoint_path /= f"trainer_{model_i}"
                    if pretrained_path:
                        current_trainer.pretrained_path /= f"trainer_{model_i}"

                current_trainer.fit(data, self.pipeline, val_data)

                if isinstance(current_trainer, DLTrainer):
                    current_trainer.checkpoint_path = checkpoint_path
                    current_trainer.pretrained_path = pretrained_path

                self.trainers.append(current_trainer)

        self.is_fitted = True

        return self

    def make_step(self, step, dataset):
        """Make a step in the direct strategy.

        Args:
            step: the step number.
            dataset: the dataset to make the step on.

        Returns:
            the updated dataset.

        """
        test_idx = index_slicer.create_idx_test(
            dataset.data,
            self.horizon,
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

        pred = self.trainers[step].predict(data, self.pipeline)
        pred = self.pipeline.inverse_transform_y(pred)

        dataset.data.loc[target_idx.reshape(-1), dataset.target_column] = pred.reshape(-1)

        return dataset
