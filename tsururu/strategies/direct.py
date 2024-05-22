from copy import deepcopy

from ..dataset.dataset import TSDataset
from ..dataset.pipeline import Pipeline
from ..dataset.slice import IndexSlicer
from ..models import Estimator
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
        step:  in how many points to take the next observation while making
            samples' matrix.
        model: base model.
        pipeline: pipeline for feature and target generation.
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
        step: int,
        model: Estimator,
        pipeline: Pipeline,
        model_horizon: int = 1,
        equal_train_size: bool = False,
    ):
        super().__init__(horizon, history, step, model, pipeline, model_horizon)
        self.equal_train_size = equal_train_size
        self.strategy_name = "direct"

    @timing_decorator
    def fit(self, dataset: TSDataset) -> "DirectStrategy":
        """Fits the direct strategy to the given dataset.

        Args:
            dataset: The dataset to fit the strategy on.

        Returns:
            self.

        """
        self.models = []

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

            for horizon in range(1, self.horizon // self.model_horizon + 1):
                target_idx = index_slicer.create_idx_target(
                    dataset.data,
                    self.horizon,
                    self.history,
                    self.step,
                    date_column=dataset.date_column,
                    delta=dataset.delta,
                )[:, (horizon - 1) * self.model_horizon : horizon * self.model_horizon]

                data["target_idx"] = target_idx

                current_model = deepcopy(self.model)
                current_model.fit(data, self.pipeline)
                self.models.append(current_model)

        else:
            for horizon in range(1, self.horizon // self.model_horizon + 1):
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

                current_model = deepcopy(self.model)
                current_model.fit(data, self.pipeline)
                self.models.append(current_model)

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

        pred = self.models[step].predict(data, self.pipeline)
        pred = self.pipeline.inverse_transform_y(pred)

        dataset.data.loc[target_idx.reshape(-1), dataset.target_column] = pred.reshape(-1)

        return dataset
