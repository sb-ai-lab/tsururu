from typing import Dict, Union

from .utils import timing_decorator
from .recursive import RecursiveStrategy
from ..dataset.slice import IndexSlicer
from ..models import ModelsFactory

from ..dataset import IndexSlicer, Pipeline, TSDataset
from ..models import Estimator


class DirRecStrategy(RecursiveStrategy):
    """A strategy that uses individual model for each point
        in the prediction horizon.

    Args:
        horizon: forecast horizon.
        history: number of previous for feature generating
            (i.e., features for observation y_t are counted from observations
            (y_{t-history}, ..., y_{t-1}).
        step:  in how many points to take the next observation while making
            samples' matrix.
        model: base model.
        pipeline: pipeline for feature and target generation.
        is_multivariate: whether the prediction mode is multivariate.

    Notes:
        1. Fit: mixture of DirectStrategy and RecursiveStrategy, fit
            individual models, but at each step expand history window.
        2. Inference: at each step makes a prediction one point ahead,
            and then uses this prediction to further generate features
            for subsequent models along with new observations.

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
        self.true_lags = {}
        self.strategy_name = "DirRecStrategy"

    @timing_decorator
    def fit(self, dataset):
        self.models = []  # len = amount of forecast points

        # Save true lags for each feature
        self.true_lags = {
            column_name: column_dict["features"]["LagTransformer"]["lags"]
            for (
                column_name,
                column_dict,
            ) in dataset.columns_params.items()
            if column_dict.get("features") and column_dict["features"].get("LagTransformer")
        }

        for horizon in range(1, self.horizon + 1):
            X, y = self._generate_X_y(
                dataset,
                train_horizon=1,
                target_horizon=horizon,
                history=dataset.history + (horizon - 1),
                n_last_horizon=1,
                is_train=True,
            )

            # Update lags for each feature
            for column_name in self.true_lags.keys():
                dataset.columns_params[column_name]["features"]["LagTransformer"][
                    "lags"
                ] += 1

            factory = ModelsFactory()
            model_params = {
                "model_name": self.model_name,
                "validation_params": self.validation_params,
                "model_params": self.model_params,
            }
            current_model = factory[model_params]
            current_model.fit(X, y)
            self.models.append(current_model)

        # Return true lags
        for column_name in self.true_lags.keys():
            dataset.columns_params[column_name]["features"]["LagTransformer"][
                "lags"
            ] = self.true_lags[column_name]
        return self

    def make_step(self, step, dataset, _):
        index_slicer = IndexSlicer()
        current_test_ids = index_slicer.create_idx_test(
            dataset.data,
            self.horizon - step,
            dataset.history + step,
            dataset.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )
        current_X, _ = self._generate_X_y(
            dataset,
            train_horizon=1,
            target_horizon=1,
            is_train=False,
            idx=current_test_ids,
            X_only=True,
        )
        if self.is_multivariate:
            current_X = self._make_multivariate_X_y(current_X, date_column=dataset.date_column)
        current_pred = self.models[step].predict(current_X)

        # Update lags for each feature
        for column_name in self.true_lags.keys():
            dataset.columns_params[column_name]["features"]["LagTransformer"][
                "lags"
            ] += 1

        dataset.data.loc[
            step + dataset.history::dataset.history + self.horizon,
            dataset.target_column,
        ] = current_pred.reshape(-1)
        dataset.data.loc[
            step + dataset.history::dataset.history + self.horizon
        ] = self._inverse_transform_y(
            dataset.data.loc[step + dataset.history::dataset.history + self.horizon]
        )
        return dataset
