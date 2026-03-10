import pytest
import numpy as np
import pandas as pd

from tsururu.dataset import IndexSlicer, Pipeline, TSDataset
from tsururu.model_training import DLTrainer, KFoldCrossValidator, MLTrainer
from tsururu.model_training.validator import Validator
from tsururu.models import CatBoost, Estimator
from tsururu.strategies.base import Strategy
from tsururu.strategies.utils import timing_decorator
from tsururu.utils.optional_imports import OptionalImport
from tsururu.strategies import (
    RecursiveStrategy,
    MIMOStrategy,
    FlatWideMIMOStrategy,
    DirectStrategy,
)

df_path = "../../datasets/global/AirPassengers.csv"
df = pd.read_csv(df_path)

TARGET_COL = "Passengers"
TIME_COL = "Month"
ID_COL = "id"

HORIZON = 4
HISTORY = 7

dataset_params = {
    "target": {"columns": [TARGET_COL], "type": "continuous"},
    "date": {"columns": [TIME_COL], "type": "datetime"},
    "id": {"columns": [ID_COL], "type": "categorical"},
}

dataset = TSDataset(data=df, columns_params=dataset_params, print_freq_period_info=True)

pipeline_params = {
    "target": {
        "columns": [TARGET_COL],
        "features": {
            "LagTransformer": {"lags": HISTORY},
        },
    },
    "date": {
        "columns": [TIME_COL],
        "features": {"DateSeasonsGenerator": {}, "LagTransformer": {"lags": HORIZON}},
    },
    "id": {
        "columns": [ID_COL],
        "features": {
            "LagTransformer": {"lags": 1},
        },
    },
}

pipeline = Pipeline.from_dict(pipeline_params, multivariate=False)

model = CatBoost
model_params = {
    "loss_function": "MultiRMSE",
    "early_stopping_rounds": 100,
    "verbose": 500,
}
validation = KFoldCrossValidator
validation_params = {"n_splits": 2}

trainer = MLTrainer(
    model, model_params, validation, validation_params, return_importance=True
)


@pytest.fixture(scope="session", autouse=True)
def dataset_fix():
    return dataset


@pytest.fixture(scope="function")
def fitted_strategy(dataset_fix, strategy_class):
    strategy = strategy_class(HORIZON, HISTORY, trainer, pipeline)
    fit_time, _ = strategy.fit(dataset_fix)
    assert fit_time is not None
    return strategy


@pytest.mark.parametrize(
    "strategy_class", [RecursiveStrategy, MIMOStrategy, FlatWideMIMOStrategy]
)
class TestCommonStrategies:

    def test_fit(self, dataset_fix, strategy_class):
        strategy = strategy_class(HORIZON, HISTORY, trainer, pipeline)
        fit_time, _ = strategy.fit(dataset_fix)
        assert fit_time is not None

    def test_predict(self, fitted_strategy, dataset_fix, strategy_class):
        forecast_time, pred = fitted_strategy.predict(dataset_fix)
        assert forecast_time is not None
        assert pred is not None

    @pytest.mark.parametrize(
        "aggregate_by_folds, return_explainer",
        [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ],
    )
    def test_feature_importance_combinations(
        self,
        fitted_strategy,
        dataset_fix,
        strategy_class,
        aggregate_by_folds,
        return_explainer,
    ):
        explainer = fitted_strategy.get_feature_importance(
            top_k=3,
            aggregate_by_folds=aggregate_by_folds,
            return_explainer=return_explainer,
            round_to=4,
        )

    def test_full_pipeline(self, dataset_fix, strategy_class):
        strategy = strategy_class(HORIZON, HISTORY, trainer, pipeline)
        strategy.fit(dataset_fix)
        forecast_time, pred = strategy.predict(dataset_fix)

        for agg in [True, False]:
            for expl in [True, False]:
                result = strategy.get_feature_importance(
                    top_k=3, aggregate_by_folds=agg, return_explainer=expl, round_to=4
                )

        train_shap = strategy.get_train_shap()
        test_shap = strategy.get_test_shap()

    @pytest.mark.smoke
    def test_smoke(self, dataset_fix, strategy_class):
        strategy = strategy_class(HORIZON, HISTORY, trainer, pipeline)
        strategy.fit(dataset_fix)
        strategy.predict(dataset_fix)
        strategy.get_feature_importance(top_k=3)
        strategy.get_train_shap()
        strategy.get_test_shap()


@pytest.fixture(scope="function")
def fitted_direct_strategy(dataset_fix):
    strategy = DirectStrategy(HORIZON, HISTORY, trainer, pipeline, model_horizon=2)
    fit_time, _ = strategy.fit(dataset_fix)
    assert fit_time is not None
    return strategy


class TestDirectStrategy:

    def test_fit(self, dataset_fix):
        strategy = DirectStrategy(HORIZON, HISTORY, trainer, pipeline, model_horizon=2)
        fit_time, _ = strategy.fit(dataset_fix)
        assert fit_time is not None

    def test_predict(self, fitted_direct_strategy, dataset_fix):
        forecast_time, pred = fitted_direct_strategy.predict(dataset_fix)
        assert forecast_time is not None
        assert pred is not None

    @pytest.mark.parametrize(
        "aggregate_by_folds, return_explainer",
        [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ],
    )
    def test_feature_importance_combinations(
        self, fitted_direct_strategy, aggregate_by_folds, return_explainer
    ):
        explainer = fitted_direct_strategy.get_feature_importance(
            top_k=3,
            aggregate_by_folds=aggregate_by_folds,
            return_explainer=return_explainer,
            round_to=4,
        )

    def test_full_pipeline(self, dataset_fix):
        strategy = DirectStrategy(HORIZON, HISTORY, trainer, pipeline, model_horizon=2)
        strategy.fit(dataset_fix)
        forecast_time, pred = strategy.predict(dataset_fix)

        for agg in [True, False]:
            for expl in [True, False]:
                result = strategy.get_feature_importance(
                    top_k=3, aggregate_by_folds=agg, return_explainer=expl, round_to=4
                )

        train_shap = strategy.get_train_shap()
        test_shap = strategy.get_test_shap()

    @pytest.mark.smoke
    def test_smoke(self, dataset_fix):
        strategy = DirectStrategy(HORIZON, HISTORY, trainer, pipeline, model_horizon=2)
        strategy.fit(dataset_fix)
        strategy.predict(dataset_fix)
        strategy.get_feature_importance(top_k=3)
        strategy.get_train_shap()
        strategy.get_test_shap()
