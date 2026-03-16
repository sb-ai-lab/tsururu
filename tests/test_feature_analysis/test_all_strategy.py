import pandas as pd
import pytest

from tsururu.dataset import Pipeline, TSDataset
from tsururu.model_training import KFoldCrossValidator, MLTrainer
from tsururu.models import CatBoost
from tsururu.strategies import (
    DirectStrategy,
    FlatWideMIMOStrategy,
    MIMOStrategy,
    RecursiveStrategy,
)

df = pd.read_csv("datasets/global/AirPassengers.csv")

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
        _ = fitted_strategy.get_feature_importance(
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
                _ = strategy.get_feature_importance(
                    top_k=3, aggregate_by_folds=agg, return_explainer=expl, round_to=4
                )

        _ = strategy.get_train_shap()
        _ = strategy.get_test_shap()

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
        _ = fitted_direct_strategy.get_feature_importance(
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
                _ = strategy.get_feature_importance(
                    top_k=3, aggregate_by_folds=agg, return_explainer=expl, round_to=4
                )

        _ = strategy.get_train_shap()
        _ = strategy.get_test_shap()

    @pytest.mark.smoke
    def test_smoke(self, dataset_fix):
        strategy = DirectStrategy(HORIZON, HISTORY, trainer, pipeline, model_horizon=2)
        strategy.fit(dataset_fix)
        strategy.predict(dataset_fix)
        strategy.get_feature_importance(top_k=3)
        strategy.get_train_shap()
        strategy.get_test_shap()
