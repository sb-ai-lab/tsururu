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
from tsururu.strategies import DirectStrategy


df_path = "/media/ssd-3t/dtsarenov/shap_research/tsururu/datasets/global/AirPassengers.csv"
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
        "features": {
            "DateSeasonsGenerator": {},
            "LagTransformer": {"lags": HORIZON}
        },
    },
    "id": {
        "columns": [ID_COL],
        "features": {
            "LagTransformer": {"lags": 1},
        },
    }
}

pipeline = Pipeline.from_dict(pipeline_params, multivariate=False)

model = CatBoost
model_params = {"loss_function": "MultiRMSE", "early_stopping_rounds": 100, "verbose": 500}
validation = KFoldCrossValidator
validation_params = {"n_splits": 2}

trainer = MLTrainer(model, model_params, validation, validation_params, return_importance=True)


@pytest.fixture(scope="module")
def dataset_fix():
    return dataset


@pytest.fixture(scope="module")
def fitted_strategy(dataset_fix):
    strategy = DirectStrategy(HORIZON, HISTORY, trainer, pipeline, model_horizon=2)
    fit_time, _ = strategy.fit(dataset_fix)
    assert fit_time is not None
    return strategy


def test_fit(dataset_fix):
    """Тест fit()."""
    strategy = DirectStrategy(HORIZON, HISTORY, trainer, pipeline, model_horizon=2)
    fit_time, _ = strategy.fit(dataset_fix)
    assert fit_time is not None
    print(f"Fit OK, time: {fit_time}s")


def test_predict(fitted_strategy, dataset_fix):
    """Тест predict()."""
    forecast_time, pred = fitted_strategy.predict(dataset_fix)
    assert forecast_time is not None
    assert pred is not None
    print(f"Predict OK, time: {forecast_time}s, pred: {getattr(pred, 'shape', 'N/A')}")


@pytest.mark.parametrize(
    "aggregate_by_folds, return_explainer, result", 
    [
        (False, False, None),
        (False, True, not None), 
        (True, False, None),
        (True, True, not None),
    ]
)
def test_feature_importance_combinations(fitted_strategy, aggregate_by_folds, return_explainer, result):
    """Тест всех комбинаций get_feature_importance()."""
    explainer = fitted_strategy.get_feature_importance(
        top_k=3, 
        aggregate_by_folds=aggregate_by_folds, 
        return_explainer=return_explainer, 
        round_to=4
    )
    print(f"Importance OK: agg={aggregate_by_folds}, expl={return_explainer}, type={type(result)}")


def test_train_shap(fitted_strategy):
    """Тест get_train_shap()."""
    result = fitted_strategy.get_train_shap()
    assert result is not None
    print(f"Train SHAP OK: {type(result)}")


def test_test_shap(fitted_strategy):
    """Тест get_test_shap()."""
    result = fitted_strategy.get_test_shap()
    assert result is not None
    print(f"Test SHAP OK: {type(result)}")


def test_full_pipeline(fitted_strategy, dataset_fix):
    """Полный пайплайн."""
    forecast_time, pred = fitted_strategy.predict(dataset_fix)
    
    for agg in [True, False]:
        for expl in [True, False]:
            result = fitted_strategy.get_feature_importance(
                top_k=3, aggregate_by_folds=agg, return_explainer=expl, round_to=4
            )
    
    train_shap = fitted_strategy.get_train_shap()
    test_shap = fitted_strategy.get_test_shap()
    
    print("FULL PIPELINE OK")
    print(f"   Predict: {getattr(pred, 'shape', 'N/A')}")
    print(f"   Train SHAP: {type(train_shap)}")
    print(f"   Test SHAP: {type(test_shap)}")


@pytest.mark.smoke
def test_smoke(dataset_fix):
    strategy = DirectStrategy(HORIZON, HISTORY, trainer, pipeline, model_horizon=2)
    strategy.fit(dataset_fix)
    strategy.predict(dataset_fix)
    strategy.get_feature_importance(top_k=3)
    strategy.get_train_shap()
    strategy.get_test_shap()
