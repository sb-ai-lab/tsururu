import logging

import pandas as pd

from tsururu.dataset import IndexSlicer, Pipeline, TSDataset
from tsururu.model_training.trainer import MLTrainer
from tsururu.model_training.validator import KFoldCrossValidator
from tsururu.models import CatBoost
from tsururu.strategies import MIMOStrategy

index_slicer = IndexSlicer()


HORIZON = 3
HISTORY = 7

DATASET_PARAMS = {
    "target": {
        "columns": ["value"],
        "type": "continuous",
    },
    "date": {
        "columns": ["date"],
        "type": "datetime",
    },
    "id": {
        "columns": ["id"],
        "type": "categorical",
    },
}

PIPELINE_PARAMS = {
    "target": {
        "columns": ["value"],
        "features": {
            "DifferenceNormalizer": {
                "regime": "delta",
                "transform_features": True,
                "transform_target": False,
            },
            "LagTransformer": {"lags": 7},
        },
    },
    "date": {
        "columns": ["date"],
        "features": {
            "DateSeasonsGenerator": {
                "seasonalities": ["doy", "m", "wd"],
                "from_target_date": True,
            },
            "LagTransformer": {"lags": 1},
        },
    },
}


def test_assert_for_nans_in_nn_X(caplog):
    df = pd.read_csv("./datasets/global/simulated_data_to_check.csv")

    dataset = TSDataset(
        data=df,
        columns_params=DATASET_PARAMS,
        print_freq_period_info=False,
    )
    pipeline = Pipeline.from_dict(PIPELINE_PARAMS, multivariate=False)

    # Configure the model parameters
    model = CatBoost
    model_params = {
        "loss_function": "MultiRMSE",
        "early_stopping_rounds": 100,
        "n_estimators": 100,
        "verbose": 1000,
    }

    # Configure the validation parameters
    validation = KFoldCrossValidator
    validation_params = {
        "n_splits": 2,
    }

    trainer = MLTrainer(
        model,
        model_params,
        validation,
        validation_params,
    )

    strategy = MIMOStrategy(HORIZON, HISTORY, trainer, pipeline)

    with caplog.at_level(logging.WARNING):
        _, _ = strategy.fit(dataset)
        assert "It seems that there are NaN values in the input data." in caplog.text
        assert "Try to check pipeline configuration" in caplog.text
        assert "NaN values can be caused by division by zero" in caplog.text
