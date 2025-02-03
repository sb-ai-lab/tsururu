from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import pytest

from tsururu.dataset import IndexSlicer, Pipeline

index_slicer = IndexSlicer()

HORIZON = 3
HISTORY = 3

PIPELINE_PARAMS_STANDARD = {
    "target": {
        "columns": ["value"],
        "features": {
            "LagTransformer": {"lags": 3},
        },
    },
    "date": {
        "columns": ["date"],
        "features": {
            "DateSeasonsGenerator": {
                "seasonalities": ["doy", "m", "wd"],
                "from_target_date": True,
            },
            "LagTransformer": {"lags": 3},
        },
    },
    "id": {
        "columns": ["id"],
        "features": {"LagTransformer": {"lags": 1}},
    },
}

transformers = ["StandardScalerTransformer", "DifferenceNormalizer", "LastKnownNormalizer"]
regimes = ["ratio", "delta"]
transform_features_list = [True, False]
transform_target_list = [True, False]

PIPELINE_CONFIGURATIONS = {}

for transformer, regime, transform_features, transform_target in product(
    transformers, regimes, transform_features_list, transform_target_list
):
    current_config = deepcopy(PIPELINE_PARAMS_STANDARD)

    if transform_target == False and transform_features == False:
        continue

    if transformer == "StandardScalerTransformer":
        if regime == "ratio":
            current_config["target"]["features"] = {
                transformer: {
                    "transform_features": transform_features,
                    "transform_target": transform_target,
                },
                "LagTransformer": {"lags": 3},
            }
        else:
            continue
    elif transformer == "LastKnownNormalizer":
        current_config["target"]["features"] = {
            "LagTransformer": {"lags": 3},
            transformer: {
                "regime": regime,
                "transform_features": transform_features,
                "transform_target": transform_target,
            },
        }
    elif transformer == "DifferenceNormalizer":
        current_config["target"]["features"] = {
            transformer: {
                "regime": regime,
                "transform_features": transform_features,
                "transform_target": transform_target,
            },
            "LagTransformer": {"lags": 3},
        }
    PIPELINE_CONFIGURATIONS[f"{transformer}_{regime}_{transform_features}_{transform_target}"] = (
        current_config
    )


@pytest.mark.parametrize(
    "pipeline_params, result_lag_2__value, result_y",
    [
        (
            PIPELINE_CONFIGURATIONS["StandardScalerTransformer_ratio_True_True"],
            np.array([-1.729454, -1.729454, -1.729454, -1.725992, -1.725992]),
            np.array([[-1.71906713], [-1.71560476], [-1.71214239]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["StandardScalerTransformer_ratio_True_False"],
            np.array([-1.729454, -1.729454, -1.729454, -1.725992, -1.725992]),
            np.array([[1003], [1004], [1005]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["StandardScalerTransformer_ratio_False_True"],
            np.array([1000, 1000, 1000, 1001, 1001]),
            np.array([[-1.71906713], [-1.71560476], [-1.71214239]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["DifferenceNormalizer_ratio_True_True"],
            np.array([np.nan, np.nan, np.nan, 1.001000, 1.001000]),
            np.array([[1.000998], [1.00099701], [1.00099602]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["DifferenceNormalizer_ratio_True_False"],
            np.array([np.nan, np.nan, np.nan, 1.001000, 1.001000]),
            np.array([[1003], [1004], [1005]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["DifferenceNormalizer_ratio_False_True"],
            np.array([1000, 1000, 1000, 1001, 1001]),
            np.array([[1.000998], [1.00099701], [1.00099602]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["DifferenceNormalizer_delta_True_True"],
            np.array([np.nan, np.nan, np.nan, 1.0, 1.0]),
            np.array([[1.0], [1.0], [1.0]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["DifferenceNormalizer_delta_True_False"],
            np.array([np.nan, np.nan, np.nan, 1.0, 1.0]),
            np.array([[1003], [1004], [1005]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["DifferenceNormalizer_delta_False_True"],
            np.array([1000, 1000, 1000, 1001, 1001]),
            np.array([[1.0], [1.0], [1.0]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["LastKnownNormalizer_ratio_True_True"],
            np.array([0.998004, 0.998004, 0.998004, 0.998006, 0.998006]),
            np.array([[1.000998], [1.00199601], [1.00299401]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["LastKnownNormalizer_ratio_True_False"],
            np.array([0.998004, 0.998004, 0.998004, 0.998006, 0.998006]),
            np.array([[1003], [1004], [1005]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["LastKnownNormalizer_ratio_False_True"],
            np.array([1000, 1000, 1000, 1001, 1001]),
            np.array([[1.000998], [1.00199601], [1.00299401]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["LastKnownNormalizer_delta_True_True"],
            np.array([-2.0, -2.0, -2.0, -2.0, -2.0]),
            np.array([[1.0], [2.0], [3.0]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["LastKnownNormalizer_delta_True_False"],
            np.array([-2.0, -2.0, -2.0, -2.0, -2.0]),
            np.array([[1003], [1004], [1005]]),
        ),
        (
            PIPELINE_CONFIGURATIONS["LastKnownNormalizer_delta_False_True"],
            np.array([1000, 1000, 1000, 1001, 1001]),
            np.array([[1.0], [2.0], [3.0]]),
        ),
    ],
)
def test_features_names(get_dataset, pipeline_params, result_lag_2__value, result_y):
    dataset = get_dataset
    pipeline = Pipeline.from_dict(pipeline_params, multivariate=False)

    features_idx = index_slicer.create_idx_data(
        dataset.data,
        HORIZON,
        HISTORY,
        step=1,
        date_column=dataset.date_column,
        delta=dataset.delta,
    )

    target_idx = index_slicer.create_idx_target(
        dataset.data,
        HORIZON,
        HISTORY,
        step=1,
        date_column=dataset.date_column,
        delta=dataset.delta,
    )
    data = Pipeline.create_data_dict_for_pipeline(dataset, features_idx, target_idx)
    data = pipeline.fit_transform(data, strategy_name="FlatWideMIMOStrategy")
    X, y = pipeline.generate(data)

    if pipeline_params["target"]["features"].get("StandardScalerTransformer"):
        result_lag_2__value_idx = pipeline.output_features == "value__standard_scaler__lag_2"
    elif pipeline_params["target"]["features"].get("DifferenceNormalizer"):
        result_lag_2__value_idx = pipeline.output_features == "value__diff_norm__lag_2"
    elif pipeline_params["target"]["features"].get("LastKnownNormalizer"):
        result_lag_2__value_idx = pipeline.output_features == "value__lag_2__last_known_norm"


    lag_2__value = X[:, result_lag_2__value_idx]

    assert np.allclose(lag_2__value[:5].flatten(), result_lag_2__value, equal_nan=True)
    assert np.allclose(y[:3], result_y)
