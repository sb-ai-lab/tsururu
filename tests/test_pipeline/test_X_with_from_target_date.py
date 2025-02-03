from copy import deepcopy

import numpy as np
import pytest

from tsururu.dataset import IndexSlicer, Pipeline

index_slicer = IndexSlicer()

HORIZON = 3
HISTORY = 3

PIPELINE_FROM_TARGET_DATE_TRUE = {
    "target": {
        "columns": ["value"],
        "features": {
            "LagTransformer": {"lags": 1},
        },
    },
    "date": {
        "columns": ["date"],
        "features": {
            "DateSeasonsGenerator": {
                "seasonalities": ["d"],
                "from_target_date": True,
            },
            "LagTransformer": {"lags": 1},
        },
    },
}

PIPELINE_FROM_TARGET_DATE_FALSE = deepcopy(PIPELINE_FROM_TARGET_DATE_TRUE)
PIPELINE_FROM_TARGET_DATE_FALSE["date"]["features"]["DateSeasonsGenerator"][
    "from_target_date"
] = False


@pytest.mark.parametrize(
    "pipeline_params, result",
    [
        (PIPELINE_FROM_TARGET_DATE_TRUE, np.array([6, 7, 8, 9, 10])),
        (PIPELINE_FROM_TARGET_DATE_FALSE, np.array([3, 4, 5, 6, 7])),
    ],
)
def test_features_names(get_dataset, pipeline_params, result):
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
    data = pipeline.fit_transform(data, strategy_name="MIMOStrategy")
    X, _ = pipeline.generate(data)

    day_column_name = pipeline.output_features == "date__season_d__lag_0"
    day_column = X[:, day_column_name]

    assert np.array_equal(day_column[:5].flatten(), result)
