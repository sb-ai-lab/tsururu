from copy import deepcopy

import numpy as np
import pytest

from tsururu.dataset import IndexSlicer, Pipeline

index_slicer = IndexSlicer()

HORIZON = 3
HISTORY = 7

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
            "LagTransformer": {"lags": 1},
        },
    },
    "id": {
        "columns": ["id"],
        "features": {"LagTransformer": {"lags": 1}},
    },
}

PIPELINE_PARAMS_WITH_ALL_EXOGS = deepcopy(PIPELINE_PARAMS_STANDARD)
PIPELINE_PARAMS_WITH_ALL_EXOGS["exog_1"] = {
    "columns": [
        "Значение времени (ч)...проверка@123__",
        "прибыль_на_АКцию (%)-@финансы",
        "объем Продаж! (Q3_2023)...анализ#данных",
        "add_feature_1",
    ],
    "features": {"LagTransformer": {"lags": 2}},
}
PIPELINE_PARAMS_WITH_ALL_EXOGS["exog_2"] = {
    "columns": [
        "кол-во_клиентов#сегмент_A__тест?",
        "показатель/успешности_доход@услуги__OK?",
        "add_feature_0",
        "add_feature_2",
    ],
    "features": {"LagTransformer": {"lags": 1}},
}

PIPELINE_PARAMS_WITH_SOME_EXOGS = deepcopy(PIPELINE_PARAMS_STANDARD)
PIPELINE_PARAMS_WITH_SOME_EXOGS["exog_1"] = {
    "columns": [
        "Значение времени (ч)...проверка@123__",
    ],
    "features": {"LagTransformer": {"lags": 2}},
}
PIPELINE_PARAMS_WITH_SOME_EXOGS["exog_2"] = {
    "columns": [
        "add_feature_0",
        "add_feature_2",
    ],
    "features": {"LagTransformer": {"lags": 1}},
}

PIPELINE_WITH_VALUE_ONLY = {
    "target": {
        "columns": ["value"],
        "features": {
            "LagTransformer": {"lags": 3},
        },
    },
}

PIPELINE_WITH_VALUE_AND_DATE = deepcopy(PIPELINE_PARAMS_STANDARD)
PIPELINE_WITH_VALUE_AND_DATE.pop("id")

PIPELINE_WITH_VALUE_AND_ID = deepcopy(PIPELINE_PARAMS_STANDARD)
PIPELINE_WITH_VALUE_AND_ID.pop("date")

PIPELINE_PARAMS_WITH_WRONG_EXOG_TRANSFORMER = deepcopy(PIPELINE_PARAMS_STANDARD)
PIPELINE_PARAMS_WITH_WRONG_EXOG_TRANSFORMER["exog_1"] = {
    "columns": [
        "add_feature_1",
    ],
    "features": {
        "StandardScalerTransformer": {"transform_features": True, "transform_target": True},
        "LagTransformer": {"lags": 2},
    },
}


@pytest.mark.parametrize(
    "pipeline_params, result",
    [
        (
            PIPELINE_PARAMS_STANDARD,
            np.array(
                [
                    "value__lag_2",
                    "value__lag_1",
                    "value__lag_0",
                    "id__lag_0",
                    "date__season_doy__lag_0",
                    "date__season_m__lag_0",
                    "date__season_wd__lag_0",
                ]
            ),
        ),
        (
            PIPELINE_PARAMS_WITH_ALL_EXOGS,
            np.array(
                [
                    "value__lag_2",
                    "value__lag_1",
                    "value__lag_0",
                    "id__lag_0",
                    "date__season_doy__lag_0",
                    "date__season_m__lag_0",
                    "date__season_wd__lag_0",
                    "Значение времени (ч)...проверка@123____lag_1",
                    "Значение времени (ч)...проверка@123____lag_0",
                    "прибыль_на_АКцию (%)-@финансы__lag_1",
                    "прибыль_на_АКцию (%)-@финансы__lag_0",
                    "объем Продаж! (Q3_2023)...анализ#данных__lag_1",
                    "объем Продаж! (Q3_2023)...анализ#данных__lag_0",
                    "add_feature_1__lag_1",
                    "add_feature_1__lag_0",
                    "кол-во_клиентов#сегмент_A__тест?__lag_0",
                    "показатель/успешности_доход@услуги__OK?__lag_0",
                    "add_feature_0__lag_0",
                    "add_feature_2__lag_0",
                ]
            ),
        ),
        (
            PIPELINE_PARAMS_WITH_SOME_EXOGS,
            np.array(
                [
                    "value__lag_2",
                    "value__lag_1",
                    "value__lag_0",
                    "id__lag_0",
                    "date__season_doy__lag_0",
                    "date__season_m__lag_0",
                    "date__season_wd__lag_0",
                    "Значение времени (ч)...проверка@123____lag_1",
                    "Значение времени (ч)...проверка@123____lag_0",
                    "add_feature_0__lag_0",
                    "add_feature_2__lag_0",
                ]
            ),
        ),
        (
            PIPELINE_WITH_VALUE_ONLY,
            np.array(
                [
                    "value__lag_2",
                    "value__lag_1",
                    "value__lag_0",
                ]
            ),
        ),
        (
            PIPELINE_WITH_VALUE_AND_DATE,
            np.array(
                [
                    "value__lag_2",
                    "value__lag_1",
                    "value__lag_0",
                    "date__season_doy__lag_0",
                    "date__season_m__lag_0",
                    "date__season_wd__lag_0",
                ]
            ),
        ),
        (
            PIPELINE_WITH_VALUE_AND_ID,
            np.array(
                [
                    "value__lag_2",
                    "value__lag_1",
                    "value__lag_0",
                    "id__lag_0",
                ]
            ),
        ),
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

    assert np.array_equal(pipeline.output_features, result)
    assert X.shape[1] == len(result)


def test_wrong_exog_transformer(get_dataset):
    with pytest.raises(AssertionError):
        _ = Pipeline.from_dict(PIPELINE_PARAMS_WITH_WRONG_EXOG_TRANSFORMER, multivariate=False)
