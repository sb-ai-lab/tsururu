import numpy as np
import pandas as pd
import pytest

from tsururu.dataset import IndexSlicer, Pipeline, TSDataset

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
            "LagTransformer": {"lags": 3},
        },
    },
    "date": {
        "columns": ["date"],
        "features": {
            "DateSeasonsGenerator": {
                "seasonalities": ["y", "m", "d", "hour", "min", "sec", "ms"],
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

BASE_PATH = "./tests/test_dataset/different_freqs_datasets/"

DF_PATH_YS = f"{BASE_PATH}simulated_data_to_check_YS.csv"
DF_PATH_Y = f"{BASE_PATH}simulated_data_to_check_Y.csv"
DF_PATH_QS = f"{BASE_PATH}simulated_data_to_check_QS.csv"
DF_PATH_Q = f"{BASE_PATH}simulated_data_to_check_Q.csv"
DF_PATH_MS = f"{BASE_PATH}simulated_data_to_check_MS.csv"
DF_PATH_M = f"{BASE_PATH}simulated_data_to_check_M.csv"
DF_PATH_W = f"{BASE_PATH}simulated_data_to_check_W.csv"
DF_PATH_3D = f"{BASE_PATH}simulated_data_to_check_3D.csv"
DF_PATH_D = f"{BASE_PATH}simulated_data_to_check_D.csv"
DF_PATH_H = f"{BASE_PATH}simulated_data_to_check_H.csv"
DF_PATH_30MIN = f"{BASE_PATH}simulated_data_to_check_30min.csv"
DF_PATH_15MIN = f"{BASE_PATH}simulated_data_to_check_15min.csv"
DF_PATH_5MIN = f"{BASE_PATH}simulated_data_to_check_5min.csv"
DF_PATH_MIN = f"{BASE_PATH}simulated_data_to_check_1min.csv"
DF_PATH_32S = f"{BASE_PATH}simulated_data_to_check_32s.csv"
DF_PATH_S = f"{BASE_PATH}simulated_data_to_check_1s.csv"
DF_PATH_1MS = f"{BASE_PATH}simulated_data_to_check_1ms.csv"

DF_PATH_28D = f"{BASE_PATH}simulated_data_to_check_28D.csv"  # wrong without pd.DateOffset


@pytest.mark.parametrize(
    "df_path, result_lag_2__season_y__date, \
        result_lag_2__season_m__date, result_lag_2__season_d__date, \
        result_lag_2__season_h__date, result_lag_2__season_min__date, \
        result_lag_2__season_sec__date, result_lag_2__season_ms__date",
    [
        (
            DF_PATH_YS,
            np.array([2027, 2028, 2029, 2030, 2031]),
            np.array([1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_Y,
            np.array([2027, 2028, 2029, 2030, 2031]),
            np.array([12, 12, 12, 12, 12]),
            np.array([31, 31, 31, 31, 31]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_QS,
            np.array([2021, 2022, 2022, 2022, 2022]),
            np.array([10, 1, 4, 7, 10]),
            np.array([1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_Q,
            np.array([2021, 2022, 2022, 2022, 2022]),
            np.array([12, 3, 6, 9, 12]),
            np.array([31, 31, 30, 30, 31]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_MS,
            np.array([2020, 2020, 2020, 2020, 2020]),
            np.array([8, 9, 10, 11, 12]),
            np.array([1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_M,
            np.array([2020, 2020, 2020, 2020, 2020]),
            np.array([8, 9, 10, 11, 12]),
            np.array([31, 30, 31, 30, 31]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_W,
            np.array([2020, 2020, 2020, 2020, 2020]),
            np.array([2, 3, 3, 3, 3]),
            np.array([23, 1, 8, 15, 22]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_3D,
            np.array([2020, 2020, 2020, 2020, 2020]),
            np.array([1, 1, 1, 1, 2]),
            np.array([22, 25, 28, 31, 3]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_D,
            np.array([2020, 2020, 2020, 2020, 2020]),
            np.array([1, 1, 1, 1, 1]),
            np.array([8, 9, 10, 11, 12]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_H,
            np.array([2020, 2020, 2020, 2020, 2020]),
            np.array([1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1]),
            np.array([7, 8, 9, 10, 11]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_30MIN,
            np.array([2020, 2020, 2020, 2020, 2020]),
            np.array([1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1]),
            np.array([3, 4, 4, 5, 5]),
            np.array([30, 0, 30, 0, 30]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_15MIN,
            np.array([2020, 2020, 2020, 2020, 2020]),
            np.array([1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1]),
            np.array([1, 2, 2, 2, 2]),
            np.array([45, 0, 15, 30, 45]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_5MIN,
            np.array([2020, 2020, 2020, 2020, 2020]),
            np.array([1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0]),
            np.array([35, 40, 45, 50, 55]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_MIN,
            np.array([2020, 2020, 2020, 2020, 2020]),
            np.array([1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0]),
            np.array([7, 8, 9, 10, 11]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_32S,
            np.array([2020, 2020, 2020, 2020, 2020]),
            np.array([1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0]),
            np.array([3, 4, 4, 5, 5]),
            np.array([44, 16, 48, 20, 52]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_S,
            np.array([2020, 2020, 2020, 2020, 2020]),
            np.array([1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([7, 8, 9, 10, 11]),
            np.array([0, 0, 0, 0, 0]),
        ),
        (
            DF_PATH_1MS,
            np.array([2020, 2020, 2020, 2020, 2020]),
            np.array([1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([7000, 8000, 9000, 10000, 11000]),
        ),
    ],
)
def test_date_features(
    df_path,
    result_lag_2__season_y__date,
    result_lag_2__season_m__date,
    result_lag_2__season_d__date,
    result_lag_2__season_h__date,
    result_lag_2__season_min__date,
    result_lag_2__season_sec__date,
    result_lag_2__season_ms__date,
):
    df = pd.read_csv(df_path)
    dataset = TSDataset(
        data=df,
        columns_params=DATASET_PARAMS,
        print_freq_period_info=False,
    )
    pipeline = Pipeline.from_dict(PIPELINE_PARAMS, multivariate=False)

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

    result_lag_2__season_y__date_idx = pipeline.output_features == "date__season_y__lag_2"
    result_lag_2__season_m__date_idx = pipeline.output_features == "date__season_m__lag_2"
    result_lag_2__season_d__date_idx = pipeline.output_features == "date__season_d__lag_2"
    result_lag_2__season_h__date_idx = pipeline.output_features == "date__season_hour__lag_2"
    result_lag_2__season_min__date_idx = pipeline.output_features == "date__season_min__lag_2"
    result_lag_2__season_sec__date_idx = pipeline.output_features == "date__season_sec__lag_2"
    result_lag_2__season_ms__date_idx = pipeline.output_features == "date__season_ms__lag_2"

    assert np.array_equal(
        X[:, result_lag_2__season_y__date_idx][:5].flatten(), result_lag_2__season_y__date
    )
    assert np.array_equal(
        X[:, result_lag_2__season_m__date_idx][:5].flatten(), result_lag_2__season_m__date
    )
    assert np.array_equal(
        X[:, result_lag_2__season_d__date_idx][:5].flatten(), result_lag_2__season_d__date
    )
    assert np.array_equal(
        X[:, result_lag_2__season_h__date_idx][:5].flatten(), result_lag_2__season_h__date
    )
    assert np.array_equal(
        X[:, result_lag_2__season_min__date_idx][:5].flatten(), result_lag_2__season_min__date
    )
    assert np.array_equal(
        X[:, result_lag_2__season_sec__date_idx][:5].flatten(), result_lag_2__season_sec__date
    )
    assert np.array_equal(
        X[:, result_lag_2__season_ms__date_idx][:5].flatten(), result_lag_2__season_ms__date
    )
    assert index_slicer.ids_from_date(dataset.data, "date") == [
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
    ]


def test_custom_offset_28days():
    df = pd.read_csv(DF_PATH_28D)
    dataset = TSDataset(
        data=df,
        columns_params=DATASET_PARAMS,
        print_freq_period_info=False,
        delta=pd.DateOffset(days=28),
    )

    PIPELINE_PARAMS["date"]["features"]["DateSeasonsGenerator"]["delta"] = pd.DateOffset(days=28)

    pipeline = Pipeline.from_dict(PIPELINE_PARAMS, multivariate=False)

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

    result_lag_2__season_y__date_idx = pipeline.output_features == "date__season_y__lag_2"
    result_lag_2__season_m__date_idx = pipeline.output_features == "date__season_m__lag_2"
    result_lag_2__season_d__date_idx = pipeline.output_features == "date__season_d__lag_2"

    assert np.array_equal(
        X[:, result_lag_2__season_y__date_idx][:5].flatten(), [2020, 2020, 2020, 2020, 2020]
    )
    assert np.array_equal(X[:, result_lag_2__season_m__date_idx][:5].flatten(), [7, 8, 9, 10, 11])
    assert np.array_equal(X[:, result_lag_2__season_d__date_idx][:5].flatten(), [15, 12, 9, 7, 4])

    assert index_slicer.ids_from_date(dataset.data, "date", delta=pd.DateOffset(days=28)) == [
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
    ]
