import numpy as np
import pandas as pd

from tsururu.dataset import IndexSlicer, Pipeline, TSDataset

index_slicer = IndexSlicer()


HORIZON = 3
HISTORY = 7

DATASET_PARAMS = {
    "target": {
        "columns": ["value"],
        "type": "continious",
    },
    "date": {
        "columns": ["date"],
        "type": "datetime",
    },
    "id": {
        "columns": ["id"],
        "type": "categorical",
    },
    "exog_1": {
        "columns": ["id2"],
        "type": "categorical",
    },
}

PIPELINE_PARAMS = {
    "target": {
        "columns": ["value"],
        "features": {
            "LagTransformer": {"lags": 1},
        },
    },
    "id": {
        "columns": ["id"],
        "features": {"LabelEncodingTransformer": {}, "LagTransformer": {"lags": 1}},
    },
    "exog_1": {
        "columns": ["id2"],
        "features": {
            "OneHotEncodingTransformer": {"drop": np.array(["k"])},
            "LagTransformer": {"lags": 1},
        },
    },
}


def int_to_str(x):
    str_map_arr = "abcdefghijklmn"
    return str_map_arr[x]


def test_categorical_encoding():
    df = pd.read_csv("./datasets/global/simulated_data_to_check.csv")
    df["id2"] = df["id"] + 2
    df["id"] = df["id"].apply(int_to_str)
    df["id2"] = df["id2"].apply(int_to_str)

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

    id__label__lag_0_idx = pipeline.output_features == "id__label__lag_0"
    assert np.all(
        np.unique(X[:, id__label__lag_0_idx], return_counts=True)[0]
        == np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    )
    assert np.all(
        np.unique(X[:, id__label__lag_0_idx], return_counts=True)[1]
        == np.array([991, 991, 991, 991, 991, 991, 991, 991, 991, 991])
    )

    id2__g_ohe__lag_0_idx = pipeline.output_features == "id2__g_ohe__lag_0"
    assert np.all(
        np.unique(X[:, id2__g_ohe__lag_0_idx], return_counts=True)[0] == np.array([0.0, 1.0])
    )
    assert np.all(
        np.unique(X[:, id2__g_ohe__lag_0_idx], return_counts=True)[1] == np.array([8919, 991])
    )

    assert np.all(
        pipeline.output_features
        == np.array(
            [
                "value__lag_0",
                "id__label__lag_0",
                "id2__c_ohe__lag_0",
                "id2__d_ohe__lag_0",
                "id2__e_ohe__lag_0",
                "id2__f_ohe__lag_0",
                "id2__g_ohe__lag_0",
                "id2__h_ohe__lag_0",
                "id2__i_ohe__lag_0",
                "id2__j_ohe__lag_0",
                "id2__l_ohe__lag_0",
            ],
        )
    )
