"""Tests for LagTransformer.from_target_date and future_exog pipeline integration."""
import numpy as np
import pandas as pd
import pytest

from tsururu.dataset import TSDataset, IndexSlicer, Pipeline
from tsururu.transformers.seq import LagTransformer


N = 10
HORIZON = 7
HISTORY = 3

index_slicer = IndexSlicer()


def make_raw_dataframe(n=N):
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "id": "0",
            "timestamp": dates,
            "target": np.arange(1.0, n + 1.0),
            "exog_default_1": np.arange(1.0, 1.0 + n),
            "exog_future": np.arange(100.0, 100.0 + n),
        }
    )


COLUMNS_PARAMS = {
    "target": {"columns": ["target"], "type": "continuous"},
    "date": {"columns": ["timestamp"], "type": "datetime"},
    "id": {"columns": ["id"], "type": "categorical"},
    "exog_1": {"columns": ["exog_default_1"], "type": "continuous"},
    "future_exog": {"columns": ["exog_future"], "type": "continuous"},
}

PIPELINE_PARAMS = {
    "target": {
        "columns": ["target"],
        "features": {"LagTransformer": {"lags": [1, 2]}},
    },
    "date": {
        "columns": ["timestamp"],
        "features": {
            "DateSeasonsGenerator": {"seasonalities": ["m"], "from_target_date": False},
            "LagTransformer": {"lags": [1]},
        },
    },
    "exog_future": {
        "columns": ["exog_future"],
        "features": {
            "LagTransformer": {"lags": [1, 0], "from_target_date": True},
        },
    },
    "exog_1": {
        "columns": ["exog_default_1"],
        "features": {
            "LagTransformer": {"lags": [0], "from_target_date": False},
        },
    },
}


@pytest.fixture
def dataset():
    return TSDataset(make_raw_dataframe(), COLUMNS_PARAMS, print_freq_period_info=False)


@pytest.fixture
def data_dict(dataset):
    features_idx = index_slicer.create_idx_data(
        dataset.data, HORIZON, HISTORY, step=1,
        date_column=dataset.date_column, delta=dataset.delta,
    )
    target_idx = index_slicer.create_idx_target(
        dataset.data, HORIZON, HISTORY, step=1,
        date_column=dataset.date_column, delta=dataset.delta,
    )
    return Pipeline.create_data_dict_for_pipeline(dataset, features_idx, target_idx)


@pytest.fixture
def fitted_pipeline(data_dict):
    pipeline = Pipeline.from_dict(PIPELINE_PARAMS, multivariate=False)
    pipeline.fit_transform(data_dict, strategy_name="MIMOStrategy")
    return pipeline, data_dict


def test_lag_from_target_date_anchors_at_idx_y():
    """
    from_target_date=True: якорь = idx_y[:, 0] (первый шаг горизонта).

    idx_X[i] = [i, i+1, i+2] -> idx_X[:, -1] = [2, 3, 4, 5, 6, 7]
    idx_y[i] = [i+3, i+4] -> idx_y[:, 0]  = [3, 4, 5, 6, 7, 8]

    При lag=0: expected = exog_future[idx_y[:, 0]] = [103..108]
    """
    raw = pd.DataFrame({"exog_future": np.arange(100.0, 110.0)})
    idx_X = np.array([[i, i + 1, i + 2] for i in range(6)])
    idx_y = np.array([[i + 3, i + 4] for i in range(6)])

    lag = LagTransformer(lags=[0], from_target_date=True)
    lag.fit({"target_column_name": "_other_"}, ["exog_future"])

    data = {
        "raw_ts_X": raw,
        "idx_X": idx_X,
        "idx_y": idx_y,
        "X": np.array([]),
        "target_column_name": "_other_",
    }
    lag.generate(data)

    # anchor = idx_y[:, 0] = [3,4,5,6,7,8] -> values [103..108]
    expected = np.arange(103.0, 109.0).reshape(-1, 1)
    np.testing.assert_array_equal(data["X"], expected)


def test_lag_from_target_date_differs_from_idx_x_anchor():
    """
    from_target_date=True -> якорь idx_y[:, 0] → [103..108]
    from_target_date=False -> якорь idx_X[:, -1] → [102..107]
    """
    raw = pd.DataFrame({"exog_future": np.arange(100.0, 110.0)})
    idx_X = np.array([[i, i + 1, i + 2] for i in range(6)])
    idx_y = np.array([[i + 3, i + 4] for i in range(6)])

    base = {
        "raw_ts_X": raw,
        "idx_X": idx_X,
        "idx_y": idx_y,
        "target_column_name": "_other_",
    }

    lag_future = LagTransformer(lags=[0], from_target_date=True)
    lag_future.fit({"target_column_name": "_other_"}, ["exog_future"])
    data_future = {**base, "X": np.array([])}
    lag_future.generate(data_future)

    lag_normal = LagTransformer(lags=[0], from_target_date=False)
    lag_normal.fit({"target_column_name": "_other_"}, ["exog_future"])
    data_normal = {**base, "X": np.array([])}
    lag_normal.generate(data_normal)

    assert not np.array_equal(data_future["X"], data_normal["X"])
    np.testing.assert_array_equal(data_future["X"].flatten(), np.arange(103.0, 109.0))
    np.testing.assert_array_equal(data_normal["X"].flatten(), np.arange(102.0, 108.0))


def test_future_exog_lag0_values_are_correct(fitted_pipeline):
    """
    exog_future__lag_0 при from_target_date=True:
      anchor = idx_y[i, 0] = i + HISTORY
      lag_0 = anchor - 0  = i + HISTORY
      expected = 100 + i + HISTORY

    При HISTORY=3, n_samples=1, i=0: expected = [103.0]
    """
    pipeline, data_dict = fitted_pipeline
    X, _ = pipeline.generate(data_dict)

    col_mask = pipeline.output_features == "exog_future__lag_0"
    assert col_mask.sum() == 1

    actual = X[:, col_mask].flatten()
    n_samples = N - HISTORY - HORIZON + 1
    expected = np.arange(100.0 + HISTORY, 100.0 + HISTORY + n_samples)  # [103.0]
    np.testing.assert_array_equal(actual, expected)


def test_future_exog_lag1_values_are_correct(fitted_pipeline):
    """
    exog_future__lag_1 при from_target_date=True:
      anchor = idx_y[i, 0] = i + HISTORY
      lag_1 = anchor - 1  = i + HISTORY - 1
      expected = 100 + i + HISTORY - 1

    При HISTORY=3, n_samples=1, i=0: expected = [102.0]
    """
    pipeline, data_dict = fitted_pipeline
    X, _ = pipeline.generate(data_dict)

    col_mask = pipeline.output_features == "exog_future__lag_1"
    assert col_mask.sum() == 1

    actual = X[:, col_mask].flatten()
    n_samples = N - HISTORY - HORIZON + 1
    expected = np.arange(100.0 + HISTORY - 1, 100.0 + HISTORY - 1 + n_samples)  # [102.0]
    np.testing.assert_array_equal(actual, expected)
