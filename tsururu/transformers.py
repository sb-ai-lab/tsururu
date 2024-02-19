from __future__ import annotations
from typing import List, Union, Tuple, Optional
from numpy.typing import NDArray
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import holidays

from .dataset import IndexSlicer

transformers_masks = {
    "raw": r"^",
    "LAG": r"lag_\d+__",
    "SEASON": r"season_\w+__",
    "TIMETONUM": r"time_to_num__",
    "LABEL": r"label_encoder__",
    "OHE": r"ohe_encoder_\S+__",
}

date_attrs = {
    "y": "year",
    "m": "month",
    "d": "day",
    "wd": "weekday",
    "doy": "dayofyear",
    "hour": "hour",
    "min": "minute",
    "sec": "second",
    "ms": "microsecond",
    "ns": "nanosecond",
}


def _seq_mult_ts(data, idx_data):
    index_slicer = IndexSlicer()
    data_seq = np.array([])

    for idx in range(len(data)):
        current_data_seq = index_slicer.get_slice(data[idx], (idx_data[idx], None))
        if data_seq.shape[0] == 0:
            data_seq = current_data_seq
        else:
            data_seq = np.hstack((data_seq, current_data_seq))
    return data_seq


class SeriesToFeaturesTransformer:
    """A transformer that is trained on the raw time series,
    and applied to generated features, targets, or both.
    """

    def __init__(self):
        self.columns = None
        self.id_column = None
        self.transform_train = None
        self.transform_target = None

    def _fit_segment(self, segment: pd.Series, columns: List[str], id_column: str) -> pd.Series:
        pass

    def _transform_segment(
        self, segment: pd.Series, columns: List[str], id_column: str
    ) -> pd.Series:
        pass

    def fit(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        columns: List[str],
        id_column: str,
        transform_train: Optional[bool],
        transform_target: Optional[bool],
    ) -> SeriesToFeaturesTransformer:
        """Fit transformer.

        Args:
            raw_ts_X: Current raw time series data for features.
            raw_ts_y: Current raw time series data for targets.
            features_X: Current dataframe with features.
            y: Current targets' array.
            columns: List with columns' names to transform.
            id_column: Id column's name from raw dataset.
            transform_train: Either to transform features.
            trainform_target: Either to transform target values.

        Returns:
            Fitted transformer.
        """
        appropriate_columns_list = []
        for raw_column_name in columns:
            column_mask = ""
            for _, transformer_mask in transformers_masks.items():
                column_mask += fr"{transformer_mask}{re.escape(raw_column_name)}$|"
            column_mask = column_mask[:-1]
            appropriate_columns_list.append(raw_ts_X.columns.str.contains(column_mask))

        self.columns = raw_ts_X.columns[np.any(appropriate_columns_list, axis=0)]

    def transform(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        X_only: bool,
    ) -> Tuple[Union[pd.DataFrame, NDArray]]:
        """Transform features / target or both.

        Args:
            raw_ts_X: Current raw time series data for features.
            raw_ts_y: Current raw time series data for targets.
            features_X: Current dataframe with features.
            y: Current targets' array.
            X_only: Transform only features data.

        Returns:
            Transformer features / target or both.
        """
        pass


class SeriesToSeriesTransformer(SeriesToFeaturesTransformer):
    """A transformer that is trained on the raw time series
    and applied to it."""

    def transform(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        X_only: bool,
    ) -> Tuple[pd.DataFrame]:
        if self.transform_train:
            raw_ts_X = raw_ts_X.groupby(self.id_column).apply(self._transform_segment).reset_index(level=self.id_column, drop=True)
        if self.transform_target and not X_only:
            raw_ts_y = raw_ts_y.groupby(self.id_column).apply(self._transform_segment).reset_index(level=self.id_column, drop=True)
        return raw_ts_X, raw_ts_y, features_X, y

    def inverse_transform_y(self, y: pd.DataFrame) -> pd.DataFrame:
        return y.groupby(self.id_column).apply(self._inverse_transform_segment).reset_index(level=self.id_column, drop=True)


class FeaturesToFeaturesTransformer(SeriesToFeaturesTransformer):
    """A transformer that is trained on generated features,
    and applied to either features or targets, or both."""

    def __init__(self):
        self.transform_train = None
        self.transform_target = None

    def fit(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        columns: List[str],
        id_column: str,
        transform_train: bool,
        transform_target: bool,
    ) -> FeaturesToFeaturesTransformer:
        """Fit transformer.

        Args:
            raw_ts_X: Current raw time series data for features.
            raw_ts_y: Current raw time series data for targets.
            features_X: Current dataframe with features.
            y: Current targets' array.
            columns: List with columns' names to transform.
            id_column: Id column's name from raw dataset.
            transform_train: Either to transform features.
            trainform_target: Either to transform target values.

        Returns:
            Fitted transformer.
        """
        self.transform_train = transform_train
        self.transform_target = transform_target


class FeaturesGenerator(SeriesToFeaturesTransformer):
    """A transformer that is trained both on the raw time series
    or features, and used for generating new features."""

    def __init__(self):
        pass

    def fit(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        columns: List[str],
        id_column: str,
        transform_train: bool,
        transform_target: bool,
    ) -> FeaturesGenerator:
        """Fit transformer.

        Args:
            raw_ts_X: Current raw time series data for features.
            raw_ts_y: Current raw time series data for targets.
            features_X: Current dataframe with features.
            y: Current targets' array.
            columns: List with columns' names to transform.
            id_column: Id column's name from raw dataset.
            transform_train: Either to transform features.
            trainform_target: Either to transform target values.

        Returns:
            Fitted transformer.
        """
        self.columns = columns


class StandardScalerTransformer(SeriesToSeriesTransformer):
    def __init__(self):
        """Standardize features by removing the mean and scaling
            to unit variance.

        self.params = {
            id_1: {
                (colname_1, 'mean'): mean, (colname_2, 'std'): std, ...
            },
            id_2: {
                (colname_1, 'mean'): mean, (colname_2, 'std'): std, ...
            },
            ...
        }
        """
        super().__init__()
        self.params = {}

    def _get_mask_mean_std(self, segment, column_name, current_id):
        column_mask = [segment.columns.str.contains(column_name)][0]
        mean = self.params[current_id][(column_name, "mean")]
        std = self.params[current_id][(column_name, "std")]
        return column_mask, mean, std

    def _transform_segment(self, segment: pd.Series) -> pd.Series:
        segment = segment.copy()
        current_id = segment[self.id_column].values[0]

        for column_name in self.columns:
            column_mask, mean, std = self._get_mask_mean_std(
                segment=segment,
                column_name=column_name,
                current_id=current_id,
            )
            segment.loc[:, column_mask] = (segment.loc[:, column_mask] - mean) / std
        return segment

    def _inverse_transform_segment(self, segment: pd.Series) -> pd.Series:
        segment = segment.copy()
        current_id = segment[self.id_column].values[0]

        for column_name in self.columns:
            column_mask, mean, std = self._get_mask_mean_std(
                segment=segment,
                column_name=column_name,
                current_id=current_id,
            )
            segment.loc[:, column_mask] = segment.loc[:, column_mask] * std + mean
        return segment

    def fit(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        columns: List[str],
        id_column: str,
        transform_train: bool,
        transform_target: bool,
    ) -> SeriesToSeriesTransformer:
        super().fit(
            raw_ts_X,
            raw_ts_y,
            features_X,
            y,
            columns,
            id_column,
            transform_train,
            transform_target,
        )
        self.columns = [
            column
            for column in self.columns
            if issubclass(raw_ts_X[column].dtype.type, np.integer)
            or issubclass(raw_ts_X[column].dtype.type, np.floating)
        ]
        stat_df = raw_ts_X.groupby(id_column)[self.columns].agg(["mean", "std"])
        self.params = stat_df.to_dict(orient="index")
        return self


class DifferenceNormalizer(SeriesToSeriesTransformer):
    """Normalize values ​​in a time series by the previous value.

    Args:
        type: "delta" to take the difference or "ratio" -- ratio
            between the current and the previous value.

    self.params: dict with last values by each id (for targets' inverse transform)
    """

    def __init__(self, regime: str = "delta"):
        super().__init__()
        self.type = regime
        self.params = None

    def _transform_segment(self, segment: pd.Series):
        segment = segment.copy()

        for current_column_name in self.columns:
            if self.type == "delta":
                segment.loc[:, current_column_name] = segment.loc[
                    :, current_column_name
                ] - segment.loc[:, current_column_name].shift(1)
            elif self.type == "ratio":
                segment.loc[:, current_column_name] = segment.loc[
                    :, current_column_name
                ] / segment.loc[:, current_column_name].shift(1)
        return segment

    def _inverse_transform_segment(self, segment: pd.Series) -> pd.Series:
        segment = segment.copy()
        current_id = segment[self.id_column].values[0]

        for current_column_name in self.columns:
            current_columns_mask = [segment.columns.str.contains(current_column_name)][0]
            current_last_value = self.params[current_id][current_column_name]
            if self.type == "delta":
                segment.loc[:, current_columns_mask] = np.cumsum(np.append(current_last_value, segment.loc[:, current_columns_mask].values))[1:]
            if self.type == "ratio":
                segment.loc[:, current_columns_mask] = np.cumprod(np.append(current_last_value, segment.loc[:, current_columns_mask].values))[1:]
        return segment

    def fit(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        columns: List[str],
        id_column: str,
        transform_train: bool,
        transform_target: bool,
    ) -> SeriesToSeriesTransformer:
        super().fit(
            raw_ts_X,
            raw_ts_y,
            features_X,
            y,
            columns,
            id_column,
            transform_train,
            transform_target,
        )
        self.columns = [
            column
            for column in self.columns
            if issubclass(raw_ts_X[column].dtype.type, np.integer)
            or issubclass(raw_ts_X[column].dtype.type, np.floating)
        ]
        last_values_df = raw_ts_X.groupby(self.id_column)[self.columns].last()
        self.params = last_values_df.to_dict(orient="index")
        return self


class LastKnownNormalizer(FeaturesToFeaturesTransformer):
    """Normalize values ​​in a time series by the last known value.

    Args:
        regime: "delta" to take the difference or "ratio" -- the ratio
            between the current and the last known value in the future.
        last_lag_substring: a substring that is included in the name
            of any columns in the feature table and denotes
            the last known (nearest) lag features.

    self.params: dict with last values by each column
    """

    def __init__(
        self,
        regime: str = "ratio",
        last_lag_substring: str = "lag_0",
    ):
        super().__init__()
        self.regime = regime
        self.last_lag_substring = last_lag_substring
        self.params = {}

    def fit(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        columns: List[str],
        id_column: str,
        transform_train: bool,
        transform_target: bool,
    ) -> FeaturesToFeaturesTransformer:
        super().fit(
            raw_ts_X,
            raw_ts_y,
            features_X,
            y,
            columns,
            id_column,
            transform_train,
            transform_target,
        )
        for column_name in columns:
            last_column_name = features_X.columns[
                (features_X.columns.str.contains(column_name))
                & (features_X.columns.str.contains(self.last_lag_substring))
            ]
            self.params[column_name] = features_X[last_column_name].values
        return self

    def transform(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        X_only: bool,
    ) -> Tuple[pd.DataFrame]:
        for column_name, last_values in self.params.items():
            columns_to_transform = features_X.columns[
                (features_X.columns.str.contains(column_name))
            ]
            if self.regime == "delta":
                if self.transform_train:
                    features_X.loc[:, columns_to_transform] = (
                        features_X[columns_to_transform] - last_values
                    )
                if self.transform_target and not X_only:
                    y = y - last_values
            elif self.regime == "ratio":
                if self.transform_train:
                    features_X.loc[:, columns_to_transform] = (
                        features_X[columns_to_transform] / last_values
                    )
                if self.transform_target and not X_only:
                    y = y / last_values
        return raw_ts_X, raw_ts_y, features_X, y

    def inverse_transform_y(self, y: pd.DataFrame) -> pd.DataFrame:
        column_name = list(self.params.keys())[0]
        if self.regime == "delta":
            y.loc[:, column_name] = y[column_name] + np.repeat(
                self.params[column_name].reshape(-1),
                len(y[column_name]) // len(self.params[column_name].reshape(-1)),
            )
        if self.regime == "ratio":
            y.loc[:, column_name] = y[column_name] * np.repeat(
                self.params[column_name].reshape(-1),
                len(y[column_name]) // len(self.params[column_name].reshape(-1)),
            )
        return y


class TimeToNumGenerator(FeaturesGenerator):
    """Datetime converted to difference with basic_date.

    Arguments:
        basic_date: date relating to which normalization takes place.
        basic_interval: what time value is between the observations.
        from_target_date: features are built from the targets' dates.
        horizon: forecast horizon.
    """

    def __init__(
        self,
        basic_date: str = "2020-01-01",
        basic_interval: str = "D",
        from_target_date: bool = False,
        horizon: Optional[int] = None,
        delta: Optional[pd.DateOffset] = None,
    ):
        super().__init__()
        self.basic_date = basic_date
        self.basic_interval = basic_interval
        self.from_target_date = from_target_date
        self.horizon = horizon
        self.delta = delta

    def fit(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        columns: List[str],
        id_column: str,
        transform_train: bool,
        transform_target: bool,
    ) -> FeaturesGenerator:
        super().fit(
            raw_ts_X,
            raw_ts_y,
            features_X,
            y,
            columns,
            id_column,
            transform_train,
            transform_target,
        )
        self._features = [f"time_to_num__{column_name}" for column_name in self.columns]
        return self

    def transform(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        _,
    ) -> Tuple[pd.DataFrame]:
        result_data = []
        for column_name in self.columns:
            time_col = raw_ts_X[column_name]
            index_slicer = IndexSlicer()

            str_time_col = time_col.apply(lambda x: str(x).split(" ")[0])
            _, time_delta = index_slicer.timedelta(str_time_col, delta=self.delta)

            if self.from_target_date:
                time_col = time_col + self.horizon * time_delta

            data = pd.to_datetime(time_col.to_numpy().reshape(-1), origin="unix")
            data_transformed = (
                (data - np.datetime64(self.basic_date)) / np.timedelta64(1, self.basic_interval)
            ).values.astype(np.float32)
            result_data.append(data_transformed)
        raw_ts_X[:, self._features] = result_data
        return raw_ts_X, raw_ts_y, features_X, y


class DateSeasonsGenerator(FeaturesGenerator):
    """Generate categorical features that reflect seasonality.
        In case when country is provided (it is possible to specify
        prov and state) indicator 'there is a holiday at that moment'
        is generated.

    Arguments:
        seasonalities: categorical features to build.
        from_target_date: features are built from the targets' dates.
        horizon: forecast horizon.
        country: country code ("RU" for Russia).
        prov: province inside country.
        state: state inside country.
    """

    def __init__(
        self,
        seasonalities: List[str] = ["doy", "m", "wd"],
        from_target_date: bool = False,
        horizon: Optional[int] = None,
        country: Optional[str] = None,
        prov: Optional[str] = None,
        state: Optional[str] = None,
        delta: Optional[pd.DateOffset] = None
    ):
        super().__init__()
        self.seasonalities = seasonalities
        self.from_target_date = from_target_date
        self.horizon = horizon
        self._country = country
        self._prov = prov
        self._state = state
        self._features = []
        self.delta = delta

    def fit(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        columns: List[str],
        id_column: str,
        transform_train: bool,
        transform_target: bool,
    ) -> FeaturesGenerator:
        super().fit(
            raw_ts_X,
            raw_ts_y,
            features_X,
            y,
            columns,
            id_column,
            transform_train,
            transform_target,
        )
        self._features = []

        for s in self.seasonalities:
            self._features.extend([f"season_{s}__{column_name}" for column_name in self.columns])
        if self._country is not None:
            self._features.extend([f"season_hol__{column_name}" for column_name in self.columns])
        return self

    def transform(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        X_only: bool,
    ) -> Tuple[pd.DataFrame]:
        result_data = []

        for column_name in self.columns:
            time_col = raw_ts_X[column_name]
            index_slicer = IndexSlicer()

            _, time_delta = index_slicer.timedelta(time_col, delta=self.delta)
            if self.from_target_date:
                time_col = time_col + self.horizon * time_delta
            time_col = pd.to_datetime(time_col.to_numpy(), origin="unix")

            new_arr = np.empty((time_col.shape[0], len(self._features)), np.int32)

            n = 0
            for seas in self.seasonalities:
                new_arr[:, n] = getattr(time_col, date_attrs[seas])
                n += 1

            if self._country is not None:
                # get years
                years = np.unique(time_col.year)
                hol = holidays.CountryHoliday(
                    self._country,
                    years=years,
                    prov=self._prov,
                    state=self._state,
                )
                new_arr[:, n] = time_col.date.isin(hol)
                n += 1
            result_data.append(new_arr)
        raw_ts_X[self._features] = np.hstack(result_data)
        return raw_ts_X, raw_ts_y, features_X, y


class LabelEncodingTransformer(SeriesToSeriesTransformer):
    """Transform categories of features into integer values.
    """
    def __init__(self):
        super().__init__()

    def fit(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        columns: List[str],
        id_column: str,
        transform_train: bool,
        transform_target: bool,
    ) -> FeaturesGenerator:
        super().fit(
            raw_ts_X,
            raw_ts_y,
            features_X,
            y,
            columns,
            id_column,
            transform_train,
            transform_target,
        )
        self._features = [f"label_encoder__{column_name}" for column_name in self.columns]
        return self

    def transform(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        X_only: bool,
    ) -> Tuple[pd.DataFrame]:
        new_arr = np.empty((len(raw_ts_X), len(self._features)), np.int32)
        for i, column_name in enumerate(self.columns):
            new_arr[:, i] = LabelEncoder().fit_transform(raw_ts_X[column_name])
        raw_ts_X[self._features] = new_arr
        return raw_ts_X, raw_ts_y, features_X, y


class OneHotEncodingTransformer(SeriesToSeriesTransformer):
    """Transform categorical features as a one-hot numeric array.

    Arguments:
        - drop: one from ['first', 'if_binary', None] or array-list of shape (n_features, )
            None : retain all features.
            ‘first’ : drop the first category in each feature. 
            ‘if_binary’ : drop the first category in each feature with two categories.
            array : drop[i] is the category in feature X[:, i] that should be dropped.
    """

    def __init__(self, drop: str = None):
        super().__init__()
        self.drop = drop

    def fit(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        columns: List[str],
        id_column: str,
        transform_train: bool,
        transform_target: bool,
    ) -> FeaturesGenerator:
        super().fit(
            raw_ts_X,
            raw_ts_y,
            features_X,
            y,
            columns,
            id_column,
            transform_train,
            transform_target,
        )
        self._features = []

        if self.drop == "first":
            for column_name in self.columns:
                for id_name in raw_ts_X[column_name].unique()[1:]:
                    self._features.append(f"ohe_encoder_{id_name}__{column_name}")

        elif self.drop == "is_binary":
            for column_name in self.columns:
                if raw_ts_X[column_name].nunique() == 2:
                    for id_name in raw_ts_X[column_name].unique()[1:]:
                        self._features.append(f"ohe_encoder_{id_name}__{column_name}")
                else:
                    for id_name in raw_ts_X[column_name].unique():
                        self._features.append(f"ohe_encoder_{id_name}__{column_name}")

        elif isinstance(self.drop, np.ndarray):
            for column_i, column_name in enumerate(self.columns):
                for id_name in np.delete(raw_ts_X[column_name].unique(), np.where(raw_ts_X[column_name].unique() == self.drop[column_i])):
                    self._features.append(f"ohe_encoder_{id_name}__{column_name}")

        else:
            for column_i, column_name in enumerate(self.columns):
                for id_name in raw_ts_X[column_name].unique():
                    self._features.append(f"ohe_encoder_{id_name}__{column_name}")
        return self

    def transform(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        X_only: bool,
    ) -> Tuple[pd.DataFrame]:
        result_data = [OneHotEncoder(drop=self.drop).fit_transform(raw_ts_X[column_name].values.reshape(-1, 1)).todense() for i, column_name in enumerate(self.columns)]
        raw_ts_X[self._features] = np.hstack(result_data)
        # new_arr = np.empty((len(raw_ts_X), len(self._features)), np.int32)
        # for i, column_name in enumerate(self.columns):
        #     new_arr[:, i] = OneHotEncoder(drop=self.drop).fit_transform(raw_ts_X[column_name].values.reshape(-1, 1))
        # raw_ts_X[self._features] = new_arr
        return raw_ts_X, raw_ts_y, features_X, y


class LagTransformer(SeriesToFeaturesTransformer):
    """Generate lag features.

    Arguments:
        lags: lags features to build.
        drop_raw_features: whether to throw out the original column.
        idx_data: indices that are used to construct attributes.
    """

    def __init__(
        self,
        lags: Union[int, List[int], np.ndarray],
        drop_raw_features: bool,
        idx_data: NDArray[np.floating],
    ):
        super().__init__()
        if isinstance(lags, list):
            self.lags = np.array(lags)
        if isinstance(lags, int):
            self.lags = np.arange(lags)
        self.drop_raw_features = drop_raw_features
        self.idx_data = idx_data

    def fit(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        columns: List[str],
        id_column: str,
        transform_train: Optional[bool],
        transform_target: Optional[bool],
    ) -> SeriesToFeaturesTransformer:
        super().fit(
            raw_ts_X,
            raw_ts_y,
            features_X,
            y,
            columns,
            id_column,
            transform_train,
            transform_target,
        )
        if self.drop_raw_features:
            self.columns = self.columns.drop(columns)

        data = raw_ts_X[self.columns]
        index_slicer = IndexSlicer()

        if isinstance(data, list):
            # (1 observation, history, features)
            sample_data = index_slicer.get_slice(data[0], (self.idx_data[0][0], None))
        else:
            sample_data = index_slicer.get_slice(data, (self.idx_data[0], None))

        # convert to accepted dtype and get attributes
        # leave only correct lags (< number of points in sample_data)
        self.current_correct_lags = self.lags.copy()[self.lags < sample_data.shape[1]]

        feats = []

        if isinstance(data, list):
            data = data[0]

        for feat in data.columns:
            feats.extend(
                ["lag" + f"_{i}" + "__" + feat for i in reversed(self.current_correct_lags)]
            )
        self._features = list(feats)
        return self

    def transform(
        self,
        raw_ts_X: pd.DataFrame,
        raw_ts_y: pd.DataFrame,
        features_X: pd.DataFrame,
        y: pd.DataFrame,
        X_only: bool,
    ) -> Tuple[pd.DataFrame]:
        data = raw_ts_X[self.columns]
        index_slicer = IndexSlicer()
        if isinstance(data, list):
            data_seq = _seq_mult_ts(data, self.idx_data)
        else:
            data_seq = index_slicer.get_slice(data, (self.idx_data, None))

        data = data_seq[:, (data_seq.shape[1] - 1) - self.current_correct_lags[::-1], :]
        data = np.moveaxis(data, 1, 2).reshape(len(data), -1)
        features_X = pd.DataFrame(data, columns=self._features)
        return raw_ts_X, raw_ts_y, features_X, y


# Factory Object
class TransformersFactory:
    def __init__(self):
        self.models = {
            "StandardScalerTransformer": StandardScalerTransformer,
            "LabelEncodingTransformer": LabelEncodingTransformer,
            "OneHotEncodingTransformer": OneHotEncodingTransformer,
            "LastKnownNormalizer": LastKnownNormalizer,
            "DifferenceNormalizer": DifferenceNormalizer,
            "TimeToNumGenerator": TimeToNumGenerator,
            "DateSeasonsGenerator": DateSeasonsGenerator,
            "LagTransformer": LagTransformer,
        }

    def get_allowed(self):
        return sorted(list(self.models.keys()))

    def __getitem__(self, params):
        return self.models[params["transformer_name"]](**params["transformer_params"])
