"""Module for transformers for datetime features."""

from typing import List, Optional, Sequence

import holidays
import numpy as np
import pandas as pd

from ..dataset.slice import IndexSlicer
from .base import FeaturesGenerator
from .utils import date_attrs

index_slicer = IndexSlicer()


class TimeToNumGenerator(FeaturesGenerator):
    """A transformer that converts datetime to difference with basic_date.

    Args:
        basic_date: date relating to which normalization takes place.
        from_target_date: if True, features are built from the targets' dates;
            otherwise, features are built from the last training dates.
        horizon: forecast horizon.
        delta: frequency of the time series.

    """

    def __init__(
        self,
        basic_date: Optional[str] = "2020-01-01",
        from_target_date: Optional[bool] = False,
        delta: Optional[pd.DateOffset] = None,
    ):
        super().__init__()
        self.basic_date = basic_date
        self.from_target_date = from_target_date
        self.delta = delta

    def fit(
        self, data: dict, input_features: Optional[Sequence[str]] = None
    ) -> "TimeToNumGenerator":
        """Fit transformer on "elongated series" and return it's instance.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            input_features: array with names of columns to transform.

        Returns:
            self.

        """
        super().fit(data, input_features)

        self.output_features = [
            f"{column_name}__time_to_num" for column_name in self.input_features
        ]
        return self

    def transform(self, data: dict) -> dict:
        """Generate features in `raw_ts_X`.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        result_data = []
        for column_name in self.input_features:
            time_col = data["raw_ts_X"][column_name]

            _, time_delta = index_slicer.timedelta(time_col, delta=self.delta)
            if self.from_target_date:
                horizon = data["target_idx"][0, -1] - data["features_idx"][0, -1]
                time_col = time_col + horizon * time_delta

            new_arr = pd.to_datetime(time_col.to_numpy().reshape(-1), origin="unix")
            data_transformed = (
                (new_arr - np.datetime64(self.basic_date)) / np.timedelta64(1, self.delta)
            ).values.astype(np.float32)

            result_data.append(data_transformed)

        result_data = np.hstack(result_data)
        if result_data.ndim == 1:
            result_data = result_data.reshape(-1, 1)

        data["raw_ts_X"][self.output_features] = result_data

        return data


class DateSeasonsGenerator(FeaturesGenerator):
    """A transformer that generates features that reflect seasonality.

    Args:
        seasonalities: features to build.
        from_target_date: features are built from the targets' dates.
        horizon: forecast horizon.
        country: country code ("RU" for Russia).
        prov: province inside country.
        state: state inside country.

    Notes:
        1. In case when country is provided (it is possible to specify prov
            and state) indicator 'there is a holiday at that moment' will be
            generated.

    """

    def __init__(
        self,
        seasonalities: Optional[List[str]] = ["doy", "m", "wd", "d", "y"],
        from_target_date: Optional[bool] = False,
        country: Optional[str] = None,
        prov: Optional[str] = None,
        state: Optional[str] = None,
        delta: Optional[pd.DateOffset] = None,
    ):
        super().__init__()
        self.seasonalities = seasonalities
        self.from_target_date = from_target_date
        self._country = country
        self._prov = prov
        self._state = state
        self.delta = delta

    def fit(
        self, data: dict, input_features: Optional[Sequence[str]] = None
    ) -> "DateSeasonsGenerator":
        """Fit transformer on "elongated series" and return it's instance.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            input_features: array with names of columns to transform.

        Returns:
            self.

        """
        super().fit(data, input_features)

        self.output_features = []

        for s in self.seasonalities:
            self.output_features.extend(
                [f"{column_name}__season_{s}" for column_name in self.input_features]
            )
        if self._country is not None:
            self.output_features.extend(
                [f"{column_name}__season_hol" for column_name in self.input_features]
            )

        return self

    def transform(self, data: dict) -> dict:
        """Generate features in `raw_ts_X`.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.

        """
        result_data = []

        for column_name in self.input_features:
            time_col = data["raw_ts_X"][column_name]

            _, time_delta = index_slicer.timedelta(time_col, delta=self.delta)
            if self.from_target_date:
                horizon = data["idx_y"][0, -1] - data["idx_X"][0, -1]
                time_col = time_col + horizon * time_delta
            time_col = pd.to_datetime(time_col.to_numpy(), origin="unix")

            new_arr = np.empty((time_col.shape[0], len(self.output_features)), np.int32)

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
                dates, names = zip(*hol.items())
                new_arr[:, n] = np.isin(time_col.date, dates)
                n += 1
            result_data.append(new_arr)

        data["raw_ts_X"][self.output_features] = np.hstack(result_data)

        return data
