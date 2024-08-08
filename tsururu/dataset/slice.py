"""Module for creating indexes and manipulating data."""

import re
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.tseries.offsets import MonthEnd
from scipy import stats as st


class IndexSlicer:
    """Combines ways to create indexes and manipulate data."""

    @staticmethod
    def _timedelta_above_daily_freq(
        d_multiplier: int,
        check_end_regex: str,
        d_from_series: int,
        freq_name: str,
        inferred_freq: str,
    ) -> Tuple[Union[pd.DateOffset, MonthEnd], str]:
        """Calculate the timedelta based on the given parameters for
            the frequencies above daily.

        Args:
            d_multiplier: the multiplier for the number of months.
            check_end_regex: the regular expression to check if the
                inferred frequency is an end frequency.
            d_from_series: the number of periods from the series.
            freq_name: the name of the frequency.
            inferred_freq: the inferred frequency.

        Returns:
            the calculated timedelta and the information about the
                frequency and period.

        """
        if inferred_freq and re.match(check_end_regex, inferred_freq):
            delta = MonthEnd(d_multiplier * d_from_series)
            freq_period_info = f"freq: {freq_name}End; period: {d_from_series}"
        else:
            delta = pd.DateOffset(months=d_multiplier * d_from_series)
            freq_period_info = f"freq: {freq_name}; period: {d_from_series}"

        return delta, freq_period_info

    def timedelta(
        self,
        x: np.ndarray,
        delta: Optional[pd.DateOffset] = None,
        return_freq_period_info: bool = False,
    ) -> Union[Tuple[np.ndarray, pd.DateOffset], Tuple[np.ndarray, pd.DateOffset, str]]:
        """Returns the difference between neighboring observations in
            the array in terms of delta and the delta itself.

        Args:
            x: array with datetime points.
            delta: custom offset if needed.
            return_freq_period_info: either to return information about
                inferred frequency and period.

        Returns:
            difference between neighboring observations and the delta
                itself; if return_freq_period_info is True, return
                information about inferred frequency and period.

        Raises:
            AssertionError: if the frequency and period are failed
                to be defined.

        Notes:
            1. It is used to correctly generate indexes for
                observations without mixing observations
                with different IDs.

        """

        if not is_datetime(x):
            x = pd.to_datetime(x)

        if delta is None:
            inferred_freq = pd.infer_freq(x[-3:])  # Need at least 3 dates to infer frequency
            delta = x.diff().iloc[-1]

            # N Years
            if delta > pd.Timedelta(days=360) and (delta.days % 365 == 0 or delta.days % 366 == 0):
                delta, freq_period_info = self._timedelta_above_daily_freq(
                    d_multiplier=12,
                    check_end_regex=r"\b\d*A-|\b\d*YE-",
                    d_from_series=x.dt.year.diff().values[-1],
                    freq_name="Year",
                    inferred_freq=inferred_freq,
                )

            # N Quarters and Months
            elif delta > pd.Timedelta(days=27):
                if delta > pd.Timedelta(days=88):
                    check_end_regex = r"\b\d*Q-|\b\d*QE-"
                else:
                    check_end_regex = r"\b\d*M\b|\b\d*ME\b"
                delta, freq_period_info = self._timedelta_above_daily_freq(
                    d_multiplier=1,
                    check_end_regex=check_end_regex,
                    d_from_series=st.mode(x.dt.month.diff())[0],
                    freq_name="Month",
                    inferred_freq=inferred_freq,
                )

            # N Days
            elif delta >= pd.Timedelta(days=1):
                freq_period_info = f"freq: Day; period: {delta.days}"

            # N Hours; Min; Sec; etc
            elif delta <= pd.Timedelta(days=1):
                freq_period_info = f"freq: less then Day (Hour, Min, Sec, etc); period: {delta.total_seconds()} seconds"
        else:
            freq_period_info = f"Custom OffSet: {delta}"

        assert delta, "either or both frequency and period are failed to be defined."

        if return_freq_period_info:
            return x.diff().fillna(delta).values, delta, freq_period_info

        return x.diff().fillna(delta).values, delta

    @staticmethod
    def get_cols_idx(
        data: pd.DataFrame, columns: Union[str, Sequence[str]]
    ) -> Union[int, np.ndarray]:
        """Get numeric index of columns by column names.

        Args:
            data: source dataframe.
            columns: sequence of columns or single column.

        Returns:
            sequence of int indexes or single int.

        """
        if type(columns) is str:
            idx = data.columns.get_loc(columns)
        else:
            idx = data.columns.get_indexer(columns)

        return idx

    @staticmethod
    def get_slice(data: pd.DataFrame, k: Tuple[np.ndarray]) -> np.ndarray:
        """Get 3d slice.

        Args:
            data: source dataframe.
            k: tuple of integer sequences.

        Returns:
            slice.

        """
        rows, cols = k
        if cols is None:
            if isinstance(data, np.ndarray):
                new_data = data[rows, :]
            else:
                # new_data = data.iloc[:, :].values[rows]
                new_data = data.values[rows]
        else:
            if isinstance(data, np.ndarray):
                new_data = data[rows, cols]
            else:
                new_data = data.iloc[:, cols].values[rows]

        if len(new_data.shape) == 2:
            return np.expand_dims(new_data, axis=0)
        return new_data

    def ids_from_date(
        self,
        data: pd.DataFrame,
        date_column: str,
        delta: Optional[bool] = None,
        return_delta: bool = False,
    ) -> List[int]:
        """Find indexes by which the dataset can be divided into
            segments that are "identical" in terms of time stamps, but
            different in terms of some identifier.

        Args:
            data: source dataframe.
            date_column: date column name in source dataframe.
            delta: custom offset if needed.
            return_delta: whether to return value of delta.

        Returns:
            indexes of the ends of segments; if return_delta is True,
                return value of delta.

        """
        _, time_delta = self.timedelta(data[date_column], delta=delta)
        ids = (
            np.argwhere(
                data[date_column][1:].values != (data[date_column] + time_delta)[:-1].values
            )
            + 1
        )
        if return_delta:
            return list(ids.reshape(-1)), time_delta

        return list(ids.reshape(-1))

    def _rolling_window(
        self,
        a: np.ndarray,
        window: int,
        step: int,
        from_last: bool = True,
    ) -> np.ndarray:
        """Generate a rolling window view of a numpy array.

        Args:
            a: the input array.
            window: the size of the window.
            step: the step size between windows.
            from_last: whether to start the window from last element.

        Returns:
            the rolling window view of the input array.

        """
        sliding_window = np.lib.stride_tricks.sliding_window_view(a, window)

        return sliding_window[(len(a) - window) % step if from_last else 0 :][::step]

    def _create_idx_data(
        self, data: np.ndarray, horizon: int, history: int, step: int, *_
    ) -> np.ndarray:
        """Create index data for train observations' windows.

        Args:
            data: the input data array.
            horizon: the number of steps to predict into the future.
            history: the number of past steps to consider.
            step: the step size between each window.

        Returns:
            the index data array for train observations' windows.

        """
        return self._rolling_window(np.arange(len(data))[:-horizon], history, step)

    def _create_idx_target(
        self,
        data: np.ndarray,
        horizon: int,
        history: int,
        step: int,
        n_last_horizon: Optional[int],
    ) -> np.ndarray:
        """Create index data for targets.

        Args:
            data: the input data array.
            horizon: the number of steps to predict into the future.
            history: the number of past steps to consider.
            step: the step size between each window.
            n_last_horizon: how many last points we wish to leave.

        Returns:
            the index data array for targets.

        """
        return self._rolling_window(np.arange(len(data))[history:], horizon, step)[
            :, -n_last_horizon:
        ]

    def _create_idx_test(
        self, data: np.ndarray, horizon: int, history: int, step: int, *_
    ) -> np.ndarray:
        """Create index data for test observations' windows.

        Args:
            data: the input data array.
            horizon: the number of steps to predict into the future.
            history: the number of past steps to consider.
            step: the step size between each window.

        Returns:
            the index data array for test observations' windows.

        """
        return self._rolling_window(np.arange(len(data)), history, step)[-(horizon + 1) : -horizon]

    def _get_ids(
        self,
        func,
        data: np.ndarray,
        horizon: int,
        history: int,
        step: int,
        ids: np.ndarray,
        cond: int = 0,
        n_last_horizon: Optional[int] = None,
    ) -> np.ndarray:
        """Get indices for creating windows of data.

        Args:
            func: the function to create index data.
            data: the input data array.
            horizon: the number of steps to predict into the future.
            history: the number of past steps to consider.
            step: the step size between each window.
            ids: indexes of the ends of segments.
            cond: the condition for segment length.
            n_last_horizon: how many last points to leave.

        Returns:
            the index data array.

        """
        prev = 0
        inds = []
        for i, split in enumerate(ids + [len(data)]):
            if isinstance(data, np.ndarray):
                segment = data[prev:split]
            else:
                segment = data.iloc[prev:split]
            if len(segment) >= cond:
                ind = func(segment, horizon, history, step, n_last_horizon) + prev
                inds.append(ind)
            prev = split
        inds = np.vstack(inds)

        return inds

    def create_idx_data(
        self,
        data: pd.DataFrame,
        horizon: int,
        history: int,
        step: int,
        ids: Optional[np.ndarray] = None,
        date_column: Optional[str] = None,
        delta: Optional[pd.DateOffset] = None,
    ):
        """Find indices that, when applied to the original dataset,
            can be used to obtain windows for building
            train observations' features.

        Args:
            data: source dataframe.
            horizon: the number of steps to predict into the future.
            history: the number of past steps to consider.
            step: number of points to take the next observation.
            ids: indexes of the ends of segments.
            date_column: date column name in source dataframe,
                needs in the absence of ids.

        Returns:
            indices of train observations' windows.

        """
        if ids is None:
            ids = self.ids_from_date(data, date_column, delta=delta)

        seq_idx_data = self._get_ids(
            self._create_idx_data,
            data,
            horizon,
            history,
            step,
            ids,
            history + horizon,
        )

        return seq_idx_data

    def create_idx_test(
        self,
        data: pd.DataFrame,
        horizon: int,
        history: int,
        step: int,
        ids: Optional[np.ndarray] = None,
        date_column: Optional[str] = None,
        delta: Optional[pd.DateOffset] = None,
    ):
        """Find indices that, when applied to the original dataset,
            can be used to obtain windows for building
            test observations' features.

        Arguments:
            data: source dataframe.
            horizon: the number of steps to predict into the future.
            history: the number of past steps to consider.
            step: number of points to take the next observation.
            ids: indexes of the ends of segments.
            date_column: date column name in source dataframe,
                needs in the absence of ids.

        Returns:
            indices of test observations' windows.

        """
        if ids is None:
            ids = self.ids_from_date(data, date_column, delta=delta)

        seq_idx_test = self._get_ids(
            self._create_idx_test,
            data,
            horizon,
            history,
            step,
            ids,
            history,
        )

        return seq_idx_test

    def create_idx_target(
        self,
        data: pd.DataFrame,
        horizon: int,
        history: int,
        step: int,
        ids: Optional[np.ndarray] = None,
        date_column: Optional[str] = None,
        delta: Optional[pd.DateOffset] = None,
        n_last_horizon: Optional[int] = None,
    ):
        """Find indices that, when applied to the original dataset,
            can be used to obtain targets.

        Arguments:
            data: source dataframe.
            horizon: number of points to prediction.
            history: number of points to use for prediction.
            step: number of points to take the next observation.
            ids: indexes of the ends of segments.
            date_column: date column name in source dataframe,
                needs in the absence of ids.
            n_last_horizon: how many last points we wish to leave.

        Returns:
            indices of targets.
        """
        if ids is None:
            ids = self.ids_from_date(data, date_column, delta=delta)

        if n_last_horizon is None:
            n_last_horizon = horizon

        seq_idx_target = self._get_ids(
            self._create_idx_target,
            data,
            horizon,
            history,
            step,
            ids,
            history + horizon,
            n_last_horizon,
        )

        return seq_idx_target
