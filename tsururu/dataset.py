from typing import Optional, List, Union, Tuple
from numpy.typing import NDArray

import numpy as np
import pandas as pd

from pandas.api.types import is_datetime64_any_dtype as is_datetime


class IndexSlicer:
    """A class that combines ways to create indexes and with the help
    of which the data is manipulated.
    """

    @staticmethod
    def timedelta(x: Tuple[NDArray[Union[np.integer, np.floating]], pd.Timedelta]):
        """
        Returns the difference between neighboring observations
            in the array in terms of delta and the delta itself.
        Then it is used to correctly generate indexes for observations
            without mixing observations with different IDs.

        Arguments:
            x: array with datetime points.

        Returns:
            The difference and the delta.
        """
        if not is_datetime(x):
            x = pd.to_datetime(x)

        delta = x.diff().iloc[-1]

        if delta <= pd.Timedelta(days=1):
            return x.diff().fillna(delta).values, delta

        if delta > pd.Timedelta(days=360):
            d = x.dt.year.diff()
            delta = d.iloc[-1]
            return d.fillna(delta).values, delta

        elif delta > pd.Timedelta(days=27):
            d = x.dt.month.diff() + 12 * x
            delta = d.iloc[-1]
            return d.fillna(delta).values, delta

        else:
            return x.diff().fillna(delta).values, delta

    @staticmethod
    def get_cols_idx(data: pd.DataFrame, columns: List):
        """Get numeric index of columns by column names.

        Arguments:
            data: source dataframe.
            columns: sequence of columns of single column.

        Returns:
            sequence of int indexes or single int.
        """
        if type(columns) is str:
            idx = data.columns.get_loc(columns)
        else:
            idx = data.columns.get_indexer(columns)
        return idx

    @staticmethod
    def get_slice(
        data: pd.DataFrame,
        k: Tuple[NDArray[int], NDArray[int]],
    ) -> pd.DataFrame:
        """Get 3d slice.

        Arguments:
            data: source dataframe.
            k: Tuple[IntIdx, IntIdx]; Tuple of integer sequences.

        Returns:
            slice.
        """
        rows, cols = k
        if cols is None:
            if isinstance(data, np.ndarray):
                new_data = data[rows, :]
            else:
                new_data = data.iloc[:, :].values[rows]
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
        return_delta: bool = False,
    ) -> List[int]:
        """Find indexes by which the dataset can be divided into
            segments that are "identical" in terms of time stamps,
            but different in terms of some identifier.

        Arguments:
            data: source dataframe.
            date_column: date column name in source dataframe.

        Returns:
            Indexes of the ends of segments.
        """
        vals, time_delta = self.timedelta(pd.to_datetime(data[date_column]))
        ids = list(np.argwhere(vals != time_delta).flatten())
        if return_delta:
            return ids, time_delta
        return ids

    def _rolling_window(
        self, a: NDArray[np.floating], window: int, step: int, from_last: bool = True
    ):
        sliding_window = np.lib.stride_tricks.sliding_window_view(a, window)
        return sliding_window[(len(a) - window) % step if from_last else 0 :][::step]

    def _create_idx_data(
        self,
        data: NDArray[np.floating],
        horizon: int,
        history: int,
        step: int,
        _,
        __,
    ):
        return self._rolling_window(np.arange(len(data))[:-horizon], history, step)

    def _create_idx_target(
        self,
        data: NDArray[np.floating],
        horizon: int,
        history: int,
        step: int,
        _,
        n_last_horizon: Optional[int],
    ):
        return self._rolling_window(np.arange(len(data))[history:], horizon, step)[
            :, -n_last_horizon:
        ]

    def _create_idx_test(
        self,
        data: NDArray[np.floating],
        horizon: int,
        history: int,
        step: int,
        _,
        __,
    ):
        return self._rolling_window(np.arange(len(data)), history, step)[-(horizon + 1) : -horizon]

    def _get_ids(
        self,
        func,
        data: NDArray[np.floating],
        horizon: int,
        history: int,
        step: int,
        ids: NDArray[np.integer],
        cond: int = 0,
        n_last_horizon: Optional[int] = None,
    ):
        prev = 0
        inds = []
        for i, split in enumerate(ids + [len(data)]):
            if isinstance(data, np.ndarray):
                segment = data[prev:split]
            else:
                segment = data.iloc[prev:split]
            if len(segment) >= cond:
                ind = func(segment, horizon, history, step, i, n_last_horizon) + prev
                inds.append(ind)
            prev = split
        inds = np.vstack(inds)
        return inds

    def create_idx_data(
        self,
        data: NDArray[np.floating],
        horizon: int,
        history: int,
        step: int,
        ids: Optional[NDArray[np.integer]] = None,
        date_column: Optional[str] = None,
    ):
        """Find indices that, when applied to the original dataset,
            can be used to obtain windows for building
            train observations' features.

        Arguments:
            data: source dataframe.
            horizon: number of points to prediction.
            history: number of points to use for prediction.
            step: number of points to take the next observation.
            ids: indexes of the ends of segments.
            date_column: date column name in source dataframe,
                needs in the absence of ids.

        Returns:
            indices of train observations' windows.
        """
        if ids is None:
            ids = self.ids_from_date(data, date_column)

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
        data: NDArray[np.floating],
        horizon: int,
        history: int,
        step: int,
        ids: Optional[NDArray[np.integer]] = None,
        date_column: Optional[str] = None,
    ):
        """Find indices that, when applied to the original dataset,
            can be used to obtain windows for building
            test observations' features.

        Arguments:
            data: source dataframe.
            horizon: number of points to prediction.
            history: number of points to use for prediction.
            step: number of points to take the next observation.
            ids: indexes of the ends of segments.
            date_column: date column name in source dataframe,
                needs in the absence of ids.

        Returns:
            indices of test observations' windows.
        """
        if ids is None:
            ids = self.ids_from_date(data, date_column)

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
        data: NDArray[np.floating],
        horizon: int,
        history: int,
        step: int,
        ids: Optional[NDArray[np.integer]] = None,
        date_column: Optional[str] = None,
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
            ids = self.ids_from_date(data, date_column)

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


class TSDataset:
    """Class for initializing data from pandas, including:
    -- reading data;
    -- handling id and date columns.
    Args:
        data: source dataframe.
        columns_and_features_params: cols roles, types and transformers
            for example:
                {
                    "target": {
                        "column": ["value"],
                        "type": "continious",
                        "features": [LagTransformer(...)]
                    },
                    "date": {...},
                    ...
                }.
        history: number of points to use for prediction.
        step: number of points to take the next observation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        columns_and_features_params: dict,
        history: int,
        step: int = 1,
    ):
        # Columns typing
        for _, role_dict in columns_and_features_params.items():
            column_name = role_dict["column"][0]
            column_type = role_dict["type"]
            if column_type == "continious":
                data[column_name] = data[column_name].astype("float")
            elif column_type == "datetime":
                data[column_name] = pd.to_datetime(data[column_name])

        self.seq_data = data
        self.columns_and_features_params = columns_and_features_params
        self.history = history
        self.step = step
        self.id_column = columns_and_features_params["id"]["column"][0]
        self.target_column = columns_and_features_params["target"]["column"][0]
        self.date_column = columns_and_features_params["date"]["column"][0]

    def make_padded_test(
        self,
        horizon: int,
        test_last: bool = True,
        id_col_name: Union[None, str, List[str]] = None,
    ):
        """Generate a test dataframe with empty values in place of
            future predicted values.

        Arguments:
            horizon: number of points to prediction.
            test_last: whether test data are built by the last point.
            id_col_name: column to divide time series
                (sometimes different from the original id column).
        """

        def _crop_segment(
            segment: NDArray[Union[np.floating, np.str_]],
            test_last: bool,
        ) -> NDArray[Union[np.floating, np.str_]]:
            if test_last:
                return segment[-self.history :]
            return segment[-self.history - horizon : -horizon]

        def _pad_segment(
            segment: NDArray[Union[np.floating, np.str_]],
            horizon: int,
            time_delta: pd.Timedelta,
            date_col_id: Optional[int],
            id_col_id: Optional[Union[str, NDArray[np.str_]]],
        ) -> NDArray[Union[np.floating, np.str_]]:
            result = np.full((horizon, segment.shape[1]), np.nan)

            last_date = segment[-1, date_col_id]
            new_dates = pd.date_range(last_date + time_delta, periods=horizon, freq=time_delta)
            result[:, date_col_id] = new_dates

            if isinstance(id_col_id, np.ndarray):
                for i in range(len(id_col_id)):
                    result[:, id_col_id[i]] = np.repeat(segment[0, id_col_id[i]], horizon)
            else:
                result[:, id_col_id] = np.repeat(segment[0, id_col_id], horizon)
            return result

        index_slicer = IndexSlicer()
        columns = self.seq_data.columns
        date_col_id = index_slicer.get_cols_idx(self.seq_data, self.date_column)
        if id_col_name is None:
            id_col_name = self.id_column
        id_col_id = index_slicer.get_cols_idx(self.seq_data, id_col_name)

        # Find indices for segments
        ids, time_delta = index_slicer.ids_from_date(
            self.seq_data, self.date_column, return_delta=True
        )

        data = self.seq_data.to_numpy()

        segments = np.split(data, ids)
        segments = [_crop_segment(segment, test_last) for segment in segments]

        # Find padded parts for each segment
        padded_segments_results = [
            _pad_segment(segment, horizon, time_delta, date_col_id, id_col_id)
            for segment in segments
        ]

        # Concatenate together
        result = np.vstack(np.concatenate((segments, padded_segments_results), axis=1))
        result = pd.DataFrame(result, columns=columns)
        result[self.date_column] = pd.to_datetime(result[self.date_column])
        result[self.id_column] = result[self.id_column].astype("int")
        other = [col for col in columns if col not in [self.id_column, self.date_column]]
        result[other] = result[other].astype("float")
        return result
