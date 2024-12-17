from typing import Optional, Sequence, Union
import polars as pl
import pandas as pd
import numpy as np
from numba import jit
from .slice import IndexSlicer, IndexSlicerPolars
import logging

slicer = IndexSlicer() # IndexSlicerPolars() 
logger = logging.getLogger(__name__)

class TSDatasetPolars:
    """Class for initializing data from pandas DataFrame.

    Args:
        data: source dataframe.
        columns_params: columns' roles and types.
        delta: the pd.DateOffset class for datetime features.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        columns_params: dict,
        delta: pd.DateOffset = None,
        print_freq_period_info: bool = True,
    ):
        # Convert Pandas DataFrame to Polars DataFrame
        # data = pl.from_pandas(data)

        # Columns typing
        for _, role_dict in columns_params.items():
            column_name = role_dict["columns"][0]
            column_type = role_dict["type"]
            if column_type == "continious":
                data = data.with_columns(data[column_name].cast(pl.Float64))
            elif column_type == "datetime":
                data = data.with_columns(data[column_name].str.strptime(pl.Datetime))

        self.data = data
        self.columns_params = columns_params
        self.delta = delta

        self._check_single_column()

        self.id_column = columns_params["id"]["columns"][0]
        self.target_column = columns_params["target"]["columns"][0]
        self.date_column = columns_params["date"]["columns"][0]

        self.data = data.sort([self.id_column, self.date_column])

        self._check_regular(print_freq_period_info)

    def _check_single_column(self):
        """Check that `target`, `id`, `date` columns contain only one column."""
        for role in ["target", "date", "id"]:
            assert (
                len(self.columns_params[role]["columns"]) == 1
            ), f"The `columns` container for role {role} should contain only one column"

    def _check_regular(self, print_freq_period_info):
        """Check that the data is regular."""
        ts_count = self.data[self.id_column].n_unique()

        _, delta, info = slicer.timedelta(
            self.data[self.date_column].to_pandas(), self.delta, return_freq_period_info=True
        )

        if print_freq_period_info:
            logger.info(info)

        min_data = self.data.select(pl.col(self.date_column).min()).item()
        max_data = self.data.select(pl.col(self.date_column).max()).item()

        reconstructed_data = pd.date_range(start=min_data, end=max_data, freq=delta)
        reconstructed_data = np.tile(reconstructed_data, ts_count)

        original_dates = self.data[self.date_column].to_numpy()

        if (
            reconstructed_data.shape[0] != self.data.shape[0]
            or not np.array_equal(reconstructed_data, original_dates)
        ):
            logger.warning(
                """
                It seems that the data is not regular. Please check the data and frequency info.
                For multivariate regimes, it is critical to have regular data.
                For global regimes, each regular part of the time series will be processed as a separate time series.
                """
            )

    def _crop_segment(
        self, segment: np.ndarray, test_last: bool, horizon: int, history: int
    ) -> np.ndarray:
        """Crop a segment of data based on the history and horizon."""
        if test_last:
            return segment[-history:]
        return segment[-history - horizon : -horizon]

    @staticmethod
    def _pad_segment(
        segment: np.ndarray,
        horizon: int,
        time_delta: Union[pd.Timedelta, pd.DateOffset],
        date_col_id: int,
        id_col_id: Sequence[int],
    ) -> np.ndarray:
        """Pad a segment of data with new rows based on the horizon."""
        result = np.full((horizon, segment.shape[1]), np.nan, dtype=object)

        last_date = segment[-1, date_col_id]
        new_dates = pd.date_range(last_date + time_delta, periods=horizon, freq=time_delta)
        result[:, date_col_id] = new_dates

        if isinstance(id_col_id, np.ndarray):
            for i in range(len(id_col_id)):
                result[:, id_col_id[i]] = np.repeat(segment[0, id_col_id[i]], horizon)
        else:
            result[:, id_col_id] = np.repeat(segment[0, id_col_id], horizon)

        return result

    def make_padded_test(
        self,
        horizon: int,
        history: int,
        test_last: bool = True,
        test_all: bool = False,
        step: Optional[int] = None,
        id_column_name: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """Generate a test dataframe with new rows containing NaN targets."""
        if test_all:
            current_test_ids = slicer.create_idx_data(
                self.data.to_pandas(),
                horizon,
                history,
                step,
                date_column=self.date_column,
            )
            extended_data = slicer.get_slice(self.data.to_pandas(), (current_test_ids, None))
            extended_data = pd.DataFrame(
                extended_data.reshape(-1, extended_data.shape[-1]),
                columns=self.data.columns,
            )
            extended_data["segment_col"] = np.repeat(
                np.arange(len(extended_data) // history), history
            )
            id_column_name = ["segment_col", self.id_column]
        else:
            extended_data = self.data.to_pandas()

        columns = self.data.columns
        date_col_id = slicer.get_cols_idx(extended_data, self.date_column)
        if id_column_name is None:
            id_column_name = self.id_column
        id_col_id = slicer.get_cols_idx(extended_data, id_column_name)

        ids, time_delta = slicer.ids_from_date(
            extended_data, self.date_column, delta=self.delta, return_delta=True
        )

        if test_all:
            ids = list(np.unique(extended_data.segment_col, return_index=True)[1])[1:]

        data = extended_data.to_numpy()

        segments = np.split(data, ids)
        segments = [
            self._crop_segment(segment, test_last, horizon, history) for segment in segments
        ]

        padded_segments_results = [
            self._pad_segment(segment, horizon, time_delta, date_col_id, id_col_id)
            for segment in segments
        ]

        result = np.vstack(np.concatenate((segments, padded_segments_results), axis=1))
        if test_all:
            result = pd.DataFrame(result, columns=list(columns) + ["segment_col"])
        else:
            result = pd.DataFrame(result, columns=columns)
        result[self.date_column] = pd.to_datetime(result[self.date_column])
        other = [col for col in columns if col not in [self.id_column, self.date_column]]
        result[other] = result[other].astype("float")

        return result


class TSDataset:
    """Class for initializing data from pandas DataFrame.

    Args:
        data: source dataframe.
        columns_params: columns' roles and types:
            for example:
                {
                    "target": {
                        "columns": ["value"],
                        "type": "continious",
                    },
                    "date": {...},
                    "id": {...},
                    "exog_1": {...},
                    "exog_2": {...},
                    ...,
                }.
        delta: the pd.DateOffset class. Usually generated
            automatically, but can be externally specified. Needs to
            create datetime features and new values.

    Notes:
        1. If the printed information about freq and period values is
            not correct for user's task, user should redefine `delta`
            parameter.

    """

    def _check_single_column(self):
        """Check that `target`, `id`, `date` columns contains only one
            column.

        Raises:
            AssertionError: if the `columns` container for one of the
                roles from `target`, `date`, `id` contains more than
                one column.

        """
        for role in ["target", "date", "id"]:
            assert (
                len(self.columns_params[role]["columns"]) == 1
            ), f"the `columns` container for role {role} should contain only one column"

    def _check_regular(self, print_freq_period_info):
        """Check that the data is regular.

        Raises:
            AssertionError: if the data is not regular.

        """
        ts_count = self.data.loc[:, self.id_column].nunique()

        _, delta, info = slicer.timedelta(
            self.data[self.date_column], self.delta, return_freq_period_info=True
        )

        if print_freq_period_info:
            logger.info(info)
            
        # Try to reconstruct regular data
        min_data = self.data.min()
        max_data = self.data.max()
        
        reconstructed_data = pd.date_range(
            start=min_data[self.date_column],
            end=max_data[self.date_column],
            freq=delta,
        )
        reconstructed_data = np.tile(reconstructed_data, ts_count)

        if (
            reconstructed_data.shape[0] != self.data.shape[0]
            or not np.all(reconstructed_data == self.data[self.date_column].values)
        ):
            logger.warning(
                f"""
                It seems that the data is not regular. Please, check the data and the frequency info.                
                For multivariate regime it is critical to have regular data.
                For global regime each regular part of time series will be processed as separate time series.           
                """
            )

    def __init__(
        self,
        data: pd.DataFrame,
        columns_params: dict,
        delta: pd.DateOffset = None,
        print_freq_period_info: bool = True,
    ):
        # Columns typing
        for _, role_dict in columns_params.items():
            column_name = role_dict["columns"][0]
            column_type = role_dict["type"]
            if column_type == "continious":
                data[column_name] = data[column_name].astype("float")
            elif column_type == "datetime":
                data[column_name] = pd.to_datetime(data[column_name])

        self.data = data
        self.columns_params = columns_params
        self.delta = delta

        self._check_single_column()

        self.id_column = columns_params["id"]["columns"][0]
        self.target_column = columns_params["target"]["columns"][0]
        self.date_column = columns_params["date"]["columns"][0]

        self.data = data.sort_values([self.id_column, self.date_column])

        self._check_regular(print_freq_period_info)

    def _crop_segment(
        self,
        segment: np.ndarray,
        test_last: bool,
        horizon: int,
        history: int,
    ) -> np.ndarray:
        """Crop a segment of data based on the history and horizon.

        Args:
            segment: the input segment of data to crop.
            test_last: if True, return the last history elements of the
                segment. Otherwise, return the elements between history
                and history + horizon.

        Returns:
            the cropped segment of data.

        """
        if test_last:
            return segment[-history:]

        return segment[-history - horizon : -horizon]

    @staticmethod
    def _pad_segment(
        segment: np.ndarray,
        horizon: int,
        time_delta: Union[pd.Timedelta, pd.DateOffset],
        date_col_id: int,
        id_col_id: Sequence[int],
    ) -> np.ndarray:
        """Pad a segment of data with new rows based on the horizon.

        Args:
            segment: the input segment of data to pad.
            horizon: the number of new rows to add to the segment.
            time_delta: the time_delta to use for the new date values.
            date_col_id: the index of the date column in the segment.
            id_col_id: the index of the id columns in the segment.

        Returns:
            padded segment of data.

        """
        result = np.full((horizon, segment.shape[1]), np.nan, dtype=object)

        last_date = segment[-1, date_col_id]
        new_dates = pd.date_range(last_date + time_delta, periods=horizon, freq=time_delta)
        result[:, date_col_id] = new_dates

        if isinstance(id_col_id, np.ndarray):
            for i in range(len(id_col_id)):
                result[:, id_col_id[i]] = np.repeat(segment[0, id_col_id[i]], horizon)
        else:
            result[:, id_col_id] = np.repeat(segment[0, id_col_id], horizon)

        return result

    def make_padded_test(
        self,
        horizon: int,
        history: int,
        test_last: bool = True,
        test_all: bool = False,
        step: Optional[int] = None,
        id_column_name: Optional[Union[str, Sequence[str]]] = None,
    ):
        """Generate a test dataframe with new rows with NaN targets.

        Args:
            horizon: number of points to prediction
                (number of new rows to add to each segment).
            history: number of previous for feature generating.
            test_last: if True, return generated test data built by
                the last point.
            test_all: if True, return generated test data for all
                points (like rolling forecast).
            step:  in how many points to take the next observation while
                making samples' matrix.
                Needs for test_all=True.
            id_column_name: name of the column(s) by which the data is
                split (in some cases it is different from the originalß
                id column(s)).

        Notes:
            1. The new rows are filled with NaN target values,
                generated datetimes values (based on the time_delta)
                and the same id values as the last row of the segment.

        Returns:
            the padded test dataset.

        """
        if test_all:
            current_test_ids = slicer.create_idx_data(
                self.data,
                horizon,
                history,
                step,
                date_column=self.date_column,
            )
            extended_data = slicer.get_slice(self.data, (current_test_ids, None))
            extended_data = pd.DataFrame(
                extended_data.reshape(-1, extended_data.shape[-1]),
                columns=self.data.columns,
            )
            extended_data_nrows = extended_data.shape[0]

            extended_data["segment_col"] = np.repeat(
                np.arange(extended_data_nrows // history), history
            )
            id_column_name = ["segment_col", self.id_column]
        else:
            extended_data = self.data

        columns = self.data.columns
        date_col_id = slicer.get_cols_idx(extended_data, self.date_column)
        if id_column_name is None:
            id_column_name = self.id_column
        id_col_id = slicer.get_cols_idx(extended_data, id_column_name)

        # Find indices for segments
        ids, time_delta = slicer.ids_from_date(
            extended_data, self.date_column, delta=self.delta, return_delta=True
        )

        if test_all:
            ids = list(np.unique(extended_data.segment_col, return_index=True)[1])[1:]

        data = extended_data.to_numpy()

        segments = np.split(data, ids)
        segments = [
            self._crop_segment(segment, test_last, horizon, history) for segment in segments
        ]

        # Find padded parts for each segment
        padded_segments_results = [
            self._pad_segment(segment, horizon, time_delta, date_col_id, id_col_id)
            for segment in segments
        ]

        # Concatenate together
        result = np.vstack(np.concatenate((segments, padded_segments_results), axis=1))
        if test_all:
            result = pd.DataFrame(result, columns=list(columns) + ["segment_col"])
        else:
            result = pd.DataFrame(result, columns=columns)
        result[self.date_column] = pd.to_datetime(result[self.date_column])
        other = [col for col in columns if col not in [self.id_column, self.date_column]]
        result[other] = result[other].astype("float")

        return result

class TSDatasetNumba:
    """Class for initializing data from pandas DataFrame.

    Args:
        data: source dataframe.
        columns_params: columns' roles and types.
        delta: the pd.DateOffset class. Usually generated
            automatically, but can be externally specified.

    Notes:
        1. If the printed information about freq and period values is
            not correct for user's task, user should redefine `delta`
            parameter.
    """

    def _check_single_column(self):
        """Check that `target`, `id`, `date` columns contain only one column."""
        for role in ["target", "date", "id"]:
            assert (
                len(self.columns_params[role]["columns"]) == 1
            ), f"The `columns` container for role {role} should contain only one column."

    def _check_regular(self, print_freq_period_info):
        """Check that the data is regular."""
        ts_count = self.data.loc[:, self.id_column].nunique()

        _, delta, info = slicer.timedelta(
            self.data[self.date_column], self.delta, return_freq_period_info=True
        )

        if print_freq_period_info:
            logger.info(info)
            
        # Try to reconstruct regular data
        min_data = self.data.min()
        max_data = self.data.max()
        
        reconstructed_data = pd.date_range(
            start=min_data[self.date_column],
            end=max_data[self.date_column],
            freq=delta,
        )
        reconstructed_data = np.tile(reconstructed_data, ts_count)

        if (
            reconstructed_data.shape[0] != self.data.shape[0]
            or not np.all(reconstructed_data == self.data[self.date_column].values)
        ):
            logger.warning(
                "The data seems irregular. Ensure regularity for better processing."
            )

    def __init__(
        self,
        data: pd.DataFrame,
        columns_params: dict,
        delta: pd.DateOffset = None,
        print_freq_period_info: bool = True,
    ):
        # Columns typing
        for _, role_dict in columns_params.items():
            column_name = role_dict["columns"][0]
            column_type = role_dict["type"]
            if column_type == "continious":
                data[column_name] = data[column_name].astype("float")
            elif column_type == "datetime":
                data[column_name] = pd.to_datetime(data[column_name])

        self.data = data
        self.columns_params = columns_params
        self.delta = delta

        self._check_single_column()

        self.id_column = columns_params["id"]["columns"][0]
        self.target_column = columns_params["target"]["columns"][0]
        self.date_column = columns_params["date"]["columns"][0]

        self.data = data.sort_values([self.id_column, self.date_column])

        self._check_regular(print_freq_period_info)

    @staticmethod
    @jit(nopython=True)
    def _crop_segment(
        segment: np.ndarray,
        test_last: bool,
        horizon: int,
        history: int,
    ) -> np.ndarray:
        """Crop a segment of data based on the history and horizon."""
        if test_last:
            return segment[-history:]
        return segment[-history - horizon : -horizon]

    @staticmethod
    @jit(nopython=True)
    def _pad_segment(
        segment: np.ndarray,
        horizon: int,
        time_delta: Union[pd.Timedelta, pd.DateOffset],
        date_col_id: int,
        id_col_id: Sequence[int],
    ) -> np.ndarray:
        """Pad a segment of data with new rows based on the horizon."""
        result = np.full((horizon, segment.shape[1]), np.nan, dtype=np.float64)

        last_date = segment[-1, date_col_id]
        for i in range(horizon):
            result[i, date_col_id] = last_date + (i + 1) * time_delta

        if isinstance(id_col_id, np.ndarray):
            for i in range(len(id_col_id)):
                result[:, id_col_id[i]] = segment[0, id_col_id[i]]
        else:
            result[:, id_col_id] = segment[0, id_col_id]

        return result

    def make_padded_test(
        self,
        horizon: int,
        history: int,
        test_last: bool = True,
        test_all: bool = False,
        step: Optional[int] = None,
        id_column_name: Optional[Union[str, Sequence[str]]] = None,
    ):
        """Generate a test dataframe with new rows with NaN targets."""
        if test_all:
            current_test_ids = slicer.create_idx_data(
                self.data,
                horizon,
                history,
                step,
                date_column=self.date_column,
            )
            extended_data = slicer.get_slice(self.data, (current_test_ids, None))
            extended_data = pd.DataFrame(
                extended_data.reshape(-1, extended_data.shape[-1]),
                columns=self.data.columns,
            )
            extended_data_nrows = extended_data.shape[0]

            extended_data["segment_col"] = np.repeat(
                np.arange(extended_data_nrows // history), history
            )
            id_column_name = ["segment_col", self.id_column]
        else:
            extended_data = self.data

        columns = self.data.columns
        date_col_id = slicer.get_cols_idx(extended_data, self.date_column)
        if id_column_name is None:
            id_column_name = self.id_column
        id_col_id = slicer.get_cols_idx(extended_data, id_column_name)

        # Find indices for segments
        ids, time_delta = slicer.ids_from_date(
            extended_data, self.date_column, delta=self.delta, return_delta=True
        )

        if test_all:
            ids = list(np.unique(extended_data.segment_col, return_index=True)[1])[1:]

        data = extended_data.to_numpy()

        segments = np.split(data, ids)
        segments = [
            self._crop_segment(segment, test_last, horizon, history) for segment in segments
        ]

        # Find padded parts for each segment
        padded_segments_results = [
            self._pad_segment(segment, horizon, time_delta, date_col_id, id_col_id)
            for segment in segments
        ]

        # Concatenate together
        result = np.vstack(np.concatenate((segments, padded_segments_results), axis=1))
        if test_all:
            result = pd.DataFrame(result, columns=list(columns) + ["segment_col"])
        else:
            result = pd.DataFrame(result, columns=columns)
        result[self.date_column] = pd.to_datetime(result[self.date_column])
        other = [col for col in columns if col not in [self.id_column, self.date_column]]
        result[other] = result[other].astype("float")

        return result


class TSDatasetNumbaPolars:
    """Class for initializing data from pandas DataFrame.

    Args:
        data: source dataframe.
        columns_params: columns' roles and types.
        delta: the pd.DateOffset class. Usually generated
            automatically, but can be externally specified.

    Notes:
        1. If the printed information about freq and period values is
            not correct for user's task, user should redefine `delta`
            parameter.
    """

    def _convert_to_polars(self, data: pd.DataFrame) -> pl.DataFrame:
        """Convert a pandas DataFrame to a Polars DataFrame."""
        return pl.from_pandas(data)

    def _convert_to_pandas(self, data: pl.DataFrame) -> pd.DataFrame:
        """Convert a Polars DataFrame to a Pandas DataFrame."""
        return data.to_pandas()

    def _check_single_column(self):
        """Check that `target`, `id`, `date` columns contain only one column."""
        for role in ["target", "date", "id"]:
            assert (
                len(self.columns_params[role]["columns"]) == 1
            ), f"The `columns` container for role {role} should contain only one column."

    def _check_regular(self, print_freq_period_info):
        """Check that the data is regular."""
        ts_count = self.data[self.id_column].n_unique()

        _, delta, info = slicer.timedelta(
            self.data[self.date_column].to_pandas(), self.delta, return_freq_period_info=True
        )

        if print_freq_period_info:
            logger.info(info)

        min_date = self.data[self.date_column].min()
        max_date = self.data[self.date_column].max()

        reconstructed_data = pd.date_range(start=min_date, end=max_date, freq=delta)
        reconstructed_data = np.tile(reconstructed_data, ts_count)

        if len(reconstructed_data) != self.data.shape[0] or not np.all(
            reconstructed_data == self.data[self.date_column].to_pandas().values
        ):
            logger.warning(
                "The data seems irregular. Ensure regularity for better processing."
            )

    def __init__(
        self,
        data: pd.DataFrame,
        columns_params: dict,
        delta: pd.DateOffset = None,
        print_freq_period_info: bool = True,
    ):
        # Convert input pandas DataFrame to Polars DataFrame
        # data = self._convert_to_polars(data)

        # Columns typing
        for _, role_dict in columns_params.items():
            column_name = role_dict["columns"][0]
            column_type = role_dict["type"]
            if column_type == "continious":
                data = data.with_columns(pl.col(column_name).cast(pl.Float64))
            elif column_type == "datetime":
                data = data.with_columns(pl.col(column_name).str.to_datetime())

        self.data = data
        self.columns_params = columns_params
        self.delta = delta

        self._check_single_column()

        self.id_column = columns_params["id"]["columns"][0]
        self.target_column = columns_params["target"]["columns"][0]
        self.date_column = columns_params["date"]["columns"][0]

        self.data = self.data.sort([self.id_column, self.date_column])

        self._check_regular(print_freq_period_info)

    @staticmethod
    @jit(nopython=True)
    def _crop_segment(
        segment: np.ndarray,
        test_last: bool,
        horizon: int,
        history: int,
    ) -> np.ndarray:
        """Crop a segment of data based on the history and horizon."""
        if test_last:
            return segment[-history:]
        return segment[-history - horizon : -horizon]

    @staticmethod
    @jit(nopython=True)
    def _pad_segment(
        segment: np.ndarray,
        horizon: int,
        time_delta: Union[pd.Timedelta, pd.DateOffset],
        date_col_id: int,
        id_col_id: Sequence[int],
    ) -> np.ndarray:
        """Pad a segment of data with new rows based on the horizon."""
        result = np.full((horizon, segment.shape[1]), np.nan, dtype=np.float64)

        last_date = segment[-1, date_col_id]
        for i in range(horizon):
            result[i, date_col_id] = last_date + (i + 1) * time_delta

        if isinstance(id_col_id, np.ndarray):
            for i in range(len(id_col_id)):
                result[:, id_col_id[i]] = segment[0, id_col_id[i]]
        else:
            result[:, id_col_id] = segment[0, id_col_id]

        return result

    def make_padded_test(
        self,
        horizon: int,
        history: int,
        test_last: bool = True,
        test_all: bool = False,
        step: Optional[int] = None,
        id_column_name: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """Generate a test dataframe with new rows with NaN targets."""
        extended_data = self.data

        if test_all:
            current_test_ids = slicer.create_idx_data(
                extended_data.to_pandas(),
                horizon,
                history,
                step,
                date_column=self.date_column,
            )
            extended_data = slicer.get_slice(extended_data.to_pandas(), (current_test_ids, None))
            extended_data = pl.DataFrame(extended_data)

            extended_data = extended_data.with_columns(
                pl.Series("segment_col", np.repeat(np.arange(len(extended_data) // history), history))
            )
            id_column_name = ["segment_col", self.id_column]

        columns = extended_data.columns
        date_col_id = columns.index(self.date_column)
        id_col_id = [columns.index(col) for col in (id_column_name or [self.id_column])]

        ids, time_delta = slicer.ids_from_date(
            extended_data.to_pandas(), self.date_column, delta=self.delta, return_delta=True
        )

        if test_all:
            ids = list(np.unique(extended_data["segment_col"].to_numpy(), return_index=True)[1])[1:]

        data = extended_data.to_numpy()

        segments = np.split(data, ids)
        segments = [
            self._crop_segment(segment, test_last, horizon, history) for segment in segments
        ]

        padded_segments_results = [
            self._pad_segment(segment, horizon, time_delta, date_col_id, id_col_id)
            for segment in segments
        ]

        # Concatenate together
        result = np.vstack(np.concatenate((segments, padded_segments_results), axis=1))
        result_df = pl.DataFrame(result, schema=columns)
        result_df = result_df.with_columns(
            pl.col(self.date_column).cast(pl.Datetime),
            *[
                pl.col(col).cast(pl.Float64)
                for col in columns
                if col not in [self.id_column, self.date_column]
            ]
        )

        # return self._convert_to_pandas(result_df)
        return result_df
