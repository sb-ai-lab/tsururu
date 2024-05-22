"""Module for initializing and manipulating time series data."""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .slice import IndexSlicer

slicer = IndexSlicer()


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
                    ...
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

    def _print_freq_period_info(self):
        """Print the frequency and period information for data for
        validation purposes.

        """
        info = slicer.timedelta(
            self.data[self.date_column], delta=self.delta, return_freq_period_info=True
        )[2]

        print(info)

    def __init__(
        self,
        data: pd.DataFrame,
        columns_params: dict,
        delta: pd.DateOffset = None,
        print_freq_period_info: bool = False,
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

        if print_freq_period_info:
            self._print_freq_period_info()

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

    def make_padded_test(
        self,
        horizon: int,
        history: int,
        test_last: bool = True,
        id_column_name: Optional[Union[str, Sequence[str]]] = None,
    ):
        """Generate a test dataframe with new rows with NaN targets.

        Args:
            horizon: number of points to prediction
                (number of new rows to add to each segment).
            test_last: if True, return generated test data
                corresponding to the last observation only.
            id_column_name: name of the column(s) by which the data is
                split (in some cases it is different from the original
                id column(s)).

        Notes:
            1. The new rows are filled with NaN target values,
                generated datetimes values (based on the time_delta)
                and the same id values as the last row of the segment.

        Returns:
            the padded test dataset.

        """
        index_slicer = IndexSlicer()
        columns = self.data.columns
        date_col_id = index_slicer.get_cols_idx(self.data, self.date_column)
        if id_column_name is None:
            id_column_name = self.id_column
        id_col_id = index_slicer.get_cols_idx(self.data, id_column_name)

        # Find indices for segments
        ids, time_delta = index_slicer.ids_from_date(
            self.data, self.date_column, delta=self.delta, return_delta=True
        )

        data = self.data.to_numpy()

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
        result = pd.DataFrame(result, columns=columns)
        result[self.date_column] = pd.to_datetime(result[self.date_column])
        result[self.id_column] = result[self.id_column].astype("int")
        other = [col for col in columns if col not in [self.id_column, self.date_column]]
        result[other] = result[other].astype("float")

        return result
