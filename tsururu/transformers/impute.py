from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .base import SeriesToSeriesTransformer


class MissingValuesImputer(SeriesToSeriesTransformer):
    """Imputes missing values in time series data using various strategies.

    Args:
        impute_inf: whether to impute infinite values additionally to missing values.
        regime: the strategy to use for imputation. Options are 'mean' or 'lag'.
            if None, the transformer will fill missing values with a constant value.
        constant_value: the constant value to fill remaining missing values
            after applying the chosen regime.
        transform_features: whether to transform features.
        transform_target: whether to transform target.
        window: the size of the window for the mean imputation strategy.
            if -1, the window size is the length of the series.
            if window size is bigger than the length of the series, the window size
                is the length of the series.
        weighted_alpha: the alpha value for weighting in the mean imputation strategy.
            the bigger the alpha, the more recent values are weighted.
        lag: the lag value for the lag imputation strategy.

    """

    def __init__(
        self,
        impute_inf: bool = False,
        regime: Optional[str] = None,
        constant_value: Optional[float] = None,
        transform_features: bool = True,
        transform_target: bool = True,
        window: Optional[int] = None,
        weighted_alpha: Optional[float] = None,
        lag: Optional[int] = None,
    ):
        super().__init__(transform_features=transform_features, transform_target=transform_target)
        self.impute_inf = impute_inf
        self.regime = regime
        self.constant_value = constant_value
        self.window = window if window is not None else -1
        self.weighted_alpha = weighted_alpha if weighted_alpha is not None else 0
        self.lag = lag if lag is not None else 1

    def fit(self, data: dict, input_features: Sequence[str]) -> SeriesToSeriesTransformer:
        """Fit transformer on 'elongated series' and return its instance.

        Args:
            data: dictionary with current states of 'elongated series',
                  arrays with features and targets, name of id, date and target
                  columns and indices for features and targets.
            input_features: array with names of columns to transform.

        Returns:
            self.
        """
        super().fit(data, input_features)
        self.output_features = [f"{column}__imputed" for column in self.input_features]
        return self

    def _transform_segment(self, segment: pd.Series, id_column_name: str) -> pd.Series:
        """Transform segment (points with similar id) of 'elongated series'
            for features' and targets' further generation.

        Args:
            segment: segment of 'elongated series' to transform.
            id_column_name: name of id column.

        Returns:
            transformed segment of 'elongated series'.
        """
        for i, column_name in enumerate(self.input_features):
            if self.regime == "mean":
                segment.loc[:, self.output_features[i]] = segment.loc[:, column_name].copy()
                segment.loc[:, self.output_features[i]] = self._fill_mean(
                    segment.loc[:, self.output_features[i]]
                )
            elif self.regime == "lag":
                segment.loc[:, self.output_features[i]] = segment.loc[:, column_name].copy()
                segment.loc[:, self.output_features[i]] = self._fill_lag(
                    segment.loc[:, self.output_features[i]]
                )
            # Fill remaining missing values with constant value
            if self.output_features[i] in segment.columns:
                segment.loc[:, self.output_features[i]] = segment.loc[:, self.output_features[i]].fillna(
                    self.constant_value
                )
                if self.impute_inf:
                    segment.loc[:, self.output_features[i]] = segment.loc[:, self.output_features[i]].replace(
                        [np.inf, -np.inf], self.constant_value
                    )
            else:
                segment.loc[:, self.output_features[i]] = segment.loc[:, column_name].fillna(
                    self.constant_value
                )
                if self.impute_inf:
                    segment.loc[:, self.output_features[i]] = segment.loc[:, self.output_features[i]].replace(
                        [np.inf, -np.inf], self.constant_value
                    )

        return segment

    def _fill_mean(self, series: pd.Series) -> pd.Series:
        """Fill missing values using mean with optional weighting.

        Args:
            series: series with missing values to be filled.

        Returns:
            series with missing values filled using mean.

        """
        filled_series = series.copy()
        if self.window == -1:
            window_size = len(series)
        else:
            window_size = self.window
            
        idx_list = series[series.isnull()].index
        if self.impute_inf:
            idx_list = idx_list.union(series.index[~np.isfinite(series)])

        for idx in idx_list:
            if idx >= window_size:
                window = series.loc[idx - window_size : idx]
            else:
                window = series.loc[:idx]

            try:
                if self.weighted_alpha > 0:
                    weights = np.exp(np.linspace(-self.weighted_alpha, 0, len(window)))
                    mean_value = np.average(
                        window.dropna(), weights=weights[: len(window.dropna())]
                    )
                else:
                    mean_value = window.mean()
            except:
                mean_value = series.loc[idx]

            filled_series.at[idx] = mean_value

        return filled_series

    def _fill_lag(self, series: pd.Series) -> pd.Series:
        """Fill missing values using lagged values.

        Args:
            series: series with missing values to be filled.

        Returns:
            series with missing values filled using lagged values.

        """
        filled_series = series.copy()

        idx_list = series[series.isnull()].index
        if self.impute_inf:
            idx_list = idx_list.union(series.index[~np.isfinite(series)])
        
        for idx in idx_list:
            try:
                current_lag = series.loc[idx - self.lag]
            except:
                current_lag = series.loc[idx]

            filled_series.at[idx] = current_lag

        return filled_series

    def transform(self, data: dict) -> dict:
        """Transform 'elongated series' for features' and targets' further
            generation and update self.params.

        Args:
            data: dictionary with current states of 'elongated series',
                  arrays with features and targets, name of id, date and target
                  columns and indices for features and targets.

        Returns:
            current states of `data` dictionary.
        """
        data = super().transform(data)
        return data

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Inverse transforms on the target variable y.

        Args:
            y: the target variable to be inversed.

        Returns:
            the inversed target variable.

        Notes:
            This method does not perform any transformation on the target variable
                as it is not needed for this transformer.

        """
        return y
