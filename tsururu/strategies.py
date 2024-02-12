import time
from copy import deepcopy
from typing import Dict, Tuple, List, Union, Optional, Callable
from numpy.typing import NDArray
from itertools import product

import numpy as np
import pandas as pd

from .transformers import (
    StandardScalerTransformer,
    LastKnownNormalizer,
    TransformersFactory,
    DifferenceNormalizer,
)

from .dataset import TSDataset, IndexSlicer
from .models import ModelsFactory


def timing_decorator(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return execution_time, result

    return wrapper


class Strategy:
    """Defines what features and targets will be built for subsequent
        training of the baseline model.

    Arguments:
        horizon: forecast horizon.
        validation_params: execution params for base model,
            for example: {
                "type": "KFold",
                "n_splits": 3,
                "loss_function": "MAE",
            }.
        model_name: model name from the model factory.
        model_params: base model's params,
            for example: {
                "loss_function": "MultiRMSE",
                "early_stopping_rounds": 100,
            }.
        is_multivariate: whether the prediction mode is multivariate.
        get_num_iterations: whether to get total number of trees.
    """

    def __init__(
        self,
        horizon: int,
        validation_params: Dict[str, Union[str, int]],
        model_name: str,
        model_params: Dict[str, Union[str, int]],
        is_multivariate: bool,
        get_num_iterations: bool,
    ):
        self.horizon = horizon
        self.validation_params = validation_params
        self.model_name = model_name
        self.model_params = model_params
        self.is_multivariate = is_multivariate
        self.get_num_iterations = get_num_iterations

        self.models = []
        self.fitted_transformers = {}
        self.inverse_transformers = []
        self.id_feature_column = None

    @staticmethod
    def _make_preds_df(
        dataset: TSDataset,
        horizon: int,
        id_column_name: Optional[str] = None,
    ) -> pd.DataFrame:
        if id_column_name is None:
            id_column_name = dataset.id_column

        columns_list = [id_column_name, dataset.date_column, dataset.target_column]

        index_slicer = IndexSlicer()
        # Get dataframe with predictions only
        target_ids = index_slicer.create_idx_target(
            data=dataset.seq_data,
            horizon=horizon,
            history=dataset.history,
            step=horizon + dataset.history,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )

        columns_ids = index_slicer.get_cols_idx(dataset.seq_data, columns_list)
        data = index_slicer.get_slice(
            dataset.seq_data,
            (target_ids, columns_ids),
        )
        pred_df = pd.DataFrame(np.vstack(data), columns=columns_list)
        return pred_df

    @staticmethod
    def _backtest_generator(
        dataset: TSDataset,
        cv: int,
        horizon: int,
    ):
        index_slicer = IndexSlicer()
        segments_ids = index_slicer.ids_from_date(
            dataset.seq_data,
            dataset.date_column,
            delta=dataset.delta
        )
        segments_ids = [0] + segments_ids + [len(dataset.seq_data)]

        for val_idx in range(cv):
            full_train = np.array([])
            full_test = np.array([])

            for i in range(len(segments_ids) - 1):
                if len(full_train) > 0:
                    full_train = np.vstack(
                        (
                            full_train,
                            np.arange(
                                segments_ids[i],
                                segments_ids[i + 1] - horizon * (val_idx + 1),
                            ),
                        )
                    )
                    full_test = np.vstack(
                        (
                            full_test,
                            np.arange(
                                segments_ids[i + 1] - horizon * (val_idx + 1),
                                segments_ids[i + 1] - horizon * (val_idx),
                            ),
                        )
                    )
                else:
                    full_train = np.arange(
                        segments_ids[i], segments_ids[i + 1] - horizon * (val_idx + 1)
                    )
                    full_test = np.arange(
                        segments_ids[i + 1] - horizon * (val_idx + 1),
                        segments_ids[i + 1] - horizon * (val_idx),
                    )

            yield (full_train, full_test)

    @staticmethod
    def _make_multivariate_X_y(
        X: pd.DataFrame, y: NDArray[np.floating]
    ) -> Tuple[pd.DataFrame, NDArray[np.floating]]:
        raise NotImplementedError()

    def _generate_X_y(
        self,
        dataset: TSDataset,
        train_horizon: int,
        target_horizon: int,
        is_train: bool,
        history: str = None,
        idx: Optional[NDArray[np.floating]] = None,
        n_last_horizon: Optional[int] = None,
        X_only: bool = False,
    ):
        raise NotImplementedError()

    def make_step(self, dataset: TSDataset):
        raise NotImplementedError()

    @timing_decorator
    def fit(self, dataset: TSDataset):
        raise NotImplementedError()

    def back_test(
        self, dataset: TSDataset, cv: int
    ) -> Union[List, NDArray[Union[np.floating, np.str_]]]:
        ids_list = []
        test_list = []
        preds_list = []
        fit_time_list = []
        forecast_time_list = []
        num_iterations_list = []

        for train_idx, test_idx in self._backtest_generator(dataset, cv, self.horizon):
            current_train = dataset.seq_data.iloc[train_idx.reshape(-1)]
            current_test = dataset.seq_data.iloc[test_idx.reshape(-1)]
            current_dataset = TSDataset(
                current_train,
                dataset.columns_and_features_params,
                dataset.history,
                dataset.step,
            )

            fit_time, _ = self.fit(current_dataset)
            forecast_time, current_pred = self.predict(current_dataset)

            test_list.append(np.asarray(current_test[dataset.target_column].values))
            preds_list.append(np.asarray(current_pred[dataset.target_column].values))
            fit_time_list.append(fit_time)
            forecast_time_list.append(forecast_time)

            if self.get_num_iterations:
                num_iterations = sum(
                    [self.models[i_model].num_iterations for i_model in range(len(self.models))]
                )
                num_iterations_list.append(num_iterations)

            if dataset.seq_data[dataset.id_column].nunique() > 1:
                ids_list.append(np.asarray(current_pred[dataset.id_column].values))
        return (
            ids_list,
            test_list,
            preds_list,
            fit_time_list,
            forecast_time_list,
            num_iterations_list,
        )

    @timing_decorator
    def predict(self, dataset: TSDataset) -> NDArray[np.floating]:
        raise NotImplementedError()


class RecursiveStrategy(Strategy):
    """A strategy that uses a single model to predict all points
        in the forecast horizon.

    Fit: the model is fitted to predict one point ahead.
    Inference: a prediction is iteratively made one point ahead and
        then this prediction is used to build further features.

    Arguments:
        horizon: forecast horizon.
        validation_params: execution params for base model,
            for example: {
                "type": "KFold",
                "n_splits": 3,
                "loss_function": "MAE",
            }.
        model_name: model name from the model factory.
        model_params: base model's params,
            for example: {
                "loss_function": "MultiRMSE",
                "early_stopping_rounds": 100,
            }.
        is_multivariate: whether the prediction mode is multivariate.
        get_num_iterations: whether to get total number of trees.
        k: how many points to predict at a time,
            if k > 1, then it's an intermediate strategy between
            RecursiveStrategy and MIMOStrategy.
        reduced: if true, features are at once formed for all test
            observations, unavailable values are replaced by NaN.
    """

    def __init__(
        self,
        horizon: int,
        validation_params: Dict[str, Union[str, int]],
        model_name: str,
        model_params: Dict[str, Union[str, int]],
        is_multivariate: bool = False,
        get_num_iterations: bool = False,
        k: int = 1,
        reduced: bool = False,
    ):
        super().__init__(
            horizon,
            validation_params,
            model_name,
            model_params,
            is_multivariate,
            get_num_iterations,
        )
        self.strategy_name = "RecursiveStrategy"
        self.k = k
        self.reduced = reduced

    def _make_multivariate_X_y(
        self,
        X: pd.DataFrame,
        date_column: "str",
        y: Optional[NDArray[np.floating]] = None,
    ):
        idx_slicer = IndexSlicer()

        date_features_colname = X.columns[X.columns.str.contains(date_column)].values
        other_features_colname = np.setdiff1d(
            X.columns.values,
            np.concatenate([self.id_feature_column.columns.values, date_features_colname]),
        )

        date_features_idx = idx_slicer.get_cols_idx(X, date_features_colname)
        other_features_idx = idx_slicer.get_cols_idx(X, other_features_colname)

        segments_ids = np.append(np.unique(self.id_feature_column, return_index=1)[1], len(X))
        segments_ids_array = np.array(
            [
                np.arange(segments_ids[segment_id - 1], segments_ids[segment_id])
                for segment_id in range(1, len(segments_ids))
            ]
        ).T

        date_features_array = idx_slicer.get_slice(
            X, (segments_ids_array[:, 0], date_features_idx)
        ).reshape(len(segments_ids_array), len(date_features_colname))
        other_features_array = idx_slicer.get_slice(
            X, (segments_ids_array, other_features_idx)
        ).reshape(
            len(segments_ids_array),
            len(other_features_colname) * (len(segments_ids) - 1),
        )

        final_other_features_colname = [
            f"{feat}__{i}"
            for i, feat in product(range(len(segments_ids) - 1), other_features_colname)
        ]
        final_X = pd.DataFrame(
            np.hstack((date_features_array, other_features_array)),
            columns=np.hstack((date_features_colname, final_other_features_colname)),
        )

        if y is not None:
            final_y = idx_slicer.get_slice(y, (segments_ids_array, None)).reshape(
                len(segments_ids_array), -1
            )
            return final_X, final_y
        return final_X

    def _generate_X_y(
        self,
        dataset: TSDataset,
        train_horizon: int,
        target_horizon: int,
        is_train: bool,
        history: str = None,
        idx: Optional[NDArray[np.floating]] = None,
        n_last_horizon: Optional[int] = None,
        X_only: bool = False,
    ):
        transformers_factory = TransformersFactory()
        index_slicer = IndexSlicer()

        if n_last_horizon is None:
            n_last_horizon = target_horizon
        if history is None:
            history = dataset.history
        if idx is None:
            idx = index_slicer.create_idx_data(
                dataset.seq_data,
                train_horizon,
                history,
                dataset.step,
                date_column=dataset.date_column,
                delta=dataset.delta,
            )

        final_X = pd.DataFrame()
        y = self._generate_y(dataset, target_horizon, n_last_horizon)

        for role, column_params in dataset.columns_and_features_params.items():
            current_X = pd.DataFrame()
            raw_ts_X = dataset.seq_data.copy()
            raw_ts_y = dataset.seq_data.copy()
            raw_ts_y_copy = raw_ts_y.copy()
            if column_params.get("features") is None:
                if column_params["drop_raw_feature"] and role != "id":
                    continue
                else:
                    transformer_init_params = {
                        "transformer_name": "LagTransformer",
                        "transformer_params": {
                            "lags": 1,
                            "idx_data": idx,
                            "drop_raw_features": False,
                        },
                    }
                    current_transformer = transformers_factory[transformer_init_params]
                    current_transformer.fit(
                        raw_ts_X=raw_ts_X,
                        raw_ts_y=raw_ts_y,
                        features_X=current_X,
                        y=y,
                        columns=column_params["column"],
                        id_column=dataset.id_column,
                        transform_train=None,
                        transform_target=None,
                    )
                    (raw_ts_X, raw_ts_y, current_X, y) = current_transformer.transform(
                        raw_ts_X, raw_ts_y, current_X, y, X_only
                    )
                    if role == "id":
                        self.id_feature_column = current_X
                    final_X = pd.concat((final_X, current_X), axis=1)
            else:
                for transformer_name, transformer_params in column_params["features"].items():
                    transformer_init_params = {
                        "transformer_name": transformer_name,
                        "transformer_params": {
                            param: transformer_params[param]
                            for param in transformer_params
                            if param not in ["transform_train", "transform_target"]
                        },
                    }

                    # Add additional params to certain transformers
                    if (
                        transformer_name in ["DateSeasonsGenerator", "TimeToNumGenerator"]
                    ):
                        transformer_init_params["transformer_params"]["delta"] = dataset.delta

                    if (
                        transformer_name in ["DateSeasonsGenerator", "TimeToNumGenerator"]
                        and transformer_params.get("from_target_date") is not None
                    ):
                        transformer_init_params["transformer_params"]["horizon"] = self.horizon

                    if transformer_name == "LagTransformer":
                        transformer_init_params["transformer_params"]["idx_data"] = idx
                        transformer_init_params["transformer_params"][
                            "drop_raw_features"
                        ] = column_params["drop_raw_feature"]

                    current_transformer = transformers_factory[transformer_init_params]

                    # Use transformer's
                    if transformer_name == "StandardScalerTransformer":
                        if is_train:
                            current_transformer.fit(
                                raw_ts_X,
                                raw_ts_y,
                                current_X,
                                y,
                                column_params["column"],
                                dataset.id_column,
                                transformer_params.get("transform_train"),
                                transformer_params.get("transform_target"),
                            )

                            self.fitted_transformers[tuple(column_params["column"])] = {
                                "StandardScalerTransformer": (current_transformer)
                            }

                            if transformer_params.get("transform_target"):
                                if len(self.inverse_transformers) == 0:
                                    self.inverse_transformers.append(current_transformer)
                                else:
                                    for i in range(len(self.inverse_transformers)):
                                        if isinstance(
                                            self.inverse_transformers[i],
                                            StandardScalerTransformer,
                                        ):
                                            self.inverse_transformers[i] = current_transformer
                        else:
                            current_transformer = self.fitted_transformers[
                                tuple(column_params["column"])
                            ]["StandardScalerTransformer"]
                    else:
                        current_transformer.fit(
                            raw_ts_X,
                            raw_ts_y,
                            current_X,
                            y,
                            column_params["column"],
                            dataset.id_column,
                            transformer_params.get("transform_train"),
                            transformer_params.get("transform_target"),
                        )

                    if (
                        transformer_name in ["LastKnownNormalizer", "DifferenceNormalizer"]
                        and not is_train
                        and transformer_params.get("transform_target")
                    ):
                        if len(self.inverse_transformers) == 0:
                            self.inverse_transformers.append(current_transformer)
                        else:
                            for i in range(len(self.inverse_transformers)):
                                if isinstance(
                                    self.inverse_transformers[i], LastKnownNormalizer
                                ) or isinstance(
                                    self.inverse_transformers[i], DifferenceNormalizer
                                ):
                                    self.inverse_transformers[i] = current_transformer

                    (raw_ts_X, raw_ts_y, current_X, y) = current_transformer.transform(
                        raw_ts_X, raw_ts_y, current_X, y, X_only
                    )

                    if (len(raw_ts_y) != len(raw_ts_y_copy)) or (
                        raw_ts_y != raw_ts_y_copy
                    ).any().any():
                        target_dataset = deepcopy(dataset)
                        target_dataset.seq_data = raw_ts_y
                        y = self._generate_y(target_dataset, target_horizon, n_last_horizon)
                final_X = pd.concat((final_X, current_X), axis=1)
        return final_X, y

    def _inverse_transform_y(self, dataset):
        dataset = dataset.copy()
        for transfomer in self.inverse_transformers[::-1]:
            dataset = transfomer.inverse_transform_y(dataset)
        return dataset

    def _generate_y(self, dataset, horizon=1, n_last_horizon=None):
        index_slicer = IndexSlicer()
        idx_target = index_slicer.create_idx_target(
            dataset.seq_data,
            horizon,
            dataset.history,
            dataset.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
            n_last_horizon=n_last_horizon,
        )
        return index_slicer.get_slice(
            dataset.seq_data[[dataset.target_column]], (idx_target, None)
        ).squeeze(axis=-1)

    @timing_decorator
    def fit(self, dataset, horizon=None):
        if horizon is None:
            horizon = self.k

        X, y = self._generate_X_y(dataset, horizon, horizon, is_train=True)

        if self.is_multivariate:
            X, y = self._make_multivariate_X_y(X, y, dataset.date_column)

        factory = ModelsFactory()
        model_params = {
            "model_name": self.model_name,
            "validation_params": self.validation_params,
            "model_params": self.model_params,
            "get_num_iterations": self.get_num_iterations,
        }
        model = factory[model_params]
        model.fit(X, y)
        self.models.append(model)
        return self

    def make_step(self, step, dataset, horizon=None):
        if horizon is None:
            horizon = self.horizon
        index_slicer = IndexSlicer()
        current_test_ids = index_slicer.create_idx_test(
            dataset.seq_data,
            horizon - step * self.k,
            dataset.history,
            dataset.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )

        # which NaN observations we should replace with preds
        current_target_ids = index_slicer.create_idx_target(
            dataset.seq_data,
            horizon,
            dataset.history,
            dataset.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )[:, self.k * step:self.k * (step + 1)]
        X_current, _ = self._generate_X_y(
            dataset,
            train_horizon=self.k,
            target_horizon=self.k,
            is_train=False,
            idx=current_test_ids,
            X_only=True,
        )
        if self.is_multivariate:
            X_current = self._make_multivariate_X_y(X_current, date_column=dataset.date_column)
        current_pred = self.models[0].predict(X_current)
        dataset.seq_data.loc[
            current_target_ids.reshape(-1), dataset.target_column
        ] = current_pred.reshape(-1)
        dataset.seq_data.loc[current_target_ids.reshape(-1)] = self._inverse_transform_y(
            dataset.seq_data.loc[current_target_ids.reshape(-1)]
        )
        return dataset

    @timing_decorator
    def predict(self, dataset, train=False):
        if train:
            index_slicer = IndexSlicer()
            current_test_ids = index_slicer.create_idx_data(
                dataset.seq_data,
                self.horizon,
                dataset.history,
                self.k,
                date_column=dataset.date_column,
                delta=dataset.delta,
            )

            extended_data = index_slicer.get_slice(dataset.seq_data, (current_test_ids, None))
            extended_data = pd.DataFrame(
                extended_data.reshape(-1, extended_data.shape[-1]),
                columns=dataset.seq_data.columns,
            )
            extended_data_nrows = extended_data.shape[0]
            new_seq_params = {
                "history": dataset.history,
                "step": dataset.step,
                "columns_roles": dataset.columns_roles,
            }
            extended_dataset = TSDataset(
                extended_data, new_seq_params, dataset.transformers_params
            )
            extended_dataset.seq_data["segment_col"] = np.repeat(
                np.arange(extended_data_nrows // extended_dataset.history),
                extended_dataset.history,
            )
            new_data = extended_dataset.make_padded_test(
                self.horizon,
                id_column_name=["segment_col", dataset.id_column_name],
            )

        else:
            new_data = dataset.make_padded_test(self.horizon)

        new_dataset = deepcopy(dataset)
        new_dataset.seq_data = new_data

        if self.reduced:
            index_slicer = IndexSlicer()
            current_test_ids = index_slicer.create_idx_data(
                new_dataset.seq_data,
                self.k,
                new_dataset.history,
                self.k,
                date_column=dataset.date_column,
                delta=dataset.delta,
            )
            X, _ = self._generate_X_y(
                new_dataset,
                train_horizon=self.k,
                target_horizon=self.k,
                is_train=False,
                idx=current_test_ids,
                X_only=True,
            )
            if self.is_multivariate:
                X = self._make_multivariate_X_y(X, date_column=dataset.date_column)
            pred = self.models[0].predict(X)
            nan_mask = np.isnan(new_dataset.seq_data[dataset.target_column].astype("float"))
            new_dataset.seq_data.loc[nan_mask, dataset.target_column] = np.hstack(pred)
            new_dataset.seq_data.loc[nan_mask] = self._inverse_transform_y(
                new_dataset.seq_data.loc[nan_mask]
            )

        else:
            for step in range(self.horizon // self.k):
                new_dataset = self.make_step(step, new_dataset, self.horizon)

        # Get dataframe with predictions only
        if train:
            pred_df = self._make_preds_df(new_dataset, self.horizon, id_column_name="segment_col")
            pred_df = pred_df.merge(
                new_dataset.seq_data[["date", "segment_col", "id"]],
                on=("date", "segment_col"),
            )
        else:
            pred_df = self._make_preds_df(new_dataset, self.horizon)
        return pred_df


class DirectStrategy(RecursiveStrategy):
    """A strategy that uses an individual model for each point
        in the forecast horizon.

    Fit: the models is fitted to predict certain point in the
        forecasting horizon (number of models = horizon).
    Inference: each model predict one point.

    Arguments:
        horizon: forecast horizon.
        validation_params: execution params for base model,
            for example: {
                "type": "KFold",
                "n_splits": 3,
                "loss_function": "MAE",
            }.
        model_name: model name from the model factory.
        model_params: base model's params,
            for example: {
                "loss_function": "MultiRMSE",
                "early_stopping_rounds": 100,
            }.
        is_multivariate: whether the prediction mode is multivariate.
        get_num_iterations: whether to get total number of trees.
        k: how many points to predict at a time,
            if k > 1, then it's an intermediate strategy between
            RecursiveStrategy and MIMOStrategy.
        equal_train_size: if true, all models are trained with the same
            training sample (which is equal to the training sample
            of the last model if equal_train_size=false).
    """

    def __init__(
        self,
        horizon: int,
        validation_params: Dict[str, Union[str, int]],
        model_name: str,
        model_params: Dict[str, Union[str, int]],
        is_multivariate: bool = False,
        get_num_iterations: bool = False,
        k: int = 1,
        equal_train_size: bool = False,
    ):
        super().__init__(
            horizon,
            validation_params,
            model_name,
            model_params,
            is_multivariate,
            get_num_iterations,
            k,
        )
        self.equal_train_size = equal_train_size

    @timing_decorator
    def fit(self, dataset):
        self.models = []
        factory = ModelsFactory()

        if self.equal_train_size:
            X, y = self._generate_X_y(
                dataset,
                self.horizon,
                self.horizon,
                is_train=True,
                n_last_horizon=self.horizon,
            )
            for horizon in range(1, self.horizon // self.k + 1):
                current_y = y[:, (horizon - 1) * self.k:horizon * self.k]
                if self.is_multivariate:
                    current_X, current_y = self._make_multivariate_X_y(
                        X, current_y, dataset.date_column
                    )
                else:
                    current_X = X

                model_params = {
                    "model_name": self.model_name,
                    "validation_params": self.validation_params,
                    "model_params": self.model_params,
                    "get_num_iterations": self.get_num_iterations,
                }
                current_model = factory[model_params]
                current_model.fit(current_X, current_y)
                self.models.append(current_model)
        else:
            for horizon in range(1, self.horizon // self.k + 1):
                X, y = self._generate_X_y(
                    dataset,
                    self.k * horizon,
                    self.k * horizon,
                    is_train=True,
                    n_last_horizon=self.k,
                )
                if self.is_multivariate:
                    X, y = self._make_multivariate_X_y(X, y, dataset.date_column)

                model_params = {
                    "model_name": self.model_name,
                    "validation_params": self.validation_params,
                    "model_params": self.model_params,
                    "get_num_iterations": self.get_num_iterations,
                }
                current_model = factory[model_params]
                current_model.fit(X, y)
                self.models.append(current_model)
        return self

    def make_step(self, step, dataset, horizon=None):
        index_slicer = IndexSlicer()
        current_test_ids = index_slicer.create_idx_test(
            dataset.seq_data,
            self.horizon,
            dataset.history,
            dataset.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )
        current_target_ids = index_slicer.create_idx_target(
            dataset.seq_data,
            self.horizon,
            dataset.history,
            dataset.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )[:, self.k * step:self.k * (step + 1)]

        current_X, _ = self._generate_X_y(
            dataset,
            train_horizon=horizon,
            target_horizon=horizon,
            is_train=False,
            idx=current_test_ids,
            X_only=True,
        )

        if self.is_multivariate:
            current_X = self._make_multivariate_X_y(current_X, date_column=dataset.date_column)

        current_pred = self.models[step].predict(current_X)

        dataset.seq_data.loc[
            current_target_ids.reshape(-1), dataset.target_column
        ] = current_pred.reshape(-1)
        dataset.seq_data.loc[current_target_ids.reshape(-1)] = self._inverse_transform_y(
            dataset.seq_data.loc[current_target_ids.reshape(-1)]
        )
        return dataset


class MIMOStrategy(RecursiveStrategy):
    """A strategy that uses one model that learns to predict
        the entire prediction horizon.

    Fit: the model is fitted to predict a vector which length is equal
        to the length of the prediction horizon).
    Inference: the model makes a vector of predictions.

    Arguments:
        horizon: forecast horizon.
        validation_params: execution params for base model,
            for example: {
                "type": "KFold",
                "n_splits": 3,
                "loss_function": "MAE",
            }.
        model_name: model name from the model factory.
        model_params: base model's params,
            for example: {
                "loss_function": "MultiRMSE",
                "early_stopping_rounds": 100,
            }.
        is_multivariate: whether the prediction mode is multivariate.
        get_num_iterations: whether to get total number of trees.
    """

    def __init__(
        self,
        horizon: int,
        validation_params: Dict[str, Union[str, int]],
        model_name: str,
        model_params: Dict[str, Union[str, int]],
        is_multivariate: bool = False,
        get_num_iterations: bool = False,
    ):
        super().__init__(
            horizon,
            validation_params,
            model_name,
            model_params,
            is_multivariate=is_multivariate,
            get_num_iterations=get_num_iterations,
        )
        self.strategy_name = "MIMOStrategy"

    @timing_decorator
    def fit(self, dataset):
        super().fit(dataset, self.horizon)

    @timing_decorator
    def predict(self, dataset):
        target_column_name, date_column_name = (
            dataset.columns_and_features_params["target"]["column"][0],
            dataset.columns_and_features_params["date"]["column"][0],
        )

        new_data = dataset.make_padded_test(self.horizon)
        new_dataset = deepcopy(dataset)
        new_dataset.seq_data = new_data

        index_slicer = IndexSlicer()
        current_test_ids = index_slicer.create_idx_test(
            new_dataset.seq_data,
            self.horizon,
            dataset.history,
            dataset.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )
        current_X, _ = self._generate_X_y(
            new_dataset,
            train_horizon=self.horizon,
            target_horizon=self.horizon,
            is_train=False,
            idx=current_test_ids,
            X_only=True,
        )

        if self.is_multivariate:
            current_X = self._make_multivariate_X_y(current_X, date_column=date_column_name)
        current_pred = self.models[0].predict(current_X)

        nan_mask = np.isnan(new_dataset.seq_data[target_column_name].astype("float"))
        new_dataset.seq_data.loc[nan_mask, target_column_name] = np.hstack(current_pred)
        new_dataset.seq_data.loc[nan_mask] = self._inverse_transform_y(
            new_dataset.seq_data.loc[nan_mask]
        )

        # Get dataframe with predictions only
        pred_df = self._make_preds_df(new_dataset, self.horizon)
        return pred_df


class DirRecStrategy(RecursiveStrategy):
    """A strategy that uses individual model for each point
        in the prediction horizon.

    Fit: mixture of DirectStrategy and RecursiveStrategy, fit individual
        models, but at each step expand history window.

    Inference: at each step makes a prediction one point ahead, and then
        uses this prediction to further generate features for subsequent
        models along with new observations.

    Arguments:
        horizon: forecast horizon.
        validation_params: execution params for base model,
            for example: {
                "type": "KFold",
                "n_splits": 3,
                "loss_function": "MAE",
            }.
        model_name: model name from the model factory.
        model_params: base model's params,
            for example: {
                "loss_function": "MultiRMSE",
                "early_stopping_rounds": 100,
            }.
        is_multivariate: whether the prediction mode is multivariate.
        get_num_iterations: whether to get total number of trees.
    """

    def __init__(
        self,
        horizon: int,
        validation_params: Dict[str, Union[str, int]],
        model_name: str,
        model_params: Dict[str, Union[str, int]],
        is_multivariate: bool = False,
        get_num_iterations: bool = False,
    ):
        super().__init__(
            horizon,
            validation_params,
            model_name,
            model_params,
            is_multivariate=is_multivariate,
            get_num_iterations=get_num_iterations,
        )
        self.strategy_name = "DirRecStrategy"
        self.true_lags = {}

    @timing_decorator
    def fit(self, dataset):
        self.models = []  # len = amount of forecast points

        # Save true lags for each feature
        self.true_lags = {
            column_name: column_dict["features"]["LagTransformer"]["lags"]
            for (
                column_name,
                column_dict,
            ) in dataset.columns_and_features_params.items()
            if column_dict.get("features") and column_dict["features"].get("LagTransformer")
        }

        for horizon in range(1, self.horizon + 1):
            X, y = self._generate_X_y(
                dataset,
                train_horizon=1,
                target_horizon=horizon,
                history=dataset.history + (horizon - 1),
                n_last_horizon=1,
                is_train=True,
            )

            # Update lags for each feature
            for column_name in self.true_lags.keys():
                dataset.columns_and_features_params[column_name]["features"]["LagTransformer"][
                    "lags"
                ] += 1

            if self.is_multivariate:
                X, y = self._make_multivariate_X_y(X, y, dataset.date_column)

            factory = ModelsFactory()
            model_params = {
                "model_name": self.model_name,
                "validation_params": self.validation_params,
                "model_params": self.model_params,
                "get_num_iterations": self.get_num_iterations,
            }
            current_model = factory[model_params]
            current_model.fit(X, y)
            self.models.append(current_model)

        # Return true lags
        for column_name in self.true_lags.keys():
            dataset.columns_and_features_params[column_name]["features"]["LagTransformer"][
                "lags"
            ] = self.true_lags[column_name]
        return self

    def make_step(self, step, dataset, _):
        index_slicer = IndexSlicer()
        current_test_ids = index_slicer.create_idx_test(
            dataset.seq_data,
            self.horizon - step,
            dataset.history + step,
            dataset.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )
        current_X, _ = self._generate_X_y(
            dataset,
            train_horizon=1,
            target_horizon=1,
            is_train=False,
            idx=current_test_ids,
            X_only=True,
        )
        if self.is_multivariate:
            current_X = self._make_multivariate_X_y(current_X, date_column=dataset.date_column)
        current_pred = self.models[step].predict(current_X)

        # Update lags for each feature
        for column_name in self.true_lags.keys():
            dataset.columns_and_features_params[column_name]["features"]["LagTransformer"][
                "lags"
            ] += 1

        dataset.seq_data.loc[
            step + dataset.history::dataset.history + self.horizon,
            dataset.target_column,
        ] = current_pred.reshape(-1)
        dataset.seq_data.loc[
            step + dataset.history::dataset.history + self.horizon
        ] = self._inverse_transform_y(
            dataset.seq_data.loc[step + dataset.history::dataset.history + self.horizon]
        )
        return dataset


class FlatWideMIMOStrategy(MIMOStrategy):
    """A strategy that uses a single model for all points
        in the prediction horizon.

    Fit: mixture of DirectStrategy and MIMOStrategy, fit one
        model, but uses deployed over horizon DirectStrategy's features.

    Inference: similarly.

    Arguments:
        horizon: forecast horizon.
        validation_params: execution params for base model,
            for example: {
                "type": "KFold",
                "n_splits": 3,
                "loss_function": "MAE",
            }.
        model_name: model name from the model factory.
        model_params: base model's params,
            for example: {
                "loss_function": "MultiRMSE",
                "early_stopping_rounds": 100,
            }.
        is_multivariate: whether the prediction mode is multivariate.
        get_num_iterations: whether to get total number of trees.
    """

    def __init__(
        self,
        horizon: int,
        validation_params: Dict[str, Union[str, int]],
        model_name: str,
        model_params: Dict[str, Union[str, int]],
        is_multivariate: bool = False,
        get_num_iterations: bool = False,
    ):
        super().__init__(
            horizon,
            validation_params,
            model_name,
            model_params,
            is_multivariate=is_multivariate,
            get_num_iterations=get_num_iterations,
        )
        self.strategy_name = "FlatWideMIMOStrategy"

    @staticmethod
    def _make_multivariate_X_y(X, date_column, y=None):
        idx_slicer = IndexSlicer()

        id_feature_colname = np.array(["ID"])
        fh_feature_colname = np.array(["FH"])
        date_features_colname = X.columns[X.columns.str.contains(date_column)].values
        other_features_colname = np.setdiff1d(
            X.columns.values,
            np.concatenate([id_feature_colname, date_features_colname, fh_feature_colname]),
        )

        date_features_idx = idx_slicer.get_cols_idx(X, date_features_colname)
        other_features_idx = idx_slicer.get_cols_idx(X, other_features_colname)
        fh_feature_idx = idx_slicer.get_cols_idx(X, fh_feature_colname)

        segments_ids = np.append(np.unique(X[id_feature_colname], return_index=1)[1], len(X))

        segments_ids_array = np.array(
            [
                np.arange(segments_ids[segment_id - 1], segments_ids[segment_id])
                for segment_id in range(1, len(segments_ids))
            ]
        ).T
        date_features_array = idx_slicer.get_slice(
            X, (segments_ids_array[:, 0], date_features_idx)
        ).reshape(len(segments_ids_array), len(date_features_colname))
        fh_feature_array = idx_slicer.get_slice(
            X, (segments_ids_array[:, 0], fh_feature_idx)
        ).reshape(len(segments_ids_array), -1)
        other_features_array = idx_slicer.get_slice(
            X, (segments_ids_array, other_features_idx)
        ).reshape(
            len(segments_ids_array),
            len(other_features_colname) * (len(segments_ids) - 1),
        )

        final_other_features_colname = [
            f"{feat}__{i}"
            for i, feat in product(range(len(segments_ids) - 1), other_features_colname)
        ]
        final_X = pd.DataFrame(
            np.hstack((fh_feature_array, date_features_array, other_features_array)),
            columns=np.hstack(
                (
                    fh_feature_colname,
                    date_features_colname,
                    final_other_features_colname,
                )
            ),
        )

        if y is not None:
            final_y = idx_slicer.get_slice(y, (segments_ids_array, None)).reshape(
                len(segments_ids_array), -1
            )
            return final_X, final_y
        return final_X

    def _generate_X_y(
        self,
        dataset: TSDataset,
        train_horizon: int,
        target_horizon: int,
        is_train: bool,
        history: str = None,
        idx: Optional[NDArray[np.floating]] = None,
        n_last_horizon: Optional[int] = None,
        X_only: bool = False,
    ):
        direct_df, direct_y = super()._generate_X_y(
            dataset,
            train_horizon,
            target_horizon,
            is_train,
            history,
            idx,
            n_last_horizon,
            X_only,
        )
        direct_lag_index_dict = {}

        fh_array = np.arange(1, self.horizon + 1)

        id_mask = direct_df.columns.str.contains(dataset.id_column)
        if sum(id_mask) > 0:
            id_count = len(direct_df.loc[:, id_mask].value_counts())
        else:
            id_count = 1
        direct_lag_index_dict["ID"] = np.repeat(
            np.arange(id_count),
            repeats=(len(direct_df) / id_count * len(fh_array)),
        )
        direct_lag_index_dict["FH"] = np.tile(fh_array, len(direct_df.index))
        direct_lag_index_df = pd.DataFrame(direct_lag_index_dict)

        direct_df = direct_df.loc[:, ~id_mask]
        features_df = pd.DataFrame(
            np.repeat(direct_df.values, self.horizon, axis=0),
            columns=direct_df.columns,
        )
        final_df = direct_lag_index_df.merge(features_df, left_index=True, right_index=True)

        final_y = direct_y.reshape(-1, 1)
        return final_df, final_y

    @timing_decorator
    def predict(self, dataset):
        new_data = dataset.make_padded_test(self.horizon)
        new_dataset = deepcopy(dataset)
        new_dataset.seq_data = new_data

        # Make predictions and fill NaNs
        index_slicer = IndexSlicer()
        test_ids = index_slicer.create_idx_test(
            new_dataset.seq_data,
            self.horizon,
            dataset.history,
            dataset.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )
        X, _ = self._generate_X_y(
            new_dataset,
            train_horizon=1,
            target_horizon=1,
            is_train=False,
            idx=test_ids,
            X_only=True,
        )
        if self.is_multivariate:
            X = self._make_multivariate_X_y(X, date_column=dataset.date_column)
        pred = self.models[0].predict(X)

        nan_mask = np.isnan(new_dataset.seq_data[dataset.target_column].astype("float"))
        if self.is_multivariate:
            new_dataset.seq_data.loc[nan_mask, dataset.target_column] = pred.T.reshape(-1)
        else:
            new_dataset.seq_data.loc[nan_mask, dataset.target_column] = np.hstack(pred)

        # Get dataframe with predictions only
        pred_df = self._make_preds_df(new_dataset, self.horizon)
        return pred_df


# Factory Object
class StrategiesFactory:
    def __init__(self):
        self.models = {
            "RecursiveStrategy": RecursiveStrategy,
            "DirectStrategy": DirectStrategy,
            "MIMOStrategy": MIMOStrategy,
            "DirRecStrategy": DirRecStrategy,
            "FlatWideMIMOStrategy": FlatWideMIMOStrategy,
        }

    def get_allowed(self):
        return sorted(list(self.models.keys()))

    def __getitem__(self, params):
        return self.models[params["strategy_name"]](**params["strategy_params"])
