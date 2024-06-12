from typing import Dict, Union, Optional

import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit

from ..dataset.pipeline import Pipeline


class Estimator:
    """Base class for all models.

    Args:
        model_params: parameters for the model.
            Individually defined for each model.

    """

    def __init__(self, model_params: Dict[str, Union[str, int]]):
        self.model_params = model_params

        self.model = None
        self.score = None

    def fit_one_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> "Estimator":
        """Fits the model on one fold using the input data.

        Args:
            X_train: features array.
            y_train: target array.
            X_val: validation features array.
            y_val: validation target array.

        Returns:
            the fitted model.

        """
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions using the trained model.

        Args:
            X: features array.

        Returns:
            array of predicted values.

        """
        return self.model.predict(X)


class MLEstimator(Estimator):
    def initialize_validator(self):
        """Initialization of the sample generator.

        Returns:
            generator object.

        """
        if self.validation_params["type"] == "KFold":
            # Set default params if params are None
            for param, default_value in [
                ("n_splits", 3),
                ("shuffle", True),
                ("random_state", 42),
            ]:
                if self.validation_params.get(param) is None:
                    self.validation_params[param] = default_value

            cv = KFold(**{k: v for k, v in self.validation_params.items() if k != "type"})

        elif self.validation_params["type"] == "TS_expanding_window":
            cv = TimeSeriesSplit(n_splits=self.validation_params["n_splits"])

        return cv


class StatEstimator(Estimator):
    @staticmethod
    def _inspect_raw_ts_X_columns(data: dict, pipeline: Pipeline):
        res = {
            "target_column": data["target_column_name"],
            "date_column": data["date_column_name"],
            "id_column": data["id_column_name"],
        }

        for transformer in pipeline.transformers.transformers_list:
            if transformer.input_features == data["target_column_name"]:
                res["target_column"] = transformer.output_features
            elif transformer.input_features == data["date_column_name"]:
                res["date_column"] = transformer.output_features
            elif transformer.input_features == data["id_column_name"]:
                res["id_column"] = transformer.output_features

        return res

    def _rename_raw_ts_X_columns(self, data: dict, pipeline: Pipeline):
        current_column_names = self._inspect_raw_ts_X_columns(data, pipeline)
        data["raw_ts_X"][current_column_names["target_column"]] == "y"
        data["raw_ts_X"][current_column_names["date_column"]] == "ds"
        data["raw_ts_X"][current_column_names["id_column"]] == "unique_id"

        return data

    def _initialize_model(self):
        raise NotImplementedError()

    def __init__(
        self,
        model_params: Dict[str, Union[str, int]],
        model_name: str,
    ):
        super().__init__(model_params)
        self.model_name = model_name

    def fit(self, data: dict, pipeline: Pipeline) -> "StatEstimator":
        data = self._rename_raw_ts_X_columns(data, pipeline)

        model = self._initialize_model()
        fitted_model = model.fit(data["raw_ts_X"])

        self.models.append(fitted_model)

    def predict(self, data: dict, pipeline: Pipeline) -> np.ndarray:
        data = self._rename_raw_ts_X_columns(data, pipeline)

        horizon = pipeline.target_ids.shape[1]
        y_pred = self.models[0].predict(h=horizon)[self.model_name].values

        return y_pred


class BaselineEstimator:
    @staticmethod
    def _inspect_raw_ts_X_columns(data: dict, pipeline: Pipeline):
        res = {
            "target_column": data["target_column_name"],
            "date_column": data["date_column_name"],
            "id_column": data["id_column_name"],
        }

        for transformer in pipeline.transformers.transformers_list:
            if transformer.input_features == data["target_column_name"]:
                res["target_column"] = transformer.output_features
            elif transformer.input_features == data["date_column_name"]:
                res["date_column"] = transformer.output_features
            elif transformer.input_features == data["id_column_name"]:
                res["id_column"] = transformer.output_features

        return res

    def __init__(self):
        pass

    def predict(self, data: dict, pipeline) -> np.ndarray:
        raise NotImplementedError()
