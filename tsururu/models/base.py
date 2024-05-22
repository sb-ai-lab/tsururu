from typing import Dict, Union

import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit


class Estimator:
    """Base class for all models.

    Args:
        validation_params: execution params (type, cv, loss),
            for example: {
                "type": "KFold",
                "n_splits": 3,
                "loss_function": "MAE",
            }.
        model_params: base model's params,
            for example: {
                "loss_function": "MultiRMSE",
                "early_stopping_rounds": 100,
            }.

    """

    def __init__(
        self,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        self.validation_params = validation_params
        self.model_params = model_params

        self.models = []
        self.scores = []
        self.columns = None

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

    def fit(self, data: dict, pipeline) -> "Estimator":
        """Fits the model using the input data and pipeline.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            pipeline: data preprocessing pipeline.

        Returns:
            the fitted model.

        """
        raise NotImplementedError()

    def predict(self, data: dict, pipeline) -> np.ndarray:
        """Generates predictions using the trained model.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            pipeline: data preprocessing pipeline.

        Returns:
            array of predicted values.

        """
        raise NotImplementedError()
