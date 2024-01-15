from typing import Dict, Union
from numpy.typing import NDArray

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit


class Estimator:
    """ "A class of underlying model that is used to derive predictions
    using a feature format and targets defined by the strategy.

    Arguments:
        get_num_iterations: whether to get total number of trees.
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
        get_num_iterations: bool,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        self.models = []
        self.scores = []
        self.validation_params = validation_params
        self.model_params = model_params
        self.get_num_iterations = get_num_iterations
        if self.get_num_iterations:
            self.num_iterations = []

    def initialize_validator(self):
        """Initialization of the sample generator for training the model
            according to the passed parameters.

        Returns:
            Generator object.
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

    def fit(self, X: pd.DataFrame, y: NDArray[np.floating]) -> None:
        """Initialization and training of the model according to the
            passed parameters.

        Arguments:
            X: source train data.
            y: source train argets.
        """
        raise NotImplementedError()

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]:
        """Obtaining model predictions.

        Arguments:
            X: source test data.

        Returns:
            array with models' preds.
        """
        raise NotImplementedError()
