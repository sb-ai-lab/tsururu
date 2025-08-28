from typing import Dict, Optional, Union

import numpy as np

from tsururu.models.ml_base import Estimator
from tsururu.utils.optional_imports import OptionalImport

Pool = OptionalImport("catboost.Pool")
CatBoostRegressor = OptionalImport("catboost.CatBoostRegressor")
GradientBoosting = OptionalImport("py_boost.GradientBoosting")
RandomSamplingSketch = OptionalImport("py_boost.multioutput.sketching.RandomSamplingSketch")
RandomProjectionSketch = OptionalImport("py_boost.multioutput.sketching.RandomProjectionSketch")


class CatBoost(Estimator):
    """CatBoost is a class that performs cross-validation
        using CatBoostRegressor.

    Args:
        model_params: parameters for the model,
            for example: {
                "loss_function": "MultiRMSE",
                "early_stopping_rounds": 100,
            }.

    """

    def __init__(self, model_params: Dict[str, Union[str, int]]):
        super().__init__(model_params)

    def fit_one_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> "CatBoost":
        train_dataset = Pool(data=X_train, label=y_train)
        eval_dataset = Pool(data=X_val, label=y_val)

        # Set default params if params are None
        for param, default_value in [
            ("loss_function", "MultiRMSE"),
            ("thread_count", -1),
            ("random_state", 42),
            ("early_stopping_rounds", 100),
        ]:
            if self.model_params.get(param) is None:
                self.model_params[param] = default_value

        self.model = CatBoostRegressor(**self.model_params)

        self.model.fit(
            train_dataset,
            eval_set=eval_dataset,
            use_best_model=True,
            plot=False,
        )

        self.score = self.model.best_score_["validation"][f"{self.model_params['loss_function']}"]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class PyBoost(Estimator):
    """PyBoost is a class that performs cross-validation
        using PyBoostRegressor.

    Args:
        model_params: parameters for the model,
            for example: {
                "sketch": RandomSamplingSketch(10),  # PyBoostFullRegressor_CV, RandomProjectionSketch(1)
                "use_hess": True,
                "loss": "mse",
                "ntrees": 150000,
                "es": 100,
                "versbose": 1000
            }.

    Notes:
        Source: https://github.com/sberbank-ai-lab/Py-Boost

    """

    def __init__(self, model_params: Dict[str, Union[str, int]]):
        super().__init__(model_params)

    def fit_one_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> "PyBoost":
        # Set default params if params are None
        for param, default_value in [
            ("multioutput_sketch", RandomSamplingSketch(10)),
            ("use_hess", True),
            ("loss", "mse"),
            ("ntrees", 150000),
            ("es", 100),
            ("verbose", 1000),
        ]:
            if self.model_params.get(param) is None:
                self.model_params[param] = default_value

        self.model = GradientBoosting(**self.model_params)

        X_train = X_train.astype("float")
        X_val = X_val.astype("float")

        self.model.fit(
            X_train,
            y_train,
            eval_sets=[
                {"X": X_val, "y": y_val},
            ],
        )

        self.score = self.model.history[self.model.best_round]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
