from typing import Dict, Optional, Union

import numpy as np

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:
    Pool = None
    CatBoostRegressor = None
from .base import MLEstimator


class CatBoost(MLEstimator):
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

    def _initialize_model(self):
        # Set default params if params are None
        for param, default_value in [
            ("loss_function", "MultiRMSE"),
            ("thread_count", -1),
            ("random_state", 42),
            ("early_stopping_rounds", 100),
        ]:
            if self.model_params.get(param) is None:
                self.model_params[param] = default_value
        return CatBoostRegressor(**self.model_params)

    def fit_one_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> "CatBoost":
        train_dataset = Pool(data=X_train, label=y_train)
        eval_dataset = Pool(data=X_val, label=y_val)

        self.model = self._initialize_model()

        self.model.fit(
            train_dataset,
            eval_set=eval_dataset,
            use_best_model=True,
            plot=False,
        )

        self.score = self.model.best_score_["validation"][f"{self.model_params['loss_function']}"]

        return self
