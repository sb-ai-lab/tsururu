import logging
from typing import Dict, Optional, Union

import numpy as np

from ..utils.logging import LoggerStream

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:
    Pool = None
    CatBoostRegressor = None
from .base import Estimator

logger = logging.getLogger(__name__)


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
        self.trainer_type = "MLTrainer"

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
            ("verbose", 100),
        ]:
            if self.model_params.get(param) is None:
                self.model_params[param] = default_value

        self.model = CatBoostRegressor(**self.model_params)

        self.model.fit(
            train_dataset,
            eval_set=eval_dataset,
            use_best_model=True,
            plot=False,
            log_cout=LoggerStream(logger, verbose_eval=self.model_params["verbose"]),
        )

        self.score = self.model.best_score_["validation"][f"{self.model_params['loss_function']}"]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
