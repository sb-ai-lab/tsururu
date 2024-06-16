from typing import Dict, Optional, Union

import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
except:
    RandomForestRegressor = None
    mean_squared_error = None

from .base import MLEstimator


class RandomForest(MLEstimator):
    """RandomForest is a class that performs cross-validation
        using RandomForest from scikit-learn.
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

    Args:
        model_params: parameters for the CatBoostRegressor model,
            for example: {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            }.

    """

    def __init__(self, model_params: Dict[str, Union[str, int]]):
        super().__init__(model_params)

    def _initialize_model(self):
        for param, default_value in [
            ("n_estimators", 100),
            ("criterion", 'squared_error'),
            ("random_state", 42),
            ("max_features", 1.0),
            ("verbose", 2),
            ("n_jobs", -1),
        ]:
            if self.model_params.get(param) is None:
                self.model_params[param] = default_value
        return RandomForestRegressor(**self.model_params)
