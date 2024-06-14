"""Algorithms for time series forecasting."""

from .base import BaselineEstimator, Estimator, MLEstimator
from .baselines import DriftMethod, MeanMethod, NaiveMethod, SeasonalNaiveMethod
from .boost import CatBoost
from .linear import LassoRegression, LinRegression, RidgeRegression
from .random_forest import RandomForest
from .stats import ARIMA, ETS, Theta


# Factory Object
class ModelsFactory:
    def __init__(self):
        self.models = {
            "CatBoost": CatBoost,
            "LinRegression": LinRegression,
            "LassoRegression": LassoRegression,
            "RidgeRegression": RidgeRegression,
            "RandomForest": RandomForest,
            "DriftMethod": DriftMethod,
            "MeanMethod": MeanMethod,
            "NaiveMethod": NaiveMethod,
            "SeasonalNaiveMethod": SeasonalNaiveMethod,
            "ETS": ETS,
            "ARIMA": ARIMA,
            "Theta": Theta,
        }

    def get_allowed(self):
        return sorted(list(self.models.keys()))

    def __getitem__(self, params):
        return self.models[params["model_name"]](
            params["validation_params"],
            params["model_params"],
        )

    def create_model(self, model_name, model_params):
        return self.models[model_name](**model_params)


__all__ = [
    "ModelsFactory",
    "Estimator",
    "BaselineEstimator",
    "MLEstimator",
    "CatBoost",
    "LinRegression",
    "LassoRegression",
    "RidgeRegression",
    "RandomForest",
    "DriftMethod",
    "MeanMethod",
    "NaiveMethod",
    "SeasonalNaiveMethod",
    "ETS",
    "ARIMA",
    "Theta",
]
