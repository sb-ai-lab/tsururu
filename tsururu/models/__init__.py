"""Algorithms for time series forecasting."""


from .base import Estimator, BaselineEstimator, MLEstimator
from .boost import CatBoost
from .baselines import (
    DriftMethod,
    MeanMethod,
    NaiveMethod,
    SeasonalNaiveMethod,
)
from .stats import (
    ETS,
    ARIMA,
    Theta,
)
from .linear import (
    LinearRegression_CV,
    LassoRegression_CV,
    RidgeRegression_CV,
)
from .random_forest import RandomForest_CV


# Factory Object
class ModelsFactory:
    def __init__(self):
        self.models = {
            "CatBoostRegressor_CV": CatBoost,
            "LinearRegression_CV": LinearRegression_CV,
            "LassoRegression_CV": LassoRegression_CV,
            "RidgeRegression_CV": RidgeRegression_CV,
            "RandomForest_CV": RandomForest_CV,
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
    "LinearRegression_CV",
    "LassoRegression_CV",
    "RidgeRegression_CV",
    "RandomForest_CV",
    "DriftMethod",
    "MeanMethod",
    "NaiveMethod",
    "SeasonalNaiveMethod",
    "ETS",
    "ARIMA",
    "Theta"
]
