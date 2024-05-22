"""Algorithms for time series forecasting."""

from .base import Estimator
from .boost import CatBoostRegressor_CV


# Factory Object
class ModelsFactory:
    def __init__(self):
        self.models = {
            "CatBoostRegressor_CV": CatBoostRegressor_CV,
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


__all__ = ["CatBoostRegressor_CV", "ModelsFactory", "Estimator"]
