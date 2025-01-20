"""Algorithms for time series forecasting."""

from .base import Estimator
from .boost import CatBoost
from .torch_based.dlinear import DLinear_NN
from .torch_based.patch_tst import PatchTST_NN


# Factory Object
class ModelsFactory:
    def __init__(self):
        self.models = {
            "CatBoost": CatBoost,
            "DLinear_NN": DLinear_NN,
            "PatchTST_NN": PatchTST_NN,
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


__all__ = ["CatBoost", "Estimator", "ModelsFactory", "DLinear_NN", "PatchTST_NN"]
