"""Algorithms for time series forecasting."""

from tsururu.models.boost import CatBoost, PyBoost
from tsururu.models.ml_base import Estimator
from tsururu.models.torch_based.cycle_net import CycleNet_NN
from tsururu.models.torch_based.dlinear import DLinear_NN
from tsururu.models.torch_based.gpt import GPT4TS_NN
from tsururu.models.torch_based.patch_tst import PatchTST_NN
from tsururu.models.torch_based.times_net import TimesNet_NN


# Factory Object
class ModelsFactory:
    def __init__(self):
        self.models = {
            "CatBoost": CatBoost,
            "PyBoost": PyBoost,
            "DLinear_NN": DLinear_NN,
            "PatchTST_NN": PatchTST_NN,
            "GPT4TS_NN": GPT4TS_NN,
            "TimesNet_NN": TimesNet_NN,
            "CycleNet_NN": CycleNet_NN,
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
    "Estimator",
    "ModelsFactory",
    "CatBoost",
    "PyBoost",
    "DLinear_NN",
    "PatchTST_NN",
    "GPT4TS_NN",
    "TimesNet_NN",
    "CycleNet_NN",
]
