from .models import CatBoostRegressor_CV
from .nn_models import AutoformerRegressor_NN, PatchTSTRegressor_NN

# Factory Object
class ModelsFactory:
    def __init__(self):
        self.models = {
            "CatBoostRegressor_CV": CatBoostRegressor_CV,
            "AutoformerRegressor_NN": AutoformerRegressor_NN,
            "PatchTSTRegressor_NN": PatchTSTRegressor_NN,
            }

    def get_allowed(self):
        return sorted(list(self.models.keys()))

    def __getitem__(self, params):
        return self.models[params["model_name"]](
            params["get_num_iterations"],
            params["validation_params"],
            params["model_params"],
        )