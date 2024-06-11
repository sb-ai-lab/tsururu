from typing import Dict, Union

import numpy as np

try:
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
except:
    LinearRegression = None
    Lasso = None
    Ridge = None

from .base import StatEstimator


class ETS_Model(StatEstimator):
    def __init__(
        self,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(validation_params, model_params, "AutoETS")

    def _initialize_model(self):
        return AutoETS(**self.model_params)

class ARIMA_Model(StatEstimator):
    def __init__(
        self,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(validation_params, model_params, "AutoARIMA")

    def _initialize_model(self):
        return AutoARIMA(**self.model_params)
    

class Theta_Model(StatEstimator):
    def __init__(
        self,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(validation_params, model_params, "AutoTheta")

    def _initialize_model(self):
        return AutoTheta(**self.model_params)
