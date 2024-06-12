from typing import Dict, Union

import numpy as np

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

from .base import StatEstimator


class ETS_Model(StatEstimator):
    def __init__(self, model_params: Dict[str, Union[str, int]]):
        super().__init__(model_params=model_params, model_name="AutoETS")
        self.model = self._initialize_model()

    def _initialize_model(self):
        return AutoETS(**self.model_params)

    def new(self):
        return AutoETS(**self.model_params)

class ARIMA_Model(StatEstimator):
    def __init__(self, model_params: Dict[str, Union[str, int]]):
        super().__init__(model_params=model_params, model_name="AutoARIMA")
        self.model = self._initialize_model()

    def _initialize_model(self):
        return AutoARIMA(**self.model_params)

    def new(self):
        return AutoARIMA(**self.model_params)

class Theta_Model(StatEstimator):
    def __init__(self, model_params: Dict[str, Union[str, int]]):
        super().__init__(model_params=model_params, model_name="AutoTheta")
        self.model = self._initialize_model()

    def _initialize_model(self):
        return AutoTheta(**self.model_params)

    def new(self):
        return AutoTheta(**self.model_params)
