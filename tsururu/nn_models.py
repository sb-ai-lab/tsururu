from typing import Dict, Union

import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import Autoformer, PatchTST

from .base_model import Estimator


class AutoformerRegressor_NN(Estimator):
    def __init__(
        self,
        get_num_iterations: bool,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]]
        ):
        
        self.model_params = model_params
        self.get_num_iterations = get_num_iterations
        self.validation_params = validation_params
        freq = self.model_params.pop('freq')
        self.model = Autoformer(**self.model_params)
        self.nf = NeuralForecast(models=[self.model], freq=freq) 
        
        super().__init__(get_num_iterations, validation_params, model_params)
    
    def fit(self, X: pd.DataFrame):
        
        self.nf.fit(df=X)
        
    def predict(self, X: pd.DataFrame):
        
        forecasts = self.nf.predict(static_df=X)
        return forecasts
    
    
class PatchTSTRegressor_NN(Estimator):
    def __init__(
        self,
        get_num_iterations: bool,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]]
        ):
        
        self.model_params = model_params
        self.get_num_iterations = get_num_iterations
        self.validation_params = validation_params
        self.model = PatchTST(**self.model_params)
        self.nf = NeuralForecast(models=[self.model], freq='D') 
        
        super().__init__(get_num_iterations, validation_params, model_params)
    
    def fit(self, X: pd.DataFrame):
        
        self.nf.fit(df=X)
        
    def predict(self, test_df):
        forecasts = self.nf.predict(static_df=test_df)
        return forecasts