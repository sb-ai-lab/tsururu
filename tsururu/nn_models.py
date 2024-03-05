from typing import Dict, Union

import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import Autoformer, Informer, PatchTST

from .base_model import Estimator


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class DLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len
        
        self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            
            
    @staticmethod
    def my_Layernorm(channels):
        """
        Special designed layernorm for the seasonal part
        """

        layernorm = nn.LayerNorm(channels)

        def forward(x):
            x_hat = layernorm(x)
            bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
            return x_hat - bias
        
        return forward


    @staticmethod
    def moving_avg(kernel_size, stride):
        """
        Moving average block to highlight the trend of time series
        """

        avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

        def forward(x):
            # padding on the both ends of time series
            front = x[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=1)
            x = avg(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
            return x
        
        return forward


    @staticmethod
    def series_decomp(kernel_size):
        """
        Series decomposition block
        """

        moving_avg = DLinear.moving_avg(kernel_size, stride=1)

        def forward(x):
            moving_mean = moving_avg(x)
            res = x - moving_mean
            return res, moving_mean
        
        return forward

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]



# class AutoformerRegressor_NN(Estimator):
#     def __init__(
#         self,
#         get_num_iterations: bool,
#         validation_params: Dict[str, Union[str, int]],
#         model_params: Dict[str, Union[str, int]]
#         ):
        
#         self.model_params = model_params
#         self.get_num_iterations = get_num_iterations
#         self.validation_params = validation_params
#         # freq = self.model_params.pop('freq')
#         self.model = Autoformer(**self.model_params)
#         self.nf = NeuralForecast(models=[self.model], freq='T') 
        
#         super().__init__(get_num_iterations, validation_params, model_params)
    
#     def fit(self, X: pd.DataFrame):
        
#         self.nf.fit(df=X)
        
#     def predict(self, X: pd.DataFrame):
        
#         forecasts = self.nf.predict(static_df=X)
#         return forecasts
    
# class InformerRegressor_NN(Estimator):
#     def __init__(
#         self,
#         get_num_iterations: bool,
#         validation_params: Dict[str, Union[str, int]],
#         model_params: Dict[str, Union[str, int]]
#         ):
        
#         self.model_params = model_params
#         self.get_num_iterations = get_num_iterations
#         self.validation_params = validation_params
#         # freq = self.model_params.pop('freq')
#         self.model = Informer(**self.model_params)
#         self.nf = NeuralForecast(models=[self.model], freq='T') 
        
#         super().__init__(get_num_iterations, validation_params, model_params)
    
#     def fit(self, X: pd.DataFrame):
        
#         self.nf.fit(df=X)
        
#     def predict(self, X: pd.DataFrame):
        
#         forecasts = self.nf.predict(static_df=X)
#         return forecasts
    
    
# class PatchTSTRegressor_NN(Estimator):
#     def __init__(
#         self,
#         get_num_iterations: bool,
#         validation_params: Dict[str, Union[str, int]],
#         model_params: Dict[str, Union[str, int]]
#         ):
        
#         self.model_params = model_params
#         self.get_num_iterations = get_num_iterations
#         self.validation_params = validation_params
#         # freq = self.model_params.pop('freq')
#         self.model = PatchTST(**self.model_params)
#         self.nf = NeuralForecast(models=[self.model], freq='T') 
        
#         super().__init__(get_num_iterations, validation_params, model_params)
    
#     def fit(self, X: pd.DataFrame):
        
#         self.nf.fit(df=X)
        
#     def predict(self, X):
#         forecasts = self.nf.predict(static_df=X)
#         return forecasts