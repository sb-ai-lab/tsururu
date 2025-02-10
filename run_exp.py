import yaml
import argparse
import torch

from pathlib import Path

from copy import deepcopy
from torch.optim import lr_scheduler
from validation import get_train_val_test_datasets

import os

from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from validation import get_train_val_test_datasets, path_to_tsururu_format, get_fitted_scaler_on_train

from tsururu.dataset import Pipeline
from tsururu.model_training.trainer import DLTrainer
from tsururu.model_training.validator import HoldOutValidator
from tsururu.models import DLinear_NN, PatchTST_NN, TimesNet_NN, TimeMixer_NN, GPT4TS_NN, CycleNet_NN
from tsururu.models.torch_based.time_mixer import TimeMixer_NN
from tsururu.strategies import DirectStrategy, MIMOStrategy, RecursiveStrategy
from tsururu.transformers import (
    LagTransformer,
    SequentialTransformer,
    StandardScalerTransformer,
    TargetGenerator,
    UnionTransformer,
    DateSeasonsGenerator,
    MissingValuesImputer,
    CycleGenerator,
)
import random
from torch.optim import lr_scheduler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

MODELS = {
    "DLinear_NN": DLinear_NN,
    "PatchTST_NN": PatchTST_NN,
    "GPT4TS_NN": GPT4TS_NN,
    "TimeMixer_NN": TimeMixer_NN,
    "TimesNet_NN": TimesNet_NN,
    "CycleNet_NN": CycleNet_NN,
}

SCHEDULERS = {
    "LambdaLR": LambdaLR,
    "CosineAnnealingLR": CosineAnnealingLR,
}

DATASET_PARAMS = {
    "target": {
        "columns": ["value"],
        "type": "continious",
    },
    "date": {
        "columns": ["date"],
        "type": "datetime",
    },
    "id": {
        "columns": ["id"],
        "type": "categorical",
    }
}


def run_experiment(config_path):
    # Load YAML config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    horizon = config["horizon"]
    history = config["history"]

    dataset_tsururu_path = path_to_tsururu_format(config["data"]["root_path"], config["data"]["data_path"])

    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        dataset_path=dataset_tsururu_path,
        columns_params=DATASET_PARAMS,  
        train_size=0.7,
        test_size=0.2,
        history=history,
    )

    ss = StandardScalerTransformer(transform_features=True, transform_target=True)

    lag = LagTransformer(lags=history)
    target_generator = TargetGenerator()
    
    union_1 = UnionTransformer(transformers_list=[lag, target_generator])
    seq_1 = SequentialTransformer(transformers_list=[ss, union_1], input_features=["value"])
    transformers = [seq_1]

    if config["model"]["model_type"] == "TimesNet_NN":
        datetime = DateSeasonsGenerator(seasonalities=["hour", "wd", "d", "doy"], from_target_date=True)
        datetime_lag = LagTransformer(lags=history)
        datetime_ss = StandardScalerTransformer(
            transform_features=True, transform_target=False, agg_by_id=False
        )
        datetime_imp = MissingValuesImputer(
            regime="constant", constant_value=0, transform_features=True, transform_target=False
        )
        seq_2 = SequentialTransformer(
            transformers_list=[datetime, datetime_ss, datetime_imp, datetime_lag], input_features=["date"]
        )

        transformers.append(seq_2)

    if config["model"]["model_type"] == "CycleNet_NN":
        cycle = CycleGenerator(cycle=config["model"]["model_params"]["cycle_len"])
        cycle_lag = LagTransformer(lags=history)

        seq_2 = SequentialTransformer(
            transformers_list=[cycle, cycle_lag], input_features=["date"]
        )

        transformers.append(seq_2)

    union = UnionTransformer(transformers_list=transformers)
    pipeline = Pipeline(union, multivariate=config["multivaraiate"])

    model_class = MODELS[config["model"]["model_type"]]
    model_params = config["model"]["model_params"]

    validation = HoldOutValidator
    validation_params = {"validation_data": val_dataset}

    scheduler = SCHEDULERS[config["scheduler"]["scheduler_type"]]
    if config["scheduler"]["scheduler_type"] == "LambdaLR":
        scheduler_params = {"lr_lambda": eval(config["scheduler"]["scheduler_params"]["lr_lambda"])}
    else:
        scheduler_params = config["scheduler"]

    trainer_params = config["trainer_params"]
    trainer = DLTrainer(
        model_class, 
        model_params, 
        validation, 
        validation_params,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        **trainer_params
    )

    strategy = MIMOStrategy(
        pipeline=pipeline,
        trainer=trainer,
        horizon=horizon,
        history=history,
    )

    strategy.fit(train_dataset)

    _, current_pred = strategy.predict(test_dataset, test_all=True, inverse_transform=False)

    stat_df = get_fitted_scaler_on_train(dataset_tsururu_path)

    scaled_test_dataset = deepcopy(test_dataset.data)
    scaled_test_dataset = scaled_test_dataset.merge(stat_df, left_on='id', right_index=True)
    scaled_test_dataset['value'] = (scaled_test_dataset['value'] - scaled_test_dataset['mean']) / scaled_test_dataset['std']

    current_pred = current_pred.rename(columns={'value': 'pred'})
    scaled_test_dataset = scaled_test_dataset.rename(columns={'value': 'true'})

    merged = scaled_test_dataset.merge(current_pred, on=['date', 'id'])

    mae = mean_absolute_error(merged['true'], merged['pred'])
    mse = mean_squared_error(merged['true'], merged['pred'])

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run time series forecasting experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    
    run_experiment(args.config)
