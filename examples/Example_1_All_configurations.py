import argparse
import os
import glob
import random
import re
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from tqdm import tqdm

from tsururu.dataset import Pipeline, TSDataset
from tsururu.model_training.trainer import DLTrainer, MLTrainer
from tsururu.model_training.validator import KFoldCrossValidator
from tsururu.models.boost import CatBoost
from tsururu.models.torch_based.dlinear import DLinear_NN
from tsururu.strategies import (
    DirectStrategy,
    FlatWideMIMOStrategy,
    MIMOStrategy,
    RecursiveStrategy,
)

warnings.filterwarnings("ignore")

#################################
#   Constants
#################################

CURRENT_DIR = Path(__file__).parent

# Data
DATE_COLUMN = "date"
ID_COLUMN = "id"
TARGET_COLUMN = "value"

# Forecasting task
HISTORY = 52
HORIZON = 4
STEP = 1

VALIDATION = KFoldCrossValidator
VALIDATION_PARAMS = {"n_splits": 3}

# Optimization
BATCH_SIZE = 32
LEARNING_RATE = 0.05

# Model, Strategy, Preprocessing
MODELS = [
    (DLinear_NN, {"moving_avg": 25, "individual": False, "enc_in": 7}, DLTrainer),
    (CatBoost, {}, MLTrainer),
]

STRATEGIES_OVER_TIME = [
    "RecursiveStrategy",
    "DirectStrategy",
    "MIMOStrategy",
    "FlatWideMIMOStrategy",
]
MULTIVARIATE = [
    True, 
    False
]

DATE_FEATURES = [
    "without_normalization",
    "with_normalization_over_all",
    False,
]

ID_FEATURES = [
    "with_ohe",
    "with_le_without_normalization",
    "with_le_normalization_over_all",
    False,
]

INDIVIDUAL_MODEL_HORIZON = [None, 1, 2]

TRANSFORMERS = {
    None,
    "StandardScalerTransformer",
    "DifferenceNormalizer",
    "LastKnownNormalizer",
}
TRANSFORMERS_REGIMES = [None, "delta", "ratio"]
TRANSFORM_TARGET = [True, False]
TRANSFORM_FEATURES = [True, False]


# Training on GPU / CPU
cuda_device_available = torch.cuda.is_available()
cuda_device_number = 0
if cuda_device_available:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_number)
    device = torch.device(f"cuda:{cuda_device_number}")
else:
    device = torch.device("cpu")


def lradj(epoch):
    if epoch < 3:
        return 1
    else:
        return 0.9 ** ((epoch - 3) // 1)


def get_metrics(x):
    res_dict = {
        "mae": mean_absolute_error(x[TARGET_COLUMN], x["y_pred"]),
        "rmse": root_mean_squared_error(x[TARGET_COLUMN], x["y_pred"]),
        "mape": mean_absolute_percentage_error(x[TARGET_COLUMN], x["y_pred"]),
    }
    return pd.Series(res_dict)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed_everything()

    #################################
    #   0. CLI parameters
    #################################

    parser = argparse.ArgumentParser(
        description="Check all strategies, regimes and preprocessings."
    )

    parser.add_argument(
        "--df_path",
        type=Path,
        default=Path("datasets/global/simulated_data_to_check.csv"),
        help="Path to the dataframe CSV file.",
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        default=Path("./logs/all_configuraion.txt"),
        help="Path to the log file.",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        default=Path("./results/"),
        help="Path to the results file.",
    )

    args = parser.parse_args()

    args.df_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.parent.mkdir(parents=True, exist_ok=True)

    #################################
    #   1. dataset, pipeline, model, validation -> trainer
    #################################

    df = pd.read_csv(args.df_path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    train_test_date_split = df.loc[df[ID_COLUMN] == 0, DATE_COLUMN].values[-HORIZON]
    train = df.loc[df[DATE_COLUMN] < train_test_date_split]
    test = df.loc[df[DATE_COLUMN] >= train_test_date_split]

    dataset_params = {
        "target": {
            "columns": [TARGET_COLUMN],
            "type": "continuous",
        },
        "date": {
            "columns": [DATE_COLUMN],
            "type": "datetime",
        },
        "id": {
            "columns": [ID_COLUMN],
            "type": "categorical",
        },
    }

    dataset = TSDataset(
        data=train,
        columns_params=dataset_params,
    )

    pipeline_params = {
        "target": {
            "columns": [TARGET_COLUMN],
            "features": {},
        }
    }

    optimizer_params = {
        "lr": LEARNING_RATE,
    }

    sch, sch_params = lr_scheduler.LambdaLR, {"lr_lambda": lradj}

    dl_trainer_params = {
        "device": device,
        "num_workers": 0,
        "best_by_metric": True,
        "batch_size": BATCH_SIZE,
        "optimizer_params": optimizer_params,
        "scheduler": sch,
        "scheduler_params": sch_params,
        "save_k_best": 1,
        "save_to_dir": False,
    }
    ml_trainer_params = {}

    #################################
    #   2. Loop through model, strategies, regimes, preprocessings
    #################################

    df_list = []

    sys.stdout = open(args.log_path, "a")

    for (
        model,
        strategy_over_time,
        multivariate,
        date_features,
        id_features,
        model_horizon,
        transformer,
        transformer_regime,
        transform_target,
        transform_features,
    ) in tqdm(
        product(
            MODELS,
            STRATEGIES_OVER_TIME,
            MULTIVARIATE,
            DATE_FEATURES,
            ID_FEATURES,
            INDIVIDUAL_MODEL_HORIZON,
            TRANSFORMERS,
            TRANSFORMERS_REGIMES,
            TRANSFORM_TARGET,
            TRANSFORM_FEATURES,
        ),
        total=(
            len(MODELS)
            * len(STRATEGIES_OVER_TIME)
            * len(MULTIVARIATE)
            * len(DATE_FEATURES)
            * len(ID_FEATURES)
            * len(INDIVIDUAL_MODEL_HORIZON)
            * len(TRANSFORMERS)
            * len(TRANSFORMERS_REGIMES)
            * len(TRANSFORM_TARGET)
            * len(TRANSFORM_FEATURES)
        ),
    ):
        # Cases when we skip the combination
        if (
            strategy_over_time == "MIMOStrategy" or strategy_over_time == "FlatWideMIMOStrategy"
        ) and model_horizon is not None:
            continue

        if (
            strategy_over_time == "RecursiveStrategy" or strategy_over_time == "DirectStrategy"
        ) and model_horizon is None:
            continue

        if strategy_over_time == "FlatWideMIMOStrategy" and date_features is False:
            continue

        if multivariate is True and id_features is not False:
            continue

        if transformer == "StandardScalerTransformer" or transformer is None:
            if (
                transformer_regime is not None
                or transform_target is not False
                or transform_features is not False
            ):
                continue
        else:
            if (
                transformer_regime is None
                or transform_target is False
                and transform_features is False
            ):
                continue

        print(
            f"""
            Model: {model[0].__name__}
            Strategy: {strategy_over_time}
            Multivariate: {multivariate}
            Date features: {date_features}
            ID features: {id_features}
            Model horizon: {model_horizon}
            Transformer: {transformer}
            Transformer regime: {transformer_regime}
            Transform target: {transform_target}
            Transform features: {transform_features}
            """
        )

        model_name = (
            f"model_{model[0].__name__}/strategy_time_{strategy_over_time}__mult_{multivariate}/"
            f"dataset_{args.df_path.stem}__hist_{HISTORY}__hor_{HORIZON}__model_hor_{model_horizon}/"
            f"datetime_{date_features}__id_{id_features}__"
            f"transformer_{transformer}__regime_{transformer_regime}__"
            f"tr_target_{transform_target}__tr_features_{transform_features}"
        )

        #################################
        #   4. pipeline -> trainer
        #################################
        pipeline_params = {
            "target": {
                "columns": ["value"],
                "features": {},
            }
        }

        # Change pipeline params and initialize pipeline
        # Target
        if transformer == "StandardScalerTransformer":
            pipeline_params["target"]["features"] = {
                transformer: {
                    "transform_features": transform_features,
                    "transform_target": transform_target,
                },
                "LagTransformer": {"lags": HISTORY},
            }
        elif transformer == "DifferenceNormalizer":
            pipeline_params["target"]["features"] = {
                transformer: {
                    "transform_features": transform_features,
                    "transform_target": transform_target,
                    "regime": transformer_regime,
                },
                "MissingValueImputer": {
                    "constant_value": 0,
                    "transform_features": transform_features,
                    "transform_target": transform_target,
                },
                "LagTransformer": {"lags": HISTORY},
            }
        elif transformer == "LastKnownNormalizer":
            pipeline_params["target"]["features"] = {
                "LagTransformer": {"lags": HISTORY},
                transformer: {
                    "transform_features": transform_features,
                    "transform_target": transform_target,
                    "regime": transformer_regime,
                },
            }
        else:
            pipeline_params["target"]["features"] = {"LagTransformer": {"lags": HISTORY}}

        # Date
        date_features_dict = {
            "DateSeasonsGenerator": {
                "seasonalities": ["doy", "m", "wd", "hour"],
                "from_target_date": True,
            },
            "MissingValueImputer": {
                "constant_value": 0,
                "transform_features": True,
                "transform_target": False,
            },
        }

        if date_features == "without_normalization":
            scaler = None
        elif date_features == "with_normalization_over_all":
            scaler = {
                "StandardScalerTransformer": {
                    "transform_features": True,
                    "transform_target": False,
                    "agg_by_id": False,
                }
            }
        else:
            pipeline_params.pop("date", None)
            scaler = None

        if scaler is not None:
            lags_value = HORIZON if strategy_over_time == "FlatWideMIMOStrategy" else 1
            date_features = {
                **date_features_dict,
                **scaler,
                "LagTransformer": {"lags": lags_value},
            }
            pipeline_params["date"] = {"columns": ["date"], "features": date_features}

        # ID
        if id_features == "with_ohe":
            pipeline_params["id"] = {
                "columns": ["id"],
                "features": {
                    "OneHotEncodingTransformer": {
                        "drop": "first",
                    },
                    "LagTransformer": {"lags": 1},
                },
            }
        elif id_features == "with_le_without_normalization":
            pipeline_params["id"] = {
                "columns": ["id"],
                "features": {
                    "LabelEncodingTransformer": {},
                    "LagTransformer": {"lags": 1},
                },
            }
        elif id_features == "with_le_normalization_over_all":
            pipeline_params["id"] = {
                "columns": ["id"],
                "features": {
                    "LabelEncodingTransformer": {},
                    "StandardScalerTransformer": {
                        "transform_features": True,
                        "transform_target": False,
                        "agg_by_id": False,
                    },
                    "LagTransformer": {"lags": 1},
                },
            }
        else:
            if "id" in pipeline_params:
                del pipeline_params["id"]

        pipeline = Pipeline.from_dict(pipeline_params, multivariate=multivariate)

        model, model_params, trainer = model
        
        if trainer == MLTrainer:
            trainer = trainer(model, model_params, VALIDATION, VALIDATION_PARAMS, **ml_trainer_params)
        else:
            trainer = trainer(model, model_params, VALIDATION, VALIDATION_PARAMS, **dl_trainer_params)

        try:
            if strategy_over_time == "RecursiveStrategy":
                strategy = RecursiveStrategy(
                    HORIZON, HISTORY, trainer, pipeline, STEP, model_horizon=model_horizon
                )
            elif strategy_over_time == "DirectStrategy":
                strategy = DirectStrategy(
                    HORIZON, HISTORY, trainer, pipeline, STEP, model_horizon=model_horizon
                )
            elif strategy_over_time == "MIMOStrategy":
                strategy = MIMOStrategy(HORIZON, HISTORY, trainer, pipeline, STEP)
            elif strategy_over_time == "FlatWideMIMOStrategy":
                strategy = FlatWideMIMOStrategy(HORIZON, HISTORY, trainer, pipeline, STEP)

            fit_time, _ = strategy.fit(dataset)
            forecast_time, current_pred = strategy.predict(dataset)
            current_pred = current_pred.rename(columns={TARGET_COLUMN: "y_pred"})
            current_pred = current_pred.merge(test, on=[DATE_COLUMN, ID_COLUMN])

            (args.results_path / model_name).parent.mkdir(parents=True, exist_ok=True)
            
            current_pred.to_csv(
                os.path.join(args.results_path / f"{model_name}_pred.csv"), index=False
            )

            metrics = current_pred.groupby(ID_COLUMN).apply(lambda x: get_metrics(x)).reset_index()
            metrics.loc["mean"] = metrics.mean()
            metrics.loc["fit_time"] = fit_time
            metrics.loc["forecast_time"] = forecast_time

            metrics = metrics.reset_index()
            metrics = metrics.drop(columns=[ID_COLUMN])
            metrics = metrics.rename(columns={"index": ID_COLUMN})

            metrics.to_csv(
                os.path.join(args.results_path / f"{model_name}_metrics.csv"), index=False
            )
            print("Success!")

        except:
            print("Fail!")

    # Aggregate results
    print("Aggregating results...")
    for file in glob.glob(str(args.results_path / "*/*/*/*.csv")):
        if file.endswith("_metrics.csv"):
            current_df = pd.read_csv(file)
            current_df["model"] = re.search(r"model_(.*?)\/", file).group(1)
            current_df["strategy_over_time"] = re.search(r"strategy_time_(.*?)__", file).group(1)
            current_df["multivariate"] = re.search(
                r"mult_(.*?)\/", file
            ).group(1)
            current_df["date_features"] = re.search(r"datetime_(.*?)__", file).group(1)
            current_df["id_features"] = re.search(r"id_(.*?)__", file).group(1)
            current_df["model_horizon"] = re.search(r"model_hor_(.*?)\/", file).group(1)
            current_df["transformer"] = re.search(r"transformer_(.*?)__", file).group(1)
            current_df["transformer_regime"] = re.search(r"regime_(.*?)__", file).group(1)
            current_df["transform_target"] = re.search(r"tr_target_(.*?)__", file).group(1)
            current_df["transform_features"] = re.search(r"tr_features_(.*?)_", file).group(1)
            df_list.append(current_df)

    df = pd.concat(df_list)
    df.to_csv(args.results_path / "all_results.csv", index=False)

    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
