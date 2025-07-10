import argparse
import glob
import os
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
from constants import all_models_params
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from tqdm import tqdm

from tsururu.dataset import Pipeline
from tsururu.examples.all_configurations_benchmark.scripts.validation import (
    get_train_val_test_datasets,
)
from tsururu.model_training.trainer import DLTrainer, MLTrainer
from tsururu.model_training.validator import HoldOutValidator
from tsururu.models.boost import PyBoost
from tsururu.models.torch_based.cycle_net import CycleNet_NN
from tsururu.models.torch_based.dlinear import DLinear_NN
from tsururu.models.torch_based.gpt import GPT4TS_NN
from tsururu.models.torch_based.patch_tst import PatchTST_NN
from tsururu.models.torch_based.times_net import TimesNet_NN
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
TRAIN_SIZE = 0.7
TEST_SIZE = 0.2
HISTORY = 96
HORIZON = 24
STEP = 1

# Optimization
BATCH_SIZE = 128
LEARNING_RATE = 0.0001

# Model, Strategy, Preprocessing
MODELS = [
    (PyBoost, {}, MLTrainer),
    (DLinear_NN, all_models_params["ILI"]["DLinear"], DLTrainer),
    (PatchTST_NN, all_models_params["ILI"]["PatchTST"], DLTrainer),
    (GPT4TS_NN, all_models_params["ILI"]["GPT4TS"], DLTrainer),
    (TimesNet_NN, all_models_params["ILI"]["TimesNet"], DLTrainer),
    (CycleNet_NN, all_models_params["ILI"]["CycleNet"], DLTrainer),
]

STRATEGIES_OVER_TIME = [
    "MIMOStrategy",
    "FlatWideMIMOStrategy",
    "RecursiveStrategy",
]
MULTIVARIATE = [True, False]
CHANNEL_INDEPENDENT_LIST = [True, False]

DATE_FEATURES = ["with_normalization_over_all", False]
ID_FEATURES = ["with_le_normalization_over_all", False]

INDIVIDUAL_MODEL_HORIZON = [None, 1, 6]

TRANSFORMERS = {
    None,
    "DifferenceNormalizer",
    "LastKnownNormalizer",
}
TRANSFORMERS_REGIMES = [
    None,
    "delta",
    # "ratio"
]
TRANSFORM_TARGET = [True]
TRANSFORM_FEATURES = [True]


# Training on GPU / CPU
cuda_device_available = torch.cuda.is_available()
cuda_device_number = 0
if cuda_device_available:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_number)
    device = torch.device(f"cuda:{cuda_device_number}")
else:
    device = torch.device("cpu")


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
        default=Path("datasets/global/demand_forecasting_kernels.csv"),
        help="Path to the dataframe CSV file.",
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        default=Path("./logs/demand_tr.txt"),
        help="Path to the log file.",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        default=Path("./results/demand"),
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

    dataset_params = {
        "target": {
            "columns": [TARGET_COLUMN],
            "type": "continious",
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

    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        args.df_path, dataset_params, TRAIN_SIZE, TEST_SIZE, HISTORY
    )

    pipeline_params = {
        "target": {
            "columns": [TARGET_COLUMN],
            "features": {},
        }
    }

    validation = HoldOutValidator
    validation_params = {"validation_data": val_dataset}

    optimizer_params = {
        "lr": LEARNING_RATE,
    }

    sch, sch_params = lr_scheduler.CosineAnnealingLR, {"eta_min": 1e-8}

    dl_trainer_params = {
        "device": device,
        "num_workers": 10,
        "best_by_metric": True,
        "batch_size": BATCH_SIZE,
        "optimizer_params": optimizer_params,
        "scheduler": sch,
        "scheduler_params": sch_params,
        "scheduler_after_epoch": False,
        "save_k_best": 1,
        "save_to_dir": False,
        "n_epochs": 50,
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
        channel_independent,
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
            CHANNEL_INDEPENDENT_LIST,
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
            * len(CHANNEL_INDEPENDENT_LIST)
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
        if multivariate is False and channel_independent is True:
            continue

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
                or transform_target is False
                or transform_features is False
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
            CI: {channel_independent}
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
            f"model_{model[0].__name__}/strategy_time_{strategy_over_time}__mult_{multivariate}__ci_{channel_independent}/"
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
                "StandardScalerTransformer": {
                    "transform_features": True,
                    "transform_target": True,
                },
                transformer: {
                    "transform_features": transform_features,
                    "transform_target": transform_target,
                    "regime": transformer_regime,
                },
                "MissingValuesImputer": {
                    "constant_value": 0,
                    "transform_features": transform_features,
                    "transform_target": transform_target,
                },
                "LagTransformer": {"lags": HISTORY},
            }
        elif transformer == "LastKnownNormalizer":
            pipeline_params["target"]["features"] = {
                "StandardScalerTransformer": {
                    "transform_features": True,
                    "transform_target": True,
                },
                "LagTransformer": {"lags": HISTORY},
                transformer: {
                    "transform_features": transform_features,
                    "transform_target": transform_target,
                    "regime": transformer_regime,
                },
            }
        else:
            pipeline_params["target"]["features"] = {
                "StandardScalerTransformer": {
                    "transform_features": True,
                    "transform_target": True,
                },
                "LagTransformer": {"lags": HISTORY},
            }
        # Date
        date_features_dict = {
            "DateSeasonsGenerator": {
                "seasonalities": ["doy", "m"],
                "from_target_date": True,
            },
            "MissingValuesImputer": {
                "constant_value": 0,
                "transform_features": True,
                "transform_target": False,
            },
        }

        if date_features == "without_normalization":
            scaler = None
        elif date_features == "with_normalization_over_all":
            del date_features_dict["MissingValuesImputer"]
            scaler = {
                "StandardScalerTransformer": {
                    "transform_features": True,
                    "transform_target": False,
                    "agg_by_id": False,
                },
                "MissingValuesImputer": {
                    "constant_value": 0,
                    "transform_features": True,
                    "transform_target": False,
                },
            }
        else:
            pipeline_params.pop("date", None)
            scaler = None

        if scaler is not None:
            lags_value = HORIZON if strategy_over_time == "FlatWideMIMOStrategy" else HISTORY
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
                    "LagTransformer": {"lags": HISTORY},
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
                    "LagTransformer": {"lags": HISTORY},
                },
            }
        else:
            if "id" in pipeline_params:
                del pipeline_params["id"]

        model, model_params, trainer = model

        if model_params.get("cycle_len"):
            pipeline_params["cycle"] = {
                "columns": ["date"],
                "features": {
                    "CycleGenerator": {
                        "cycle": 24,
                    },
                    "LagTransformer": {"lags": HISTORY},
                },
            }

        pipeline = Pipeline.from_dict(pipeline_params, multivariate=multivariate)

        if trainer != MLTrainer:
            model_params["channel_independent"] = channel_independent

        if trainer == MLTrainer and channel_independent is True:
            continue

        if trainer == MLTrainer:
            trainer = trainer(
                model, model_params, validation, validation_params, **ml_trainer_params
            )
        else:
            trainer = trainer(
                model, model_params, validation, validation_params, **dl_trainer_params
            )

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
            fit_time, _ = strategy.fit(dataset=train_dataset)
            forecast_time_test, current_pred_test = strategy.predict(test_dataset, test_all=True)
            forecast_time_val, current_pred_val = strategy.predict(val_dataset, test_all=True)

            current_pred_test = current_pred_test.rename(columns={TARGET_COLUMN: "y_pred"})
            current_pred_test = current_pred_test.merge(
                test_dataset.data, on=[DATE_COLUMN, ID_COLUMN]
            )

            current_pred_val = current_pred_val.rename(columns={TARGET_COLUMN: "y_pred"})
            current_pred_val = current_pred_val.merge(
                val_dataset.data, on=[DATE_COLUMN, ID_COLUMN]
            )

            (args.results_path / model_name).parent.mkdir(parents=True, exist_ok=True)

            current_pred_test.to_csv(
                os.path.join(args.results_path / f"{model_name}__pred_test.csv"), index=False
            )

            current_pred_val.to_csv(
                os.path.join(args.results_path / f"{model_name}__pred_val.csv"), index=False
            )

            metrics_test = (
                current_pred_test.groupby(ID_COLUMN).apply(lambda x: get_metrics(x)).reset_index()
            )
            metrics_test.loc["mean"] = metrics_test.mean()
            metrics_test.loc["fit_time"] = fit_time
            metrics_test.loc["forecast_time"] = forecast_time_test

            metrics_test = metrics_test.reset_index()
            metrics_test = metrics_test.drop(columns=[ID_COLUMN])
            metrics_test = metrics_test.rename(columns={"index": ID_COLUMN})

            metrics_test.to_csv(
                os.path.join(args.results_path / f"{model_name}__metrics_test.csv"), index=False
            )

            metrics_val = (
                current_pred_val.groupby(ID_COLUMN).apply(lambda x: get_metrics(x)).reset_index()
            )
            metrics_val.loc["mean"] = metrics_val.mean()
            metrics_val.loc["fit_time"] = fit_time
            metrics_val.loc["forecast_time"] = forecast_time_val

            metrics_val = metrics_val.reset_index()
            metrics_val = metrics_val.drop(columns=[ID_COLUMN])
            metrics_val = metrics_val.rename(columns={"index": ID_COLUMN})

            metrics_val.to_csv(
                os.path.join(args.results_path / f"{model_name}__metrics_val.csv"), index=False
            )
            print("Success!")

        except:
            print("Fail!")

    # Aggregate results
    print("Aggregating test results...")
    for file in glob.glob(str(args.results_path / "*/*/*/*.csv")):
        if file.endswith("_metrics_test.csv"):
            current_df = pd.read_csv(file)
            current_df["model"] = re.search(r"model_(.*?)\/", file).group(1)
            current_df["strategy_over_time"] = re.search(r"strategy_time_(.*?)__", file).group(1)
            current_df["multivariate"] = re.search(r"mult_(.*?)\/", file).group(1)
            current_df["channel_independence"] = re.search(r"ci_(.*?)\/", file).group(1)
            current_df["date_features"] = re.search(r"datetime_(.*?)__", file).group(1)
            current_df["id_features"] = re.search(r"id_(.*?)__", file).group(1)
            current_df["model_horizon"] = re.search(r"model_hor_(.*?)\/", file).group(1)
            current_df["transformer"] = re.search(r"transformer_(.*?)__", file).group(1)
            current_df["transformer_regime"] = re.search(r"regime_(.*?)__", file).group(1)
            current_df["transform_target"] = re.search(r"tr_target_(.*?)__", file).group(1)
            current_df["transform_features"] = re.search(r"tr_features_(.*?)_", file).group(1)
            df_list.append(current_df)

    df = pd.concat(df_list)
    df.to_csv(args.results_path / "all_results_test.csv", index=False)

    print("Aggregating val results...")
    for file in glob.glob(str(args.results_path / "*/*/*/*.csv")):
        if file.endswith("_metrics_val.csv"):
            current_df = pd.read_csv(file)
            current_df["model"] = re.search(r"model_(.*?)\/", file).group(1)
            current_df["strategy_over_time"] = re.search(r"strategy_time_(.*?)__", file).group(1)
            current_df["multivariate"] = re.search(r"mult_(.*?)\/", file).group(1)
            current_df["channel_independence"] = re.search(r"ci_(.*?)\/", file).group(1)
            current_df["date_features"] = re.search(r"datetime_(.*?)__", file).group(1)
            current_df["id_features"] = re.search(r"id_(.*?)__", file).group(1)
            current_df["model_horizon"] = re.search(r"model_hor_(.*?)\/", file).group(1)
            current_df["transformer"] = re.search(r"transformer_(.*?)__", file).group(1)
            current_df["transformer_regime"] = re.search(r"regime_(.*?)__", file).group(1)
            current_df["transform_target"] = re.search(r"tr_target_(.*?)__", file).group(1)
            current_df["transform_features"] = re.search(r"tr_features_(.*?)_", file).group(1)
            df_list.append(current_df)

    df = pd.concat(df_list)
    df.to_csv(args.results_path / "all_results_val.csv", index=False)

    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
