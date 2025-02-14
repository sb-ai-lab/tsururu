import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from tsururu.examples.utils.validation import get_train_val_test_datasets
from tqdm import tqdm
import re
from joblib import Memory

memory = Memory(".cache", verbose=0)

RESULTS_DIRECTORY = [Path("results")]
DATASETS_DIRECTORY = Path("datasets/global")

NORMALIZED = True

# Data
DATE_COLUMN = "date"
ID_COLUMN = "id"
TARGET_COLUMN = "value"

TRAIN_SIZE = 0.7
TEST_SIZE = 0.2
HISTORY = 96
HORIZON = 24

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

@memory.cache
def get_df_with_train_mean_and_std(dataset_path):
    train_dataset, _, _ = get_train_val_test_datasets(
        dataset_path, dataset_params, TRAIN_SIZE, TEST_SIZE, HISTORY
    )
    stat_df = (
        train_dataset.data
        .groupby(train_dataset.data[train_dataset.id_column])[train_dataset.target_column]
        .agg(["mean", "std"])
    )
    return stat_df


def read_old_metrics_get_times(metrics_path):
    """
    Считывает файла *metrics_test.csv или *metrics_val.csv fit_time, forecast_time.
    Возвращает кортеж (fit_time, forecast_time).
    Если файла нет или не найдено, вернёт (None, None).
    """
    if not metrics_path.is_file():
        return None, None

    df_old = pd.read_csv(metrics_path)
    fit_time = None
    forecast_time = None

    row_fit = df_old.loc[df_old["id"] == "fit_time"]
    if len(row_fit) > 0:
        fit_time = float(row_fit["mae"].values[0])

    row_forecast = df_old.loc[df_old["id"] == "forecast_time"]
    if len(row_forecast) > 0:
        forecast_time = float(row_forecast["mae"].values[0])

    return fit_time, forecast_time


def process_file(
    model_pred_path: Path,
    old_metrics_path: Path,
    out_csv: Path,
    normalized: bool,
):
    """
    - MAE, RMSE для текущей модели считаем на основе model_pred_path.
    - fit_time, forecast_time берём из old_metrics_path.
    """

    # fit_time и forecast_time
    fit_time, forecast_time = read_old_metrics_get_times(old_metrics_path)

    df_model = pd.read_csv(model_pred_path)
    df_model.rename(columns={"y_pred": "y_pred_model", "value": "y_true"}, inplace=True)

    if normalized:      
        dataset_name = re.search(r'\/dataset_([^\/]+?)__', str(model_pred_path)).group(1)
        stat_df = get_df_with_train_mean_and_std(DATASETS_DIRECTORY / f"{dataset_name}.csv")

    results = []
    for grp_id, grp_data in df_model.groupby("id"):
        y_true = grp_data["y_true"].values
        y_pred_model = grp_data["y_pred_model"].values

        if np.isnan(y_true).any() or np.isnan(y_pred_model).any():
            continue

        if normalized:
            try:
                y_true = (y_true - stat_df.loc[grp_id, "mean"]) / (stat_df.loc[grp_id, "std"] + 1e-6) 
                y_pred_model = (y_pred_model - stat_df.loc[grp_id, "mean"]) / (stat_df.loc[grp_id, "std"] + 1e-6) 
            except:
                continue

        try:
            mae_current = mean_absolute_error(y_true, y_pred_model)
            rmse_current = root_mean_squared_error(y_true, y_pred_model)
        except:
            continue

        results.append(
            {
                "id": grp_id,
                "mae": mae_current,
                "rmse": rmse_current,
            }
        )

    df_res = pd.DataFrame(results, columns=["id", "mae", "rmse"])
    df_res.sort_values(by="id", inplace=True)
    df_stats = pd.DataFrame(
        {
            "id": ["mean", "median", "std"],
            "mae": [df_res["mae"].mean(), df_res["mae"].median(), df_res["mae"].std(ddof=1)],
            "rmse": [df_res["rmse"].mean(), df_res["rmse"].median(), df_res["rmse"].std(ddof=1)],
        }
    )
    df_res = pd.concat([df_res, df_stats], ignore_index=True)

    df_extra = pd.DataFrame(
        {
            "id": ["fit_time", "forecast_time"],
            "mae": [fit_time, forecast_time],
            "rmse": [fit_time, forecast_time],
        }
    )
    df_res = pd.concat([df_res, df_extra], ignore_index=True)

    df_res.to_csv(out_csv, index=False)


def main(
    results_dirs: Path,
    normalized: bool = False,
):
    """
    Обходит каждую директорию из results_dirs,
    ищет там *pred_test.csv и *pred_val.csv,
    и для каждого такого файла формирует соответствующие *metrics_test_v2.csv / *metrics_val_v2.csv
    """
    for results_dir in tqdm(results_dirs):
        print(f"Обрабатываем директорию: {results_dir}")
        for root, dirs, files in os.walk(results_dir):
            for file in tqdm(files):
                if file.endswith("pred_test.csv"):
                    model_pred_path = Path(root) / file
                    old_metrics_path = model_pred_path.with_name(
                        file.replace("pred_test.csv", "metrics_test.csv")
                    )
                    if normalized:
                        out_csv = model_pred_path.with_name(
                            file.replace("pred_test.csv", "metrics_test_v2_normalized.csv")
                        )
                    else:
                        out_csv = model_pred_path.with_name(
                            file.replace("pred_test.csv", "metrics_test_v2.csv")
                        )

                    process_file(model_pred_path, old_metrics_path, out_csv, normalized)

                elif file.endswith("pred_val.csv"):
                    model_pred_path = Path(root) / file
                    old_metrics_path = model_pred_path.with_name(
                        file.replace("pred_val.csv", "metrics_val.csv")
                    )
                    if normalized:
                        out_csv = model_pred_path.with_name(
                            file.replace("pred_val.csv", "metrics_val_v2_normalized.csv")
                        )
                    else:
                        out_csv = model_pred_path.with_name(
                            file.replace("pred_val.csv", "metrics_val_v2.csv")
                        )

                    process_file(model_pred_path, old_metrics_path, out_csv, normalized)

    # get aggregated df with results
    if NORMALIZED:
        metrics_test_path_suffix = f"_metrics_test_v2_normalized.csv"
        metrics_val_path_suffix = f"_metrics_val_v2_normalized.csv"
    else:
        metrics_test_path_suffix = "_metrics_test_v2.csv"
        metrics_val_path_suffix = "_metrics_val_v2.csv"

    results_df_list = []

    for root_dir in results_dirs:
        for dirpath, dirnames, _ in os.walk(root_dir):
            for dirname in dirnames:
                base_path = Path(dirpath) / dirname
                filenames = list(base_path.glob("*/*/*.csv"))
                metric_files_test = [
                    str(f)
                    for f in filenames
                    if str(f).endswith(metrics_test_path_suffix)
                    and not str(f).startswith("all_results")
                    and "pred_test" not in str(f)
                ]
                metric_files_val = [
                    str(f)
                    for f in filenames
                    if str(f).endswith(metrics_val_path_suffix)
                    and not str(f).startswith("all_results")
                    and "pred_val" not in str(f)
                ]

                paired_files = {}
                for test_file in metric_files_test:
                    base_name = test_file.replace(metrics_test_path_suffix, "")
                    paired_files[base_name] = [test_file, None]

                for val_file in metric_files_val:
                    base_name = val_file.replace(metrics_val_path_suffix, "")
                    if base_name in paired_files:
                        paired_files[base_name][1] = val_file

                for base_name, files_pair in paired_files.items():
                    test_file, val_file = files_pair

                    if test_file is None or val_file is None:
                        print(f"fail {test_file} or {val_file}")
                        continue

                    path_parts = test_file.split(os.sep)

                    base_part = path_parts[-1]
                    dataset_part = path_parts[-2]
                    strategy_part = path_parts[-3]
                    model_part = path_parts[-4]

                    dataset_split = dataset_part.split("__")
                    dataset_name = dataset_split[0].replace("dataset_", "")
                    hist_name = dataset_split[1].replace("hist_", "")
                    hor_name = dataset_split[2].replace("hor_", "")
                    model_hor_name = dataset_split[3].replace("model_hor_", "")

                    strategy_split = strategy_part.split("__")
                    strategy_time_name = strategy_split[0].replace("strategy_time_", "")
                    mult_name = strategy_split[1].replace("mult_", "")
                    ci_name = strategy_split[2].replace("ci_", "")

                    model_name = model_part.replace("model_", "")

                    base_split = base_part.split("__")
                    datetime_val = base_split[0].replace("datetime_", "")
                    id_val = base_split[1].replace("id_", "")
                    transformer_val = base_split[2].replace("transformer_", "")
                    regime_val = base_split[3].replace("regime_", "")
                    tr_target_val = base_split[4].replace("tr_target_", "")
                    tr_features_val = base_split[5].replace("tr_features_", "")

                    df_test = pd.read_csv(test_file)
                    df_val = pd.read_csv(val_file)

                    df_test_mean = df_test[df_test["id"] == "mean"]
                    df_val_mean = df_val[df_val["id"] == "mean"]

                    if df_test_mean.empty or df_val_mean.empty:
                        continue

                    mae_test = df_test_mean["mae"].values[0]
                    rmse_test = df_test_mean["rmse"].values[0]
                    fit_time_test = df_test[df_test["id"] == "fit_time"]["mae"].values[0]
                    forecast_time_test = df_test[df_test["id"] == "forecast_time"]["mae"].values[0]

                    mae_val = df_val_mean["mae"].values[0]
                    rmse_val = df_val_mean["rmse"].values[0]
                    fit_time_val = df_val[df_val["id"] == "fit_time"]["mae"].values[0]
                    forecast_time_val = df_val[df_val["id"] == "forecast_time"]["mae"].values[0]

                    new_row = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "strategy_time": strategy_time_name,
                        "mult": mult_name,
                        "ci": ci_name,
                        "hist": hist_name,
                        "hor": hor_name,
                        "model_hor": model_hor_name,
                        "dateteime": datetime_val,
                        "id": id_val,
                        "transformer": transformer_val,
                        "regime": regime_val,
                        "tr_target": tr_target_val,
                        "tr_features": tr_features_val,
                        "mae_test": mae_test,
                        "rmse_test": rmse_test,
                        "fit_time_test": fit_time_test,
                        "forecast_time_test": forecast_time_test,
                        "mae_val": mae_val,
                        "rmse_val": rmse_val,
                        "fit_time_val": fit_time_val,
                        "forecast_time_val": forecast_time_val,
                    }
                    results_df_list.append(new_row)

    results_df = pd.DataFrame(results_df_list)
    results_df.to_csv(f"agg_results_{str(date.today())}__normalized_{NORMALIZED}.csv", index=False)


if __name__ == "__main__":
    main(RESULTS_DIRECTORY, NORMALIZED)
