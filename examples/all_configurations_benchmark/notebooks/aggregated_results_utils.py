from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from critdd import Diagram


def plot_random_time_series(df: pd.DataFrame, n: int = 5) -> None:
    """Plot n random time series from the DataFrame.

    Args:
        df: the DataFrame containing the time series data with columns 'id', 'date', and 'value'.
        n: the number of random time series to plot.

    """
    unique_ids = df["id"].unique()
    selected_ids = np.random.choice(unique_ids, size=n, replace=False)

    plt.figure(figsize=(12, 6))
    for uid in selected_ids:
        subset = df[df["id"] == uid]
        plt.plot(subset["date"], subset["value"], label=f"ID: {uid}")

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(f"Randomly Selected Time Series (n={n})")
    plt.legend()
    plt.show()


def get_cd_diagram_for_dataset(
    df: pd.DataFrame, dataset: Union[str, bool], save_latex: bool = False
) -> str:
    """Get the critical diagram for a specific dataset.

    Args:
        current_df: the DataFrame containing the results.
        dataset: the name of the dataset or False, if current_df has no dataset column
            (it constists of only one dataset).
        save_latex: whether to save the CD diagram in LaTeX format.

    Returns:
        A critical diagram for the specified dataset in TikZ format (str).

    """
    df = df.copy()
    if dataset:
        df = df[df["dataset"] == dataset].drop(columns=["dataset"])

    df = df.pivot(
        index=["model", "strategy_time", "datetime", "id", "mode"],
        columns="transformer_name",
        values="mae_test",
    )
    df = df.dropna()
    diagram = Diagram(
        df.to_numpy(),
        treatment_names=df.columns,
    )
    tikz_code = diagram.to_str(
        alpha=0.05,
        adjustment="holm",
        reverse_x=True,
        axis_options={"title": "critdd"},
    )

    if save_latex:
        diagram.to_file(
            f"transformer_CD_{dataset}.tex" if dataset else "transformer_CD.tex",
            alpha=0.05,
            adjustment="holm",
            reverse_x=True,
            axis_options={"title": "critdd"},
        )

    return tikz_code


def get_model_type(model_name: str) -> str:
    """Determine the type of model based on its name.

    Args:
        model_name: the name of the model.

    Returns:
        A string indicating the type of model:
            "NN" for neural networks or "Boosting" for boosting models.

    """
    if "NN" in model_name.upper():
        return "NN"
    else:
        return "Boosting"


def calc_avg_rank_median_mae_for_param(
    df_subset: pd.DataFrame, param_group: str, metric_col: str = "mae_test"
) -> pd.DataFrame:
    """Calculate average rank and median metric for a given parameter group.

    Args:
        df_subset: the DataFrame containing the results.
        param_group: the name of the parameter group to analyze.
        metric_col: the name of the metric column to use for calculations.

    Returns:
        A DataFrame containing the average rank and median metric for each parameter value.

    """
    pivot_index = [
        c
        for c in df_subset.columns
        if c
        not in [
            param_group,
            metric_col,
            "rmse_test",
            "fit_time_test",
            "forecast_time_test",
            "mae_val",
            "rmse_val",
            "fit_time_val",
            "forecast_time_val",
        ]
    ]
    pivoted = df_subset.pivot_table(
        index=pivot_index, columns=param_group, values=metric_col, aggfunc="mean"
    )

    pivoted = pivoted.dropna(axis=0, how="any")
    ranked = pivoted.rank(axis=1, ascending=True, method="min")
    avg_ranks = ranked.mean(axis=0)

    median_mae_by_param = df_subset.groupby(param_group)[metric_col].median()

    results = pd.DataFrame(
        {
            "param_value": avg_ranks.index,
            "avg_rank": avg_ranks.values,
            "median_mae": [median_mae_by_param.get(x, np.nan) for x in avg_ranks.index],
        }
    )

    return results


def get_hparams_comparison_table_for_dataset(
    df: pd.DataFrame, dataset: Union[str, bool] = False, save_latex: bool = False
) -> pd.DataFrame:
    """Get a table comparing hyperparameters for a specific dataset.

    Args:
        df: the DataFrame containing the results.
        dataset: the name of the dataset or False, if df has no dataset column
            (it constists of only one dataset).
        save_latex: whether to save the table in LaTeX format.

    Returns:
        A DataFrame comparing hyperparameters for the specified dataset.

    """
    df = df.copy()
    if dataset:
        df = df[df["dataset"] == dataset].drop(columns=["dataset"])

    df["model_type"] = df["model"].apply(get_model_type)
    df = df[~((df["model_type"] == "Boosting") & (df["mode"] == "multivariate CI"))]

    # param_groups = ["mode", "strategy_time", "datetime", "id"]
    param_groups = {
        "mode": "Mode",
        "strategy_time": "Prediction Strategy",
        "datetime": "Datetime Features",
        "id": "ID Features",
    }

    big_results = []

    for param_group in param_groups.keys():
        df_nn = df[df["model_type"] == "NN"]
        nn_res = calc_avg_rank_median_mae_for_param(df_nn, param_group, metric_col="mae_test")

        df_boost = df[df["model_type"] == "Boosting"]
        boost_res = calc_avg_rank_median_mae_for_param(
            df_boost, param_group, metric_col="mae_test"
        )

        df_all = df
        overall_res = calc_avg_rank_median_mae_for_param(
            df_all, param_group, metric_col="mae_test"
        )

        merged = nn_res.merge(boost_res, on="param_value", how="outer", suffixes=("_nn", "_boost"))
        merged = merged.merge(overall_res, on="param_value", how="outer")

        merged.rename(
            columns={"avg_rank": "avg_rank_all", "median_mae": "median_mae_all"}, inplace=True
        )

        rows = []
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    ("Hyperparam"): param_groups[param_group],
                    ("Value"): row.param_value,
                    ("NN", "Rank"): row.avg_rank_nn,
                    ("NN", "Median MAE"): row.median_mae_nn,
                    ("Boosting", "Rank"): row.avg_rank_boost,
                    ("Boosting", "Median MAE"): row.median_mae_boost,
                    ("Overall", "Rank"): row.avg_rank_all,
                    ("Overall", "Median MAE"): row.median_mae_all,
                }
            )

        big_results.extend(rows)

    final_df = pd.DataFrame(big_results)
    final_df.sort_values(by=["Hyperparam", "Value"], inplace=True)

    final_df.set_index(["Hyperparam", "Value"], inplace=True)
    final_df = final_df[
        [
            ("NN", "Rank"),
            ("NN", "Median MAE"),
            ("Boosting", "Rank"),
            ("Boosting", "Median MAE"),
            ("Overall", "Rank"),
            ("Overall", "Median MAE"),
        ]
    ]

    if save_latex:
        latex_df = final_df.copy()
        latex_df.columns = pd.MultiIndex.from_tuples(latex_df.columns, names=["Model", "Metric"])
        latex_df.to_latex(
            f"hparams_comparison_{dataset}.tex" if dataset else "hparams_comparison.tex",
            multirow=True,
            multicolumn=True,
            caption="Comparison of hyperparameters.",
            label="tab:hparams",
            float_format="%.4f",
        )

    return final_df


def get_top10_test_val_table_for_dataset(
    df: pd.DataFrame, dataset: Union[str, bool] = False, save_latex: bool = False
) -> pd.DataFrame:
    """Get a table with the top 10 models based on test and validation metrics.

    Args:
        df: the DataFrame containing the results.
        dataset: the name of the dataset or False, if df has no dataset column
            (it constists of only one dataset).
        save_latex: whether to save the table in LaTeX format.

    Returns:
        A DataFrame containing the top 10 models based on test and validation metrics.

    """
    df = df.copy()
    if dataset:
        df = df[df["dataset"] == dataset].drop(columns=["dataset"])

    grouped = df.groupby(["model", "strategy_time"], as_index=False).agg(
        {"mae_test": "min", "mae_val": "min"}
    )

    df_test = grouped.sort_values("mae_test", ascending=True).head(10).copy()

    df_test.rename(
        columns={
            "model": "Model (test)",
            "strategy_time": "Strategy (test)",
            "mae_test": "MAE (test)",
        },
        inplace=True,
    )

    df_test.index = range(1, len(df_test) + 1)
    df_test["rank_test"] = df_test.index

    df_val = grouped.sort_values("mae_val", ascending=True).head(10).copy()
    df_val.rename(
        columns={
            "model": "Model (val)",
            "strategy_time": "Strategy (val)",
            "mae_val": "MAE (val)",
        },
        inplace=True,
    )
    df_val.index = range(1, len(df_val) + 1)
    df_val["rank_val"] = df_val.index

    df_final = pd.concat([df_test, df_val], axis=1)

    cols = [
        "Model (test)",
        "Strategy (test)",
        "MAE (test)",
        "Model (val)",
        "Strategy (val)",
        "MAE (val)",
    ]
    df_final = df_final[cols]
    df_final.index.name = "rank"

    # Delete "_NN" suffix from model names
    df_final["Model (test)"] = df_final["Model (test)"].str.replace("_NN", "", regex=False)
    df_final["Model (val)"] = df_final["Model (val)"].str.replace("_NN", "", regex=False)

    if save_latex:
        df_final.to_latex(
            f"comparison of models_{dataset}.tex" if dataset else "comparison_of_models.tex",
            multirow=True,
            multicolumn=True,
            caption="Comparison of models.",
            label="tab:hparams",
            float_format="%.4f",
        )

    return df_final
