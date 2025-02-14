import pandas as pd
import os
from pathlib import Path

from tsururu.dataset import TSDataset


def path_to_tsururu_format(root_path, data_path):
    dataset_path = Path(root_path) / data_path

    dataset_tsururu_path = Path(root_path) / "tsururu_format"
    output_file = dataset_tsururu_path / data_path

    # If the transformed file already exists, return the path
    if output_file.exists():
        return output_file

    df = pd.read_csv(dataset_path)
    df_melted = pd.melt(df, id_vars=['date']).rename(columns={"variable": "id"})

    os.makedirs(dataset_tsururu_path, exist_ok=True)
    df_melted.to_csv(output_file, index=False)

    return output_file


def expand_val_with_train(train_data, val_data, id_column, date_column, history):
    L_split_data = train_data[date_column].values[(len(train_data) - history)]
    L_last_train_data = train_data[train_data[date_column] >= L_split_data]
    val_data_expanded = pd.concat((L_last_train_data, val_data))
    val_data_expanded = val_data_expanded.sort_values([id_column, date_column]).reset_index(
        drop=True
    )
    return val_data_expanded


def expand_test_with_val_and_train(
    train_data, val_data, test_data, id_column, date_column, history
):
    unqiue_id_cnt = val_data[id_column].nunique()
    L_split_data = val_data[date_column].values[
        (
            (len(val_data) - history)
            if (len(val_data) // val_data[id_column].nunique() - history) > 0
            else 0
        )
    ]
    L_last_val_data = val_data[val_data[date_column] >= L_split_data]
    if len(val_data) // unqiue_id_cnt - history < 0:
        if (len(train_data) - (history - len(L_last_val_data) / unqiue_id_cnt)) > 0:
            L_split_data = train_data[date_column].values[
                (
                    len(train_data) // unqiue_id_cnt
                    - (history - len(L_last_val_data) // unqiue_id_cnt)
                )
            ]
        else:
            L_split_data = 0
        L_last_train_data = train_data[train_data[date_column] >= L_split_data]
        test_data_expanded = pd.concat((L_last_train_data, L_last_val_data, test_data))
    else:
        test_data_expanded = pd.concat((L_last_val_data, test_data))
    test_data_expanded = test_data_expanded.sort_values([id_column, date_column]).reset_index(
        drop=True
    )
    return test_data_expanded


def get_train_val_test_datasets(dataset_path, columns_params, train_size, test_size, history):
    data = pd.read_csv(dataset_path)

    date_column = columns_params["date"]["columns"][0]
    id_column = columns_params["id"]["columns"][0]

    if dataset_path.parts[-1] in ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]:
        train_val_split_data = "2017-06-26 00:00:00"
        val_test_slit_data = "2017-10-23 23:00:00"
        # + 4 * 30 * 24 часов от val_test_slit_data
        test_split_data = "2018-02-21 00:00:00"
    else:
        # TODO: Не подходит для M!!!
        train_val_split_data = data[date_column].values[
            int(data[date_column].nunique() * train_size)
        ]
        val_test_slit_data = data[date_column].values[
            int(data[date_column].nunique() * (1 - test_size))
        ]
        test_split_data = None

    train_data = data[data[date_column] < train_val_split_data]
    val_data = data[
        (data[date_column] >= train_val_split_data) & (data[date_column] <= val_test_slit_data)
    ]
    test_data = data[data[date_column] > val_test_slit_data]
    if test_split_data:
        test_data = test_data[test_data[date_column] < test_split_data]

    val_data = expand_val_with_train(train_data, val_data, id_column, date_column, history)
    test_data_expanded = expand_test_with_val_and_train(
        train_data, val_data, test_data, id_column, date_column, history
    )

    # train, val and test TSDataset initialization
    train_dataset = TSDataset(
        data=train_data,
        columns_params=columns_params,
    )

    val_dataset = TSDataset(
        data=val_data,
        columns_params=columns_params,
    )

    test_dataset = TSDataset(
        data=test_data_expanded,
        columns_params=columns_params,
    )

    return train_dataset, val_dataset, test_dataset


def get_fitted_scaler_on_train(
    df_path, date_column="date", id_column="id", train_size=0.7, test_size=0.2
):
    df_path = Path(df_path)
    data = pd.read_csv(df_path)

    if df_path.parts[-1] in ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]:
        train_val_split_data = "2017-06-25 23:00:00"
    else:
        train_val_split_data = data[date_column].values[
            int(data[date_column].nunique() * train_size)
        ]

    train_data = data[data[date_column] < train_val_split_data]
    train_data = train_data.drop(date_column, axis=1)

    stat_df = train_data.groupby(id_column).agg(["mean", "std"])
    stat_df.columns = ["mean", "std"]

    return stat_df