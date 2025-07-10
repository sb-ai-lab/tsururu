import os
from pathlib import Path

import pandas as pd

from tsururu.dataset import TSDataset


def path_to_tsururu_format(root_path: str, data_path: str) -> Path:
    """Converts a dataset from CSV format to Tsururu format.

    Args:
        root_path: the root directory where the dataset is located.
        data_path: the path to the dataset file relative to the root directory.

    Returns:
        The path to the transformed dataset in Tsururu format.

    """
    dataset_path = Path(root_path) / data_path

    dataset_tsururu_path = Path(root_path) / "tsururu_format"
    output_file = dataset_tsururu_path / data_path

    # If the transformed file already exists, return the path
    if output_file.exists():
        return output_file

    df = pd.read_csv(dataset_path)
    df_melted = pd.melt(df, id_vars=["date"]).rename(columns={"variable": "id"})

    os.makedirs(dataset_tsururu_path, exist_ok=True)
    df_melted.to_csv(output_file, index=False)

    return output_file


def expand_val_with_train(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    id_column: str,
    date_column: str,
    history: int,
) -> pd.DataFrame:
    """Expands the validation dataset with the last `history` number of rows from the training dataset.

    Args:
        train_data: the training dataset.
        val_data: the validation dataset.
        id_column: the name of the column containing unique identifiers.
        date_column: the name of the column containing date information.
        history: the number of rows to include from the end of the training dataset.

    Returns:
        A DataFrame containing the expanded validation dataset.

    """
    L_split_data = train_data[date_column].values[(len(train_data) - history)]
    L_last_train_data = train_data[train_data[date_column] >= L_split_data]
    val_data_expanded = pd.concat((L_last_train_data, val_data))
    val_data_expanded = val_data_expanded.sort_values([id_column, date_column]).reset_index(
        drop=True
    )

    return val_data_expanded


def expand_test_with_val_and_train(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    id_column: str,
    date_column: str,
    history: int,
) -> pd.DataFrame:
    """Expands the test dataset with the last `history` number of rows from the validation and training datasets.

    Note:
        If the validation dataset is smaller than `history`,
            it will use the last rows from the training dataset to fill the gap.

    Args:
        train_data: the training dataset.
        val_data: the validation dataset.
        test_data: the test dataset.
        id_column: the name of the column containing unique identifiers.
        date_column: the name of the column containing date information.
        history: the number of rows to include from the end of the validation dataset.

    Returns:
        A DataFrame containing the expanded test dataset.

    """
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


def get_train_val_test_datasets(
    dataset_path: Path,
    columns_params: dict,
    train_size: float,
    test_size: float,
    history: int,
) -> tuple[TSDataset, TSDataset, TSDataset]:
    """Splits the dataset into train, validation, and test sets.

    Args:
        dataset_path: the path to the dataset file.
        columns_params: a dictionary containing parameters for the dataset columns.
        train_size: the proportion of the dataset to include in the training set.
        test_size: the proportion of the dataset to include in the test set.
        history: the number of rows to include from the end of the training dataset for validation.

    Returns:
        A tuple containing the train, validation, and test datasets as TSDataset objects.

    """
    data = pd.read_csv(dataset_path)

    date_column = columns_params["date"]["columns"][0]
    id_column = columns_params["id"]["columns"][0]

    if dataset_path.parts[-1] in ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]:
        train_val_split_data = "2017-06-26 00:00:00"
        val_test_slit_data = "2017-10-23 23:00:00"
        test_split_data = "2018-02-21 00:00:00"
    else:
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
    df_path: str,
    date_column: str = "date",
    id_column: str = "id",
    train_size: float = 0.7,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Calculates the mean and standard deviation for each unique id in the training dataset.

    Args:
        df_path: the path to the dataset file.
        date_column: the name of the column containing date information.
        id_column: the name of the column containing unique identifiers.
        train_size: the proportion of the dataset to include in the training set.
        test_size: the proportion of the dataset to include in the test set.

    Returns:
        A DataFrame containing the mean and standard deviation for each unique id.

    """
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
