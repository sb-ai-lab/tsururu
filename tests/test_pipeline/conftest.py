import pandas as pd
import pytest

from tsururu.dataset import TSDataset


@pytest.fixture(scope="package")
def get_dataset():
    df = pd.read_csv("./datasets/global/simulated_data_to_check.csv")

    # Add some features with different names
    for i, i_name in enumerate(
        [
            "Значение времени (ч)...проверка@123__",
            "кол-во_клиентов#сегмент_A__тест?",
            "прибыль_на_АКцию (%)-@финансы",
            "объем Продаж! (Q3_2023)...анализ#данных",
            "показатель/успешности_доход@услуги__OK?",
            "add_feature_0",
            "add_feature_1",
            "add_feature_2",
        ]
    ):
        df[f"{i_name}"] = df["value"] + 1000 * (i + 1)

    dataset_params = {
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
        },
        "exog_1": {
            "columns": [
                "Значение времени (ч)...проверка@123__",
                "прибыль_на_АКцию (%)-@финансы",
                "feature_0",
                "feature_2",
            ],
            "type": "continious",
        },
        "exog_2": {
            "columns": [
                "кол-во_клиентов#сегмент_A__тест?",
                "объем Продаж! (Q3_2023)...анализ#данных",
                "feature_1",
            ],
            "type": "continious",
        },
    }

    return TSDataset(
        data=df,
        columns_params=dataset_params,
        print_freq_period_info=False,
    )
