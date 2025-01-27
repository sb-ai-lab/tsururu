![tsururu_logo](imgs/tsururu_logo.png)

# Tsururu (TSForesight) – a time series forecasting strategies framework

Tsururu is a Python-based library that provides a wide range of multi-series and multi-point-ahead prediction strategies, compatible with any underlying model, including neural networks. 

While much attention is currently focused on selecting models for time series forecasting, the crucial aspect of how to perform training and inference often goes overlooked. Tsururu aims to address this gap. 

Also tsururu provides various preprocessing techniques.

<a name="quicktour"></a>
## Quick tour

```python
from tsururu.dataset import Pipeline, TSDataset
from tsururu.model_training.trainer import MLTrainer
from tsururu.model_training.validator import KFoldCrossValidator
from tsururu.models.boost import CatBoost
from tsururu.strategies import RecursiveStrategy

dataset_params = {
    "target": {"columns": ["value"]},
    "date": {"columns": ["date"]},
    "id": {"columns": ["id"]},
}

dataset = TSDataset(
    data=pd.read_csv(df_path),
    columns_params=dataset_params,
)

pipeline = Pipeline.easy_setup(
    dataset_params, {"target_lags": 3, "date_lags": 1}, multivariate=False
)
trainer = MLTrainer(model=CatBoost, validator=KFoldCrossValidator)
strategy = RecursiveStrategy(horizon=3, history=7, trainer=trainer, pipeline=pipeline)

fit_time, _ = strategy.fit(dataset)
forecast_time, current_pred = strategy.predict(dataset)
```

<a name="installation"></a>
## Installation
To install Tsururu on your machine from PyPI:
```bash
# Base functionality:
pip install -U tsururu

# For partial installation use corresponding option
# Extra dependencies: [catboost, torch] or use 'all' to install all dependencies
pip install -U tsururu[catboost]
```

<a name="examples"></a>
## Other tutorials and examples

* [Tutorial_1_Quick_Start](https://github.com/sb-ai-lab/tsururu/blob/main/examples/Tutorial_1_Quick_start.ipynb) for simple usage examples
* [Tutorial_2_Strategies](https://github.com/sb-ai-lab/tsururu/blob/main/examples/Tutorial_2_Strategies.ipynb) covers forecasting strategies.
* [Tutorial_3_Transformers_and_Pipeline](https://github.com/sb-ai-lab/tsururu/blob/main/examples/Tutorial_3_Transformers_and_Pipeline.ipynb) provides a description of available data preprocessing techniques.
* [Tutorial_4_Neural_Networks](https://github.com/sb-ai-lab/tsururu/blob/main/examples/Tutorial_4_Neural_Networks.ipynb) demonstrates working with neural networks.
* [Example_1_All_configurations](https://github.com/sb-ai-lab/tsururu/blob/main/examples/Example_1_All_configurations.py) script for benchmarking multiple configurations from available strategies, models and preprocessing methods on a dataset.

<a name="description"></a>
## Multi-series prediction strategies:
- _Local-modelling_:
  - An individual model for each time series. 
  - Each time series is modeled independently of the others.
- _Global-modelling_:
  - A single model for all time series.
  - Features created from each series do not overlap with other series. Series are related but modeled separately.
- _Multivariate-modelling_:
  - A single model for all time series. 
  - Features created from each series are concatenated at each time step. Try to capture dependencies between the series at the same time point.

## Multi-point-ahead prediction strategies:
- _Recursive:_ 
	- One model is used for the entire forecast horizon. 
	- training: The model is trained to predict one point ahead.
	- prediction: The model iteratively predicts each point, using previous predictions to update the features in the test data.
	- Note 1: There is an option to use a “reduced” version, where features are generated for all test observations at once, and unavailable values are filled with NaN.
	- Note 2: Recursive can also be combined with the MIMO strategy, allowing the model to predict model_horizon points ahead at each step.
- _Direct:_ 
	- Individual models are trained for each point in the forecast horizon.
	- Note 1: There is an option to use "equal_train_size" option, where all models can be trained on the same X_train set, formed for the last model predicting h point. Only the target variable (y) is updated for each model, reducing the time spent generating new training sets.
	- Note 2: Direct can also be combined with MIMO, where each individual model predicts model_horizon points ahead.
- _MultiOutput (MIMO - Multi-input-multi-output):_
 	- One model is trained and used for the entire forecast horizon at once. 
	- Note 1: This strategy can also accommodate exogenous features (for local- or global-modelling strategies).
- _FlatWideMIMO:_.
	- A hybrid of Direct and MIMO. One model is trained, but Direct’s features are deployed across the forecast horizon.
	- Note 1: To use FlatWideMIMO with date-related features, h lags of them must be included (with help of LagTransformer).


## Preprocessing
- _StandardScalerTransformer_: scales features to have zero mean and unit variance.
- _DifferenceNormalizer_: transforms features by subtracting or dividing by their previous value.
- _TimeToNumGenerator_ and _DateSeasonsGenerator_: generates seasonal features (e.g., month, quarter, day of the week) from date information.
- _LabelEncodingTransformer_ and _OneHotEncodingTransformer_: encodes categorical features.
- _MissingValuesImputer_: handles missing values by imputing them with a chosen strategy.
- _LagTransformer_: generates lagged features. 
- _LastKnownNormalizer_: normalizes lagged features by the last known value in history, either by subtracting it or dividing by it.

<a name="license"></a>
# License
This project is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/sb-ai-lab/tsururu/blob/master/LICENSE) file for more details.

[Back to top](#toc)