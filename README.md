![tsururu_logo](tsururu_logo.png)

# Tsururu – a time series forecasting strategies framework

Much attention is now paid to what models to use for time series forecasting, but not to how exactly to perform training and inference. 

Tsururu is a Python-based library which aims at overcoming the aforementioned problems and provides a large number of multi-series and multi-point-ahead prediction strategies that can be used with any underlying model, including neural networks. 

Also tsururu provides various preprocessing techniques.

## Multi-series prediction strategies:
- _Local-modelling_:
  - An individual model for each time series. 
  - Each time series as independent from others.
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

## Installation

To install tsururu via pip you can use:

`pip install -U tsururu`

## Quick tour

For usage example please see:

* [Tutorial_1_Quick_Start](https://github.com/sb-ai-lab/tsururu/blob/main/Tutorial_1_Quick_start.ipynb) for simple usage examples

More examples are coming soon.
