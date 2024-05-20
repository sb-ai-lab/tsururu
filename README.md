![tsururu_logo](tsururu_logo.png)

# Tsururu – a time series forecasting strategies framework

Much attention is now paid to what models to use for time series forecasting, but not to how exactly to perform training and inference. 

Tsururu is a Python-based library which aims at overcoming the aforementioned problems and provides a large number of multi-series and multi-point-ahead prediction strategies that can be used with any underlying model, including neural networks. 

Also tsururu provides various preprocessing techniques.

## Multi-series prediction strategies:
- _Local-modelling_:
    - Individual model for each time series.
- _Global-modelling_:
    - One model for all time series;
    - features made up of individual series do not overlap. 
- _Multivariate-modelling_:
    - One model for all time series;
    - features made up of individual series corresponding to the same time point are concatenated for all time series.

## Multi-point-ahead prediction strategies:
- _Recursive:_ 
    - one model for all points of the forecast horizon;
    - *training*: the model is trained to predict one point ahead;
    - *prediction*: a prediction is iteratively made one point ahead, and then this prediction is used to further shape the features in the test data. 
- _Recursive-reduced:_
    - one model for all points in the prediction horizon;
    - *training*: the model is trained to predict one point ahead;
    - *prediction*: features are generated for all test observations at once, unavailable values are replaced by NaN.
- _Direct:_ 
    - individual models for each point in the prediction horizon. 
- _MultiOutput (MIMO - Multi-input-multi-output):_
    - one model that learns to predict the entire prediction horizon. 
    - __Also, this strategy supports the presence of exogenous features (only for local- or global-modelling).__
- _FlatWideMIMO:_.
    - mixture of Direct and MIMO, fit one model, but uses deployed over horizon Direct's features.

## Installation

To install tsururu via pip you can use:

`pip install -U tsururu`

## Quick tour

For usage example please see:

* [Tutorial_1_Quick_Start](https://github.com/sb-ai-lab/tsururu/blob/main/Tutorial_1_Quick_start.ipynb) for simple usage examples

More examples are coming soon.
