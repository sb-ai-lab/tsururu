![tsururu_logo](tsururu_logo.png)

# Tsururu – time series forecasting strategies framework

Tsururu is a framework which provides instruments for the time series forecasting task.

## Ways to work with multiple time series:
- _Local-modelling_:
    - Individual model for each time series.
- _Global-modelling_:
    - One model for all time series;
    - features made up of individual series do not overlap. 
- _Multivariate-modelling_:
    - One model for all time series;
    - features made up of individual series corresponding to the same time point are concatenated for all time series.

## Prediction Strategies
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
- _FlatWideMIMO:_.
    - mixture of Direct and MIMO, fit one model, but uses deployed over horizon Direct's features.

A detailed description of the strategies can be found in Tutorial 2.
