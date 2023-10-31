# Time Series Forecasting Strategies Library.

A framework which provides instruments for the following time series forecasting tasks and strategies:

## Quick tour
[`Tutorial`](https://github.com/sb-ai-lab/tsururu/tutorial.ipynb)

## Ways to work with multiple time series:
- _Local-modelling_:
    - Individual model for each time series.
- _Global-modelling_:
    - One model for all time series;
    - features for individual observations do not overlap in the context of different time series. 
- _Multivariate-modelling_:
    - One model for all time series;
    - features for observations corresponding to the same time point are concatenated for all time series, and the output of the model for a single observation from the test sample is a vector of predicted values whose length is equal to the number of time series under consideration.

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
    - An individual model for each point in the prediction horizon. 
- _DirRec:_
    - An individual model for each point in the prediction horizon. 
    - *learning* and *prediction*: iteratively builds test data for an individual model at each step, makes a prediction one point ahead, and then uses this prediction to further generate features for subsequent models. 
- _MultiOutput (MIMO - Multi-input-multi-output):_
    - one model that learns to predict the entire prediction horizon (the model output for one observation from the test sample is a vector of predicted values whose length is equal to the length of the prediction horizon). 
- _FlatWideMIMO:_.
    - mixture of Direct and MIMO, fit one model, but uses deployed over horizon Direct's features.
