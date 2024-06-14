from typing import Dict, Union

try:
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
except:
    LinearRegression = None
    Lasso = None
    Ridge = None
    mean_squared_error = None

from .base import MLEstimator


class LinRegression(MLEstimator):
    """LinRegression is a class that performs cross-validation
        using LinearRegression from scikit-learn.
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    Args:
        model_params: parameters for the CatBoostRegressor model,
            for example: {
                "fit_intercept": False,
            }.

    """

    def _initialize_model(self):
        return LinearRegression(**self.model_params)

    def __init__(self, model_params: Dict[str, Union[str, int]]):
        super().__init__(model_params)


class LassoRegression(LinRegression):
    """LassoRegression is a class that performs cross-validation
        using LassoRegression from scikit-learn.
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

    Args:
        model_params: parameters for the CatBoostRegressor model,
            for example: {
                "fit_intercept": False,
                "alpha": 0.1,
                "max_iter": 1000,
                "tol": 1e-4,
            }.

    """

    def __init__(self, model_params: Dict[str, Union[str, int]]):
        super().__init__(model_params)

    def _initialize_model(self):
        return Lasso(**self.model_params)


class RidgeRegression(LinRegression):
    """RidgeRegression is a class that performs cross-validation
        using RidgeRegression from scikit-learn.
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

    Args:
        model_params: parameters for the CatBoostRegressor model,
            for example: {
                "fit_intercept": False,
                "alpha": 0.1,
                "max_iter": 1000,
                "tol": 1e-4,
            }.

    """

    def __init__(self, model_params: Dict[str, Union[str, int]]):
        super().__init__(model_params)

    def _initialize_model(self):
        return Ridge(**self.model_params)
