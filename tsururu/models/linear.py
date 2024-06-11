from typing import Dict, Union

import numpy as np

try:
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
except:
    LinearRegression = None
    Lasso = None
    Ridge = None

from ..dataset import Pipeline
from .base import MLEstimator


class LinearRegression_CV(MLEstimator):
    """LinearRegression_CV is a class that performs cross-validation
        using LinearRegression from scikit-learn.
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    Args:
        validation_params: execution params (type, cv, loss),
            for example: {
                "type": "KFold",
                "n_splits": 3,
                "loss_function": "MAE",
            }.
        model_params: parameters for the CatBoostRegressor model,
            for example: {
                "fit_intercept": False,
            }.

    """

    def _initialize_model(self):
        return LinearRegression(**self.model_params)

    def __init__(
        self,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(validation_params, model_params)

    def fit(self, data: dict, pipeline: Pipeline) -> "LinearRegression_CV":
        """Fits the LinearRegression models using the input data and
            pipeline.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            pipeline: fitted data preprocessing pipeline.

        Returns:
            the fitted model.

        """
        X, y = pipeline.generate(data)

        # Initialize columns' order and reorder columns
        self.features_argsort = np.argsort(pipeline.output_features)
        X = X[:, self.features_argsort]

        # Initialize cv object
        cv = self.initialize_validator()

        # Fit models
        for i, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = self._initialize_model()
            model.fit(X_train, y_train)

            self.models.append(model)
            score = model.score(X_test, y_test)
            self.scores.append(score)

            print(f"Fold {i}:")
            print(f"{self.model_params['loss_function']}: {score}")

        print(f"Mean {self.model_params['loss_function']}: {np.mean(self.scores).round(4)}")
        print(f"Std: {np.std(self.scores).round(4)}")

    def predict(self, data: dict, pipeline: Pipeline) -> np.ndarray:
        """Generates predictions using the trained model.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            pipeline: fitted data preprocessing pipeline.

        Returns:
            array of predicted values.

        """
        X, _ = pipeline.generate(data)

        # Reorder columns
        X = X[:, self.features_argsort]

        models_preds = [model.predict(X) for model in self.models]
        y_pred = np.mean(models_preds, axis=0)

        return y_pred


class LassoRegression_CV(LinearRegression_CV):
    """LassoRegression_CV is a class that performs cross-validation
        using LassoRegression from scikit-learn.
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

    Args:
        validation_params: execution params (type, cv, loss),
            for example: {
                "type": "KFold",
                "n_splits": 3,
                "loss_function": "MAE",
            }.
        model_params: parameters for the CatBoostRegressor model,
            for example: {
                "fit_intercept": False,
                "alpha": 0.1,
                "max_iter": 1000,
                "tol": 1e-4,
            }.

    """

    def __init__(
        self,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(validation_params, model_params)

    def _initialize_model(self):
        return Lasso(**self.model_params)


class RidgeRegression_CV(LinearRegression_CV):
    """RidgeRegression_CV is a class that performs cross-validation
        using RidgeRegression from scikit-learn.
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

    Args:
        validation_params: execution params (type, cv, loss),
            for example: {
                "type": "KFold",
                "n_splits": 3,
                "loss_function": "MAE",
            }.
        model_params: parameters for the CatBoostRegressor model,
            for example: {
                "fit_intercept": False,
                "alpha": 0.1,
                "max_iter": 1000,
                "tol": 1e-4,
            }.

    """

    def __init__(
        self,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(validation_params, model_params)

    def _initialize_model(self):
        return Ridge(**self.model_params)
