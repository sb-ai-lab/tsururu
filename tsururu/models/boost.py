from typing import Dict, Union

import numpy as np

try:
    from catboost import CatBoostRegressor, Pool
except:
    Pool = None
    CatBoostRegressor = None

from ..dataset import Pipeline
from .base import Estimator


class CatBoostRegressor_CV(Estimator):
    """CatBoostRegressor_CV is a class that performs cross-validation
        using CatBoostRegressor.

    Args:
        validation_params: execution params (type, cv, loss),
            for example: {
                "type": "KFold",
                "n_splits": 3,
                "loss_function": "MAE",
            }.
        model_params: parameters for the CatBoostRegressor model,
            for example: {
                "loss_function": "MultiRMSE",
                "early_stopping_rounds": 100,
            }.

    """

    def __init__(
        self,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(validation_params, model_params)

    def fit(self, data: dict, pipeline: Pipeline) -> "CatBoostRegressor_CV":
        """Fits the CatBoostRegressor models using the input data and
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

            train_dataset = Pool(data=X_train, label=y_train)
            eval_dataset = Pool(data=X_test, label=y_test)

            # Set default params if params are None
            for param, default_value in [
                ("loss_function", "MultiRMSE"),
                ("thread_count", -1),
                ("random_state", 42),
                ("early_stopping_rounds", 100),
            ]:
                if self.model_params.get(param) is None:
                    self.model_params[param] = default_value

            model = CatBoostRegressor(**self.model_params)

            model.fit(
                train_dataset,
                eval_set=eval_dataset,
                use_best_model=True,
                plot=False,
            )

            self.models.append(model)

            score = model.best_score_["validation"][f"{self.model_params['loss_function']}"]
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
