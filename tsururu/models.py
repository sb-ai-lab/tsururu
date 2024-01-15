from importlib.metadata import PackageNotFoundError
from typing import Dict, Union
from numpy.typing import NDArray

import numpy as np
import pandas as pd

from .base_model import Estimator

try:
    from catboost import Pool
    from catboost import CatBoostRegressor
except PackageNotFoundError:
    Pool = None
    CatBoostRegressor = None


class CatBoostRegressor_CV(Estimator):
    def __init__(
        self,
        get_num_iterations: bool,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(get_num_iterations, validation_params, model_params)

    def fit(self, X: pd.DataFrame, y: NDArray[np.floating]) -> None:
        # Initialize cv object
        cv = self.initialize_validator()

        # Fit models
        for i, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
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

        if self.get_num_iterations:
            self.num_iterations = sum(
                [
                    self.models[i_cv].get_best_iteration()
                    for i_cv in range(self.validation_params["n_splits"])
                ]
            )

        print(f"Mean {self.model_params['loss_function']}: {np.mean(self.scores).round(4)}")
        print(f"Std: {np.std(self.scores).round(4)}")

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]:
        models_preds = [model.predict(X) for model in self.models]
        y_pred = np.mean(models_preds, axis=0)
        return y_pred

