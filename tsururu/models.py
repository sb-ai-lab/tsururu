from importlib.metadata import PackageNotFoundError
from typing import Dict, List, Union
from numpy.typing import NDArray

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV

try:
    from catboost import Pool
    from catboost import CatBoostRegressor
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    from statsforecast.models import AutoETS
    from statsforecast.models import AutoTheta

except PackageNotFoundError:
    Pool = None
    CatBoostRegressor = None
    LinearRegression = None
    Lasso = None
    Ridge = None
    RandomForestRegressor = None
    AutoARIMA = None
    AutoETS = None
    AutoTheta = None


class Estimator:
    """ "A class of underlying model that is used to derive predictions
    using a feature format and targets defined by the strategy.

    Arguments:
        get_num_iterations: whether to get total number of trees.
        validation_params: execution params (type, cv, loss),
            for example: {
                "type": "KFold",
                "n_splits": 3,
                "loss_function": "MAE",
            }.
        model_params: base model's params,
            for example: {
                "loss_function": "MultiRMSE",
                "early_stopping_rounds": 100,
            }.
    """

    def __init__(
        self,
        get_num_iterations: bool,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        self.models = []
        self.scores = []
        self.validation_params = validation_params
        self.model_params = model_params
        self.get_num_iterations = get_num_iterations
        if self.get_num_iterations:
            self.num_iterations = []
        self.columns = None

    def initialize_validator(self):
        """Initialization of the sample generator for training the model
            according to the passed parameters.

        Returns:
            Generator object.
        """
        if self.validation_params["type"] == "KFold":
            # Set default params if params are None
            for param, default_value in [
                ("n_splits", 3),
                ("shuffle", True),
                ("random_state", 42),
            ]:
                if self.validation_params.get(param) is None:
                    self.validation_params[param] = default_value

            cv = KFold(**{k: v for k, v in self.validation_params.items() if k != "type"})

        elif self.validation_params["type"] == "TS_expanding_window":
            cv = TimeSeriesSplit(n_splits=self.validation_params["n_splits"])
        return cv

    def fit(self, X: pd.DataFrame, y: NDArray[np.floating]) -> None:
        """Initialization and training of the model according to the
            passed parameters.

        Arguments:
            X: source train data.
            y: source train argets.
        """
        raise NotImplementedError()

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]:
        """Obtaining model predictions.

        Arguments:
            X: source test data.

        Returns:
            array with models' preds.
        """
        raise NotImplementedError()


class CatBoostRegressor_CV(Estimator):
    def __init__(
        self,
        get_num_iterations: bool,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(get_num_iterations, validation_params, model_params)

    def fit(self, X: pd.DataFrame, y: NDArray[np.floating]) -> None:
        # Initialize columns' order and reorder columns
        self.columns = sorted(X.columns)
        X = X[self.columns]

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
        # Reorder columns
        X = X[self.columns]

        models_preds = [model.predict(X) for model in self.models]
        y_pred = np.mean(models_preds, axis=0)
        return y_pred


class LinearRegression_CV(Estimator):
    def __init__(
        self,
        get_num_iterations: bool,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(get_num_iterations, validation_params, model_params)

    def fit(self, X: pd.DataFrame, y: NDArray[np.floating]) -> None:
        cv = self.initialize_validator()
        param_grid = {**self.model_params}

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = LinearRegression()
            grid_search = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_absolute_error')
            grid_search.fit(X_train, y_train)

            self.models.append(grid_search)
            score = grid_search.best_score_
            self.scores.append(score)

            print(f"Fold {len(self.models)}: Best Score: {score}")

        print(f"Mean score: {np.mean(self.scores)}")
        print(f"Std: {np.std(self.scores)}")

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]: 
        predictions = [grid_search.best_estimator_.predict(X) for grid_search in self.models]
        y_pred = np.mean(predictions, axis=0)
        return y_pred


class Lasso_CV(Estimator):
    def __init__(
        self,
        get_num_iterations: bool,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(get_num_iterations, validation_params, model_params)

    def fit(self, X: pd.DataFrame, y: NDArray[np.floating]) -> None:
        cv = self.initialize_validator()
        param_grid = {**self.model_params}

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = Lasso()
            grid_search = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_absolute_error')
            grid_search.fit(X_train, y_train)

            self.models.append(grid_search)
            score = grid_search.best_score_
            self.scores.append(score)

            print(f"Fold {len(self.models)}: Best Score: {score}")

        print(f"Mean score: {np.mean(self.scores)}")
        print(f"Std: {np.std(self.scores)}")

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]:
        predictions = [grid_search.best_estimator_.predict(X) for grid_search in self.models]
        y_pred = np.mean(predictions, axis=0)
        return y_pred


class Ridge_CV(Estimator):
    def __init__(
        self,
        get_num_iterations: bool,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(get_num_iterations, validation_params, model_params)

    def fit(self, X: pd.DataFrame, y: NDArray[np.floating]) -> None:
        cv = self.initialize_validator()
        param_grid = {**self.model_params}

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = Ridge()
            grid_search = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_absolute_error')
            grid_search.fit(X_train, y_train)

            self.models.append(grid_search)
            score = grid_search.best_score_
            self.scores.append(score)

            print(f"Fold {len(self.models)}: Best Score: {score}")

        print(f"Mean score: {np.mean(self.scores)}")
        print(f"Std: {np.std(self.scores)}")

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]:
        predictions = [grid_search.best_estimator_.predict(X) for grid_search in self.models]
        y_pred = np.mean(predictions, axis=0)
        return y_pred


class RandomForest_CV(Estimator):
    def __init__(
        self,
        get_num_iterations: bool,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(get_num_iterations, validation_params, model_params)

    def fit(self, X: pd.DataFrame, y: NDArray[np.floating]) -> None: 
        cv = self.initialize_validator()

        for i, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for param, default_value in [
                ("n_estimators", 100),
                ("criterion", 'squared_error'),
                ("random_state", 42),
                ("max_features", 1.0),
                ("verbose", 2),
                ("n_jobs", -1),
            ]:
                if self.model_params.get(param) is None:
                    self.model_params[param] = default_value

            model = RandomForestRegressor(**self.model_params)

            model.fit(X_train, y_train)

            self.models.append(model)
            score = model.score(X_test, y_test)
            self.scores.append(score)

            print(f"Fold {i}: Score: {score}")

        print(f"Mean Score: {np.mean(self.scores).round(4)}")
        print(f"Std: {np.std(self.scores).round(4)}")

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]:     
        models_preds = [model.predict(X) for model in self.models]
        y_pred = np.mean(models_preds, axis=0)
        return y_pred


class ETS_Model(Estimator):
    def __init__(
        self,
        get_num_iterations: bool,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(get_num_iterations, validation_params, model_params)

    def fit(self, X: np.array) -> None:
        model = AutoETS(**self.model_params)
        unique_ids = X['id'].unique()

        for unique_id in unique_ids:
            series_data = X[X['id'] == unique_id]['value']
            fitted_model = model.fit(series_data.values)
            self.models.append((unique_id, fitted_model))
            #print(f'Model {unique_id+1} has been fitted!')

    def predict(self, horizon) -> List:
        frc = []
        for model_id, model in self.models:
            forecast = model.predict(h=horizon)
            frc.append(forecast)
        return frc
    

class AutoARIMA_Model(Estimator):
    def __init__(
        self,
        get_num_iterations: bool,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(get_num_iterations, validation_params, model_params)

    def fit(self, X: np.array) -> None:
        model = AutoARIMA(**self.model_params)
        unique_ids = X['id'].unique()

        for unique_id in unique_ids:
            series_data = X[X['id'] == unique_id]['value']
            fitted_model = model.fit(series_data.values)
            self.models.append((unique_id, fitted_model))
            #print(f'Model {unique_id+1} has been fitted!')

    def predict(self, horizon) -> List:
        frc = []
        for model_id, model in self.models:
            forecast = model.predict(h=horizon)
            frc.append(forecast)
        return frc
    

class AutoTheta_Model(Estimator):
    def __init__(
        self,
        get_num_iterations: bool,
        validation_params: Dict[str, Union[str, int]],
        model_params: Dict[str, Union[str, int]],
    ):
        super().__init__(get_num_iterations, validation_params, model_params)

    def fit(self, X: np.array) -> None:
        model = AutoTheta(**self.model_params)
        unique_ids = X['id'].unique()

        for unique_id in unique_ids:
            series_data = X[X['id'] == unique_id]['value']
            if len(series_data) >= 20000:
                fitted_model = model.fit(series_data[:20000].values)
                self.models.append((unique_id, fitted_model))
                print(f'Model {unique_id+1} has been fitted!')
            else:
                fitted_model = model.fit(series_data.values)
                self.models.append((unique_id, fitted_model))
                print(f'Model {unique_id+1} has been fitted!')

    def predict(self, horizon) -> List:
        frc = []
        for model_id, model in self.models:
            forecast = model.predict(h=horizon)
            frc.append(forecast)
        return frc
    

class MeanMethod:
    """
    Mean Method entails predicting that all future values will be the same,
    specifically equal to the historical data's average, often referred to as the "mean".
    This approach simplifies forecasts by assuming a constant value.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def predict(self):
        df_filled = self.dataframe.copy()
        df_filled['value'] = df_filled.groupby('id')['value'].transform(lambda x: x.fillna(x.mean()))

        return df_filled


class NaiveMethod:
    """
    For naive forecasts, we simply set all forecasts to be the value of the last observation
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def predict(self):
        df_filled = self.dataframe.copy()
        df_filled['value'] = df_filled.groupby('id')['value'].fillna(method='ffill')

        return df_filled


class SeasonalNaiveMethod:
    """
    This class implements the seasonal naive forecast method. It predicts future values
    based on the values from the corresponding season in the previous cycle.
    """

    def __init__(self, dataframe: pd.DataFrame, season_length: int):
        self.dataframe = dataframe
        self.season_length = season_length

    def predict(self):
        df_filled = self.dataframe.copy()
        if df_filled.shape[0] < self.season_length:
            raise ValueError("Dataframe does not have enough data for the given season length.")

        season_indices = df_filled.index % self.season_length

        for i in range(self.season_length):
            season_data = df_filled[season_indices == i]
            df_filled.loc[season_indices == i, 'value'] = season_data['value'].fillna(method='ffill')

        return df_filled


class DriftMethod:
    """
    Drift Forecasting Method fills missing values based on the drift method,
    where the drift is the average change seen in the historical data.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def predict(self):
        df_filled = self.dataframe.copy()

        for id in df_filled['id'].unique():
            series_data = df_filled[df_filled['id'] == id].copy()

            non_nan_indices = series_data[series_data['value'].notna()].index

            if len(non_nan_indices) < 2:
                raise ValueError(f"Not enough data to compute drift for series {id}.")

            y_t = series_data.loc[non_nan_indices[-1], 'value']
            y_1 = series_data.loc[non_nan_indices[0], 'value']
            T = non_nan_indices[-1] - non_nan_indices[0]

            drift = (y_t - y_1) / T

            last_known_index = non_nan_indices[-1]
            for i in range(last_known_index + 1, series_data.index[-1] + 1):
                h = i - last_known_index
                df_filled.at[i, 'value'] = y_t + h * drift

        return df_filled
    

# Factory Object
class ModelsFactory:
    def __init__(self):
        self.models = {
            "CatBoostRegressor_CV": CatBoostRegressor_CV,
            "LinearRegression_CV": LinearRegression_CV,
            "Lasso_CV": Lasso_CV,
            "Ridge_CV": Ridge_CV,
            "RandomForest_CV": RandomForest_CV,
            "ETS_Model": ETS_Model,
            'AutoARIMA_Model': AutoARIMA_Model,
            'AutoTheta_Model': AutoTheta_Model,
        }

    def get_allowed(self):
        return sorted(list(self.models.keys()))

    def __getitem__(self, params):
        return self.models[params["model_name"]](
            params["get_num_iterations"],
            params["validation_params"],
            params["model_params"],
        )
