import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

from tsururu.dataset import TSDataset
from tsururu.strategies import StrategiesFactory

    
class LagTransformerParams:
    def __init__(self, column_role):
        self.lags = st.number_input('Select number of lags', min_value=1, value=10, key=f'{column_role}_lags')

    def to_config(self):
        return {
            'lags': self.lags
        }

class NormalizerParams:
    def __init__(self, column_role):
        self.transform_train = st.checkbox('Transform train?', key=f'{column_role}_transform_train')
        self.transform_target = st.checkbox('Transform target?', key=f'{column_role}_transform_target')
    
    @staticmethod
    def input_normalizer_params(column_role):
        normalizer_name = st.selectbox(
            f'Select the {column_role} normalizer',
            ['None', 'StandardScaler', 'LastKnown', 'Difference'],
            index=0,
        ) 
        if normalizer_name == 'StandardScaler':
            return StandardScalerParams(column_role)
        elif normalizer_name == 'LastKnown':
            return LastKnownNormalizerParams(column_role)
        elif normalizer_name == 'Difference':
            return DifferenceNormalizerParams(column_role)
        else:
            return None


class StandardScalerParams(NormalizerParams):
    name = 'StandardScalerTransformer'
    
    def __init__(self, column_role):
        super().__init__(column_role)
        
    def to_config(self):
        config = {
            'transform_train': self.transform_train,
            'transform_target': self.transform_target,
        }
        return config
        
        
class DifferenceNormalizerParams(NormalizerParams):
    name = 'DifferenceNormalizer'
    
    def __init__(self, column_role):
        super().__init__(column_role)
        self.regime = st.selectbox(
            'Select regime',
            ['delta', 'ratio'],
            index=None,
            key=f'{column_role}_regime'
        )

    def to_config(self):
        config = {
            'transform_train': self.transform_train,
            'transform_target': self.transform_target,
            'regime': self.regime,
        }
        return config
        
        
class LastKnownNormalizerParams(NormalizerParams):
    name = 'LastKnownNormalizer'
    
    def __init__(self, column_role):
        super().__init__(column_role)
        self.regime = st.selectbox('Select regime', ['delta', 'ratio'], index=None, key=f'{column_role}_regime')
        
    def to_config(self):
        config = {
            'transform_train': self.transform_train,
            'transform_target': self.transform_target,
            'regime': self.regime,
        }
        return config
        
class DateSeasonsGeneratorParams:
    seasonality_mapping = {
        'year': 'y',
        'month': 'm',
        'day': 'd',
        'weekday': 'wd',
        'dayofyear': 'doy',
        'hour': 'h',
        'minute': 'min',
        'second': 'sec',
        'microsecond': 'ms',
        'nanosecond': 'ns',
    }
    
    def __init__(self, column_role):
        self.date_seasons = st.multiselect(
            'Select date seasons',
            self.seasonality_mapping.keys(),
            default=['dayofyear', 'month', 'weekday'],
            key=f'{column_role}_date_seasons'
        )
        self.from_target_date = st.checkbox('Use target date?', key=f'{column_role}_from_target_date')

    def to_config(self):
        config = {
            'seasonalities': [self.seasonality_mapping[seasonality] for seasonality in self.date_seasons],
            'from_target_date': self.from_target_date,
        }
        return config

class BaseColumnParams:
    def __init__(self, column_role, default_column_name, available_columns):
        index = None
        if default_column_name in available_columns:
            index = available_columns.index(default_column_name)
        self.name = st.selectbox(
            f'Select the {column_role} column',
            available_columns,
            index=index,
        )
        

class DateColumnParams(BaseColumnParams):
    column_role = 'date'
    default_column_name = 'date'
    
    def __init__(self, available_columns):
        super().__init__(self.column_role, self.default_column_name, available_columns)
        self.lags = LagTransformerParams(self.column_role)
        self.use_date_seasons_generator = st.checkbox(
            'Use date seasons generator?',
            key=f'{self.column_role}_use_date_seasons_generator',
        )
        if self.use_date_seasons_generator:
            self.date_seasons_generator = DateSeasonsGeneratorParams(self.column_role)

    def to_config(self):
        config = {
            'column': [self.name],
            'type': 'datetime',
            'drop_raw_feature': True,
            'features': {
                'LagTransformer': self.lags.to_config()
            }
        }
        if self.use_date_seasons_generator:
            config['features'] = {
                'DateSeasonsGenerator': self.date_seasons_generator.to_config(),
                'LagTransformer': self.lags.to_config(),
            }
        return config

class TargetColumnParams(BaseColumnParams):
    column_role = 'target'
    default_column_name = 'value'
    
    def __init__(self, available_columns):
        super().__init__(self.column_role, self.default_column_name, available_columns)
        self.lags = LagTransformerParams(self.column_role)
        self.normalizer = NormalizerParams.input_normalizer_params(self.column_role)

    def to_config(self):
        config = {
            'column': [self.name],
            'type': 'continious',
            'drop_raw_feature': False,
            'features': {
                'LagTransformer': self.lags.to_config()
            }
        }
        if self.normalizer:
            if self.normalizer.name == 'LastKnownNormalizer':
                config['features'] = {
                    'LagTransformer': self.lags.to_config(),
                    'LastKnownNormalizer': self.normalizer.to_config(),
                }
            else:
                config['features'] = {
                    self.normalizer.name: self.normalizer.to_config(),
                    'LagTransformer': self.lags.to_config(),
                }
        return config

class IdColumnParams(BaseColumnParams):
    column_role = 'id'
    default_column_name = 'id'
    
    def __init__(self, available_columns):
        super().__init__(self.column_role, self.default_column_name, available_columns)
        self.drop_column = st.checkbox('Drop column?', key=f'{self.column_role}_drop_column')

    def to_config(self):
        return {
            'column': [self.name],
            'type': 'categorical',
            'drop_raw_feature': self.drop_column
        }

class CatBoostRegressorCVParams:
    def __init__(self):
        self.loss_function = st.selectbox(
            'Select the loss function',
            ['MultiRMSE'],
            index=0,
            key='loss_function'
        )
        self.early_stopping_rounds = 100
        self.verbose = 0
        
    def to_config(self):
        config = {
            'loss_function': self.loss_function,
            'early_stopping_rounds': self.early_stopping_rounds,
            'verbose': self.verbose,
        }
        return config

class ValidationParams:
    def __init__(self):
        self.type = st.selectbox(
            'Select the validation type',
            ['KFold'],
            index=0,
            key='validation_type'
        )
        self.n_splits = 3
        
    def to_config(self):
        config = {
            'type': self.type,
            'n_splits': self.n_splits,
        }
        return config

class BaseStrategyParams:
    def __init__(self, strategy_name):
        self.horizon = st.number_input(
            'Select the horizon',
            min_value=1,
            value=7,
            key='horizon',
            help='Number of points to predict'
        )
        self.model_name = st.selectbox(
            'Select the model',
            ['CatBoostRegressor_CV'],
            index=0,
            key='model_name'
        )
        self.model_params = {
            'CatBoostRegressor_CV': CatBoostRegressorCVParams,
        }[self.model_name]()
        self.validation_params = ValidationParams()
        if strategy_name in ('RecursiveStrategy', 'DirectStrategy'):
            self.k = st.number_input(
                'Select the k',
                min_value=1,
                value=1,
                key='k',
                help='Horizon for individual model'
            )
        
    def to_config(self):
        config = {
            'horizon': self.horizon,
            'model_name': self.model_name,
            'model_params': self.model_params.to_config(),
            'validation_params': self.validation_params.to_config(),
        }
        if self.k:
            config['k'] = self.k
        return config

class StrategyParams:
    def __init__(self):
        self.is_multivariate = st.checkbox('Is multivariate?', key='is_multivariate')
        self.strategy_name = st.selectbox(
            'Select the strategy',
            ['RecursiveStrategy', 'DirectStrategy', 'DirRecStrategy', 'MIMOStrategy', 'FlatWideMIMOStrategy'],
            index=0,
            key='strategy_name'
        )
        self.strategy_params = {
            'RecursiveStrategy': BaseStrategyParams,
            'DirectStrategy': BaseStrategyParams,
            'DirRecStrategy': BaseStrategyParams,
            'MIMOStrategy': BaseStrategyParams,
            'FlatWideMIMOStrategy': BaseStrategyParams,
        }[self.strategy_name](self.strategy_name)
        
    def to_config(self):
        config = {
            'is_multivariate': self.is_multivariate,
            'strategy_name': self.strategy_name,
            'strategy_params': self.strategy_params.to_config(),
        }
        return config
    

class Params:
    def __init__(self):
        self.df = None
        self.target_column = None
        self.date_column = None
        self.id_column = None
        self.strategy = None
        self.available_columns = []
        
    def update_available_columns(self):
        self.available_columns = [
            col 
            for col in self.df.columns 
            if col not in [
                self.target_column.name if self.target_column else None,
                self.date_column.name if self.date_column else None,
                self.id_column.name if self.id_column else None,
            ]
        ]
        
    def get_columns_and_features_params(params):
        target = params.target_column.to_config()
        date = params.date_column.to_config()
        id = params.id_column.to_config()
        return {"target": target, "date": date, "id": id}
    
    def get_strategy_params(params):
        return params.strategy.to_config()


def get_results(
    cv,
    regime,
    y_true=None,
    y_pred=None,
    ids=None,
) -> pd.DataFrame:
    def _get_fold_value(
        value, idx: int
    ):
        if value is None:
            return [None]
        if isinstance(value[idx], float):
            return value[idx]
        if isinstance(value[idx], np.ndarray):
            return value[idx].reshape(-1)
        raise TypeError(f"Unexpected value type. Value: {value}")

    df_res_dict = {}

    for idx_fold in range(cv):
        # Fill df_res_dict
        for name, value in [("y_true", y_true), ("y_pred", y_pred)]:
            df_res_dict[f"{name}_{idx_fold+1}"] = _get_fold_value(
                value, idx_fold
            )
        if regime != "local":
            df_res_dict[f"id_{idx_fold+1}"] = _get_fold_value(ids, idx_fold)

    # Save datasets to specified directory
    df_res = pd.DataFrame(df_res_dict)
    return df_res

def get_metrics(true_pred_df):
    metrics = {}
    for metric_name, metric_func in [
        ("RMSE", lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)),
        ("MAE", mean_absolute_error),
        ("MAPE, %", lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred) * 100),
    ]:
        metrics[metric_name] = metric_func(true_pred_df["y_true_1"], true_pred_df["y_pred_1"])
    return metrics

def get_metrics_by_ids(true_pred_df):
    metric_df = true_pred_df.groupby("id_1")[["y_true_1", "y_pred_1"]].apply(
        lambda x: pd.Series(metrics:=get_metrics(x), index=metrics.keys())
    )
    metric_df.index.rename("id", inplace=True)
    return metric_df

def draw_pred_true(test_df, true_pred_df, last_n_points=200):
    show_legend_flag = True
    
    titles = []
    for current_id in true_pred_df["id_1"].unique():
        titles.append(
            f"id={current_id}"
        )
        
    # Make subplots' grid
    fig = make_subplots(
        rows=true_pred_df["id_1"].nunique(), 
        subplot_titles=tuple(titles), 
        horizontal_spacing=0.025, 
        vertical_spacing=0.05
    )

    # Fill subplots
    for row_ax, current_id in enumerate(true_pred_df["id_1"].unique()):
        current_true_pred_df = true_pred_df[
            (true_pred_df["id_1"] == current_id)
        ].copy()
        current_test_df = test_df[
            (test_df["id"] == current_id)    
        ]
        test_dates = current_test_df["date"].iloc[-current_true_pred_df.shape[0] - 1:]
        current_test_df = current_test_df.iloc[:-current_true_pred_df.shape[0]]
        if (current_test_df.shape[0] > last_n_points):
            current_test_df = current_test_df.iloc[-last_n_points:]

        # Построим график
        fig.add_trace(go.Scatter(
            x=current_test_df["date"], y=current_test_df["value"], 
            mode='lines', name='train', line=go.scatter.Line(color="black"), 
            showlegend=show_legend_flag
            ), row=row_ax+1, col=1
        )
        fig.add_trace(go.Scatter(
            x=test_dates, y=current_true_pred_df["y_true_1"], 
            mode='lines', name='y_true', line=go.scatter.Line(color="green"), showlegend=show_legend_flag
            ), row=row_ax+1, col=1
        )
        fig.add_trace(go.Scatter(
            x=test_dates, y=current_true_pred_df["y_pred_1"], 
            mode='lines', name='y_pred', line=go.scatter.Line(color="blue"), showlegend=show_legend_flag
            ), row=row_ax+1, col=1
        )
        show_legend_flag = False

    fig.update_layout(
        autosize=False,
        width=800,
        height=1600,
        title="True and predicted values",
    )

    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="values")
    
    fig.update_annotations(font_size=8)

    st.plotly_chart(fig)

def show_dataset_preview(df):
    st.header('Data preview:')
    st.write(df)
    
    fig = px.line(df, x="date", y="value", color='id')
    fig.update_layout(
        autosize=False,
        width=600,
        height=400,
        title="Data preview by id",
    )
    st.plotly_chart(fig)
    

def main():
    params = Params()

    st.title('tsururu: Time Series Prediction')

    with st.sidebar.expander("Data", expanded=True):
        uploaded_csv = st.file_uploader('Upload your data', type='csv')
        
    if uploaded_csv:
        params.df = pd.read_csv(uploaded_csv)
        params.update_available_columns()
        
        show_dataset_preview(params.df)
        
        with st.sidebar.expander("Target Column", expanded=True):
            params.target_column = TargetColumnParams(params.available_columns)

        with st.sidebar.expander("Date Column", expanded=True):
            params.date_column = DateColumnParams( params.available_columns)
            
        with st.sidebar.expander("Id Column", expanded=True):
            params.id_column = IdColumnParams(params.available_columns)
            
        with st.sidebar.expander("Modelling", expanded=True):
            params.strategy = StrategyParams()         
    
    if not params.df is None and st.button('**Run model**'):
        # Configure the model parameters
        strategy_params = params.get_strategy_params()

        strategies_factory = StrategiesFactory()
        
        dataset = TSDataset(
            data=params.df,
            columns_and_features_params=params.get_columns_and_features_params(),
            history=30,
        )

        strategy = strategies_factory[strategy_params]

        ids, test, pred, fit_time, forecast_time, num_iterations = strategy.back_test(dataset, cv=1)
        
        st.session_state['result'] = get_results(cv=1, regime="global", y_true=test, y_pred=pred, ids=ids)
        
    if not params.df is None and 'result' in st.session_state:
        st.header('Metrics:')
        metrics_by_id_df = get_metrics_by_ids(st.session_state['result'])
        metrics = get_metrics(st.session_state['result'])
        metric_cols = st.columns(len(metrics))
        for metric_idx, (metric_name, metric_value) in enumerate(metrics.items()):
            metric_value_str = f'{metric_value:.2f}'
            if metric_name == 'MAPE':
                metric_value_str += ' %'
            metric_cols[metric_idx].metric(label=metric_name, value=metric_value_str)
        st.write(metrics_by_id_df)
        
        st.header('Prediction visualization:') 
        last_n_points = st.slider('Select last n points', min_value=1, max_value=1000, value=200)
        draw_pred_true(params.df, st.session_state['result'], last_n_points)
    

if __name__ == '__main__':
    main()