from enum import Enum
import streamlit as st
import pandas as pd

from tsururu.dataset import TSDataset

    
class LagTransformerParams:
    def __init__(self, column_role):
        self.lags = st.number_input('Select number of lags', min_value=1, value=10, key=f'{column_role}_lags')


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
    def __init__(self, column_role):
        super().__init__(column_role)
        
        
class DifferenceNormalizerParams(NormalizerParams):
    def __init__(self, column_role):
        super().__init__(column_role)
        self.regime = st.selectbox('Select regime', ['delta', 'ratio'], index=None, key=f'{column_role}_regime')
        
        
class LastKnownNormalizerParams(NormalizerParams):
    def __init__(self, column_role):
        super().__init__(column_role)
        self.regime = st.selectbox('Select regime', ['delta', 'ratio'], index=None, key=f'{column_role}_regime')
        
        
class DateSeasonsGeneratorParams:
    def __init__(self, column_role):
        self.date_seasons = st.multiselect(
            'Select date seasons',
            ['year', 'month', 'day', 'weekday', 'hour', 'dayofyear', 'minute', 'second', 'microsecond', 'nanosecond'],
            default=['dayofyear', 'month', 'weekday'],
            key=f'{column_role}_date_seasons'
        )
        self.from_target_date = st.checkbox('Use target date?', key=f'{column_role}_from_target_date')


class BaseColumnParams:
    def __init__(self, column_role, available_columns):
        self.name = st.selectbox(f'Select the {column_role} column', available_columns, index=None)


class DateColumnParams(BaseColumnParams):
    column_role = 'date'
    
    def __init__(self, available_columns):
        super().__init__(self.column_role, available_columns)
        self.lags = LagTransformerParams(self.column_role)
        use_date_seasons_generator = st.checkbox(
            'Use date seasons generator?',
            key=f'{self.column_role}_use_date_seasons_generator',
        )
        if use_date_seasons_generator:
            self.date_seasons_generator = DateSeasonsGeneratorParams(self.column_role)


class TargetColumnParams(BaseColumnParams):
    column_role = 'target'
    
    def __init__(self, available_columns):
        super().__init__(self.column_role, available_columns)
        self.lags = LagTransformerParams(self.column_role)
        self.normalizer = NormalizerParams.input_normalizer_params(self.column_role)


class IdColumnParams(BaseColumnParams):
    column_role = 'id'
    
    def __init__(self, available_columns):
        super().__init__(self.column_role, available_columns)
        self.drop_column = st.checkbox('Drop column?', key=f'{self.column_role}_drop_column')


class Params:
    def __init__(self):
        self.df = None
        self.target_column = None
        self.date_column = None
        self.id_column = None
        self.available_columns = []
        
    def update_available_columns(self):
        self.available_columns = [
            col 
            for col in self.df.columns 
            if col not in [self.target_column, self.date_column, self.id_column]
        ]
    
params = Params()

st.title('tsururu: Time Series Prediction')

with st.sidebar.expander("Data", expanded=True):
    # Download data
    uploaded_csv = st.file_uploader('Upload your data', type='csv')
    if uploaded_csv:
        params.df = pd.read_csv(uploaded_csv)
        params.update_available_columns()
        
        with st.sidebar.expander("Target Column", expanded=True):
            params.target_column = TargetColumnParams(params.available_columns)

        with st.sidebar.expander("Date Column", expanded=True):
            params.date_column = DateColumnParams( params.available_columns)
            
        with st.sidebar.expander("Id Column", expanded=True):
            params.id_column = IdColumnParams(params.available_columns)
            

    



    

# Initialize TSDataset
#dataset = TSDataset(df)
