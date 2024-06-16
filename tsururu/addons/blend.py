"""Blender."""

import numpy as np
import pandas as pd
from typing import Tuple, Sequence
from tsururu.dataset import TSDataset
from tsururu.strategies.base import Strategy


class BlenderBase:
    """Base class for blending models or strategies.
    
    """
    def __init__(self):
        self.params = None

    def fit_predict(self, predictions: Sequence[np.ndarray]) -> np.ndarray:
        """Blend predictions.
        
        """
        raise NotImplementedError


class MeanBlender(BlenderBase):
    """Simple average level predictions.
    
    Args:
        predictions: list of arrays obtained from different models or strategies. 
            for example:
                all_predictions = [preds1, preds2, preds3, ...]
    
    Returns:
        the mean of predictions.
    
    """
    def __init__(self):
        self.params = None

    def fit_predict(self, predictions: Sequence[np.ndarray]) -> np.ndarray:
        self.predictions = predictions

        mean_preds = np.mean(predictions, axis=0)

        return mean_preds
    
class BestModel(BlenderBase):
    def __init__(self):
        super().__init__()
        self.average_scores = []

    def fit_predict(self, strategies: Sequence, dataset:TSDataset) -> Tuple[np.ndarray, Strategy]:
        """Find the best strategy and model, and make predictions.
        
        """
        best_score = float('inf')
        best_strategy = None
        best_predictions = None

        for strategy in strategies:
            print('='*20)
            print(f'Fitting strategy: {strategy.strategy_name} with model: {strategy.model}')
            print('='*20)
            strategy.fit(dataset)

            model_scores = [np.mean(model.scores) for model in strategy.models]
            
            avg_strategy_score = np.mean(model_scores)
            self.average_scores.append(avg_strategy_score)
            
            if avg_strategy_score < best_score:
                best_score = avg_strategy_score
                best_strategy = strategy
                _, best_predictions_df = strategy.predict(dataset)
                best_predictions = best_predictions_df[dataset.target_column]
                best_strategy.models = [strategy.models[np.argmin(model_scores)]]
        
        return (np.array(best_predictions), best_strategy)

    
class ClassicBlender(BlenderBase):
    def __init__(self, strategies: Sequence[Strategy], meta_model, horizon):
        """
        Args:
            strategies: Sequence of strategies with models.
            meta_model: Meta-model to blend the predictions of individual strategies.
            val_split_ratio: Ratio to split training data into train and validation sets for the blender.
            
        Returns:
            self.
        """
        self.strategies = strategies
        self.meta_model = meta_model
        self.horizon=horizon

    def fit(self, dataset: TSDataset):
        train_data = []
        val_data = []

        # Split the dataset by time series' id
        for _, group in dataset.data.groupby(dataset.id_column):
            split_index = int(len(group)-self.horizon)
            train_data.append(group.iloc[:split_index])
            val_data.append(group.iloc[split_index:])
        
        train_data = pd.concat(train_data)
        val_data = pd.concat(val_data)

        train_dataset = TSDataset(train_data, dataset.columns_params, dataset.delta)
        val_dataset = TSDataset(val_data, dataset.columns_params, dataset.delta)

        meta_features = []
        meta_targets = None

        # Train each strategy
        for strategy in self.strategies:
            _, _ = strategy.fit(train_dataset)
            _, current_preds = strategy.predict(train_dataset)
            meta_features.append(current_preds.Demand)
            
            if meta_targets is None:
                matching_dates = val_dataset.data[val_dataset.date_column].isin(current_preds.Date)
                matching_ids = val_dataset.data[val_dataset.id_column].isin(current_preds.id)
                true_values = val_dataset.data[matching_dates & matching_ids][val_dataset.target_column].values
            
                meta_targets = true_values

        # Prepare meta-features and targets for meta-model training
        meta_features = np.column_stack(meta_features)
        print(meta_features.shape)
        meta_targets = np.array(meta_targets)
        print(meta_targets.shape)

        # Train the meta-model
        self.meta_model.fit(meta_features, meta_targets)
        print("Meta-model training complete.")
        
        self.trained_meta_model = self.meta_model

        return self

    def predict(self, dataset: TSDataset) -> pd.DataFrame:
        """
        Predict using the ClassicBlender on the given dataset.

        Args:
            dataset: The dataset to predict the target values.

        Returns:
            A pandas DataFrame containing the predicted target values.
        """

        if not hasattr(self, 'trained_meta_model'):
            raise ValueError("The meta-model is not trained. Please call the `fit` method before `predict`.")

        meta_features = []
        pred_dfs = []

        for strategy in self.strategies:
            _, preds = strategy.predict(dataset)
            print(f"Predictions shape for strategy {strategy}: {preds.shape}")
            meta_features.append(preds.Demand.values)
            pred_dfs.append(preds)

        meta_features = np.column_stack(meta_features)
        print(f"Meta features shape for prediction: {meta_features.shape}")
        blended_preds = self.trained_meta_model.predict(meta_features)
        blended_preds_df = pred_dfs[0][['id', 'Date']].copy()
        blended_preds_df[dataset.target_column] = blended_preds

        print("Blending predictions complete.")
        return blended_preds_df
        