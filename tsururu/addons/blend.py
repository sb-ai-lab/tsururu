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
            strategy.fit(dataset)
            
            model_scores = [np.mean(model.scores) for model in strategy.models]
            
            avg_strategy_score = np.mean(model_scores)
            self.average_scores.append(avg_strategy_score)
            
            if avg_strategy_score < best_score:
                best_score = avg_strategy_score
                best_strategy = strategy
                _, best_predictions_df = strategy.predict(dataset)
                best_predictions = best_predictions_df["value"]
                best_strategy.models = [strategy.models[np.argmin(model_scores)]]
        
        return (np.array(best_predictions), best_strategy)

    
class ClassicBlender(BlenderBase):
    def __init__(self, strategies: Sequence[Strategy], meta_model, val_split_ratio):
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
        self.val_split_ratio = val_split_ratio

    def fit(self, dataset: TSDataset):
        train_data = []
        val_data = []

        # Split the dataset by time series' id
        for id, group in dataset.data.groupby('id'):
            split_index = int(len(group) * (self.val_split_ratio))
            train_data.append(group.iloc[:split_index])
            val_data.append(group.iloc[split_index:])
        
        train_data = pd.concat(train_data)
        #print(train_data)
        val_data = pd.concat(val_data)
        #print(val_data)

        train_dataset = TSDataset(train_data, dataset.columns_params, dataset.delta)
        print(train_dataset.data)
        val_dataset = TSDataset(val_data, dataset.columns_params, dataset.delta)
        print(val_dataset.data)

        meta_features = []
        meta_targets = None

        # Train each strategy
        for strategy in self.strategies:
            
            strategy.fit(train_dataset)
            _, val_preds = strategy.predict(train_dataset)
            print(val_preds)
            meta_features.append(val_preds.value)
            
            if meta_targets is None:
                matching_dates = val_dataset.data[val_dataset.date_column].isin(val_preds['date'])
                matching_ids = val_dataset.data[val_dataset.id_column].isin(val_preds['id'])
                true_values = val_dataset.data[matching_dates & matching_ids][val_dataset.target_column].values

                print(f"True values shape: {true_values.shape}")
                print(true_values)
            
                meta_targets = true_values

        # Prepare meta-features and targets for meta-model training
        meta_features = np.column_stack(meta_features)
        print(f"Meta features: {meta_features}")
        meta_targets = np.array(meta_targets)
        print(f"Meta targets: {meta_targets}")

        # Train the meta-model
        self.meta_model.fit(meta_features, meta_targets)
        print("Meta-model training complete.")

        return self

    def predict(self, dataset: TSDataset) -> pd.DataFrame:
        """
        Predict using the ClassicBlender on the given dataset.

        Args:
            dataset: The dataset to predict the target values.

        Returns:
            A pandas DataFrame containing the predicted target values.
        """

        pass
        