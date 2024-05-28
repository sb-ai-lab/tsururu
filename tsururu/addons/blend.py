"""Blender."""

import numpy as np
from typing import Tuple, Sequence
from tsururu.dataset import TSDataset
from tsururu.strategies.base import Strategy


class BlenderBase:
    """Base class for blending models or strategies
    
    """
    def __init__(self):
        self.params = None

    def fit_predict(self, predictions: Sequence[np.ndarray]) -> np.ndarray:
        """Blend predictions
        
        """
        raise NotImplementedError


class MeanBlender(BlenderBase):
    """Simple average level predictions.
    
    Args:
        predictions: list of arrays obtained from different models or strategies. 
            for example:
                all_predictions = [preds1, preds2, preds3, ...]
    
    Returns:
        the mean of predictions
    
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

    def fit_predict(self, strategies: Sequence, dataset:TSDataset, cv=1) -> Tuple[np.ndarray, Strategy]:
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