"""Blender."""

import numpy as np
from typing import Sequence


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