from typing import Dict, Any, Optional

import numpy as np


class Estimator:
    """Base class for all models.

    Args:
        model_params: parameters for the model.
            Individually defined for each model.

    """

    def __init__(self, model_params: Dict[str, Any]):
        self.model_params = model_params

        self.model = None
        self.score = None

    def fit_one_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> "Estimator":
        """Fits the model on one fold using the input data.

        Args:
            X_train: features array.
            y_train: target array.
            X_val: validation features array.
            y_val: validation target array.

        Returns:
            the fitted model.

        """
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions using the trained model.

        Args:
            X: features array.

        Returns:
            array of predicted values.

        """
        return self.model.predict(X)
