from ..dataset.pipeline import Pipeline
from ..models import Estimator
from .recursive import RecursiveStrategy


class MIMOStrategy(RecursiveStrategy):
    """A strategy that uses one model that learns to predict
        the entire prediction horizon.

    Arguments:
        horizon: forecast horizon.
        history: number of previous for feature generating
            (i.e., features for observation y_t are counted from observations
            (y_{t-history}, ..., y_{t-1}).
        step:  in how many points to take the next observation while making
            samples' matrix.
        model: base model.
        pipeline: pipeline for feature and target generation.

    Notes:
        1. Technically, `MIMOStrategy` is a `RecursiveStrategy` or
            `DirectStrategy` for which the horizon of the individual model
            (`model_horizon`) coincides with the full prediction horizon
            (`horizon`).
        2. Fit: the model is fitted to predict a vector which length is equal
            to the length of the prediction horizon.
        3. Inference: the model makes a vector of predictions.

    """

    def __init__(
        self,
        horizon: int,
        history: int,
        step: int,
        model: Estimator,
        pipeline: Pipeline,
    ):
        super().__init__(horizon, history, step, model, pipeline, model_horizon=horizon)
        self.strategy_name = "MIMOStrategy"
