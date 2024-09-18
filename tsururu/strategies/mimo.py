from typing import Union

from ..dataset.pipeline import Pipeline
from ..model_training.trainer import DLTrainer, MLTrainer
from .recursive import RecursiveStrategy


class MIMOStrategy(RecursiveStrategy):
    """A strategy that uses one model that learns to predict
        the entire prediction horizon.

    Arguments:
        horizon: forecast horizon.
        history: number of previous for feature generating
            (i.e., features for observation y_t are counted from observations
            (y_{t-history}, ..., y_{t-1}).
        trainer: trainer with model params and validation params.
        pipeline: pipeline for feature and target generation.
        step:  in how many points to take the next observation while making
            samples' matrix.

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
        trainer: Union[MLTrainer, DLTrainer],
        pipeline: Pipeline,
        step: int = 1,
    ):
        super().__init__(horizon, history, trainer, pipeline, step, model_horizon=horizon)
        self.strategy_name = "MIMOStrategy"
