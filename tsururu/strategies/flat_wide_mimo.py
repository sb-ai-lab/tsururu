from typing import Union

from ..dataset.pipeline import Pipeline
from ..model_training.trainer import DLTrainer, MLTrainer
from .mimo import MIMOStrategy


class FlatWideMIMOStrategy(MIMOStrategy):
    """A strategy that uses a single model for all points
        in the prediction horizon.

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
        1. Fit: mixture of DirectStrategy and MIMOStrategy, fit one
            model, but uses deployed over horizon DirectStrategy's features.
        2. Inference: similarly.

    """

    def __init__(
        self,
        horizon: int,
        history: int,
        trainer: Union[MLTrainer, DLTrainer],
        pipeline: Pipeline,
        step: int = 1,
    ):
        super().__init__(horizon, history, trainer, pipeline, step)
        self.strategy_name = "FlatWideMIMOStrategy"
