from ..dataset import Pipeline
from ..models import Estimator
from .mimo import MIMOStrategy


class FlatWideMIMOStrategy(MIMOStrategy):
    """A strategy that uses a single model for all points
        in the prediction horizon.

    Arguments:
        horizon: forecast horizon.
        history: number of previous for feature generating
            (i.e., features for observation y_t are counted from observations
            (y_{t-history}, ..., y_{t-1}).
        step:  in how many points to take the next observation while making
            samples' matrix.
        model: base model.
        pipeline: pipeline for feature and target generation.
        is_multivariate: whether the prediction mode is multivariate.

    Notes:
        1. Fit: mixture of DirectStrategy and MIMOStrategy, fit one
            model, but uses deployed over horizon DirectStrategy's features.
        2. Inference: similarly.

    """

    def __init__(
        self,
        horizon: int,
        history: int,
        step: int,
        model: Estimator,
        pipeline: Pipeline,
        is_multivariate: bool = False,
    ):
        super().__init__(horizon, history, step, model, pipeline, is_multivariate)
        self.strategy_name = "FlatWideMIMOStrategy"
