"""Time series forecasting strategies."""

from .direct import DirectStrategy
from .flat_wide_mimo import FlatWideMIMOStrategy
from .mimo import MIMOStrategy
from .recursive import RecursiveStrategy


# Factory Object
class StrategiesFactory:
    def __init__(self):
        self.models = {
            "RecursiveStrategy": RecursiveStrategy,
            "DirectStrategy": DirectStrategy,
            "MIMOStrategy": MIMOStrategy,
            "FlatWideMIMOStrategy": FlatWideMIMOStrategy,
        }

    def get_allowed(self):
        return sorted(list(self.models.keys()))

    def __getitem__(self, params):
        return self.models[params["strategy_name"]](**params["strategy_params"])

    def create_strategy(self, strategy_name, strategy_params):
        return self.models[strategy_name](**strategy_params)


__all__ = [
    "RecursiveStrategy",
    "DirectStrategy",
    "MIMOStrategy",
    "FlatWideMIMOStrategy",
    "StrategiesFactory",
]
