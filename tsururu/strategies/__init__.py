"""Time series forecasting strategies."""

from tsururu.strategies.direct import DirectStrategy
from tsururu.strategies.flat_wide_mimo import FlatWideMIMOStrategy
from tsururu.strategies.mimo import MIMOStrategy
from tsururu.strategies.recursive import RecursiveStrategy


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
