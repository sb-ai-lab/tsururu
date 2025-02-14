"""Algorithms for time series forecasting."""

from .base import SequentialTransformer, Transformer, UnionTransformer
from .categorical import LabelEncodingTransformer, OneHotEncodingTransformer
from .datetime import CycleGenerator, DateSeasonsGenerator, TimeToNumGenerator
from .impute import MissingValuesImputer
from .numeric import (
    DifferenceNormalizer,
    LastKnownNormalizer,
    StandardScalerTransformer,
)
from .seq import LagTransformer, TargetGenerator


# Factory Object
class TransformersFactory:
    def __init__(self):
        self.transformers = {
            "Transformer": Transformer,  # "base" transformer
            "UnionTransformer": UnionTransformer,
            "SequentialTransformer": SequentialTransformer,
            "StandardScalerTransformer": StandardScalerTransformer,
            "LastKnownNormalizer": LastKnownNormalizer,
            "DifferenceNormalizer": DifferenceNormalizer,
            "LabelEncodingTransformer": LabelEncodingTransformer,
            "OneHotEncodingTransformer": OneHotEncodingTransformer,
            "TimeToNumGenerator": TimeToNumGenerator,
            "DateSeasonsGenerator": DateSeasonsGenerator,
            "CycleGenerator": CycleGenerator,
            "LagTransformer": LagTransformer,
            "TargetGenerator": TargetGenerator,
            "MissingValuesImputer": MissingValuesImputer,
        }

    def get_allowed(self):
        return sorted(list(self.transformers.keys()))

    def __getitem__(self, params):
        return self.transformers[params["transformer_name"]](**params["transformer_params"])

    def create_transformer(self, transformer_name, transformer_params):
        return self.transformers[transformer_name](**transformer_params)


__all__ = [
    "Transformer",
    "UnionTransformer",
    "SequentialTransformer",
    "StandardScalerTransformer",
    "LastKnownNormalizer",
    "DifferenceNormalizer",
    "LabelEncodingTransformer",
    "OneHotEncodingTransformer",
    "TimeToNumGenerator",
    "DateSeasonsGenerator",
    "CycleGenerator",
    "LagTransformer",
    "TargetGenerator",
    "TransformersFactory",
    "MissingValuesImputer",
]
