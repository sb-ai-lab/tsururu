import logging
import sys

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel("DEBUG")

# if root logger has handlers, propagate messages up and let root logger process them
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler(sys.stdout))
    _logger.propagate = False


__all__ = ["dataset", "transformers", "models", "strategies", "model_training"]
