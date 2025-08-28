"""Utils for optional imports (e.g. catboost, torch)."""

import importlib
from abc import ABC


class OptionalImport:
    """Class to handle lazy loading of optional imports.

    Args:
        module_name: name of the module to import.
            Can be dotted attr/class (e.g. 'torch.utils.data.Dataset') or
            a single module name (e.g. 'torch')

    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module = None  # cache for the imported module or class

    def _get_parts_of_module(self):
        """Split the module name into parts:
        - root module name
        - module path (if dotted)
        - attr/class name (if dotted)
        """
        if "." in self.module_name:
            module_path, attr = self.module_name.rsplit(".", 1)
        else:
            module_path, attr = self.module_name, None
        root = module_path.split(".")[0]
        return root, module_path, attr

    def __getattr__(self, item):
        """Explicitly resolve and return the attribute from the imported module."""
        if self._module is None:
            root, module_path, attr = self._get_parts_of_module()
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                raise ImportError(f"To use this, please install the '{root}' package.") from e
            self._module = getattr(module, attr) if attr is not None else module
        return getattr(self._module, item)

    def __call__(self, *args, **kwargs):
        """Explicitly resolve and return the target."""
        if self._module is None:
            root, module_path, attr = self._get_parts_of_module()
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                raise ImportError(f"To use this, please install the '{root}' package.") from e
            self._module = getattr(module, attr) if attr is not None else module
        return self._module(*args, **kwargs)

    def __mro_entries__(self, bases):
        """Let the class be used as a base class."""
        root, module_path, attr = self._get_parts_of_module()
        if attr is None:
            return (ABC,)  # if it's a module, return ABC
        try:
            module = importlib.import_module(module_path)
            target = getattr(module, attr)
            return (target,) if isinstance(target, type) else (ABC,)
        except Exception:
            return (ABC,)
