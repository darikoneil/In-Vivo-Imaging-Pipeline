from __future__ import annotations
from collections import OrderedDict
import importlib
from typing import Optional, Callable, Tuple
from json_tricks import dumps, loads
import numpy as np


class FunctionWrapper:
    """
    This is a class for wrapping functions into a single step pipeline
    """
    def __init__(self):
        self.config = OrderedDict()

    def wrap_function(self, Function: Callable, **kwargs) -> Self:
        """
        Stores functions into a dictionary where key is the function name and the value is a tuple:
        (Module/Package, Function, Parameters)

        :param Function: Function to add
        :type Function: Callable
        :param kwargs: Parameters to pass
        :rtype: Any
        """
        # noinspection PyArgumentList
        self.config[Function.__name__] = (Function.__module__, Function, dict(**kwargs))

    def __json_encode__(self):
        # noinspection PyTypeChecker
        for _key in self.config.keys():
            self.config[_key] = {
                "Module": self.config[_key][0],
                "Function": _key,
                "Parameters": self.config[_key][2],
            }
        return {"config": self.config}

    def __json_decode__(self, **attrs):
        self.config = attrs["config"]
        for _key in self.config.keys():
            self.config[_key] = (self.config[_key].get("Module"),
                                 importlib.import_module(self.config[_key].get("Module")).__dict__.get(_key),
                                 self.config[_key].get("Parameters"))
