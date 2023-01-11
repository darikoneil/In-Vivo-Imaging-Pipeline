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

    def __str__(self):

        def make_single_step():
            nonlocal _key
            nonlocal _step
            # Add (Step, Function): Function(Input
            _return_string = "".join([str((_step, _key)), ": ", _key, "(Input"])
            # Add Parameters (Step, Function(Input, Parameter=Value
            for _sub_key in self.config.get(_key)[2]:
                _return_string = "".join([_return_string, ", ", _sub_key, "=",
                                          str(self.config.get(_key)[2].get(_sub_key))])
            # Enclose and add import information (Step, Function(Input, Parameter=Value) from Module
            _return_string = "".join([_return_string, ") from ", self.config.get(_key)[0]])

            return _return_string

        _config_step = enumerate(self.config.keys())
        _print_string = "\nWrapped Pipeline: "

        # noinspection PyTypeChecker
        for _step, _key in enumerate(self.config.keys()):
            _print_string = "".join([_print_string, "\n", make_single_step()])

        return _print_string

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

    def __json_encode__(self) -> Self:
        """
        Json encoder for wrapped functions

        :rtype: Any
        """
        # Here we save each functions as a nested dictionary containing keys module, function (name), parameters
        # noinspection PyTypeChecker
        for _key in self.config.keys():
            self.config[_key] = {
                "Module": self.config[_key][0],
                "Function": _key,
                "Parameters": self.config[_key][2],
            }
        return {"config": self.config}

    def __json_decode__(self, **attrs) -> Self:
        """
        Json decoder for wrapped functions

        :param attrs: attributes from json
        :rtype: Any
        """
        # Here we extract the appropriate functions dynamically imported the module and grabbing the function from
        # its namespace using __dict__.get() to access the functions using the name as a key
        self.config = attrs["config"]
        for _key in self.config.keys():
            self.config[_key] = (self.config[_key].get("Module"),
                                 importlib.import_module(self.config[_key].get("Module")).__dict__.get(_key),
                                 self.config[_key].get("Parameters"))
