from __future__ import annotations
from typing import Optional, Callable, Tuple
from modified_json_tricks import dumps, loads
from os import getcwd


def wrapper_config_generator(Functions: Tuple[Callable], Parameters: Tuple[dict], Name: Optional[str] = "config.json",
                             SavePath: Optional[str] = os.getcwd()) -> None:
    """
    Generates a configuration file for wrapper functions *experimental*. Pass a tuple of functions and a tuple of
    dictionaries containing any key-value pairs that are associated with each function.

    :param Functions: Callables to be wrapped
    :type Functions: tuple
    :param Parameters: Parameters associated with callables to be wrapped
    :type Parameters: tuple[dict]
    :param Name: Name of file
    :type Name: str
    :param SavePath: Path to save file
    :type SavePath: str
    :rtype: None
    """

    # parse user input
    try:
        if ".json" not in Name:
            Name = "".join([Name, ".json"])
        _config_filename = "".join([SavePath, Name])
    except TypeError:
        print("Name and SavePath must be str")
        return

    _serialized_parameters = dumps(Config, indent=0, maintain_tuples=True)

    with open(_config_filename, "w") as _file:
        _file.write(_serialized_parameters)
        print("Configuration File Generated.")
    _file.close()


def wrapper_config_reader(File: Optional[str]) -> dict:
    """
    Reads config into a dicitonary containing a tuple of callables and
    a tuple of dictionaries containing associated parameters

    :param File: Configuration File (Optional)
    :type File: str
    :return: Configuration
    :rtype: dict
    """

    _config_filename = "".join([getcwd(), "\\config.json"])

    with open(_config_filename, "r") as _file:
        _config = loads(_config_filename)

    # need to unwrap here

    return _config
