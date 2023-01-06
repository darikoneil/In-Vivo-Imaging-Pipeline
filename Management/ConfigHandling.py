from __future__ import annotations
from typing import Optional
from modified_json_tricks import dumps, loads
from os import getcwd


def config_generator(Config: Optional[dict] = None) -> None:
    """
    Generates a configuration file

    :param Config: Configuration to write to file (Optional)
    :type Config: dict
    :rtype: None
    """

    if Config is None:
        Config = {
            "preprocessing": {
                "shutter artifact length": 1000,
                "grouped-z project bin size": 3,
                "median filter tensor size": (7, 3, 3),
                "test": {0, 1, 2}
            },
            "postprocesssing": {
                "color": "blue",
                "name": "test"
            }
        }

    _config_filename = "".join([getcwd(), "\\config.json"])

    _serialized_parameters = dumps(Config, indent=0, maintain_tuples=True)

    with open(_config_filename, "w") as _file:
        _file.write(_serialized_parameters)
        print("Configuration File Generated.")
    _file.close()


def config_reader(File: Optional[str]) -> dict:
    """
    Reads config into a dictionary

    :param File: Configuration File (Optional)
    :type File: str
    :return: Configuration
    :rtype: dict
    """

    _config_filename = "".join([getcwd(), "\\config.json"])

    with open(_config_filename, "r") as _file:
        _config = loads(_config_filename)

    return _config
