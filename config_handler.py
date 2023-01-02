from json_tricks import dumps, loads
from os import getcwd



def config_generator() -> None:
    """
    Generates a configuration file

    :return:
    """

    _parameters = {
        "shutter artifact length": 1000,
        "chunk size": 7000,
        "grouped-z project bin size": 3,
        "median filter tensor size": (7, 3, 3)
    }

    _config_filename = "".join([getcwd(), "\\config.json"])

    _serialized_parameters = dumps(_parameters)

    with open(_config_filename, "w") as _file:
        _file.write(_serialized_parameters)
        print("Configuration File Generated.")
    _file.close()


def config_reader() -> dict:
    """
    Reads config into a dictionary

    :return:
    """

    _config_filename = "".join([getcwd(), "\\config.json"])

    with open(_config_filename, "r") as _file:
        _config = loads(_config_filename)

    return _config