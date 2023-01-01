from json_tricks import dumps, loads
from os import getcwd

parameters = {
    "shutter artifact length": 1000,
    "chunk size": 7000,
    "grouped-z project bin size": 3,
    "median filter tensor size": (7, 3, 3)
}

config_filename = "".join([getcwd(), "\\config.json"])

serialized_parameters = dumps(parameters)

with open(config_filename, "w") as file:
    file.write(serialized_parameters)
    print("Configuration File Generated.")
file.close()

