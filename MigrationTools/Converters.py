import io
import numpy as np
import pickle as pkl


class RenameUnpickler(pkl.Unpickler):
    """
    See https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
    """
    def find_class(self, module, name):
        renamed_module = module
        if module == "AnalysisModules.BurrowFearConditioning":
            renamed_module = "Behavior.BurrowFearConditioning"
        elif module == "AnalysisModules.ExperimentHierarchy":
            renamed_module = "ExperimentManagement.ExperimentHierarchy"
        elif module == "AnalysisModules.Suite2PAnalysis":
            renamed_module = "Imaging.Suite2PAnalysis"
        elif module == "AnalysisModules.CascadeAnalysis":
            renamed_module = "Imaging.CascadeAnalysis"
        elif module == "AnalysisModules.FissaAnalysis":
            renamed_module = "Imaging.FissaAnalysis"
        elif module == "ImagingAnalysis.FissaModule":
            renamed_module = "Imaging.ToolWrappers.FissaModule"
        elif module == "ImagingAnalysis.CascadeModule":
            renamed_module = "Imaging.ToolWrappers.CascadeModule"
        elif module == "Imaging Analysis.ModifiedDenoising":
            renamed_module = "Imaging.DenoisingModule"
        elif module == "Imaging.FissaModule.ProcessedTracesModule":
            renamed_module = "Imaging.FissaModule.ProcessedTracesDictionary"
        elif module == "Imaging.FissaModule.ProcessedTracesDictionary":
            renamed_module = "Imaging.ToolWrappers.FissaModule.ProcessedTracesDictionary"
        elif module == "BehavioralAnalysis.BurrowFearConditioning":
            renamed_module = "Behavior.BurrowFearConditioning"
        elif module == "ImagingAnalysis.Suite2PAnalysis":
            renamed_module = "Imaging.ToolWrappers.Suite2PModule"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)


def convertFromPy27_Array(Array: np.ndarray) -> np.ndarray:
    """
    Convert a numpy array of strings in byte-form to numpy array of strings in string-form

    :param Array: An array of byte strings (e.g., b'Setup')
    :type Array: Any
    :return: decoded_array
    :rtype: Any
    """
    decoded_array = list()
    for i in range(Array.shape[0]):
        decoded_array.append("".join([chr(_) for _ in Array[i]]))
    decoded_array = np.array(decoded_array)
    return decoded_array


def convertFromPy27_Dict(Dict: dict) -> dict:
    """
    Convert a dictionary pickled in Python 2.7 to a Python 3 dictionary

    :param Dict: Dictionary to be converted
    :type Dict: dict
    :return: Converted Dictionary
    :rtype: dict
    """
    _allkeys = list(Dict.keys())
    new_dict = dict()

    for _key in range(len(_allkeys)):
        new_dict[_allkeys[_key].decode('utf-8')] = Dict.get(_allkeys[_key])

    return new_dict
