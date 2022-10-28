import io
import pickle as pkl


class RenameUnpickler(pkl.Unpickler):
    """
    See https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
    """
    def find_class(self, module, name):
        renamed_module = module
        if module == "AnalysisModules.BurrowFearConditioning":
            renamed_module = "BehavioralAnalysis.BurrowFearConditioning"
        elif module == "AnalysisModules.ExperimentHierarchy":
            renamed_module = "ExperimentManagement.ExperimentHierarchy"
        elif module == "AnalysisModules.Suite2PAnalysis":
            renamed_module = "ImagingAnalysis.Suite2PAnalysis"
        elif module == "AnalysisModules.CascadeAnalysis":
            renamed_module = "ImagingAnalysis.CascadeAnalysis"
        elif module == "AnalysisModules.FissaAnalysis":
            renamed_module = "ImagingAnalysis.FissaAnalysis"
        elif module == "ComputationalAnalysis.DecodingAnalysis":
            renamed_module = "ComputationalAnalysis.LogisticRegression"


        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)
