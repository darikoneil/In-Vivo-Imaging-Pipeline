
import numpy as np
from ImagingAnalysis.PreprocessingImages import PreProcessing
from ImagingAnalysis.Suite2PAnalysis import Suite2PModule


def processSampling(EH, Stage, SamplingRate):
    print("STARTING PROCESSING")
    EH.__dict__[Stage].addImageProcessingFolder("filtered_imaging_data")
    EH.__dict__[Stage].add_image_sampling_folder(SamplingRate)
    _TiffStack = PreProcessing.loadAllTiffs(EH.__dict__[Stage].folder_dictionary['compiled_imaging_data'].path)
    _TiffStack = PreProcessing.blockwiseFastFilterTiff(_TiffStack, Footprint=np.ones((7, 3, 3)))
    PreProcessing.saveTiffStack(_TiffStack, EH.__dict__[Stage].folder_dictionary['filtered_imaging_data'].path)
    EH.__dict__[Stage].folder_dictionary['filtered_imaging_data'].reIndex()
    EH.__dict__[Stage].addImagingAnalysis(SamplingRate)
    _TiffStack = PreProcessing.groupedZProject(_TiffStack, tuple([3, 1, 1]), np.mean)
    _sampling_string = str(SamplingRate) + "Hz"
    PreProcessing.saveTiffStack(_TiffStack,  EH.__dict__[Stage].folder_dictionary[_sampling_string].path)
    EH.__dict__[Stage].folder_dictionary[_sampling_string].reIndex()
    EH.__dict__[Stage].__dict__["imaging_" + str(SamplingRate) + "_Hz"]['suite2p'] = Suite2PModule(EH.__dict__[Stage].folder_dictionary[_sampling_string].path)
    EH.__dict__[Stage].__dict__["imaging_" + str(SamplingRate) + "_Hz"]['suite2p'].run()
    EH.__dict__[Stage].__dict__["imaging_" + str(SamplingRate) + "_Hz"]['suite2p'].load_files()
    EH.__dict__[Stage].__dict__["imaging_" + str(SamplingRate) + "_Hz"]['suite2p'].iscell, EH.__dict__[Stage]["imaging_" + str(SamplingRate) + "_Hz"]['suite2p'].stat = EH.__dict__[Stage]["imaging_" + str(SamplingRate) + "_Hz"]['suite2p'].remove_small_neurons(EH.__dict__[Stage]["imaging_" + str(SamplingRate) + "_Hz"]['suite2p'].iscell, EH.__dict__[Stage]["imaging_" + str(SamplingRate) + "_Hz"]['suite2p'].stat)
    EH.__dict__[Stage].__dict__["imaging_" + str(SamplingRate) + "_Hz"]['suite2p'].save_files()
    print("FINISHED")
