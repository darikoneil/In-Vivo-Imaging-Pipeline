
from AnalysisModules.PreprocessingImages import PreProcessing
import numpy as np
import cupy

VideoDirectory = 'D:\\AM001\\PreExposure\\Imaging\\Compiled'

complete_image = PreProcessing.loadAllTiffs(VideoDirectory)



PreProcessing.saveTiffStack(complete_image, "D:\\AM001\\PreExposure\\Imaging")

