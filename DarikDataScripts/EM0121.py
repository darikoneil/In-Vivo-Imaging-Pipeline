import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

from ExperimentManagement.ExperimentHierarchy import ExperimentData
import sklearn.svm

EM0121 = ExperimentData.loadHierarchy("D:\\EM0121")

ProcessedInferences = EM0121.PreExposure.folder_dictionary.get("10Hz").import_proc_inferences()
ProcessedTraces = EM0121.PreExposure.folder_dictionary.get("10Hz").import_proc_traces()
Firing_Rates = ProcessedInferences.firing_rates
DataFrame = EM0121.PreExposure.data_frame.copy(deep=True)
cs_ids = EM0121.PreExposure.trial_parameters.get("stimulusTypes")
NFR = Processing.normalizeSmoothFiringRates(Firing_Rates, 5) # 5 is roughly half a second
NeuralData_Matrix = Processing.trial_matrix_org(DataFrame, NFR)
NeuralData_Tensor = np.array(np.hsplit(NeuralData_Matrix, 10))
FeatureData_Tensor, FeatureData_Labels = Processing.generate_features(345, 10, EM0121.PreExposure.trial_parameters)
FeatureData_Matrix = np.hstack(FeatureData_Tensor)


Features = FeatureData_Matrix[4, :].copy()
Samples = NeuralData_Matrix.copy()
SVM = sklearn.svm.SVC()
SVM.fit(Samples.T, Features.T)
