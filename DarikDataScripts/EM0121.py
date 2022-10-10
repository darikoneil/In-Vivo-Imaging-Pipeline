from ExperimentManagement.ExperimentHierarchy import ExperimentData, CollectedImagingFolder
from BehavioralAnalysis.BurrowFearConditioning import MethodsForPandasOrganization as MFPO
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

EM0121 = ExperimentData.loadHierarchy("D:\\EM0121")
EM0121.Retrieval.loadBehavioralData()
PI = EM0121.Retrieval.folder_dictionary.get("10Hz").import_proc_inferences()

from ImagingAnalysis.StaticProcessing import Processing

NFR = Processing.normalizeSmoothFiringRates(PI.firing_rates, 3)

NFR = NFR[:, 0:np.unique(EM0121.Retrieval.data_frame["Downsampled Frame"].values).__len__()-1]

DF = EM0121.Retrieval.data_frame
MI = EM0121.Retrieval.multi_index

plus_trials = np.where(np.array(EM0121.Retrieval.trial_parameters.get("stimulusTypes")) == 0)[0]
MI2 = pd.MultiIndex.from_arrays([DF["Trial Set"].values.copy(), DF["CS"].values.copy()])
CSPlusFrames = MFPO.safe_extract(DF, None, (("Trial Set", "CS"), (0, 1), False), multi_index=MI2)
CSPlusFrames = CSPlusFrames[~CSPlusFrames["Downsampled Frame"].isnull()]

