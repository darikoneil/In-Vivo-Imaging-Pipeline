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

cs_plus_frames = MFPO.safe_extract(DF.copy(deep=True), ("CS", 0), reset=True,
                                   drop=False)
cs_plus_frames_nn = cs_plus_frames[~cs_plus_frames["Downsampled Frame"].isnull()]
cs_plus_frames_nn.reset_index(inplace=True, drop=True)
cs_plus_frames_nn.set_index("Time (s)", drop=False, inplace=True)
cs_plus_frames_nn.sort_index(inplace=True)

cs_minus_frames = MFPO.safe_extract(DF.copy(deep=True), ("CS", 1), reset=True,
                                   drop=False)
cs_minus_frames_nn = cs_minus_frames[~cs_minus_frames["Downsampled Frame"].isnull()]
cs_minus_frames_nn.reset_index(inplace=True, drop=True)
cs_minus_frames_nn.set_index("Time (s)", drop=False, inplace=True)
cs_minus_frames_nn.sort_index(inplace=True)



from ImagingAnalysis.StaticPlotting import plotFiringRateMatrix
import seaborn as sns
f1 = plt.figure(figsize=(16, 8))
a1 = f1.add_subplot(211)

sns.heatmap(NFR[:, cs_plus_frames_nn["Downsampled Frame"].values.astype(int)], ax=a1, cmap="Spectral_r")

a1.set_xticks((range(0, int(NFR[:, cs_plus_frames_nn["Downsampled Frame"].values.astype(int)].shape[1]), int(500))),
               labels=(range(0, int(NFR[:, cs_plus_frames_nn["Downsampled Frame"].values.astype(int)].shape[1] /10),
                             int(500 / 10))))
a2 = f1.add_subplot(212)

sns.heatmap(NFR[:, cs_minus_frames_nn["Downsampled Frame"].values.astype(int)], ax=a2, cmap="Spectral_r")

a2.set_xticks((range(0, int(NFR[:, cs_minus_frames_nn["Downsampled Frame"].values.astype(int)].shape[1]), int(500))),
               labels=(range(0, int(NFR[:, cs_minus_frames_nn["Downsampled Frame"].values.astype(int)].shape[1] /10),
                             int(500 / 10))))