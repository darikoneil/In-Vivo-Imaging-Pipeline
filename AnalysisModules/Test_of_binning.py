# IMPORTS
from AnalysisModules.ExperimentHierarchy import ExperimentData

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# Non-Generic Imports
from AnalysisModules.FissaAnalysis import FissaModule
from AnalysisModules.CascadeAnalysis import CascadeModule
from AnalysisModules.StaticProcessing import generateSpikeMatrix
from AnalysisModules.DecodingAnalysis import LogisticRegression
from AnalysisModules.BurrowFearConditioning import FearConditioning

# User Input
__data_folder = "H:\\DEM_Excitatory_Study\\DEM2\\Encoding\\Imaging\\10Hz"
__index_file = "H:\\DEM_Excitatory_Study\\DEM2\\Encoding\\Imaging\\10Hz\\NeuronalIndex.csv"
__features_file = "H:\\DEM_Excitatory_Study\\DEM2\\Encoding\\Imaging\\10Hz\\Features.csv"
__csIndexFile = "H:\\DEM_Excitatory_Study\\DEM2\\Encoding\\Behavior\\BehavioralExports\\DEM2_ENC_CS_INDEX.csv"

# Load
F = FissaModule(data_folder=__data_folder, index_file=__index_file)
F.pruneNonNeuronalROIs() # This step removes all non-neuronal data
F.initializeFissa()
F.loadFissaPrep()
F.loadFissaSep()
F.loadProcessedTraces()
C = CascadeModule(F.ProcessedTraces.detrended_merged_dFoF_result, F.frame_rate, model_folder="C:\\ProgramData\\Anaconda3\\envs\\suite2p\\Pretrained_models")
C.loadSpikeProb(load_path=F.output_folder)
C.loadSpikeInference(load_path=F.output_folder)
C.loadProcessedInferences(load_path=F.output_folder)

# Format Features & Spikes by CS-Valence
SpikeMatrix = generateSpikeMatrix(C.spike_time_estimates, C.spike_prob.shape[1])
FeaturesMatrix = FearConditioning.loadFeatures(__features_file)
plus_trials, minus_trials = FearConditioning.identifyTrialValence(__csIndexFile)
SpikeMatrix, FeaturesIndex, FeaturesMatrix = FearConditioning.reorganizeData(SpikeMatrix, FeaturesMatrix, C.frame_rate)
Shuffled_Plus_Spikes, Shuffled_Plus_Features = FearConditioning.shuffleTrials(SpikeMatrix[plus_trials, :, :],
                                                                              FeatureData=FeaturesMatrix[plus_trials, :, :],
                                                                              TrialSubset=[5, 6, 7, 8, 9])
Shuffled_Minus_Spikes, Shuffled_Minus_Features = FearConditioning.shuffleTrials(SpikeMatrix[minus_trials, :, :],
                                                                                FeatureData=FeaturesMatrix[minus_trials, :, :],
                                                                                TrialSubset=[5, 6, 7, 8, 9])

PlusFeatureVector = FearConditioning.collapseFeatures(Shuffled_Plus_Features, FeatureSubset=[0, 1, 2, 3, 7])
MinusFeatureVector = FearConditioning.collapseFeatures(Shuffled_Minus_Features, FeatureSubset=[0, 1, 2, 3, 7])


# Extract the trial frames & concatenate for decoding
TrialFrames = FeaturesIndex['TRIAL']
Spikes = np.append(np.concatenate(Shuffled_Plus_Spikes[:, :, TrialFrames[0]:TrialFrames[1]],
                   axis=1), np.concatenate(Shuffled_Minus_Spikes[:, :,
                                           TrialFrames[0]:TrialFrames[1]], axis=1), axis=1)
Labels = np.append(np.concatenate(PlusFeatureVector[:, :, TrialFrames[0]:TrialFrames[1]], axis=1),
                     np.concatenate(MinusFeatureVector[:, :, TrialFrames[0]:TrialFrames[1]], axis=1), axis=1)









