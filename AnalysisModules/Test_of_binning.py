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
from AnalysisModules.DecodingAnalysis import DecodingModule
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
FeaturesMatrix = DecodingModule.loadFeatures(__features_file)
plus_trials, minus_trials = FearConditioning.identifyTrialValence(__csIndexFile)
SpikeMatrix, FeaturesIndex, FeaturesMatrix = FearConditioning.reorganizeData(SpikeMatrix, FeaturesMatrix, C.frame_rate)
Shuffled_Plus_Spikes, Shuffled_Plus_Features = DecodingModule.shuffleTrials(SpikeMatrix[plus_trials, :, :],
                                                                              FeatureData=FeaturesMatrix[plus_trials, :, :],
                                                                              TrialSubset=[5, 6, 7, 8, 9])
Shuffled_Minus_Spikes, Shuffled_Minus_Features = DecodingModule.shuffleTrials(SpikeMatrix[minus_trials, :, :],
                                                                                FeatureData=FeaturesMatrix[minus_trials, :, :],
                                                                                TrialSubset=[5, 6, 7, 8, 9])

PlusFeatureVector = DecodingModule.collapseFeatures(Shuffled_Plus_Features, FeatureSubset=[0, 1, 2, 3, 7])
MinusFeatureVector = DecodingModule.collapseFeatures(Shuffled_Minus_Features, FeatureSubset=[0, 1, 2, 3, 7])


# Extract the trial frames & concatenate for decoding
IncludedFrames = [FeaturesIndex['PRE'][0], FeaturesIndex['TRIAL'][1]]
Spikes = np.append(np.concatenate(Shuffled_Plus_Spikes[:, :, IncludedFrames[0]:IncludedFrames[1]],
                   axis=1), np.concatenate(Shuffled_Minus_Spikes[:, :,
                                           IncludedFrames[0]:IncludedFrames[1]], axis=1), axis=1)
Labels = np.append(np.concatenate(PlusFeatureVector[:, :, IncludedFrames[0]:IncludedFrames[1]], axis=1),
                     np.concatenate(MinusFeatureVector[:, :, IncludedFrames[0]:IncludedFrames[1]], axis=1), axis=1)


# Now Let's Do a Full Decode
LogReg = LogisticRegression(NeuralData=Spikes, LabelData=Labels)
LogReg.neural_data, LogReg.label_data = LogReg.shuffleFrames(LogReg.neural_data, LabelData=LogReg.label_data)

LogReg.splitData()
LogReg.fitModel(penalty='l2', solver='lbfgs', max_iter=100000, multi="multinomial")
LogReg.assessFit()
LogReg.makeAllPredictions()
LogReg.commonAssessment()





