# General
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# Hierarchy
from AnalysisModules.ExperimentHierarchy import ExperimentData
from AnalysisModules.BurrowFearConditioning import Retrieval

# Fissa
from AnalysisModules.FissaAnalysis import FissaModule
from AnalysisModules.StaticProcessing import smoothTraces_TiffOrg, calculate_dFoF, mergeTraces, detrendTraces

# Cascade
from AnalysisModules.CascadeAnalysis import CascadeModule
from AnalysisModules.StaticProcessing import calculateFiringRate
from AnalysisModules.StaticUtilities import pullModels

# Logistic Regression
from AnalysisModules.StaticProcessing import pruneNaN
from AnalysisModules.DecodingAnalysis import LogisticRegression

# This is a first try/ debug script for experimental hierarchy and thereafter...

Data = ExperimentData(Directory="H:\\DEM_Excitatory_Study\\DEM3", Mouse="M4618", Study="DEM", StudyMouse="DEM3")
Data.Retrieval = Retrieval(Data.passMeta())

# Now Fissa
Data.Retrieval.Fissa = FissaModule(data_folder="H:\\DEM_Excitatory_Study\\DEM3\\Retrieval\\Imaging\\10Hz",
                                   index_file="H:\\DEM_Excitatory_Study\\DEM3\\Retrieval\\Imaging\\10Hz\\suite2p\\plane0\\DEM3_Retrieval_NeuronalIndex.csv")


Data.Retrieval.Fissa.pruneNonNeuronalROIs()
Data.Retrieval.Fissa.initializeFissa()
Data.Retrieval.Fissa.extractTraces()
Data.Retrieval.Fissa.saveFissaPrep()
Data.Retrieval.Fissa.ProcessedTraces.smoothed_raw = smoothTraces_TiffOrg(Data.Retrieval.Fissa.preparation.raw, niter=50, kappa=150, gamma=0.15)[0]
Data.Retrieval.Fissa.saveProcessedTraces()
Data.Retrieval.Fissa.preparation.raw = Data.Retrieval.Fissa.ProcessedTraces.smoothed_raw.copy()
Data.Retrieval.Fissa.passPrepToFissa()
Data.Retrieval.Fissa.separateTraces()
Data.Retrieval.Fissa.saveFissaSep()
Data.Retrieval.Fissa.ProcessedTraces.dFoF_result = calculate_dFoF(Data.Retrieval.Fissa.experiment.result, Data.Retrieval.Fissa.frame_rate, raw=Data.Retrieval.Fissa.preparation.raw, merge_after=False)
Data.Retrieval.Fissa.ProcessedTraces.merged_dFoF_result = mergeTraces(Data.Retrieval.Fissa.ProcessedTraces.dFoF_result)
Data.Retrieval.Fissa.ProcessedTraces.detrended_merged_dFoF_result = detrendTraces(Data.Retrieval.Fissa.ProcessedTraces.merged_dFoF_result, order=4, plot=False)
Data.Retrieval.Fissa.saveProcessedTraces()

# Now Cascade
Data.Retrieval.Cascade = CascadeModule(Data.Retrieval.Fissa.ProcessedTraces.detrended_merged_dFoF_result, Data.Retrieval.Fissa.frame_rate,
                                       model_folder="C:\\ProgramData\\Anaconda3\\envs\\suite2p\\Pretrained_models")
list_of_models = pullModels(Data.Retrieval.Cascade.model_folder)
Data.Retrieval.Cascade.model_name = list_of_models[22]
Data.Retrieval.Cascade.predictSpikeProb()
Data.Retrieval.Cascade.ProcessedInferences.firing_rates = calculateFiringRate(Data.Retrieval.Cascade.spike_prob, Data.Retrieval.Cascade.frame_rate)
Data.Retrieval.Cascade.saveSpikeProb(Data.Retrieval.Fissa.output_folder)
Data.Retrieval.Cascade.saveProcessedInferences(Data.Retrieval.Fissa.output_folder)
Data.Retrieval.Cascade.inferDiscreteSpikes()
Data.Retrieval.Cascade.saveSpikeInference(Data.Retrieval.Fissa.output_folder)
Data.Retrieval.Cascade.exportSpikeProb(Data.Retrieval.Fissa.output_folder)
Data.Retrieval.Cascade.exportSpikeInference(Data.Retrieval.Fissa.output_folder)

# Log Reg
Data.Retrieval.LogReg = LogisticRegression(NeuralData=Data.Retrieval.Cascade.ProcessedInferences.firing_rates)



