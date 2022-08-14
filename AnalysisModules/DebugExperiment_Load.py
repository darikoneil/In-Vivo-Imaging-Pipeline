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

Data = ExperimentData(Directory="H:\\DEM_Excitatory_Study\\DEM2", Mouse="M4584", Study="DEM", StudyMouse="DEM2")
Data.Retrieval = Retrieval(Data.passMeta())

# Now Fissa
Data.Retrieval.Fissa = FissaModule(data_folder="H:\\DEM_Excitatory_Study\\DEM3\\Retrieval\\Imaging\\10Hz",
                                   index_file="H:\\DEM_Excitatory_Study\\DEM3\\Retrieval\\Imaging\\10Hz\\suite2p\\plane0\\DEM3_Retrieval_NeuronalIndex.csv")

# Initialize
Data.Retrieval.Fissa.initializeFissa()
# Load
Data.Retrieval.Fissa.loadFissaPrep()
Data.Retrieval.Fissa.loadFissaSep()
Data.Retrieval.Fissa.loadProcessedTraces()

# Instantiate Firing Rates & Associated Data
Data.Retrieval.Cascade = CascadeModule(Data.Retrieval.Fissa.ProcessedTraces.merged_dFoF_result, Data.Retrieval.Fissa.frame_rate, model_folder="C:\\ProgramData\\Anaconda3\\envs\\suite2p\\Pretrained_models")
Data.Retrieval.Cascade.loadSpikeProb(load_path=Data.Retrieval.Fissa.output_folder)
Data.Retrieval.Cascade.loadSpikeInference(load_path=Data.Retrieval.Fissa.output_folder)
Data.Retrieval.Cascade.loadProcessedInferences(load_path=Data.Retrieval.Fissa.output_folder)

# Now Begin
from AnalysisModules.BurrowFearConditioning import identifyTrialValence, reorganizeData

# Log Reg
Data.Retrieval.LogReg = LogisticRegression(NeuralData=Data.Retrieval.Cascade.ProcessedInferences.firing_rates,
                                           FeatureDataFile=
                                           "H:\\DEM_Excitatory_Study\\DEM3\\Retrieval\\Imaging\\10Hz\\FeatureIndex.csv")

Data.Retrieval.NeuralActivity_TrialOrg, Data.Retrieval.FeatureIndex, Data.Retrieval.Features_TrialOrg = reorganizeData(Data.Retrieval.LogReg.neural_data,
                                                                                                                       Data.Retrieval.LogReg.feature_data,
                                                                                                                       Data.Retrieval.Cascade.frame_rate,
                                                                                                                       ResponseLength=15)
_num_trials = 9
_shuf_idx = np.arange(_num_trials)
np.random.shuffle(_shuf_idx)
NewActivity = Data.Retrieval.NeuralActivity_TrialOrg[_shuf_idx, :, :].copy()
NewFeatures = Data.Retrieval.Features_TrialOrg[_shuf_idx, :, :].copy()

_num_frames = 11088
_shuf_idx_2 = np.arange(_num_frames)
np.random.shuffle(_shuf_idx_2)

NAM = np.concatenate(NewActivity, axis=1)
NFM = np.concatenate(NewFeatures, axis=1)

NAM2 = NAM[:, _shuf_idx_2]
NFM2 = NFM[:, _shuf_idx_2
       ]
Data.Retrieval.LogReg.neural_data=NAM2.copy()
Data.Retrieval.LogReg.feature_data=NFM2.copy()

Data.Retrieval.LogReg.label_data = Data.Retrieval.LogReg.feature_data[5, :].copy()
# Split
Data.Retrieval.LogReg.splitData()

# Fit
Data.Retrieval.LogReg.fitModel(penalty='l1', solver='liblinear', max_iter=100000)

# Assess on Training
Data.Retrieval.LogReg.assessFit()

# Make Predictions on Withheld Data
Data.Retrieval.LogReg.makeAllPredictions()

# Extended Assessment
Data.Retrieval.LogReg.commonAssessment()
Data.Retrieval.LogReg.printAssessment()
Data.Retrieval.LogReg.plotROCs()