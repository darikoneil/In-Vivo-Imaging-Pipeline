# FISSA: Signal Extraction & Source-Separation
# Load this module to organize loading our trace data
from AnalysisModules.FissaAnalysis import FissaModule

# CASCADE: Spike Inference
from AnalysisModules.CascadeAnalysis import CascadeModule

# Visualization Packages
import matplotlib
#matplotlib inline
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Instantiate Traces & Associated Data
Data = FissaModule(data_folder="H:\\DEM_Excitatory_Study\\DEM2\\Retrieval\\Imaging\\10Hz", index_path="H:\\DEM_Excitatory_Study\\DEM2\\Retrieval\\Imaging\\10Hz\\NeuronalIndex.csv")
# Initialize
Data.initializeFissa()
# Load
Data.loadFissaPrep()
Data.loadFissaSep()
Data.loadProcessedTraces()

# Instantiate Firing Rates & Associated Data
Data.Cascade = CascadeModule(Data.ProcessedTraces.merged_dFoF_result, Data.frame_rate, model_folder="C:\\ProgramData\\Anaconda3\\envs\\suite2p\\Pretrained_models")
Data.Cascade.loadSpikeProb(load_path=Data.output_folder)
Data.Cascade.loadSpikeInference(load_path=Data.output_folder)
Data.Cascade.loadProcessedInferences(load_path=Data.output_folder)

# Logistic Regression
from AnalysisModules.DecodingAnalysis import LogisticRegression

# Instance
Data.LogReg = LogisticRegression(NeuralData=Data.Cascade.ProcessedInferences.firing_rates, FeatureDataFile="C:\\ProgramData\\Anaconda3\\envs\\Calcium-Imaging-Analysis-Pipeline\\DEM2_RET_FeatureList.csv")

# Let's prune those NaNs
from AnalysisModules.StaticProcessing import pruneNaN
Data.LogReg.label_data = Data.LogReg.feature_data[5, :]
Data.LogReg.neural_data, Data.LogReg.feature_data, Data.LogReg.label_data = pruneNaN(Data.LogReg.neural_data, FeatureData=Data.LogReg.feature_data, LabelData=Data.LogReg.label_data)

# Split
Data.LogReg.splitData()

# Fit
Data.LogReg.fitModel(penalty='l1', solver='liblinear', max_iter=100000)

# Assess on Training
Data.LogReg.assessFit()

# Make Predictions on Withheld Data
Data.LogReg.makeAllPredictions()

# Extended Assessment
Data.LogReg.commonAssessment()
Data.LogReg.printAssessment()

# Plot Data
Data.LogReg.plotROCs()

# Experiment Stuff Import
from AnalysisModules.BurrowFearConditioning import ExperimentData
from AnalysisModules.BurrowFearConditioning import identifyTrialValence, reorganizeData

# Experiment Instance
Experiment = ExperimentData()

# Add Retrieval
Experiment.add_stage("Retrieval")

# ID Trials
Experiment.Retrieval.plus_trials, Experiment.Retrieval.minus_trials = identifyTrialValence("H:\\DEM_Excitatory_Study\\DEM2\\Retrieval\\Behavior\\BehavioralExports\\DEM2_RET_CSINDEX.csv")
Experiment.Retrieval.num_trials = len(Experiment.Retrieval.plus_trials[0])+len(Experiment.Retrieval.minus_trials[0])

# Sort & Segregate Desired Measure of Activity by Trial
Experiment.Retrieval.NeuralActivityByTrial, Experiment.Retrieval.FeatureIndex = reorganizeData(Data.LogReg.neural_data, Data.LogReg.feature_data, Data.Cascade.frame_rate)
Experiment.Retrieval.PlusTrialActivity = Experiment.Retrieval.NeuralActivityByTrial[Experiment.Retrieval.plus_trials[0], :, :]
Experiment.Retrieval.MinusTrialActivity = Experiment.Retrieval.NeuralActivityByTrial[Experiment.Retrieval.minus_trials[0], :, :]