# FISSA: Signal Extraction & Source-Separation
# Load this module to organize loading our trace data
from AnalysisModules.FissaAnalysis import FissaModule

# CASCADE: Spike Inference
# Load this module to organize loading our inference data
from AnalysisModules.CascadeAnalysis import CascadeModule

# Behavioral Decoder
# Load this module to organize encoding-decoding analyses
# Interfaces Neural Decoding Package from Joshua Glaser via Kording Lab.
# Interfaces with State-Space Modeling (SSM / SSM-jax) Package from Scott Linderman Lab
# Interfaces with an annotated implementation of
# Manifold Inference from Neural Dynamics Package (MIND) from David Tank Lab
# Annotated/Implemented by Quentin RV Ferry via Tonegawa Lab
# Interfaces with PySindy Package from the Dynamics Lab at UW
# from AnalysisModules.DecodingAnalysis import DecodingModule # Not needed for this nb, as imported inline

# Options are Logistic Regression, Linear Regression, Linear Non-Linear Regression
# (NYI) Options are Principal Component Analysis, Tensor Component Analysis, Manifold Inference from Neural Dynamics,
# (NYI) Hidden-Markov Model, Recurrent Linear Switching, Non-linear Dynamical System

# Visualization Packages
import matplotlib
#%matplotlib inline
matplotlib.use('Qt5Agg')

# General Packages
import numpy as np

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

from AnalysisModules.DecodingAnalysis import LogisticRegression
Data.LogReg = LogisticRegression(NeuralData=Data.Cascade.ProcessedInferences.firing_rates, FeatureDataFile="C:\\ProgramData\\Anaconda3\\envs\\Calcium-Imaging-Analysis-Pipeline\\DEM2_RET_FeatureList.csv")
Data.LogReg.label_data = Data.LogReg.feature_data[5, :]
# Set FR == 0 when NaN
Data.LogReg.neural_data[np.isnan(Data.LogReg.neural_data)]=0
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

# Linear Regression
from AnalysisModules.DecodingAnalysis import LinearRegression
# Instance
Data.LinReg = LinearRegression(NeuralData=Data.Cascade.ProcessedInferences.firing_rates, FeatureDataFile="C:\\ProgramData\\Anaconda3\\envs\\Calcium-Imaging-Analysis-Pipeline\\DEM2_RET_FeatureList.csv")
# Pick Single Feature
Data.LinReg.label_data = Data.LinReg.feature_data[5, :]
# Set FR == 0 when NaN
Data.LinReg.neural_data[np.isnan(Data.LinReg.neural_data)]=0
# Split
Data.LinReg.splitData()
# Fit
Data.LinReg.fitModel(fit_intercept=True)
# Assess on Training
Data.LinReg.assessFit()
# Make Predictions on Withheld Data
Data.LinReg.makeAllPredictions()
# Extended Assessment
Data.LinReg.commonAssessment()
Data.LinReg.printAssessment()
