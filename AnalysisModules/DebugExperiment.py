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





