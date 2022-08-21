from AnalysisModules.FissaAnalysis import FissaModule
from AnalysisModules.StaticProcessing import smoothTraces_TiffOrg, calculate_dFoF, mergeTraces, smoothTraces
from AnalysisModules.PlottingFunctions import interactivePlot3
from AnalysisModules.CascadeAnalysis import CascadeModule

# Make the Fissa Module
Data = FissaModule(data_folder="C:\\ProgramData\\Anaconda3\\envs\\suite2p\\DebugData")

Data.initializeFissa()

Data.extractTraces()

#Data.saveFissaPrep()

#Data.saveProcessedTraces()

TrialOrgSmoothed, C = smoothTraces_TrialOrg(Data.preparation.raw)

TOS = mergeTraces(TrialOrgSmoothed)

MatrixOrgSmoothed = smoothTraces(Data.preparation.raw)

Data.ProcessedTraces.smoothed_raw = Data.preparation.raw.copy()

Data.separateTraces()

Data.saveFissaSep()

Data.saveProcessedTraces()

Data.ProcessedTraces.dFoF_result = calculate_dFoF(Data.experiment.result,
                                                                 Data.frame_rate,
                                                                 raw=Data.preparation.raw,
                                                                 merge_after=False)

Data.ProcessedTraces.merged_dFoF_result = mergeTraces(Data.ProcessedTraces.dFoF_result)

# Do The Cascade Module
Data.Cascade = CascadeModule(Data.ProcessedTraces.merged_dFoF_result, Data.frame_rate)

Data.Cascade.pullModels()

Data.Cascade.model_name = "Global_EXC_10Hz_smoothing100ms"

Data.Cascade.predictSpikeProb()


Data.Cascade.exportSpikeProb(Data.output_folder)

Data.Cascade.inferDiscreteSpikes()

Data.Cascade.exportSpikeInference(Data.output_folder)

Data.Cascade.plotTraceComparisons()

interactivePlot3(Data.preparation.raw, Data.experiment.result,
                              Data.ProcessedTraces.dFoF_result, Data.frame_rate,
                              11946)

