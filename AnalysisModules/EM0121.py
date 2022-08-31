from AnalysisModules.ExperimentHierarchy import ExperimentData, CollectedDataFolder
from AnalysisModules.BurrowFearConditioning import FearConditioning

# EM0121 = ExperimentData(Directory="D:\\EM0121", Mouse="EM0121")
# EM0121.saveHierarchy()

EM0121 = ExperimentData.loadHierarchy("D:\\EM0121")

# EM0121.saveHierarchy()
# EM0121.stages.append("Encoding")
# EM0121.stages.append("Retrieval")
# ExperimentData.generateStage(EM0121.directory, Title="Encoding")
# ExperimentData.generateStage(EM0121.directory, Title="Retrieval")
# EM0121.PreExposure = FearConditioning(EM0121.passMeta(), "PreExposure", TrialsPerStim=5, NumStim=2)
# EM0121.recordMod("Added PreExposure")
# EM0121.Encoding = FearConditioning(EM0121.passMeta(), "Encoding", TrialsPerStim=10, NumStim=2)
# EM0121.recordMod("Added Encoding")
# EM0121.Retrieval = FearConditioning(EM0121.passMeta(), "Retrieval", TrialsPerStim=5, NumStim=2)
# EM0121.recordMod("Added Retrieval")
# EM0121.saveHierarchy()
# EM0121.PreExposure.folder_dictionary['raw_behavioral_data'].reIndex()
# EM0121.PreExposure.loadBehavioralData()
# EM0121.PreExposure.recordMod()
# EM0121.recordMod("Entered PreExposure Analog Data")

# EM0121.Encoding.folder_dictionary['raw_behavioral_data'].reIndex()
# EM0121.Encoding.loadBehavioralData()
# EM0121.Encoding.recordMod()
# EM0121.recordMod("Added Encoding Analog Data")

# EM0121.Retrieval.folder_dictionary['raw_behavioral_data'].reIndex()
# EM0121.Retrieval.loadBehavioralData()
# EM0121.Retrieval.recordMod()
# EM0121.recordMod("Added Retrieval Analog Data")
# EM0121.saveHierarchy()

# EM0121.Retrieval = FearConditioning(EM0121.passMeta(), "Retrieval", TrialsPerStim=5, NumStim=2)
# EM0121.Retrieval.loadBehavioralData()
# EM0121.Retrieval.recordMod()

# EM0121.Encoding = FearConditioning(EM0121.passMeta(), "Encoding", TrialsPerStim=10, NumStim=2)
# EM0121.Encoding.loadBehavioralData()
# EM0121.Encoding.recordMod()

# EM0121.PreExposure = FearConditioning(EM0121.passMeta(), "PreExposure", TrialsPerStim=5, NumStim=2)
# EM0121.PreExposure.loadBehavioralData()
# EM0121.PreExposure.recordMod()
# EM0121.recordMod("Adjusted Behavioral Stages")

# Now Added True Logging
