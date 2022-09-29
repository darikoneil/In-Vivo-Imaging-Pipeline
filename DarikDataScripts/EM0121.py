from ExperimentManagement.ExperimentHierarchy import ExperimentData
import matplotlib
matplotlib.use('Qt5Agg')

EM0121 = ExperimentData.loadHierarchy("D:\\EM0121")
EM0121.PreExposure.loadBehavioralData()
EM0121.Encoding.loadBehavioralData()
EM0121.Retrieval.loadBehavioralData()
del EM0121.Retrieval.data
del EM0121.Encoding.data
del EM0121.PreExposure.data
EM0121.PreExposure.recordMod()
EM0121.Encoding.recordMod()
EM0121.PreExposure.recordMod()
EM0121.recordMod("Pandas Organization")
# EM0121.saveHierarchy()


