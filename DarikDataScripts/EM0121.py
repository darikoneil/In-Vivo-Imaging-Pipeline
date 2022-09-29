from ExperimentManagement.ExperimentHierarchy import ExperimentData
import matplotlib
matplotlib.use('Qt5Agg')

EM0121 = ExperimentData.loadHierarchy("D:\\EM0121")
EM0121.Retrieval.loadBehavioralData()
# EM0121.saveHierarchy()


