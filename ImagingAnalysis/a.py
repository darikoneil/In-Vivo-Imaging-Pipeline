from ExperimentManagement.ExperimentHierarchy import ExperimentData
import numpy as np
EH = ExperimentData.loadHierarchy("D:\\EM0122")
EH.Encoding.folder_dictionary.get("deep_lab_cut_data").reIndex()
EH.Encoding.folder_dictionary.get("behavioral_exports").reIndex()
EH.Encoding.loadBehavioralData()
