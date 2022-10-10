from ExperimentManagement.ExperimentHierarchy import ExperimentData
from BehavioralAnalysis.BurrowFearConditioning import FearConditioning
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

EM0121 = ExperimentData.loadHierarchy("D:\\EM0121")
EM0121.Retrieval.loadBehavioralData()
EM0121.recordMod("Organized Retrieval Image Frames")
EM0121.saveHierarchy()

_analog_file = EM0121.Retrieval.generateFileID('Analog')
AnalogData = FearConditioning.loadAnalogData(_analog_file)

_digital_file = EM0121.Retrieval.generateFileID('Digital')
DigitalData = FearConditioning.loadDigitalData(_digital_file)

_state_file = EM0121.Retrieval.generateFileID('State')
StateData = FearConditioning.loadStateData(_state_file)