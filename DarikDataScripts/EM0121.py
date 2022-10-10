from ExperimentManagement.ExperimentHierarchy import ExperimentData
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

EM0121 = ExperimentData.loadHierarchy("D:\\EM0121")
EM0121.Retrieval.loadBehavioralData()
