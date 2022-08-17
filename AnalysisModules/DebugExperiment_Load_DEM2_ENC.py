from AnalysisModules.ExperimentHierarchy import ExperimentData
__directory="H:\\DEM_Excitatory_Study\\DEM2"
__mouse="M4584"
__study="DEM"
__study_mouse="DEM2"
Data = ExperimentData(Directory=__directory, Mouse=__mouse, Study=__study, StudyMouse=__study_mouse)
from AnalysisModules.BurrowFearConditioning import Encoding
Data.Encoding = Encoding(Data.passMeta())
__data_folder = "H:\\DEM_Excitatory_Study\\DEM2\\Encoding\\Imaging\\10Hz"
__index_file = "H:\\DEM_Excitatory_Study\\DEM2\\Encoding\\Imaging\\10Hz\\NeuronalIndex.csv"
__features_file = "H:\\DEM_Excitatory_Study\\DEM2\\Encoding\\Imaging\\10Hz\\Features.csv"
# FISSA: Signal Extraction & Source-Separation
from AnalysisModules.FissaAnalysis import FissaModule

# CASCADE: Spike Inference
from AnalysisModules.CascadeAnalysis import CascadeModule
# Instantiate Fissa Module & Sub-Modules.
# Sub-Module 1 is preparation, will contain raw data (PreparationModule)
# Sub-Module 2 is experiment, will contain separation data (SeparationModule)
# Sub-Module 3 is ProcessedTraces, just a container for processed signals
Data.Encoding.Fissa = FissaModule(data_folder=__data_folder, index_file = __index_file)
# This folder contains the suite2p/plane0/___.npy files as well as the saved registered files located in the suite2p/plane0/reg_tif folder
# Initialize
Data.Encoding.Fissa.pruneNonNeuronalROIs() # This step removes all non-neuronal data
Data.Encoding.Fissa.initializeFissa()
Data.Encoding.Fissa.loadFissaPrep()
Data.Encoding.Fissa.loadFissaSep()
Data.Encoding.Fissa.loadProcessedTraces()
# Cascade
Data.Encoding.Cascade = CascadeModule(Data.Encoding.Fissa.ProcessedTraces.detrended_merged_dFoF_result, Data.Encoding.Fissa.frame_rate, model_folder="C:\\ProgramData\\Anaconda3\\envs\\suite2p\\Pretrained_models")
Data.Encoding.Cascade.loadSpikeProb(load_path=Data.Encoding.Fissa.output_folder)
Data.Encoding.Cascade.loadSpikeInference(load_path=Data.Encoding.Fissa.output_folder)
Data.Encoding.Cascade.loadProcessedInferences(load_path=Data.Encoding.Fissa.output_folder)
Data.saveHierarchy()
from AnalysisModules.StaticProcessing import generateSpikeMatrix
from AnalysisModules.DecodingAnalysis import LogisticRegression
from AnalysisModules.BurrowFearConditioning import identifyTrialValence, reorganizeData
SpikeMatrix = generateSpikeMatrix(Data.Encoding.Cascade.spike_time_estimates, Data.Encoding.Cascade.spike_prob.shape[1])

Data.Encoding.LogReg = LogisticRegression(NeuralData=SpikeMatrix, FeatureDataFile=__features_file)

OrganizedSpikes = reorganizeData(SpikeMatrix, Data.Encoding.LogReg.feature_data, Data.Encoding.Cascade.frame_rate)


__csIndexFile = "H:\\DEM_Excitatory_Study\\DEM2\\Encoding\\Behavior\\BehavioralExports\\DEM2_ENC_CS_INDEX.csv"

plus_trials, minus_trials = identifyTrialValence(__csIndexFile)
plus_trials = plus_trials[0]
minus_trials = minus_trials[0]
PlusActivity = OrganizedSpikes[0][plus_trials, :, :]
MinusActivity = OrganizedSpikes[0][minus_trials, :, :]

LearnedPlusActivity = PlusActivity[5:11, :, :]
LearnedMinusActivity = MinusActivity[5:11, :, :]

import numpy as np
SumLearnedPlusActivity = np.sum(LearnedPlusActivity, axis=0)
SumLearnedMinusActivity = np.sum(LearnedMinusActivity, axis=0)
SumAllPlus = np.sum(SumLearnedPlusActivity, axis=0)
SumAllMinus = np.sum(SumLearnedMinusActivity, axis=0)
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns

_num_trials = 5
_shufIdx = np.arange(5)
np.random.shuffle(_shufIdx)
shuffledPlus = LearnedPlusActivity[_shufIdx, :, :]
shuffledMinus = LearnedMinusActivity[_shufIdx, :, :]
LearnedFeaturesPlus = OrganizedSpikes[2][plus_trials, :, :]
LearnedFeaturesMinus = OrganizedSpikes[2][minus_trials, :, :]
shuffledFeaturesPlus = LearnedFeaturesPlus[_shufIdx, :, :]
shuffledFeaturesMinus = LearnedFeaturesMinus[_shufIdx, :, :]
NewPlus = np.concatenate(shuffledPlus[:, :, 492:887], axis=1)
NewMinus = np.concatenate(shuffledMinus[:, :, 492:887], axis=1)
NewFPlus = np.concatenate(shuffledFeaturesPlus[:, :, 492:887], axis=1)
NewFMinus = np.concatenate(shuffledFeaturesMinus[:, :, 492:887], axis=1)
NewNeural = np.append(NewPlus, NewMinus, axis=1)
NewFeatures = np.append(NewFPlus, NewFMinus, axis=1)
_shuf_frames = np.arange(NewNeural.shape[1])
np.random.shuffle(_shuf_frames)

# Plus
NewNeural = NewNeural[:, _shuf_frames]
NewFeatures = NewFeatures[:, _shuf_frames]
Data.Encoding.LogReg = LogisticRegression(NeuralData=NewNeural, FeatureData=NewFeatures)
Data.Encoding.LogReg.label_data = Data.Encoding.LogReg.feature_data[5, :]
Data.Encoding.LogReg.splitData()
Data.Encoding.LogReg.fitModel(penalty='l2', solver='lbfgs', max_iter=100000)
Data.Encoding.LogReg.assessFit()
Data.Encoding.LogReg.makeAllPredictions()
Data.Encoding.LogReg.commonAssessment()
Data.Encoding.LogReg.printAssessment()
Data.Encoding.LogReg.plotROCs()


# Minus
Data.Encoding.LogRegMinus = LogisticRegression(NeuralData=NewNeural, FeatureData=NewFeatures)
Data.Encoding.LogRegMinus.label_data = Data.Encoding.LogReg.feature_data[4, :]
Data.Encoding.LogRegMinus.splitData()
#Data.Encoding.LogRegMinus.fitModel(penalty='l1', solver='liblinear', max_iter=100000)
Data.Encoding.LogRegMinus.fitModel(penalty='l2', solver='lbfgs', max_iter=100000)
Data.Encoding.LogRegMinus.assessFit()
Data.Encoding.LogRegMinus.makeAllPredictions()
Data.Encoding.LogRegMinus.commonAssessment()
Data.Encoding.LogRegMinus.printAssessment()
Data.Encoding.LogRegMinus.plotROCs()

NeuronWeights_Plus = Data.Encoding.LogReg.internalModel.model.coef_
fun_intercept_Plus = Data.Encoding.LogReg.internalModel.model.intercept_[0]
NeuronWeights_Minus = Data.Encoding.LogRegMinus.internalModel.model.coef_
fun_intercept_Minus = Data.Encoding.LogRegMinus.internalModel.model.intercept_[0]

_num_neurons = NeuronWeights_Minus.shape[1]

#fig = plt.figure(figsize=(12, 6))
#ax1 = fig.add_subplot(111)


#for _neuron in range(_num_neurons):
#    ax1.plot([_neuron, _neuron], [0, NeuronWeights_Plus[0][_neuron]], color="#7840cc", lw=1, alpha=0.95)
#plt.show()


# Find signed distance from the hyperplane

single_state_samples = np.diag(np.diag(np.full((_num_neurons, _num_neurons), 1, dtype=np.int32)))

Plus_Signed_Distance = Data.Encoding.LogReg.internalModel.model.decision_function(single_state_samples)
Minus_Signed_Distance = Data.Encoding.LogRegMinus.internalModel.model.decision_function(single_state_samples)

plus_mean = np.mean(Plus_Signed_Distance)
plus_std = np.std(Plus_Signed_Distance)

minus_mean = np.mean(Minus_Signed_Distance)
minus_std = np.std(Minus_Signed_Distance)

EngramMinus = np.where(Minus_Signed_Distance > (minus_std+minus_std+minus_mean))
EngramPlus = np.where(Plus_Signed_Distance > (plus_std+plus_std+plus_mean))
EngramJoint = np.intersect1d(EngramMinus, EngramPlus)

Engram = np.full((978,), 0, dtype=np.int32)
Engram[EngramMinus] = 1
Engram[EngramPlus] = 2
Engram[EngramJoint] = 3

Engram = np.array(list(Engram))

import pandas as pd

_Signed_Distance_Data = {'Plus Signed Distance': Plus_Signed_Distance.squeeze(), 'Minus Signed Distance': Minus_Signed_Distance.squeeze(),
                                'Plus Signed Weights': NeuronWeights_Plus.squeeze(), 'Minus Signed Weights': NeuronWeights_Minus.squeeze(),
                                'Engram': Engram.squeeze()}

Signed_Distance_Data = pd.DataFrame(data=_Signed_Distance_Data)

axes_styling = {''}
sns.set_style('darkgrid')
kwargs = {'sizes': 25, 'alpha': 0.5, 'edgecolor': 'black', 'linewidth': 0.5}
marginal_keys = {'fill': True}
#custom_colors = {'tab': "#139fff", 'tab': ""}
custom_colors = sns.color_palette(['#40cc8b', '#139fff',  '#ff4e4b'])
sps = sns.jointplot(data=Signed_Distance_Data, x="Plus Signed Distance", y="Minus Signed Distance", hue="Engram", palette=custom_colors, kind="scatter", **kwargs, marginal_kws=marginal_keys)
#sps.plot_joint(sns.kdeplot, hue="Engram", zorder=0, levels=10, palette="flare")


ax = sps.ax_joint
# ax.set_xscale('log')
# ax.set_yscale('log')
ax.autoscale()
plt.show()

EList = np.append(EngramPlus[0], EngramMinus[0])
num_Engram_Neurons = EList.shape[0]

EPAP = LearnedPlusActivity[:, EList, 492:886]
EPAM = LearnedMinusActivity[:, EList, 492:886]
MEPAP = np.mean(EPAP, axis=0)
MEPAM = np.mean(EPAM, axis=0)
DMEPAP = MEPAP.copy()
DMEPAM = MEPAM.copy()


fig2 = plt.figure(figsize=(12, 6))

ax2 = fig2.add_subplot(211)
sh2 = sns.heatmap(DMEPAP, ax=ax2, cmap="binary")
ax2.set_xticks(range(0, 350, 50), labels=(range(0, 35, 5)))
ax2.set_yticks([0, num_Engram_Neurons], labels=[0, num_Engram_Neurons])
ax2.set_title("Spike Map: CS+")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Neuron (#)")

ax3 = fig2.add_subplot(212)
sh3 = sns.heatmap(DMEPAM, ax=ax3, cmap="binary")
ax3.set_xticks(range(0, 350, 50), labels=(range(0, 35, 5)))
ax3.set_yticks([0, num_Engram_Neurons], labels=[0, num_Engram_Neurons])
ax3.set_title("Spike Map: CS-")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Neuron (#)")

plt.show()

#import scipy.stats

#BinnedOne = scipy.stats.binned_statistic(EPAP[0, 0, :], EPAP[0, 0, :], statistic='sum', bins=39)

#TrialOneData = dict()
#for _neuron in range(num_Engram_Neurons):
#    TrialOneData[str(_neuron)] = EPAP[0, _neuron, :]

intervals = pd.interval_range(0, 394, freq=10)


PlusSums = np.full((5, 63, 39), 0, dtype=np.int32)
for _neuron in range(num_Engram_Neurons):
    for _trial in range(5):
        for _interval in range(len(intervals)):
            PlusSums[_trial, _neuron, _interval] = np.sum(EPAP[_trial, _neuron, int(intervals.values[_interval].left):int(intervals.values[_interval].right)])
PlusRafa = np.mean(PlusSums, axis=0)

MinusSums = np.full((5, 63, 39), 0, dtype=np.int32)
for _neuron in range(num_Engram_Neurons):
    for _trial in range(5):
        for _interval in range(len(intervals)):
            MinusSums[_trial, _neuron, _interval] = np.sum(EPAM[_trial, _neuron, int(intervals.values[_interval].left):int(intervals.values[_interval].right)])
MinusRafa = np.mean(MinusSums, axis=0)



fig2 = plt.figure(figsize=(12, 6))

ax2 = fig2.add_subplot(211)
sh2 = sns.heatmap(PlusRafa, ax=ax2, cmap="Spectral_r")
ax2.set_xticks(range(0, 35, 5), labels=(range(0, 35, 5)))
ax2.set_yticks([0, num_Engram_Neurons], labels=[0, num_Engram_Neurons])
ax2.set_title("Spike Map: CS+")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Neuron (#)")

ax3 = fig2.add_subplot(212)
sh3 = sns.heatmap(MinusRafa, ax=ax3, cmap="Spectral_r")
ax3.set_xticks(range(0, 35, 5), labels=(range(0, 35, 5)))
ax3.set_yticks([0, num_Engram_Neurons], labels=[0, num_Engram_Neurons])
ax3.set_title("Spike Map: CS-")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Neuron (#)")

plt.show()

EPAP = LearnedPlusActivity[:, :, 492:886]
intervals_5s = pd.interval_range(0, 394, freq=50)
PlusSums = np.full((5, 978, 7), 0, dtype=np.int32)
for _neuron in range(978):
    for _trial in range(5):
        for _interval in range(len(intervals_5s)):
            PlusSums[_trial, _neuron, _interval] = np.sum(EPAP[_trial, _neuron, int(intervals.values[_interval].left):int(intervals.values[_interval].right)])
PlusRafa = np.mean(PlusSums, axis=0)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
sns.heatmap(PlusRafa, ax=ax1, cmap="Spectral_r")
plt.show()
