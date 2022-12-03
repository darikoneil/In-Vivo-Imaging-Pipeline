import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns


from ExperimentManagement.ExperimentHierarchy import ExperimentData
from ImagingAnalysis.DataProcessing import Processing
from ComputationalAnalysis.SupportVectorMachine import SVM

EM0121 = ExperimentData.load_experiments("D:\\EM0121")
SpikeTimes = EM0121.Retrieval.folder_dictionary.get("10Hz").load_cascade_exports()[0]
SpikeTimes = Processing.generateSpikeMatrix(SpikeTimes, 26352)
SpikeTimes_Matrix = Processing.trial_matrix_org(EM0121.Retrieval.data_frame.copy(deep=True), SpikeTimes)
TrialIndex = EM0121.Retrieval.trial_parameters.get("stimulusTypes")
SpikeTimes_Tensor = np.array(np.hsplit(SpikeTimes_Matrix, 10))
FeatureData_Tensor, FeatureData_Labels = Processing.generate_features(345, 10, EM0121.Retrieval.trial_parameters)
FeatureData_Matrix = np.hstack(FeatureData_Tensor)

binned_spike_tensor = Processing.bin_data(SpikeTimes_Tensor, 10)
binned_feature_tensor = Processing.bin_data(FeatureData_Tensor, 10)

binned_spike_tensor = binned_spike_tensor[:, :, 0:30]
binned_feature_tensor = binned_feature_tensor[:, :, 0:30]
binned_feature_tensor[binned_feature_tensor > 1] = 1

svm_model = SVM(NeuralData=binned_spike_tensor, FeatureData=binned_feature_tensor, Trials=10,
                TrialIndex=TrialIndex, kernel="linear")
svm_model.shuffle_trials()
svm_model.testing_y = svm_model.testing_y[:, 0]
svm_model.training_y = svm_model.training_y[:, 0]
svm_model.fitModel()
svm_model.assessFit()
svm_model.makeAllPredictions()
svm_model.commonAssessment()
svm_model.plotROCs()

svm_model2 = SVM(NeuralData=binned_spike_tensor, FeatureData=binned_feature_tensor, Trials=10,
                TrialIndex=TrialIndex, kernel="linear")
svm_model2.shuffle_trials()
svm_model2.testing_y = svm_model2.testing_y[:, 1]
svm_model2.training_y = svm_model2.training_y[:, 1]
svm_model2.fitModel()
svm_model2.assessFit()
svm_model2.makeAllPredictions()
svm_model2.commonAssessment()
svm_model2.plotROCs()


fig1 = plt.figure()
gs = fig1.add_gridspec(5, 2)
PlusTrials = np.where(np.array(TrialIndex) == 0)[0]
MinusTrials = np.where(np.array(TrialIndex) == 1)[0]
for i in range(PlusTrials.__len__()):
    _ax = fig1.add_subplot(gs[i, 0])
    _ax.plot(np.sum(binned_spike_tensor[PlusTrials[i], :, :], axis=0))
for i in range(MinusTrials.__len__()):
    _ax = fig1.add_subplot(gs[i, 1])
    _ax.plot(np.sum(binned_spike_tensor[MinusTrials[i], :, :], axis=0))


