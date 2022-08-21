import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics, linear_model, svm
from scipy import io, stats
from Neural_Decoding.Neural_Decoding.preprocessing_funcs import get_spikes_with_history
from Neural_Decoding.Neural_Decoding.metrics import get_R2
from Neural_Decoding.Neural_Decoding.decoders import WienerCascadeDecoder
from Neural_Decoding.Neural_Decoding.decoders import WienerFilterDecoder


class DecodingFear:
    def __init__(self, featuresPath, labelsPath):

        self.featuresPath = featuresPath
        self.labelsPath = labelsPath

        self.numNeurons = int(0)
        self.numFrames = int(0)
        self.numLabels = int(0)

        self.neuralData = None
        self.labelData = None

        self.LogisticRegression = None
        self.LinearDecoder = None
        self.LinearNonLinearDecoder_Binary = None
        self.LinearNonLinearDecoder_Continuous = None
        self.SVM=None

        self.splitMode = 0 # 0 is standard train/test, 1 is train/test/valid
        self.split0 = 0.8
        self.split1 = [0.15, 0.7]
        self.shuffleBeforeSplit = bool(0) # 1 means shuffle

        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.valid_data = None
        self.valid_label = None

        self.MFRByClass = None # Mean Firing Rate By Class
        self.PFRByClass = None # Peak Firing Rate By Class
        self.FSByClass = None # Firing Stability By Class
        self.LabelClasses = None # Associated Classes for MFR, PFR, FS

    def loadData(self):
        self.neuralData = np.genfromtxt(self.featuresPath, delimiter=",")
        self.labelData = np.genfromtxt(self.labelsPath, delimiter=",")
        self.numFrames, self.numNeurons = self.neuralData.shape
        # self.numLabels = self.labelData.shape[1]
        self.numLabels = 1

    def splitData(self):
        if self.splitMode == 0:
            _trainFrames = int(self.split0*self.numFrames)

            if self.shuffleBeforeSplit == 1:
                # not yet
                a = 1
            else:
                self.train_data = self.neuralData[0:_trainFrames, :]
                self.test_data = self.neuralData[_trainFrames:self.numFrames+1, :]
                self.train_label = self.labelData[0:_trainFrames]
                self.test_label = self.labelData[_trainFrames:self.numFrames + 1]

        else:
            # not yet
            a = 1

    def useLogReg(self):
        # Initialize & Learn
        self.LogisticRegression = linear_model.LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000)
        self.LogisticRegression.fit(self.train_data, self.train_label)

        # Score
        self.LogisticRegression.trainR2 = self.LogisticRegression.score(self.train_data, self.train_label)
        self.LogisticRegression.testR2 = self.LogisticRegression.score(self.train_data, self.train_label)
        self.LogisticRegression.trainPred = self.LogisticRegression.predict(self.train_data)
        self.LogisticRegression.trainAUC = metrics.roc_auc_score(self.train_label, self.LogisticRegression.trainPred)
        self.LogisticRegression.testPred = self.LogisticRegression.predict(self.test_data)
        self.LogisticRegression.testAUC = metrics.roc_auc_score(self.test_label, self.LogisticRegression.testPred)
        self.LogisticRegression.estTrainAccuracy = metrics.accuracy_score(self.train_label,
                                                                          self.LogisticRegression.trainPred)
        self.LogisticRegression.estTestAccuracy = metrics.accuracy_score(self.test_label,
                                                                         self.LogisticRegression.testPred)
        self.LogisticRegression.balTrainAccuracy = metrics.balanced_accuracy_score(self.train_label,
                                                                                   self.LogisticRegression.trainPred)
        self.LogisticRegression.balTestAccuracy = metrics.balanced_accuracy_score(self.test_label,
                                                                                  self.LogisticRegression.testPred)

        # Visualize
        ROCfig, ROCaxis = plt.subplots(1, 1)
        metrics.plot_roc_curve(self.LogisticRegression, self.train_data, self.train_label, ax=ROCaxis)
        metrics.plot_roc_curve(self.LogisticRegression, self.test_data, self.test_label, ax=ROCaxis)
        ROCaxis.legend(['Training Dataset, AUC:' + str(round(self.LogisticRegression.trainAUC, 2)) + ', Accuracy:' + str(round(self.LogisticRegression.estTrainAccuracy,2)) + ',Balanced Accuracy:' + str(round(self.LogisticRegression.balTrainAccuracy,2)),
                        'Testing Dataset, AUC:' + str(round(self.LogisticRegression.testAUC, 2)) + ',Accuracy:' + str(round(self.LogisticRegression.estTestAccuracy,2)) + ',Balanced Accuracy:' + str(round(self.LogisticRegression.balTestAccuracy,2))])

    def initializeLogReg(self):
        self.LogisticRegression = linear_model.LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000)

    def learnLogReg(self):
        self.LogisticRegression.fit(self.train_data,self.train_label)
        self.LogisticRegression.trainR2 = self.LogisticRegression.score(self.train_data, self.train_label)
        self.LogisticRegression.testR2 = self.LogisticRegression.score(self.train_data, self.train_label)
        self.LogisticRegression.trainPred = self.LogisticRegression.predict(self.train_data)
        self.LogisticRegression.trainAUC = metrics.roc_auc_score(self.train_label, self.LogisticRegression.trainPred)
        self.LogisticRegression.testPred = self.LogisticRegression.predict(self.test_data)
        self.LogisticRegression.testAUC = metrics.roc_auc_score(self.test_label, self.LogisticRegression.testPred)
        self.LogisticRegression.estTrainAccuracy = metrics.accuracy_score(self.train_label,self.LogisticRegression.trainPred)
        self.LogisticRegression.estTestAccuracy = metrics.accuracy_score(self.test_label,self.LogisticRegression.testPred)
        self.LogisticRegression.balTrainAccuracy = metrics.balanced_accuracy_score(self.train_label,self.LogisticRegression.trainPred)
        self.LogisticRegression.balTestAccuracy = metrics.balanced_accuracy_score(self.test_label,self.LogisticRegression.testPred)

    def visualizeLogReg(self):
        ROCfig, ROCaxis = plt.subplots(1, 1)
        metrics.plot_roc_curve(self.LogisticRegression, self.train_data, self.train_label, ax=ROCaxis)
        metrics.plot_roc_curve(self.LogisticRegression, self.test_data, self.test_label, ax=ROCaxis)
        ROCaxis.legend(['Training Dataset, AUC:' + str(
            round(self.LogisticRegression.trainAUC, 2)) + ', Accuracy:' + str(
            round(self.LogisticRegression.estTrainAccuracy, 2)) + ',Balanced Accuracy:' + str(
            round(self.LogisticRegression.balTrainAccuracy, 2)),
                        'Testing Dataset, AUC:' + str(round(self.LogisticRegression.testAUC, 2)) + ',Accuracy:' + str(
                            round(self.LogisticRegression.estTestAccuracy, 2)) + ',Balanced Accuracy:' + str(
                            round(self.LogisticRegression.balTestAccuracy, 2))])

    def runSupportVectorMachine(self, FiringMetric, Kernel):
        # Kernels of note are linear and radial basis function (linear, rbf respectively)
        # If slow, use LinearSVC instead
        self.SVM = svm.SVC(kernel=Kernel)
        # Did this for maintainability
        if FiringMetric == 'MeanFiringRate':
            _X = self.MFRByClass
            _Y = self.LabelClasses
        elif FiringMetric == 'PeakFiringRate':
            _X = self.PFRByClass
            _Y = self.LabelClasses
        elif FiringMetric == 'FiringStability':
            _X = self.FSByClass
            _Y = self.LabelClasses
        else:
            _X = None # Change this to throw error later, im hungry
            _Y = None
        self.SVM.fit(_X, _Y)

   # def calculateMeanFiringRateByClass(self):

   # def calculatePeakFiringRateByClass(self);

   # def calculateFiringStabilityByClass(self):


# D = DecodingFear('C:\\ProgramData\\Anaconda3\\envs\\NeuralDecoding\\DEM2_ENC_DATA.csv', 'C:\\ProgramData\\Anaconda3\\envs\\NeuralDecoding\\DEM2_ENC_CS.csv')
#D = DecodingFear("C:\\ProgramData\\Anaconda3\\envs\\NeuralDecoding\\DEM2_ENC_BACKEND.csv","C:\\ProgramData\\Anaconda3\\envs\\NeuralDecoding\\DEM2_ENC_BACKEND_LABELS.csv")
#D.loadData()
#D.splitData()
#D.useLogReg()


D = DecodingFear("C:\\Users\\YUSTE\\Desktop\\neuralData2.csv","C:\\Users\\YUSTE\\Desktop\\labelData2.csv")
D.loadData()
D.splitData()
