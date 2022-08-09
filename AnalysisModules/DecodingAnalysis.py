import numpy as np
import pickle as pkl
import pathlib
# Imports for classes in file organized by class
# Not sure if I should use conditional imports...
# More graceful failure if certain package not installed
# But I don't want many duplicate imports if I make many decoders
# So right now, let's follow PEP8

# Logistic
from AnalysisModules.NewDecoders import LogisticRegressionDecoder

# WienerFilterDecoder
from AnalysisModules.ModifiedDecoders import WienerFilterDecoder

# WienerCascadeDecoder
from Neural_Decoding.Neural_Decoding.decoders import WienerCascadeDecoder

# Metrics functions from Neural Decoding Package from Joshua Glaser while in Kording Lab
from Neural_Decoding.Neural_Decoding.metrics import get_R2

# Metrics from sk-learn
from sklearn import metrics

# Metrics Dictionary Constructor
import itertools


class DecodingModule:
    # Generic decoding module super-class with conserved methods, properties,
    # and methods with conserved names but different implementation.
    def __init__(self, **kwargs):
        # parse inputs
        self.neural_data = kwargs.get('NeuralData', None)
        self.feature_data = kwargs.get('FeatureData', None)
        self.label_data = kwargs.get('LabelData', None)
        self.data_splits = kwargs.get('DataSplits', [0.8, 0.2])
        self.covariance_matrix = kwargs.get('CovarianceMatrix', None)
        _feature_data_file = kwargs.get('FeatureDataFile', None)
        _label_data_file = kwargs.get('LabelDataFile', None)

        # load if necessary
        if _feature_data_file is not None or _label_data_file is not None:
            self.loadFeaturesLabels(_feature_data_file, _label_data_file)

        # Initialization
        self.training_x = None
        self.training_y = None
        self.predicted_training_y = None
        self.testing_x = None
        self.testing_y = None
        self.predicted_testing_y = None
        self.validation_x = None
        self.validation_y = None
        self.predicted_validation_y = None
        self.ModelPerformance = None

    def loadFeaturesLabels(self, _feature_data_file, _label_data_file):
        if _feature_data_file is not None and self.feature_data is None:
            try:
                _feature_file_ext = pathlib.Path(_feature_data_file).suffix
                if _feature_file_ext == ".npy" or _feature_file_ext == ".npz":
                    self.feature_data = np.load(_feature_data_file, allow_pickle=True)
                elif _feature_file_ext == ".csv":
                    self.feature_data = np.genfromtxt(_feature_data_file, dtype=int, delimiter=",")
                else:
                    print("Features data in unexpected file type.")
                if self.feature_data.shape[1] != self.neural_data.shape[1] and self.neural_data is not None:
                    raise AssertionError
                # noinspection PyUnresolvedReferences
                if len(self.feature_data.shape) != len(self.neural_data.shape) and self.neural_data is not None:
                    raise ValueError
            except RuntimeError:
                print("Could not load feature data.")
            except AssertionError:
                print("The number of features samples must match the number of neural data samples.")
            except ValueError:
                print('The organization of features and neural data must match.')
        if _label_data_file is not None and self.label_data is None:
            try:
                _label_file_ext = pathlib.Path(_label_data_file).suffix
                if _label_file_ext == ".npy" or _label_file_ext == ".npz":
                    self.label_data = np.load(_label_data_file, allow_pickle=True)
                elif _label_file_ext == ".csv":
                    self.label_data = np.genfromtxt(_label_data_file, dtype=int, delimiter=",")
                else:
                    print("Labels data in unexpected file type.")
                if self.label_data.shape[1] != self.neural_data.shape[1] and self.neural_data is not None:
                    raise AssertionError
                # noinspection PyUnresolvedReferences
                if len(self.label_data.shape) != len(self.neural_data.shape) and self.neural_data is not None:
                    raise ValueError
            except RuntimeError:
                print("Could not load label data.")
            except AssertionError:
                print('The number of label samples must match the number of neural data samples')
            except ValueError:
                print("The organization of labels and neural data must match.")

    def splitData(self):
        if self.neural_data_organization == "Matrix Org":
            _training_frames = int(self.data_splits[0] * self.num_frames)
            if len(self.data_splits) == 2:
                print("Splitting data in training & testing sets")
                print("Data splits are: " +
                      str(self.data_splits[0]*100) + "% training" +
                      " vs " + str(self.data_splits[1]*100) + "% testing")
                self.training_x = self.neural_data[:, 0:_training_frames]
                self.training_y = self.label_data[0:_training_frames]
                self.testing_x = self.neural_data[:, _training_frames:self.num_frames+1]
                self.testing_y = self.label_data[_training_frames:self.num_frames+1]
            elif len(self.data_splits) == 3:
                self.training_x = self.neural_data[:, 0:_training_frames]
                self.training_y = self.neural_data[:, 0:_training_frames]
                self.testing_x = None
                self.testing_y = None
                self.validation_x = None
                self.validation_y = None

    def fitModel(self, **kwargs):
        print("1st")

    def assessFit(self, **kwargs):
        print("Blank Function")

    def commonAssessment(self, **kwargs):
        print("Blank Function")

    def fullAssessment(self, **kwargs):
        print("Blank Function")

    def makePrediction(self, **kwargs):
        print("3rd")

    def printAssessment(self):
        print("Printing Assessment of Model Performance:")
        # Iterate over key/value pairs in dict and print them
        for key, value in self.ModelPerformance.items():
            if value is not None:
                print(key, ' : ', value)

    # Here are some useful properties of the neural data that we must access often

    @property
    def neural_data_organization(self):
        if self.neural_data is not None:
            if self.neural_data.dtype == 'O':
                return "Tiff Org"
            else:
                if len(self.neural_data.shape) == 3:
                    return "Trial Org"
                elif len(self.neural_data.shape) == 2:
                    return "Matrix Org"
                else:
                    return "Unknown"
        else:
            print("Please import neural data")

    @property
    def num_neurons(self):
        if self.neural_data is not None:
            return self.neural_data.shape[0]
        else:
            print("Please import neural data")

    @property
    def num_frames(self):
        if self.neural_data is not None:
            if self.neural_data_organization == "Tiff Org":
                return np.concatenate(self.neural_data[0], axis=1)[0, :].shape[1]
            elif self.neural_data_organization == "Trial Org":
                print("Not yet implemented")
                pass
            elif self.neural_data_organization == "Matrix Org":
                return self.neural_data.shape[1]


class LogisticRegression(DecodingModule):
    # Class for easily managing Logistic Regression
    # Note inheritances from generic Decoder Module class
    def __init__(self, **kwargs):
        # noinspection PyArgumentList
        super().__init__(**kwargs)
        print("Instanced Logistic Regression")
        self.internalModel = LogisticRegressionDecoder()
        self.ModelPerformance = PerformanceMetrics("Classification", len(self.data_splits))

    def fitModel(self, **kwargs):
        print("Fitting Logistic Regression...")
        _penalty = kwargs.get('penalty', 'l1')
        _solver = kwargs.get('solver', 'liblinear')
        _max_iter = kwargs.get('max_iter', 100000)
        self.internalModel.fit(self.training_x.T, self.training_y.T, penalty=_penalty,
                               solver=_solver, max_iter=_max_iter)
        print("Finished")

    def assessFit(self, **kwargs):
        _normalize = kwargs.get('normalize', True)
        self.predicted_training_y = self.makePrediction(observed=self.training_x)
        self.ModelPerformance[('accuracy', 'training')] = self.internalModel.model.score(self.training_x.T, self.training_y.T)
        print("Training Classification Accuracy is " + str(self.ModelPerformance[('accuracy', 'training')]))

    def commonAssessment(self, **kwargs):
        _flag_valid = kwargs.get('flag_valid', False)

        # Accuracy
        if self.ModelPerformance[('accuracy', 'training')] is None:
            self.ModelPerformance[('accuracy', 'training')] = self.internalModel.model.score(self.training_x.T,
                                                                                             self.training_y.T)
        if self.ModelPerformance[('accuracy', 'testing')] is None:
            self.ModelPerformance[('accuracy', 'testing')] = self.internalModel.model.score(self.testing_x.T,
                                                                                            self.testing_y.T)
        if _flag_valid:
            if self.ModelPerformance[('accuracy', 'validation')] is None:
                self.ModelPerformance[('accuracy', 'validation')] = self.internalModel.model.score(self.validation_x.T,
                                                                                                   self.validation_y.T)

        # Precision
        if self.ModelPerformance[('precision', 'training')] is None:
            self.ModelPerformance[('precision', 'training')] = metrics.precision_score(self.training_y,
                                                                                       self.predicted_training_y)
        if self.ModelPerformance[('precision', 'testing')] is None:
            self.ModelPerformance[('precision', 'testing')] = metrics.precision_score(self.testing_y,
                                                                                      self.predicted_testing_y)
        if _flag_valid:
            if self.ModelPerformance[('precision', 'validation')] is None:
                self.ModelPerformance[('precision', 'validation')] = metrics.precision_score(self.validation_y,
                                                                                             self.predicted_validation_y)

        # Recall
        if self.ModelPerformance[('recall', 'training')] is None:
            self.ModelPerformance[('recall', 'training')] = metrics.recall_score(self.training_y,
                                                                                       self.predicted_training_y)
        if self.ModelPerformance[('recall', 'testing')] is None:
            self.ModelPerformance[('recall', 'testing')] = metrics.recall_score(self.testing_y,
                                                                                      self.predicted_testing_y)
        if _flag_valid:
            if self.ModelPerformance[('recall', 'validation')] is None:
                self.ModelPerformance[('recall', 'validation')] = metrics.recall_score(self.validation_y,
                                                                                       self.predicted_validation_y)

        # f1
        if self.ModelPerformance[('f1', 'training')] is None:
            self.ModelPerformance[('f1', 'training')] = metrics.f1_score(self.training_y,
                                                                                       self.predicted_training_y)
        if self.ModelPerformance[('f1', 'testing')] is None:
            self.ModelPerformance[('f1', 'testing')] = metrics.f1_score(self.testing_y,
                                                                                      self.predicted_testing_y)
        if _flag_valid:
            if self.ModelPerformance[('f1', 'validation')] is None:
                self.ModelPerformance[('f1', 'validation')] = metrics.f1_score(self.validation_y,
                                                                               self.predicted_validation_y)

        # balanced accuracy
        if self.ModelPerformance[('balanced_accuracy', 'training')] is None:
            self.ModelPerformance[('balanced_accuracy', 'training')] = metrics.balanced_accuracy_score(self.training_y,
                                                                                                       self.predicted_training_y)
        if self.ModelPerformance[('balanced_accuracy', 'testing')] is None:
            self.ModelPerformance[('balanced_accuracy', 'testing')] = metrics.balanced_accuracy_score(self.testing_y,
                                                                                                      self.predicted_testing_y)
        if _flag_valid:
            if self.ModelPerformance[('balanced_accuracy', 'validation')] is None:
                self.ModelPerformance[('balanced_accuracy', 'validation')] = metrics.balanced_accuracy_score(self.validation_y,
                                                                                                             self.predicted_validation_y)

        # AUC of ROC
        if self.ModelPerformance[('AUC', 'training')] is None:
            self.ModelPerformance[('AUC', 'training')] = metrics.roc_auc_score(self.training_y,
                                                                                       self.predicted_training_y)
        if self.ModelPerformance[('AUC', 'testing')] is None:
            self.ModelPerformance[('AUC', 'testing')] = metrics.roc_auc_score(self.testing_y,
                                                                                      self.predicted_testing_y)
        if _flag_valid:
            if self.ModelPerformance[('AUC', 'validation')] is None:
                self.ModelPerformance[('AUC', 'validation')] = metrics.roc_auc_score(self.validation_y,
                                                                                    self.predicted_validation_y)

        # AUC of PR
        if self.ModelPerformance[('AUC_PR', 'training')] is None:
            self.ModelPerformance[('AUC_PR', 'training')] = metrics.average_precision_score(self.training_y,
                                                                                            self.predicted_training_y)
        if self.ModelPerformance[('AUC_PR', 'testing')] is None:
            self.ModelPerformance[('AUC_PR', 'testing')] = metrics.average_precision_score(self.testing_y,
                                                                                           self.predicted_testing_y)
        if _flag_valid:
            if self.ModelPerformance[('AUC_PR', 'validation')] is None:
                self.ModelPerformance[('AUC_PR', 'validation')] = metrics.average_precision_score(self.validation_y,
                                                                                                  self.predicted_validation_y)

    def fullAssessment(self, **kwargs):
        print("Not Yet Implemented")

    def makePrediction(self, **kwargs):
        _observed = kwargs.get('observed', None)
        if _observed is not None:
            predicted = self.internalModel.predict(_observed.T)
            return predicted.T
        else:
            print("Error: Please Supply Observed Neural Activity to Generate Predicted Labels")

    def makeAllPredictions(self):
        self.predicted_testing_y = self.makePrediction(observed=self.testing_x)
        if len(self.data_splits) == 3:
            self.predicted_validation_y = self.makePrediction(observed=self.validation_x)


class LinearRegression(DecodingModule):
    # Class for easily managing Wiener Filter Decoding / Linear Regression
    # Note inheritances from generic Decoder Module class
    def __init__(self, **kwargs):
        # noinspection PyArgumentList
        super().__init__(**kwargs)
        self.internalModel = WienerFilterDecoder()
        # Instance Performance Metrics
        self.ModelPerformance = PerformanceMetrics("Regression", len(self.data_splits))
        print("Instanced Wiener Filter")

    def fitModel(self, **kwargs):
        _fit_intercept = kwargs.get('fit_intercept', False)
        _n_jobs = kwargs.get('n_jobs', None)
        print("Fitting Wiener Filter...")
        self.internalModel.fit(self.training_x.T, self.training_y.T, fit_intercept=_fit_intercept, n_jobs=_n_jobs)
        self.predicted_training_y = self.makePrediction(observed=self.training_x)
        # noinspection PyArgumentList
        print("Finished")

    def assessFit(self, **kwargs):
        _multioutput = kwargs.get('multioutput', "uniform_average")
        self.ModelPerformance[('r2', 'training')] = metrics.r2_score(self.training_y, self.predicted_training_y, multioutput=_multioutput)
        print("Training R2 is " + str(self.ModelPerformance[('r2', 'training')]))

    def commonAssessment(self, **kwargs):
        _flag_valid = kwargs.get('flag_valid', False)

        # R2
        _multioutput = kwargs.get('multioutput', "uniform_average")
        if self.ModelPerformance[('r2', 'training')] is None:
            self.ModelPerformance[('r2', 'training')] = metrics.r2_score(self.training_y,
                                                                         self.predicted_training_y,
                                                                         multioutput=_multioutput)
        if self.ModelPerformance[('r2', 'testing')] is None:
            self.ModelPerformance[('r2', 'testing')] = metrics.r2_score(self.testing_y,
                                                                         self.predicted_testing_y,
                                                                         multioutput=_multioutput)
        if _flag_valid:
            if self.ModelPerformance[('r2', 'validation')] is None:
                self.ModelPerformance[('r2', 'validation')] = metrics.r2_score(self.validation_y,
                                                                                self.predicted_validation_y,
                                                                                _multioutput=_multioutput)
        # mean_absolute_error
        if self.ModelPerformance[('mean_absolute_error', 'training')] is None:
            self.ModelPerformance[('mean_absolute_error', 'training')] = metrics.mean_absolute_error(self.training_y,
                                                                                                self.predicted_training_y)
        if self.ModelPerformance[('mean_absolute_error', 'testing')] is None:
            self.ModelPerformance[('mean_absolute_error', 'testing')] = metrics.mean_absolute_error(self.testing_y,
                                                                                               self.predicted_testing_y)
        if _flag_valid:
            if self.ModelPerformance[('mean_absolute_error', 'validation')] is None:
                self.ModelPerformance[('mean_absolute_error', 'validation')] = metrics.mean_absolute_error(self.validation_y,
                                                                                                      self.predicted_validation_y)

        # mean_squared_error
        if self.ModelPerformance[('mean_squared_error', 'training')] is None:
            self.ModelPerformance[('mean_squared_error', 'training')] = metrics.mean_squared_error(self.training_y,
                                                                                                    self.predicted_training_y)
        if self.ModelPerformance[('mean_squared_error', 'testing')] is None:
            self.ModelPerformance[('mean_squared_error', 'testing')] = metrics.mean_squared_error(self.testing_y,
                                                                                                   self.predicted_testing_y)
        if _flag_valid:
            if self.ModelPerformance[('mean_squared_error', 'validation')] is None:
                self.ModelPerformance[('mean_squared_error', 'validation')] = metrics.mean_squared_error(self.validation_y,
                                                                                                          self.predicted_validation_y)

    def fullAssessment(self, **kwargs):
        print("Not Yet Implemented")

    def makePrediction(self, **kwargs):
        _observed = kwargs.get('observed', None)
        if _observed is not None:
            predicted = self.internalModel.predict(_observed.T)
            return predicted.T
        else:
            print("Error: Please Supply Observed Neural Activity to Generate Predicted Labels")

    def makeAllPredictions(self):
        self.predicted_testing_y = self.makePrediction(observed=self.testing_x)
        if len(self.data_splits) == 3:
            self.predicted_validation_y = self.makePrediction(observed=self.validation_x)


class LinearNonLinearRegression(DecodingModule):
    # Class for easily managing Wiener Filter Decoding
    # The Wiener Filter is imported from the Neural Decoding package
    # from Joshua Glaser while in Kording Lab
    # Note inheritances from generic Decoder Module class
    def __init__(self, **kwargs):
        # noinspection PyArgumentList
        super().__init__(**kwargs)
        _degrees = kwargs.get("degree", 3)
        self.ModelPerformance = PerformanceMetrics("Regression", len(self.data_splits))
        self.internalModel = WienerCascadeDecoder(degree=_degrees)
        print("Instanced Wiener Cascade with a degree of " + _degrees)

    def fitModel(self, **kwargs):
        print("Fitting Wiener Cascade...")
        self.internalModel.fit(self.training_x, self.training_y)
        print("Finished")

    def assessFit(self, **kwargs):
        _compare_testing = kwargs.get('testing', True)
        _compare_validation = kwargs.get('validation', False)
        if _compare_testing:
            self.ModelPerformance.r2 = get_R2(self.testing_y, self.predicted_testing_y)
        if _compare_validation:
            self.ModelPerformance.r2 = get_R2(self.validation_y, self.predicted_validation_y)

    def makePrediction(self, **kwargs):
        _observed = kwargs.get('observed', None)
        if _observed is not None:
            predicted = self.internalModel.predict(_observed)
            return predicted
        else:
            print("Error: Please Supply Observed Neural Activity to Generate Predicted Labels")


def PerformanceMetrics(Type, NumberOfSplits):
    # dict constructor for a variety of performance metrics
    # Methods to calculate secondary & tertiary measures from primary measures

    # construct
    ModelPerformance = dict()

    # Metrics tiered by Most Common, Common, Uncommon
    if Type == "Regression":
        _common_metrics_list = ['r2', 'mean_absolute_error', 'mean_squared_error']
        # Common
        # r2 = None # R^2 /  Coefficient of Determination
        # mean_absolute_error = None # Mean Absolute Error

        #_uncommon_metrics_list = ['median_absolute_error', 'max_error', 'explained_variance_score']
        # Uncommon
        # median_absolute_error = None # Median Absolute Error
        # max_error = None # Max Error
        # mean_squared_error = None  # Mean Squared Error
        # explained_variance_score = None  # Explained Variance Regression Score

        # Now I append. I realize this was kinda pointless, but it makes sense
        # purely from a documentation perspective. It seems too much work to

        _metrics_list = _common_metrics_list #+ _uncommon_metrics_list
        if NumberOfSplits == 2:
            _tuples_list = ['training', 'testing']
        elif NumberOfSplits == 3:
            _tuples_list = ['training', 'testing', 'validation']
        else:
            print("More than three splits is not yet implemented.")
            print("Using default train/test")
            _tuples_list = ['training', 'testing']
        _combined = list(itertools.product(_metrics_list, _tuples_list))

        for key in _combined:
            ModelPerformance[key] = None
        return ModelPerformance

    elif Type == "Classification":

        _most_common_metrics_list = ['accuracy', 'precision', 'recall', 'f1']
        # self.accuracy = None  # (tp+tn)/(tp+fn+fp+tn)
        # self.precision = None  # PPV or Precision, = tn/(tp+fp)
        # self.recall = None  # Sensitivity, TPR, Recall, = tp/(tp+fn)
        # self.f1 = None # F-Score, F-Measure

        _common_metrics_list = ['balanced_accuracy', 'AUC', 'AUC_PR']
        # self.balanced_accuracy = None  # Balanced Accuracy
        # self.AUC = None  # Area Under the Curve of Receiver Operating Characteristic
        # ROC = TPR vs. FPR
        # self.AUC_PR = None  # Area Under the Curve of Precision-Recall Curve

        #_uncommon_metrics_list = ['specificity', 'fpr', 'fnr', 'tp', 'fn', 'fp', 'tn', 'rpp', 'rnp',
                                  #'ecost', 'markedness', 'informedness']
        # self.specificity = None  # TNR, Specificity, = tn/(tn+fp)
        # self.fpr = None  # FPR, Fallout, = fp/(tn+fp)
        # self.fnr = None  # FNR, Miss, = fn/(tp+fn)
        # self.tp = None  # True Positive
        # self.fn = None  # False Negative
        # self.fp = None  # False Positive
        # self.tn = None  # True Negative
        # self.rpp = None  # Rate of Positive Predictions
        # = (tp+fp)/(tp+fn+fp+tn)
        # self.rnp = None  # Rate of Negative Predictions
        # = (tn+fn) / (tp+fn+fp+tn)
        # self.ecost = None # Expected Cost
        # ecost = (tp*Cost(P|P)+fn*Cost(N|P)+fp* Cost(P|N)+tn*Cost(N|N))/(tp+fn+fp+tn)
        # self.markedness = None # Markedness
        # self.informedness = None # Informedness

        _metrics_list = _most_common_metrics_list + _common_metrics_list #+ _uncommon_metrics_list
        if NumberOfSplits == 2:
            _tuples_list = ['training', 'testing']
        elif NumberOfSplits == 3:
            _tuples_list = ['training', 'testing', 'validation']
        else:
            print("More than three splits is not yet implemented.")
            print("Using default train/test")
            _tuples_list = ['training', 'testing']
        _combined = list(itertools.product(_metrics_list, _tuples_list))

        for key in _combined:
            ModelPerformance[key] = None
        return ModelPerformance
