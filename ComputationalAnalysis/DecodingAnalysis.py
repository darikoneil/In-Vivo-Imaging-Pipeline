import numpy as np
import pickle as pkl
import pathlib
# Imports for classes in file organized by class
# Not sure if I should use conditional imports...
# More graceful failure if certain package not installed
# But I don't want many duplicate imports if I make many decoders
# So right now, let's follow PEP8

# Logistic
from ComputationalAnalysis.NewDecoders import LogisticRegressionDecoder

# WienerFilterDecoder
from ComputationalAnalysis.ModifiedDecoders import WienerFilterDecoder

# WienerCascadeDecoder
from Neural_Decoding.Neural_Decoding.decoders import WienerCascadeDecoder

# Long Short-Term Memory Decoder
from Neural_Decoding.Neural_Decoding.decoders import LSTMRegression

# Metrics functions from Neural Decoding Package from Joshua Glaser while in Kording Lab
from Neural_Decoding.Neural_Decoding.metrics import get_R2

# Metrics from sk-learn
from sklearn import metrics

# Metrics Dictionary Constructor
import itertools

# Visualization
from matplotlib import pyplot as plt
from ImagingAnalysis.StaticPlotting import plotROC


class DecodingModule:
    """
    This a super class passing conserved functions for decoding modules

    **Properties**
        | **imported_neural_organization** : the structure of the passed neural data
        | **imported_feature_organization** : the structure of the passed feature data
    """
    # Generic decoding module super-class with conserved methods, properties,
    # and methods with conserved names but different implementation.
    def __init__(self, **kwargs):
        """

        :key NeuralData: Neural Data, can be in the form of Neurons x Frames or Trials x Neurons x Frames
        """
        # parse inputs
        self.neural_data = kwargs.get('NeuralData', None)
        self.feature_data = kwargs.get('FeatureData', None)
        self.label_data = kwargs.get('LabelData', None)
        self.data_splits = kwargs.get('DataSplits', [0.8, 0.2])
        self.covariance_matrix = kwargs.get('CovarianceMatrix', None)
        self._num_trials = kwargs.get('Trials', None)
        self.trial_index = kwargs.get('TrialIndex', None)
        _feature_data_file = kwargs.get('FeatureDataFile', None)
        _label_data_file = kwargs.get('LabelDataFile', None)

        # load if necessary
        if _feature_data_file is not None or _label_data_file is not None:
            self.loadFeaturesLabels(_feature_data_file, _label_data_file)

        # properties
        self._imported_neural_data_org = None
        self.imported_neural_organization = self.neural_data
        self._imported_feature_data_org = None
        self.imported_feature_organization = self.feature_data

        # indices
        self.shuffle_index = None
        self.trial_order = None
        if self.num_trials is not None:
            self.trial_order = np.arange(self.num_trials)

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

        # Verbosity
        self.structural_report()

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
        _training_frames = int(self.data_splits[0] * self.num_frames)
        if len(self.data_splits) == 2:
            print("Splitting data in training & testing sets")
            print("Data splits are: " +
                    str(self.data_splits[0]*100) + "% training" +
                    " vs " + str(self.data_splits[1]*100) + "% testing")
            self.training_x = self.neural_matrix[:, 0:_training_frames]
            self.training_y = self.label_data[0:_training_frames]
            self.testing_x = self.neural_matrix[:, _training_frames:self.num_frames+1]
            self.testing_y = self.label_data[_training_frames:self.num_frames+1]
        elif len(self.data_splits) == 3:
            self.training_x = self.neural_matrix[:, 0:_training_frames]
            self.training_y = self.neural_matrix[:, 0:_training_frames]
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
                if key[0] is not "ROC" and key[0] is not "PR":
                    print(key, ' : ', value)

    def plotROCs(self, **kwargs):

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(111)
        ax1.set_title("Receiver Operating Characteristic")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        plotROC(self.ModelPerformance[('ROC', 'training')][0], self.ModelPerformance[('ROC', 'training')][1], ax=ax1)
        plotROC(self.ModelPerformance[('ROC', 'testing')][0], self.ModelPerformance[('ROC', 'testing')][1], color="#ff4e4b", ax=ax1)

        ax1.plot([0, 1], [0, 1], color="black", lw=3, ls='--', alpha=0.95)
        ax1.legend(['Training', 'Testing', 'No-Skill'])

        if ('ROC', 'validation') in self.ModelPerformance:
            plotROC(self.ModelPerformance['ROC', 'validation'][0], self.ModelPerformance['ROC', 'validation'][1], color="#40cc8b", ax=ax1)


        plt.show()
    # Here are some useful properties of the neural data that we must access often

    def saveModel(self, OutputFolder):
        print("Saving Model...")
        _output_file = OutputFolder + "//" + "GenericModel"
        _output_pickle = open(_output_file, 'wb')
        pkl.dump(self, _output_pickle)
        _output_pickle.close()
        print("Finished.")

    @property
    def imported_neural_organization(self):
        """
        :rtype: str
        """

        if self._imported_neural_data_org is not None:
            return self._imported_neural_data_org
        else:
            print("Please import neural data")
            return ""


    @imported_neural_organization.setter
    def imported_neural_organization(self, value):
        """
        :param value: Original Neural Data
        """
        if value.shape.__len__() == 2:
            self._imported_neural_data_org = "Neurons x Frames"
        elif value.shape.__len__() == 3:
            self._imported_neural_data_org = "Trials x Neurons x Frames"

    @property
    def imported_feature_organization(self):
        """
        :rtype: str
        """
        if self._imported_feature_data_org is not None:
            return self._imported_feature_data_org
        else:
            print("Please import feature data")
            return ""

    @imported_feature_organization.setter
    def imported_feature_organization(self, value):
        """
        :param value:  Original Feature Data
        """
        if value.shape.__len__() == 2:
            self._imported_feature_data_org = "Features x Frames"
        elif value.shape.__len__() == 3:
            self._imported_feature_data_org = "Trials x Features x Frames"

    @property
    def neural_matrix(self):
        if self.imported_neural_organization == "Trials x Neurons x Frames":
            return np.concatenate(self.neural_data, axis=1)
        elif self.imported_neural_organization == "Neurons x Frames":
            return self.neural_data

    @property
    def feature_matrix(self):
        if self.imported_feature_organization == "Trials x Features x Frames":
            return np.concatenate(self.feature_data, axis=1)
        elif self.imported_feature_organization == "Features x Frames":
            return self.feature_data

    @property
    def num_trials(self):
        """
        :rtype: int
        """

        if self.imported_neural_organization == "Trials x Neurons x Frames":
            return self.neural_data.shape[0]
        elif self._num_trials is not None:
            return self._num_trials
        else:
            print("The number of trials was not specified")
            return int()

    @property
    def num_neurons(self):
        if self.neural_data is not None:
            return self.neural_matrix.shape[0]
        else:
            print("Please import neural data")
            return int()

    @property
    def num_frames(self):
        if self.neural_data is not None:
            return self.neural_matrix.shape[1]
        else:
            print("Please import neural data")
            return int()

    def structural_report(self):
        print("".join(["\nNeural Data was imported in the form ", self.imported_neural_organization, " and contains ",
                       str(self.num_neurons), " neurons, ", str(self.num_frames), " frames, and ",
                       str(self.num_trials), " trials."]))

    def shuffle_trials(self):

        if self.imported_neural_organization == "Trials x Neurons x Frames" \
                and self.imported_feature_organization == "Trials x Features x Frames":
            self.shuffle_index, self.trial_order = self.shuffleByTrialIndex(self.neural_data, self.trial_index)
            _linear_index = np.concatenate(self.shuffle_index, axis=1)
            _neural_data_matrix = self.neural_matrix
            self.neural_data = np.reshape(_neural_data_matrix[:, _linear_index], self.neural_data.shape)
        else:
            print("Neural and Feature Data must be in the form of Trials x _ x Frames")

    @staticmethod
    def shuffleByTrialIndex(NeuralActivityInTrialForm, TrialIndex):
        _num_trials = NeuralActivityInTrialForm.shape[0]
        _frames_per_trial = NeuralActivityInTrialForm.shape[2]
        _num_frames = _num_trials * _frames_per_trial
        _unique_trial_types = np.unique(TrialIndex)
        _num_trial_types = _unique_trial_types.__len__()
        _frame_index = np.reshape(np.arange(_num_frames), (_num_trials, 1, _frames_per_trial))
        shuffle_index = _frame_index.copy()

        _frame_sets = []
        _trial_sets = []
        for _trial_type in _unique_trial_types:
            _trials_of_this_type = [_trial for _trial in range(_num_trials) if TrialIndex[_trial] == _trial_type]
            np.random.shuffle(_trials_of_this_type)
            _trial_sets.append(_trials_of_this_type.copy())
            _frame_sets.append(_frame_index[_trials_of_this_type, :, :])
        _frame_sets = np.asarray(_frame_sets)
        _trial_sets = np.asarray(_trial_sets)

        trial_order = []
        for _group_of_one_trial_each in range(int(_num_trials/_num_trial_types)):
            _offset = _group_of_one_trial_each*_num_trial_types
            for _trial_type in range(_num_trial_types):
                shuffle_index[_trial_type+_offset, :, :] = _frame_sets[_trial_type, _group_of_one_trial_each, :, :]
                trial_order.append(_trial_sets[_trial_type, _group_of_one_trial_each])
        trial_order = np.asarray(trial_order)
        return shuffle_index, trial_order

    @classmethod
    def shuffleFrames(cls, DataInMatrixForm, **kwargs):
        _features = kwargs.get('FeatureData', None)
        _labels = kwargs.get('LabelData', None)
        shuffle_index = np.arange(DataInMatrixForm.shape[1])
        np.random.shuffle(shuffle_index)
        shuffled_data = DataInMatrixForm[:, shuffle_index]
        if _labels is None and _features is None:
            return shuffled_data
        elif _features is not None and _labels is None:
            shuffled_features = _features[:, shuffle_index]
            return shuffled_data, shuffled_features
        elif _features is not None and _labels is not None:
            shuffled_features = _features[:, shuffle_index]
            shuffled_labels = _labels[:, shuffle_index]
            return shuffled_data, shuffled_features, shuffled_labels
        elif _features is None and _labels is not None:
            shuffled_labels = _labels[:, shuffle_index]
            return shuffled_data, shuffled_labels

    @classmethod
    def shuffleEachNeuron(cls, NeuralActivityInMatrixForm):
        _num_neurons, _num_frames = NeuralActivityInMatrixForm.shape
        shuffled_neural_data = np.zeros_like(NeuralActivityInMatrixForm).copy() # cuz paranoid
        shuffle_index = np.arange(_num_frames)

        for _neuron in range(_num_neurons):
            np.random.shuffle(shuffle_index)
            shuffled_neural_data[_neuron, :] = NeuralActivityInMatrixForm[_neuron, shuffle_index]

        return shuffled_neural_data

    @classmethod
    def collapseFeatures(cls, Features, **kwargs):
        _feature_subset = kwargs.get('FeatureSubset', None)

        if _feature_subset is not None:
            Features = Features[:, _feature_subset, :].copy()

        _num_trials, _num_features, _num_frames = Features.shape

        collapsed_features = np.zeros_like(Features).copy()

        for _feature in range(_num_features):
            for _trial in range(_num_trials):
                # Add one to _feature indicator value to account for no-feature trials
                collapsed_features[_trial, _feature, np.where(Features[_trial, _feature, :] == 1)] = _feature+1

        collapsed_features_vectorized = np.full((_num_trials, 1, _num_frames), 0, dtype=np.int32)
        for _trial in range(_num_trials):
            collapsed_features_vectorized[_trial, 0, :] = np.sum(collapsed_features[_trial, :, :], axis=0)


        return collapsed_features_vectorized

    @classmethod
    def loadFeatures(cls, FeatureFile):
        _feature_data_file = FeatureFile
        try:
            _feature_file_ext = pathlib.Path(_feature_data_file).suffix
            if _feature_file_ext == ".npy" or _feature_file_ext == ".npz":
                feature_data = np.load(_feature_data_file, allow_pickle=True)
                return feature_data
            elif _feature_file_ext == ".csv":
                feature_data = np.genfromtxt(_feature_data_file, dtype=int, delimiter=",")
                return feature_data
            else:
                print("Features data in unexpected file type.")
                raise RuntimeError
            # noinspection PyUnresolvedReferences
        except RuntimeError:
            print("Could not load feature data.")


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

    # noinspection PyTypeChecker
    def commonAssessment(self, **kwargs):
        _flag_valid = kwargs.get('flag_valid', False)
        _multi = kwargs.get('multi', False)

        # Ensure We have all our predictions already made
        if self.predicted_training_y is None or self.predicted_testing_y is None:
            self.makeAllPredictions()
        if _flag_valid:
            if self.predicted_validation_y is None:
                self.makeAllPredictions()

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
        if _multi is not True:
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
                self.ModelPerformance[('f1', 'testing')] = metrics.f1_score(self.testing_y, self.predicted_testing_y)
            if _flag_valid:
                if self.ModelPerformance[('f1', 'validation')] is None:
                    self.ModelPerformance[('f1', 'validation')] = metrics.f1_score(self.validation_y,
                                                                                   self.predicted_validation_y)

        # balanced accuracy
        if self.ModelPerformance[('balanced_accuracy', 'training')] is None:
            self.ModelPerformance[('balanced_accuracy', 'training')] = \
                metrics.balanced_accuracy_score(self.training_y, self.predicted_training_y)
        if self.ModelPerformance[('balanced_accuracy', 'testing')] is None:
            self.ModelPerformance[('balanced_accuracy', 'testing')] = \
                metrics.balanced_accuracy_score(self.testing_y, self.predicted_testing_y)

        if _flag_valid:
            if self.ModelPerformance[('balanced_accuracy', 'validation')] is None:
                self.ModelPerformance[('balanced_accuracy', 'validation')] = \
                    metrics.balanced_accuracy_score(self.validation_y, self.predicted_validation_y)

        if _multi is not True:
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
                    self.ModelPerformance[('AUC_PR', 'validation')] = \
                        metrics.average_precision_score(self.validation_y, self.predicted_validation_y)

            if self.ModelPerformance[('ROC', 'training')] is None:
                _probas = self.internalModel.model.decision_function(self.training_x.T)
                self.ModelPerformance[('ROC', 'training')] = metrics.roc_curve(self.training_y, _probas,
                                                                               drop_intermediate=False)

            if self.ModelPerformance[('ROC', 'testing')] is None:
                _probas = self.internalModel.model.decision_function(self.testing_x.T)
                self.ModelPerformance[('ROC', 'testing')] = metrics.roc_curve(self.testing_y, _probas,
                                                                              drop_intermediate=False)

            if _flag_valid:
                if self.ModelPerformance[('ROC', 'validation')] is None:
                    _probas = self.internalModel.model.decision_function(self.validation_x.T)
                    self.ModelPerformance[('ROC', 'validation')] = metrics.roc_curve(self.validation_y,
                                                                                     _probas,
                                                                                     drop_intermediate=False)

            if self.ModelPerformance[('PR', 'training')] is None:
                _probas = self.internalModel.model.decision_function(self.training_x.T)
                self.ModelPerformance[('PR', 'training')] = metrics.precision_recall_curve(self.training_y, _probas)

            if self.ModelPerformance[('PR', 'testing')] is None:
                _probas = self.internalModel.model.decision_function(self.testing_x.T)
                self.ModelPerformance[('PR', 'testing')] = metrics.precision_recall_curve(self.testing_y,
                                                                                          _probas)

            if _flag_valid:
                if self.ModelPerformance[('PR', 'validation')] is None:
                    _probas = self.internalModel.model.decision_function(self.validation_x.T)
                    self.ModelPerformance[('PR', 'validation')] = \
                        metrics.precision_recall_curve(self.validation_y, _probas)

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


class LongShortTermMemoryRegression(DecodingModule):
    # Class for easily managing LSTM Regression
    # Note inheritances from generic Decoder Module class
    def __init__(self, **kwargs):
        # noinspection PyArgumentList
        super().__init__(**kwargs)
        self.internalModel = LSTMRegression()
        # Instance Performance Metrics
        self.ModelPerformance = PerformanceMetrics("Regression", len(self.data_splits))
        print("Instanced LSTM")

    def fitModel(self, **kwargs):
        print("Fitting LSTM")
        self.internalModel.fit(self.training_x, self.training_y)
        print("Finished"
              )

    def makePrediction(self, **kwargs):
        _observed = kwargs.get('observed', None)
        if _observed is not None:
            predicted = self.internalModel.predict(_observed)
            return predicted
        else:
            print("Error: Please Supply Observed Neural Activity to Generate Predicted Labels")

    def makeAllPredictions(self, **kwargs):
        self.predicted_testing_y = self.makePrediction(observed=self.testing_x)
        if len(self.data_splits) == 3:
            self.predicted_validation_y = self.makePrediction(observed=self.validation_x)


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

        _common_metrics_list = ['balanced_accuracy', 'AUC', 'AUC_PR', 'ROC', 'PR']
        # self.balanced_accuracy = None  # Balanced Accuracy
        # self.AUC = None  # Area Under the Curve of Receiver Operating Characteristic
        # ROC = TPR vs. FPR Curves
        # self.AUC_PR = None  # Area Under the Curve of Precision-Recall Curve
        # PR = Precision-Recall Curves


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
