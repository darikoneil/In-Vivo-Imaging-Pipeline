import numpy as np
import pickle as pkl
import pathlib

# Warnin', all code in this folder is generally gross and hasty
# Metrics functions from Neural Decoding Package from Joshua Glaser while in Kording Lab
from Neural_Decoding.Neural_Decoding.metrics import get_R2

# Generics from sk-learn
from sklearn import metrics
from sklearn import model_selection

# Metrics Dictionary Constructor
import itertools

# Visualization
from matplotlib import pyplot as plt
from Imaging.StaticPlotting import plotROC


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
        self.data_splits = kwargs.get('DataSplits', [0.8, 0.2])
        self._num_trials = kwargs.get('Trials', None)
        self.trial_index = kwargs.get('TrialIndex', None)
        _feature_data_file = kwargs.get('FeatureDataFile', None)


        # clean keys from kwargs to handle unexpected keyword arguments downstream
        self.kwargs = dict() # Paper Trail
        self.kwargs.update(kwargs)
        self.cleanKwargs()

        # load if necessary
        if _feature_data_file is not None:
            self.loadFeaturesFile(_feature_data_file)

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

    def splitData(self, **kwargs):
        _shuffle = kwargs.get("shuffle", True)
        _stratify = kwargs.get("stratify", None)

        if _shuffle is None and _stratify is not None:
            AssertionError("If shuffle is False then stratify must be None")

        print("Splitting data in training & testing sets")
        print("Data splits are: " +
                str(self.data_splits[0]*100) + "% training" +
                " vs " + str(self.data_splits[1]*100) + "% testing")
        self.training_x, self.testing_x, self.training_y, self.testing_y = \
            model_selection.train_test_split(self.neural_matrix.T, self.feature_matrix.T,
                                                test_size=self.data_splits[1],
                                                train_size=self.data_splits[0])

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

    def loadFeaturesFile(self, _feature_data_file):
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

    def validate_data_sets(self):
        if self.training_x.shape[0] <= self.training_x.shape[1] and self.training_y.shape[0] <= self.training_y.shape[1]:
            AssertionError("Data splits are in the wrong shape. Check your code.")
        else:
            return True

    # noinspection PyMethodMayBeStatic
    def cleanKwargs(self):
        self.kwargs.pop("NeuralData", None)
        self.kwargs.pop("FeatureData", None)
        self.kwargs.pop("LabelData", None)
        self.kwargs.pop("DataSplits", None)
        self.kwargs.pop("CovarianceMatrix", None)
        self.kwargs.pop("Trials", None)
        self.kwargs.pop("TrialIndex", None)
        self.kwargs.pop("FeatureDataFile", None)
        self.kwargs.pop("LabelDataFile", None)

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
        if value.shape.__len__() == 1:
            self._imported_feature_data_org = "Features x Frames"
        elif value.shape.__len__() == 2:
            self._imported_feature_data_org = "Features x Frames"
        elif value.shape.__len__() == 3:
            self._imported_feature_data_org = "Trials x Features x Frames"

    @property
    def neural_matrix(self):
        if self.imported_neural_organization == "Trials x Neurons x Frames":
            return np.hstack(self.neural_data)
        elif self.imported_neural_organization == "Neurons x Frames":
            return self.neural_data

    @property
    def neural_tensor(self):
        if self.imported_neural_organization == "Trials x Neurons x Frames":
            return self.neural_data
        elif self.imported_neural_organization == "Neurons x Frames":
            return np.array(np.hsplit(self.neural_data, self.num_trials))

    @property
    def feature_matrix(self):
        if self.imported_feature_organization == "Trials x Features x Frames":
            return np.hstack(self.feature_data)
        elif self.imported_feature_organization == "Features x Frames":
            return self.feature_data

    @property
    def feature_tensor(self):
        if self.imported_feature_organization == "Trials x Features x Frames":
            return self.feature_data
        elif self.imported_feature_organization == "Features x Frames":
            return np.array(np.hsplit(self.feature_data, self.num_trials))

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

    def dep_shuffle_trials(self):
        if not self.imported_neural_organization == "Trials x Neurons x Frames":
            raise AssertionError("Neural data must be in the form Trials x Neurons x Frames")
        if not self.imported_feature_organization == "Trials x Features x Frames":
            raise AssertionError("Feature data must be in the form Trials x Features x Frames")

        self.shuffle_index, self.trial_order = self.shuffleByTrialIndex(self.neural_data, self.trial_index)
        self.neural_data = self.neural_data[self.trial_order, :, :]
        self.feature_data = self.feature_data[self.trial_order, :, :]

        self.validate_data_sets()

    def shuffle_trials(self):
        _trial_groups = self.shuffled_trials_by_group(self.num_trials, self.trial_index)
        self.training_x, self.testing_x, self.training_y, self.testing_y = self.split_by_trials(
            self.neural_tensor, self.feature_tensor, self.data_splits, _trial_groups)

        # End with shuffling the training data in case any particular method is vulnerable to sample order
        _frame_shuffle_index = np.arange(0, self.training_x.shape[0], 1)
        np.random.shuffle(_frame_shuffle_index)
        self.training_x = self.training_x[_frame_shuffle_index, :]
        if self.training_y.shape.__len__() == 1:
            self.training_y = self.training_y[_frame_shuffle_index]
        else:
            self.training_y = self.training_y[_frame_shuffle_index, :]

        self.validate_data_sets()

    def shuffle_trial_labels(self):
        if not self.imported_neural_organization == "Trials x Neurons x Frames":
            raise AssertionError("Neural data must be in the form Trials x Neurons x Frames")
        if not self.imported_feature_organization == "Trials x Features x Frames":
            raise AssertionError("Feature data must be in the form Trials x Features x Frames")

        self.shuffle_index, self.trial_order = self.shuffleByTrialIndex(self.neural_data, self.trial_index)
        self.feature_data = self.feature_data[self.trial_order, :, :]

    @classmethod
    def split_by_trials(cls, NeuralDataTensor, FeatureDataTensor, DataSplits, TrialGroups):
        assert(NeuralDataTensor.shape[0] == FeatureDataTensor.shape[0] ==
               TrialGroups[0].__len__() * TrialGroups.__len__())

        _training_groups = TrialGroups.__len__() * DataSplits[0]

        if not isinstance(_training_groups, int):
            _training_groups = int(_training_groups)
        _training_group_index = list(sum(TrialGroups[0:_training_groups], ()))
        _testing_group_index = list(sum(TrialGroups[_training_groups:], ()))

        training_x, testing_x = NeuralDataTensor[_training_group_index, :, :],  NeuralDataTensor[
                                                                                _testing_group_index, :, :]

        if FeatureDataTensor.shape.__len__() == 2:
            training_y, testing_y = FeatureDataTensor[_training_group_index, :], FeatureDataTensor[
                                                                                    _testing_group_index, :]
        else:
            training_y, testing_y = FeatureDataTensor[_training_group_index, :, :], FeatureDataTensor[
                                                                                 _testing_group_index, :, ]
        training_x = np.hstack(training_x).T
        testing_x = np.hstack(testing_x).T
        training_y = np.hstack(training_y).T
        testing_y = np.hstack(testing_y).T
        return training_x, testing_x, training_y, testing_y

    @staticmethod
    def shuffled_trials_by_group(NumTrials, TrialIndex):
        _unique_trial_types = np.unique(TrialIndex)
        _num_trial_types = _unique_trial_types.__len__()
        _num_trial_groups = NumTrials/_num_trial_types
        _trial_type_index = []
        for _type in range(_num_trial_types):
            _trial_type_index.append(np.where(TrialIndex == _unique_trial_types[_type])[0])
            np.random.shuffle(_trial_type_index[_type])
        trial_groups = list(zip(*[_trial_set for _trial_set in _trial_type_index]))
        return trial_groups

    @staticmethod
    def createTrialIndicator(NumTrials, FramesPerTrial):
        return [np.full((1, FramesPerTrial), i, dtype=int) for i in range(NumTrials)]

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
                # shuffle_index[_trial_type+_offset, :, :] = _frame_sets[_trial_type, _group_of_one_trial_each, :, :]
                trial_order.append(_trial_sets[_trial_type, _group_of_one_trial_each])
        trial_order = np.asarray(trial_order)
        return shuffle_index, trial_order

    @classmethod
    def shuffleLabels(cls, Labels):
        assert(Labels.shape.__len__() == 1)
        _shuffle_index = np.arange(Labels.shape[0])
        np.random.shuffle(_shuffle_index)
        return Labels[_shuffle_index]

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
