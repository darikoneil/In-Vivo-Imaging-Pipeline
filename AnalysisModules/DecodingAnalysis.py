import numpy as np
import pickle as pkl

# Imports for classes in file organized by class
# Not sure if I should use conditional imports...
# More graceful failure if certain package not installed
# But I don't want many duplicate imports if I make many decoders
# So right now, let's follow PEP8

# WienerFilterDecoder
from Neural_Decoding.Neural_Decoding.decoders import WienerFilterDecoder

# WienerCascadeDecoder
from Neural_Decoding.Neural_Decoding.decoders import WienerCascadeDecoder

# Metrics functions from Neural Decoding Package from Joshua Glaser while in Kording Lab
from Neural_Decoding.Neural_Decoding.metrics import get_R2


class DecodingModule:
    # Generic decoding module super-class with conserved methods, properties,
    # and methods with conserved names but different implementation.
    def __init__(self, **kwargs):
        # parse inputs
        self.neural_data = kwargs.get('NeuralData', None)
        self.feature_data = kwargs.get('FeatureData', None)
        self.label_data = kwargs.get('LabelData', None)
        self.data_splits = kwargs.get('DataSplits', [0.8, 0.2])

        # Initialization
        self.training_x = None
        self.training_y = None
        self.testing_x = None
        self.testing_y = None
        self.predicted_testing_y = None
        self.validation_x = None
        self.validation_y = None
        self.predicted_validation_y = None

        # Instance Performance Metrics
        self.ModelPerformance = PerformanceMetrics()

    def splitData(self):
        if self.neural_data_organization == "Matrix Org":
            _training_frames = int(self.data_splits[0] * self.num_frames)
            if len(self.data_splits) == 2:
                self.training_x = self.neural_data[:, 0:_training_frames]
                self.training_y = self.neural_data[:, 0:_training_frames]
                self.testing_x = self.neural_data[:, _training_frames:self.num_frames+1]
                self.testing_y = self.neural_data[:, _training_frames:self.num_frames+1]
            elif len(self.data_splits) == 3:
                self.training_x = self.neural_data[:, 0:_training_frames]
                self.training_y = self.neural_data[:, 0:_training_frames]
                self.testing_x = None
                self.testing_y = None
                self.validation_x = None
                self.validation_y = None

    def fitModel(self, **kwargs):
        print("1st")

    def assessModel(self, **kwargs):
        print("2nd)")

    def makePrediction(self, **kwargs):
        print("3rd")

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


class PerformanceMetrics:
    # Container class for a variety of performance metrics
    # Methods to calculate secondary & tertiary measures from primary measures
    def __init__(self):
        # Primary
        self.tp = None # True Positive
        self.fn = None # False Negative
        self.fp = None # False Positive
        self.tn = None # True Negative
        self.r = None  # R
        self.r2 = None # R-Squared
        self.hits = None # tp+fp

        # Secondary
        self.accuracy = None  # (tp+tn)/(tp+fn+fp+tn)
        self.precision = None  # PPV or Precision, = tn/(tp+fp)
        self.recall = None  # Sensitivity, TPR, Recall, = tp/(tp+fn)
        self.specificity = None  # TNR, Specificity, = tn/(tn+fp)

        # Tertiary
        self.balanced_accuracy = None # Balanced Accuracy
        self.fpr = None # FPR, Fallout, = fp/(tn+fp)
        self.fnr = None # FNR, Miss, = fn/(tp+fn)
        self.rpp = None  # Rate of Positive Predictions
        # = (tp+fp)/(tp+fn+fp+tn)
        self.rnp = None  # Rate of Negative Predictions
        # = (tn+fn) / (tp+fn+fp+tn)
        self.ecost = None # Expected Cost
        # ecost = (tp*Cost(P|P)+fn*Cost(N|P)+fp* Cost(P|N)+tn*Cost(N|N))/(tp+fn+fp+tn)
        self.markedness = None # Markedness
        self.informedness = None # Informedness
        self.AUC = None # Area Under the Curve of Receiver Operating Characteristic
        # ROC = TPR vs. FPR
        self.AUC_PR = None # Area Under the Curve of Precision-Recall Curve


class WienerFilter(DecodingModule):
    # Class for easily managing Wiener Filter Decoding
    # The Wiener Filter is imported from the Neural Decoding package
    # from Joshua Glaser while in Kording Lab
    # Note inheritances from generic Decoder Module class
    def __init__(self, **kwargs):
        # noinspection PyArgumentList
        super().__init__(**kwargs)
        self.internalModel = WienerFilterDecoder()


    def fitModel(self, **kwargs):
        print("Fitting Wiener Filter")
        self.internalModel.fit(self.training_x, self.training_y)

    def assessModel(self, **kwargs):
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


class WienerCascade(DecodingModule):
    # Class for easily managing Wiener Filter Decoding
    # The Wiener Filter is imported from the Neural Decoding package
    # from Joshua Glaser while in Kording Lab
    # Note inheritances from generic Decoder Module class
    def __init__(self, **kwargs):
        # noinspection PyArgumentList
        super().__init__(**kwargs)
        _degrees = kwargs.get("degree", 3)
        print("Instancing Wiener Cascade with a degree of " + _degrees)
        self.internalModel = WienerCascadeDecoder(degree=_degrees)

    def fitModel(self, **kwargs):
        print("Fitting Wiener Cascade")
        self.internalModel.fit(self.training_x, self.training_y)

    def assessModel(self, **kwargs):
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
