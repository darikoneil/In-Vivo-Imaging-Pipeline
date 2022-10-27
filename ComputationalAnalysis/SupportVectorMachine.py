# generic
import numpy as np
import pickle as pkl

# Model Specific
from sklearn import SVM

# Generics from sk-learn
from sklearn import metrics
from sklearn import model_selection

# Inheritance <-
from DecodingAnalysis import DecodingModule, PerformanceMetrics


class SupperVectorMachine(DecodingModule):
    # Class for easily managing SVM
    # Simply sklearn + interfacing functions for convenience
    def __init__(self, **kwargs):
        # noinspection PyArgumentList
        super().__init__(**kwargs)
        self.ModelPerformance = PerformanceMetrics("Classification", len(self.data_splits))
        # noinspection PyArgumentList
        self.internalModel = svm.SVC(**kwargs)
        print("Instanced Support Vector Classification")

    def fitModel(self):
        print("Fitting SVM...")
        self.internalModel.fit(self.training_x, self.training_y)
        print("Finished.")

    def assessFit(self, **kwargs):
        self.predicted_training_y = self.makePrediction(observed=self.training_x)
        self.ModelPerformance[('accuracy', 'training')] = self.internalModel.model.score(self.training_x, self.training_y)
        print("Training Classification Accuracy is " + str(self.ModelPerformance[('accuracy', 'training')]))

    def makePrediction(self, **kwargs):
        _observed = kwargs.get('observed', None)
        if _observed is not None:
            predicted = self.internalModel.predict(_observed)
            return predicted
        else:
            print("Error: Please Supply Observed Neural Activity to Generate Predicted Labels")

    def makeAllPredictions(self):
        self.predicted_testing_y = self.makePrediction(observed=self.testing_x)
        if len(self.data_splits) == 3:
            self.predicted_validation_y = self.makePrediction(observed=self.validation_x)

    def commonAssessment(self, **kwargs):
        if len(self.data_splits) == 3:
            _flag_valid = True

        if self.ModelPerformance[('accuracy', 'training')] is None:
            self.ModelPerformance[('accuracy', 'training')] = self.internalModel.model.score(self.training_x,
                                                                                             self.training_y)
        if self.ModelPerformance[('accuracy', 'testing')] is None:
            self.ModelPerformance[('accuracy', 'testing')] = self.internalModel.model.score(self.testing_x,
                                                                                            self.testing_y)
