# generic
import numpy as np
import pickle as pkl

# Model Specific
from sklearn import svm

# Generics from sk-learn
from sklearn import metrics
from sklearn import model_selection

# Inheritance <-
from ComputationalAnalysis.DecodingAnalysis import DecodingModule, PerformanceMetrics


class SVM(DecodingModule):
    # Class for easily managing SVM
    # Simply sklearn + interfacing functions for convenience
    def __init__(self, **kwargs):
        # noinspection PyArgumentList
        super().__init__(**kwargs)
        self.ModelPerformance = PerformanceMetrics("Classification", len(self.data_splits))
        # noinspection PyArgumentList
        self.internalModel = svm.SVC(**self.kwargs)
        print("Instanced Support Vector Classification")

    def fitModel(self):
        print("Fitting SVM...")
        self.internalModel.fit(self.training_x, self.training_y)
        print("Finished.")

    def assessFit(self, **kwargs):
        self.predicted_training_y = self.makePrediction(observed=self.training_x)
        self.ModelPerformance[('accuracy', 'training')] = self.internalModel.score(self.training_x, self.training_y)
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

        if self.predicted_testing_y is None:
            self.makeAllPredictions()

        if len(self.data_splits) == 3:
            _flag_valid = True

        if self.ModelPerformance[('accuracy', 'training')] is None:
            self.ModelPerformance[('accuracy', 'training')] = self.internalModel.score(self.training_x,
                                                                                             self.training_y)
        if self.ModelPerformance[('accuracy', 'testing')] is None:
            self.ModelPerformance[('accuracy', 'testing')] = self.internalModel.score(self.testing_x,
                                                                                            self.testing_y)

        if self.ModelPerformance[('precision', 'training')] is None:
            self.ModelPerformance[('precision', 'training')] = \
                metrics.precision_score(self.training_y, self.predicted_training_y)
        if self.ModelPerformance[('precision', 'testing')] is None:
            self.ModelPerformance[('precision', 'testing')] = metrics.precision_score(self.testing_y,
                                                                                      self.predicted_testing_y)

        if self.ModelPerformance[('recall', 'training')] is None:
            self.ModelPerformance[('recall', 'training')] = metrics.recall_score(self.training_y,
                                                                                 self.predicted_training_y)
        if self.ModelPerformance[('recall', 'testing')] is None:
            self.ModelPerformance[('recall', 'testing')] = metrics.recall_score(self.testing_y,
                                                                                self.predicted_testing_y)

        if self.ModelPerformance[('f1', 'training')] is None:
            self.ModelPerformance[('f1', 'training')] = metrics.f1_score(self.training_y,
                                                                         self.predicted_training_y)
        if self.ModelPerformance[('f1', 'testing')] is None:
            self.ModelPerformance[('f1', 'testing')] = \
                metrics.f1_score(self.testing_y, self.predicted_testing_y)

        if self.ModelPerformance[('balanced_accuracy', 'training')] is None:
            self.ModelPerformance[('balanced_accuracy', 'training')] = \
                metrics.balanced_accuracy_score(self.training_y, self.predicted_training_y)
        if self.ModelPerformance[('balanced_accuracy', 'testing')] is None:
            self.ModelPerformance[('balanced_accuracy', 'testing')] = \
                metrics.balanced_accuracy_score(self.testing_y, self.predicted_testing_y)

        if self.ModelPerformance[('AUC', 'training')] is None:
            self.ModelPerformance[('AUC', 'training')] = metrics.roc_auc_score(self.training_y,
                                                                               self.predicted_training_y)
        if self.ModelPerformance[('AUC', 'testing')] is None:
            self.ModelPerformance[('AUC', 'testing')] = metrics.roc_auc_score(self.testing_y,
                                                                              self.predicted_testing_y)

        if self.ModelPerformance[('AUC_PR', 'training')] is None:
            self.ModelPerformance[('AUC_PR', 'training')] = metrics.average_precision_score(self.training_y,
                                                                                            self.predicted_training_y)
        if self.ModelPerformance[('AUC_PR', 'testing')] is None:
            self.ModelPerformance[('AUC_PR', 'testing')] = \
            metrics.average_precision_score(self.testing_y, self.predicted_testing_y)

        if self.ModelPerformance[('ROC', 'training')] is None:
            _probas = self.internalModel.decision_function(self.training_x)
            self.ModelPerformance[('ROC', 'training')] = metrics.roc_curve(self.training_y, _probas,
                                                                           drop_intermediate=False)
        if self.ModelPerformance[('ROC', 'testing')] is None:
            _probas = self.internalModel.decision_function(self.testing_x)
            self.ModelPerformance[('ROC', 'testing')] = metrics.roc_curve(self.testing_y, _probas,
                                                                          drop_intermediate=False)

        if self.ModelPerformance[('PR', 'training')] is None:
            _probas = self.internalModel.decision_function(self.training_x)
            self.ModelPerformance[('PR', 'training')] = metrics.precision_recall_curve(self.training_y, _probas)
        if self.ModelPerformance[('PR', 'testing')] is None:
            _probas = self.internalModel.decision_function(self.testing_x)
            self.ModelPerformance[('PR', 'testing')] = metrics.precision_recall_curve(self.testing_y,
                                                                                      _probas)
