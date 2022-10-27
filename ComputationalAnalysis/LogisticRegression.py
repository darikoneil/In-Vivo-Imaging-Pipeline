import numpy as np
import pickle as pkl
from sklearn import linear_model
from ComputationalAnalysis.DecodingAnalysis import DecodingModule, PerformanceMetrics


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
        self.internalModel.fit(self.training_x, self.training_y, penalty=_penalty,
                               solver=_solver, max_iter=_max_iter)
        print("Finished")

    def assessFit(self, **kwargs):
        self.predicted_training_y = self.makePrediction(observed=self.training_x)
        self.ModelPerformance[('accuracy', 'training')] = self.internalModel.model.score(self.training_x, self.training_y)
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
            self.ModelPerformance[('accuracy', 'training')] = self.internalModel.model.score(self.training_x,
                                                                                             self.training_y)
        if self.ModelPerformance[('accuracy', 'testing')] is None:
            self.ModelPerformance[('accuracy', 'testing')] = self.internalModel.model.score(self.testing_x,
                                                                                            self.testing_y)
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


class LogisticRegressionDecoder(object):

    def __init__(self):
        self.model = None
        return

    def fit(self, x, y, **kwargs):
        _penalty = kwargs.get('penalty', 'l1')
        _solver = kwargs.get('solver', 'liblinear')
        _max_iter = kwargs.get('max_iter', 100000)
        _multi = kwargs.get('multi', 'auto')
        self.model = linear_model.LogisticRegression(penalty=_penalty, solver=_solver, max_iter=_max_iter, multi_class=_multi)
        self.model.fit(x, y)

    def predict(self, x):
        y_predicted = self.model.predict(x)
        return y_predicted



