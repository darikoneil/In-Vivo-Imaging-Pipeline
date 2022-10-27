import numpy as np
import pickle as pkl
from sklearn import linear_model
from ComputationalAnalysis.DecodingAnalysis import DecodingModule, PerformanceMetrics


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
                # noinspection PyArgumentList
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


class WienerFilterDecoder(object):
    """
    Class for the Wiener Filter Decoder

    There are no parameters to set.

    This simply leverages the scikit-learn linear regression.
    """

    def __init__(self):
        self.model = None
        return

    def fit(self, X_flat_train, y_train, **kwargs):
        """
        Train Wiener Filter Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """
        _fit_intercept = kwargs.get('fit_intercept', False)
        _n_jobs = kwargs.get('n_jobs', None)
        # noinspection PyAttributeOutsideInit
        self.model = linear_model.LinearRegression(fit_intercept=_fit_intercept,
                                                   n_jobs=_n_jobs)
        self.model.fit(X_flat_train, y_train)  # Train the model

    def predict(self, X_flat_test):
        """
        Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_flat_test)  # Make predictions
        return y_test_predicted