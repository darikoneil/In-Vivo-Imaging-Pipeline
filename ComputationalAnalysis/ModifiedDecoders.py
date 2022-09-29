import numpy as np
from sklearn import linear_model


class WienerFilterDecoder(object):

    """
    Class for the Wiener Filter Decoder

    There are no parameters to set.

    This simply leverages the scikit-learn linear regression.
    """

    def __init__(self):
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
        self.model = linear_model.LinearRegression(fit_intercept=_fit_intercept,
                                                   n_jobs=_n_jobs)
        self.model.fit(X_flat_train, y_train) #Train the model


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

        y_test_predicted=self.model.predict(X_flat_test) #Make predictions
        return y_test_predicted