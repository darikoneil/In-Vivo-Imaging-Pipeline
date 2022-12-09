import numpy as np
import pickle as pkl
from Theory.DecodingAnalysis import DecodingModule, PerformanceMetrics
from Neural_Decoding.Neural_Decoding.decoders import LSTMRegression


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