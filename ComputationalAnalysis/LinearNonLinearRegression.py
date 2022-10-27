import numpy as np
import pickle as pkl
from ComputationalAnalysis.DecodingAnalysis import DecodingModule, PerformanceMetrics
from Neural_Decoding.Neural_Decoding.decoders import WienerCascadeDecoder


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