import numpy as np


def generateCovarianceMatrix(NeuralActivity, ActivityMeasure, **kwargs):
    _num_neurons = kwargs.get('NumNeurons', NeuralActivity.shape[0])
    _num_frames = kwargs.get('NumFrames', NeuralActivity.shape[1])
    _bin_length = kwargs.get('BinLength', None)

    if ActivityMeasure == "Firing Rate" and _bin_length is None:
        covMatrix = np.cov(NeuralActivity)
        return covMatrix







