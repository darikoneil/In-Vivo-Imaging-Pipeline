
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution
import numpy as np


def plotNoise(self):
    # plt.figure(1)
    _noise_levels = plot_noise_level_distribution(self.traces, self.frame_rate)


def plotTraces(self, **kwargs):
    NeuronalSubset = kwargs.get('NeuronalSubset', np.random.randint(self.traces.shape[0], size=10))
    TemporalSubset = kwargs.get('TemporalSubset', [0, self.frame_rate * 30])
    plot_dFF_traces(self.traces[:, TemporalSubset[0]:TemporalSubset[1]], NeuronalSubset, self.frame_rate)


def plotTraceComparisons(self, **kwargs):
    NeuronalSubset = kwargs.get('NeuronalSubset', np.random.randint(self.traces.shape[0], size=10))
    TemporalSubset = kwargs.get('TemporalSubset', [0, self.frame_rate * 30])
    plot_dFF_traces(self.traces[:, TemporalSubset[0]:TemporalSubset[1]], NeuronalSubset, self.frame_rate,
                    spiking=self.spike_prob)


def plotSpikeInference(self, **kwargs):
    NeuronalSubset = kwargs.get('NeuronalSubset', np.random.randint(self.traces.shape[0], size=10))
    TemporalSubset = kwargs.get('TemporalSubset', [0, self.frame_rate * 30])
    plot_dFF_traces(self.traces[:, TemporalSubset[0]:TemporalSubset[1]], NeuronalSubset, self.frame_rate,
                    spiking=self.spike_prob, discrete_spikes=self.spike_time_estimates)