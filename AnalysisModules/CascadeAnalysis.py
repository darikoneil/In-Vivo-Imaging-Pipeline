# cascade imports
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import scipy.io as sio
from cascade2p import cascade
from cascade2p.utils_discrete_spikes import infer_discrete_spikes
from cascade2p import checks
checks.check_packages()


class CascadeModule:
    def __init__(self, dFoF, im_freq, **kwargs):
        self.model_folder = kwargs.get('model_folder', "Pretrained_models")
        self.traces = dFoF
        self.frame_rate = im_freq
        self.spike_prob = None
        self.model_name = None
        self.discrete_approximation = None
        self.spike_time_estimates = None
        self.data_file = None
        self.numNeurons = None
        self.numFrames = None

    def load_neurons_x_time(self, file_path, framerate):
        """Custom method to load data as 2d array with shape (neurons, nr_timepoints)"""

        if file_path.endswith('.mat'):
            self.traces = sio.loadmat(file_path)['dFoF']

        elif file_path.endswith('.npy'):
            self.traces = np.load(file_path, allow_pickle=True)
            # if saved data was a dictionary packed into a numpy array (MATLAB style): unpack
            if self.traces.shape == ():
                # noinspection PyUnresolvedReferences
                self.traces = self.traces.item()['dF_traces']
        else:
            raise Exception('This function only supports .mat or .npy files.')

        self.frame_rate = framerate

    def predictSpikeProb(self):
        self.spike_prob = cascade.predict(self.model_name, self.traces, model_folder=self.model_folder)

    def inferDiscreteSpikes(self):
        self.discrete_approximation, self.spike_time_estimates = infer_discrete_spikes(self.spike_prob, self.model_name, model_folder=self.model_folder)

    def exportSpikeProb(self, save_path):
        sio.savemat(save_path + 'spike_prob' + '.mat', {'spike_prob': self.spike_prob})

    def exportSpikeInference(self, save_path):
        sio.savemat(save_path + 'spike_times' + '.mat', {'spike_times': self.spike_time_estimates})


class ProcessedInferences:
    def __init__(self):
        self.firing_rates = None
        self.burst_events = None
        self.high_activity_frames = None
        self.firing_rates_trial_org = None
        self.mean_firing_rates_by_trial = None
