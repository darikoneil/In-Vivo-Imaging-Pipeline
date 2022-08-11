# cascade imports
import matplotlib
matplotlib.use('Qt5Agg')
import pickle as pkl
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
        self.ProcessedInferences = ProcessedInferences()

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
        self.ProcessedInferences.utilized_model_name = self.model_name
        self.spike_prob = cascade.predict(self.model_name, self.traces, model_folder=self.model_folder)

    def inferDiscreteSpikes(self):
        if self.ProcessedInferences.utilized_model_name != self.model_name:
            print("Warning! Spike probabilities inferred using " + self.ProcessedInferences.utilized_model_name +
                  ", but user has selected " + self.model_name + " for spike inference.")
            self.ProcessedInferences.utilized_model_name = self.model_name

        self.discrete_approximation, self.spike_time_estimates = infer_discrete_spikes(self.spike_prob, self.model_name, model_folder=self.model_folder)

    def saveSpikeProb(self, save_path):
        _filename = save_path + "spike_prob.npy"
        np.save(_filename, self.spike_prob, allow_pickle=True)

    def saveSpikeInference(self, save_path):
        _filename = save_path + "spike_times.npy"
        np.save(_filename, self.spike_time_estimates, allow_pickle=True)
        _filename = save_path + "discrete_approximation.npy"
        np.save(_filename, self.discrete_approximation, allow_pickle=True)

    def exportSpikeProb(self, save_path):
        sio.savemat(save_path + 'spike_prob' + '.mat', {'spike_prob': self.spike_prob})

    def exportSpikeInference(self, save_path):
        sio.savemat(save_path + 'spike_times' + '.mat', {'spike_times': self.spike_time_estimates})

    def loadSpikeProb(self, **kwargs):
        _load_path = kwargs.get('load_path', None)
        _absolute_path = kwargs.get('absolute_path', None)
        try:
            if _load_path is not None:
                print("Loading Spike Probabilities from Load Path...")
                _filename = _load_path + "spike_prob.npy"
            elif _absolute_path is not None:
                print("Loading Spike Probabilities from Provided File...")
                _filename = _absolute_path
            else:
                print("Location of Spike Probabilities File Not Adequate")
                raise RuntimeError

            self.spike_prob = np.load(_filename, allow_pickle=True)
            if type(self.spike_prob) is not np.ndarray:
                raise TypeError
            print("Finished Loading Spike Probabilities")
        except TypeError:
            print("Spike probabilities file in unexpected format")
        except RuntimeError:
            print("Unable to load spike probabilities. Check supplied path.")

    def loadSpikeInference(self, **kwargs):
        _load_path = kwargs.get('load_path', None)
        _absolute_path_spike_times = kwargs.get('spike_times_file', None)
        _absolute_path_discrete_approx = kwargs.get('discrete_approx_file', None)

        # spike times
        try:
            if _load_path is not None:
                print("Loading Spike Times from Load Path...")
                _filename_spike_times = _load_path + "spike_times.npy"
            elif _absolute_path_spike_times is not None:
                print("Loading Spike Times from Supplied Files...")
                _filename_spike_times = _absolute_path_spike_times
            else:
                print("Location of Spike Times Not Adequate")
                raise RuntimeError

            self.spike_time_estimates = np.load(_filename_spike_times, allow_pickle=True)

            if type(self.spike_time_estimates) is not np.ndarray:
                raise TypeError

        except TypeError:
            print("Spike times file in unexpected format")
        except RuntimeError:
            print("Unable to load spike times. Check supplied path.")

        # discrete approximations
        try:
            if _load_path is not None:
                print("Loading Discrete Approximations from Load Path...")
                _filename_discrete_approx = _load_path + "discrete_approximation.npy"
            elif _absolute_path_discrete_approx is not None:
                print("Loading Discrete Approximations from Supplied Files...")
                _filename_discrete_approx = _absolute_path_discrete_approx
            else:
                print("Location of Discrete Approximations Not Adequate")
                raise RuntimeError

            self.discrete_approximation = np.load(_filename_discrete_approx, allow_pickle=True)

            if type(self.discrete_approximation) is not np.ndarray:
                raise TypeError
        except TypeError:
            print("Discrete Approximations file in unexpected format")
        except RuntimeError:
            print("Unable to load Discrete Approximations. Check supplied path.")

    def saveProcessedInferences(self, save_path):
        print("Saving Processed Inferences...")
        _output_file = save_path + "ProcessedInferences"
        _output_pickle = open(_output_file, 'wb')
        pkl.dump(self.ProcessedInferences, _output_pickle)
        _output_pickle.close()
        print("Finished Saving Processed Inferences")

    def loadProcessedInferences(self, **kwargs):
        _load_path = kwargs.get('load_path')
        _absolute_path = kwargs.get('absolute_path')
        try:
            if _load_path is not None:
                _filename = _load_path + "ProcessedInferences"
            elif _absolute_path is not None:
                _filename = _absolute_path
            else:
                print("Location of Processed Inferences Not Adequate")
                raise RuntimeError

            print("Loading Processed Inferences...")
            _input_pickle = open(_filename, 'rb')
            self.ProcessedInferences = pkl.load(_input_pickle)
            _input_pickle.close()
            print("Finished Loading Processed Traces.")

        except RuntimeError:
            print("Unable to load processed inferences. Check supplied path.")


class ProcessedInferences:
    def __init__(self):
        self.firing_rates = None
        self.burst_events = None
        self.high_activity_frames = None
        self.utilized_model_name = None
