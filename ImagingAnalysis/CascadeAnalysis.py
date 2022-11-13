# cascade imports
from __future__ import annotations

import os
from typing import Tuple, List, Union, Optional
import matplotlib
matplotlib.use('Qt5Agg')
import pickle as pkl
import numpy as np
import scipy.io as sio
from cascade2p import cascade
from cascade2p.utils_discrete_spikes import infer_discrete_spikes
from cascade2p import checks
import ruamel.yaml as yaml
import tensorflow as tf
from tensorflow.python.client import device_lib


checks.check_packages()


class CascadeModule:
    """
    A module for CASCADE spike inference

    **Self Methods**
        | *load_neurons_x_time* : Loads 2D array of neurons x frames into class
        | *predictSpikeProb* : Predict spike probability
        | *inferDiscreteSpikes* : Approximate Discrete Spike Events
        | *saveSpikeProb* : Save self.spike_prob to a numpy file
        | *saveSpikeInference* : Save self.spike_time_estimates & self.discrete_approximation
        | *exportSpikeProb* : Export self.spike_prob to MATLab file
        | *exportSpikeInference* : Export self.spike_time_estimates to MATLab file
        | *loadSpikeProb* : Load Spike Probabilities into self.spike_prob
        | *loadSpikeInference* : Load Spike Inferences into self.spike_time_estimates and self.discrete_approximation
        | *saveProcessedInferences* : Save Processed Inferences to file
        | *loadProcessedInferences*: Load Processed Inferences from file
    **Class Methods**
        | *pullModels* : Retrieve the latest online models & locally-available models for spike inference
        | *downloadModel* : Downloads online model for local-availability
        | *setVerboseGPU* : Set GPU to verbose mode
        | *confirmGPU* : Confirm that tensorflow was built with CUDA, and that a GPU is available for use
    """

    def __init__(self, Traces: np.ndarray, FrameRate: float, SavePath: Optional[str] = None, **kwargs):
        self.model_folder = kwargs.get('model_folder', "Pretrained_models")
        self.save_folder = kwargs.get('save_folder', None)

        if self.save_folder is None and SavePath is None:
            self.save_path = os.getcwd()
        elif self.save_folder is None and SavePath is not None:
            self.save_path = SavePath
        elif self.save_folder is not None and SavePath is None:
            self.save_path = "".join([os.getcwd(), "\\", self.save_folder])
        elif self.save_folder is not None and SavePath is not None:
            self.save_path = "".join([SavePath, "\\", self.save_folder])

        self.traces = Traces
        self.frame_rate = FrameRate
        self.model_name = None
        self.spike_prob = None
        self.discrete_approximation = None
        self.spike_time_estimates = None
        self.data_file = None
        self.numNeurons = None
        self.numFrames = None
        # Definitely need to refactor below and maybe make num neurons, frames properties and protect model name
        self.ProcessedInferences = ProcessedInferences()

    def load_neurons_x_time(self, file_path, framerate):
        """ This function is identical to Cascade's load_neurons x time,
        described as -> "Custom method to load data as 2d array with shape (neurons, nr_timepoints)"

        With this function the returns are automatically integrated into the CascadeModule class

        With Mat Files, variable must be called 'dFoF'
        With Npy Files, variable must be called 'dF_traces' in a dictionary

        :param file_path: Filepath to an array of neurons by frames. Can be .npy or .mat
        :type file_path: str
        :param framerate: Imaging Frequency
        :type framerate: float
        """


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
        """
        Predict spike probability

        Simply a wrapper for cascade.predict

        **Requires**
            | self.model_name
            | self.traces

        **Modifies**
            | self.spike_prob
            | self.ProcessedInferences.utilized_model_name

        :rtype: None
        """
        self.ProcessedInferences.utilized_model_name = self.model_name
        self.spike_prob = cascade.predict(self.model_name, self.traces, model_folder=self.model_folder)

    def inferDiscreteSpikes(self):
        """
        Approximate Discrete Spike Events

        Simpy a wrapper for cascade.infer_discrete_spikes

        **Requires**
            | self.ProcessedInferences.utilized_model_name
            | self.model_name
            | self.model_folder
            | self.spike_prob

        **Modifies**
            | self.discrete_approximation
            | self.spike_time_estimates

        :rtype: None
        """
        if self.ProcessedInferences.utilized_model_name != self.model_name:
            print("Warning! Spike probabilities inferred using " + self.ProcessedInferences.utilized_model_name +
                  ", but user has selected " + self.model_name + " for spike inference.")
            self.ProcessedInferences.utilized_model_name = self.model_name

        self.discrete_approximation, self.spike_time_estimates = infer_discrete_spikes(self.spike_prob, self.model_name, model_folder=self.model_folder)

    def saveSpikeProb(self, save_path):
        """
        Save self.spike_prob to a numpy file

        **Requires**
            | self.spike_prob

        :param save_path: Path to save file
        :type save_path: str
        :rtype: None
        """
        _filename = save_path + "spike_prob.npy"
        np.save(_filename, self.spike_prob, allow_pickle=True)

    def saveSpikeInference(self, save_path):
        """
        Save self.spike_time_estimates & self.discrete_approximation

        **Requires**
            | self.spike_time_estimates
            | self.discrete_approximation

        :param save_path: Path to save files
        :type save_path: str
        :rtype: None
        """
        _filename = save_path + "spike_times.npy"
        np.save(_filename, self.spike_time_estimates, allow_pickle=True)
        _filename = save_path + "discrete_approximation.npy"
        np.save(_filename, self.discrete_approximation, allow_pickle=True)

    def exportSpikeProb(self, save_path):
        """
        Export self.spike_prob to MATLab file

        **Requires**
            |self.spike_prob

        :param save_path: Path to save file
        :type save_path: str
        :rtype: None
        """
        sio.savemat(save_path + 'spike_prob' + '.mat', {'spike_prob': self.spike_prob})

    def exportSpikeInference(self, save_path):
        """
        Export self.spike_time_estimates to MATLab file

        **Requires**
            | self.spike_time_estimates

        :param save_path: Path to save file
        :type save_path: str
        :rtype: None
        """
        sio.savemat(save_path + 'spike_times' + '.mat', {'spike_times': self.spike_time_estimates})

    def loadSpikeProb(self, **kwargs):
        """
        Load Spike Probabilities into self.spike_prob

        **Modifies**
            | self.spike_prob

        :keyword load_path: Directory containing the file
        :keyword absolute_path: Absolute filepath
        :rtype: None
        """
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
        """
        Load Spike Inferences into self.spike_time_estimates and self.discrete_approximation

        **Modifies**
            | self.spike_time_estimates
            | self.discrete_approximation

        :keyword load_path: Directory containing the file
        :keyword absolute_path_spike_times: Absolute filepath
        :keyword absolute_path_discrete_approx: Absolute filepath
        :rtype: None
        :param kwargs:
        :return:
        """
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
        """
        Save Processed Inferences to file

        **Requires**
            | self.ProcessedInferences
        :param save_path: Path to saved file
        :type save_path: str
        :rtype: None
        """
        print("Saving Processed Inferences...")
        _output_file = save_path + "ProcessedInferences"
        _output_pickle = open(_output_file, 'wb')
        pkl.dump(self.ProcessedInferences, _output_pickle)
        _output_pickle.close()
        print("Finished Saving Processed Inferences")

    def loadProcessedInferences(self, **kwargs):
        """
        Save Processed Inferences to file

        **Requires**
            | self.ProcessedInferences

        :keyword load_path: Path containing processed inferences
        :keyword absolute_path: Absolute filepath
        :rtype: None
        """

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

    @classmethod
    def pullModels(cls, ModelFolder):
        """
        Retrieve the latest online models & locally-available models for spike inference

        :param ModelFolder: Folder containing locally-available models
        :type ModelFolder: str
        :return: list_of_models
        :rtype: list
        """
        cascade.download_model('update_models', verbose=1, model_folder=ModelFolder)
        _available_models = ModelFolder + "\\available_models.yaml"
        yaml_file = open(_available_models)
        X = yaml.load(yaml_file, Loader=yaml.Loader)
        list_of_models = list(X.keys())
        print('\n List of available models: \n')
        for model in list_of_models:
            print(model)
        return list_of_models

    @classmethod
    def downloadModel(cls, ModelName, ModelFolder):
        """
        Downloads online model for local-availability

        See cascade.download_model

        :param ModelName: Name of online model to download
        :type ModelName: str
        :param ModelFolder: Name of folder to store model in
        :type ModelFolder: str
        :rtype: None
        """
        cascade.download_model(ModelName, verbose=1, model_folder=ModelFolder)

    @classmethod
    def setVerboseGPU(cls):
        """
        Set GPU to verbose mode

        See tf.debugging.set_log_device_placement

        :rtype: None
        """
        tf.debugging.set_log_device_placement(True)

    @classmethod
    def confirmGPU(cls):
        """
        Confirm that tensorflow was built with CUDA, and that a GPU is available for use

        See :
        device_lib.list_local_devices
        tf.test.is_built_with_cuda()
        tf.test.is_gpu_available()

        :rtype: None
        """
        print("Here are the local devices...")
        print(device_lib.list_local_devices())

        print("Is Tensorflow Built with CUDA...")
        tf.test.is_built_with_cuda()

        print("Is GPU Available for Use...")
        tf.test.is_gpu_available()


class ProcessedInferences:
    """
    Simple Container for Processed Inferences
    """
    def __init__(self):
        self.firing_rates = None
        self.burst_events = None
        self.high_activity_frames = None
        self.utilized_model_name = None
        self.spike_matrix = None
