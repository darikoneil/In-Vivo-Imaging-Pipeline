import numpy as np
import pathlib
import pickle as pkl
from AnalysisModules.ExperimentHierarchy import BehavioralStage, CollectedDataFolder
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns


class Locomotion(BehavioralStage):
    """
        Instance Factory for Locomotion Data

    **Self Methods**
        | **self.generateFileID** : Generate a file ID for a particular sort of data
        | **self.fillFolderDictionary** : Function to index subfolders containing behavioral data

    **Class Methods**
        | **cls.loadAnalogData** : Loads Analog Data from a burrow behavioral session
        | **cls.loadDigitalData** : Loads Digital Data from a burrow behavioral session
        | **cls.loadStateData** : Loads State Data from a burrow behavioral session
        | **cls.
    """
    def __init__(self, Meta, Stage, **kwargs):
        super().__init__(Meta, Stage)
        self.fillFolderDictionary()
        self.data = dict()

    @classmethod
    def loadAnalogData(cls, Filename):
        """
        Loads Analog Data from a burrow behavioral session
        :param Filename: Numpy file containing analog data
        :type Filename: str
        :return: Analog Data
        """
        try:
            analogData = np.load(Filename)
        except FileNotFoundError:
            return "ERROR"

        return analogData

    @classmethod
    def loadStateData(cls, Filename):
        """
        Loads State Data from a burrow behavioral session
        :param Filename: Numpy file containing state data
        :type Filename: str
        :return: State Data
        """
        try:
            stateData = np.load(Filename)
        except FileNotFoundError:
            return "ERROR"

        return stateData

    @classmethod
    def loadDictionaryData(cls, Filename):
        """
        Loads Dictionary Data from a burrow behavioral session
        :param Filename: Numpy file containing dictionary data
        :type Filename: str
        :return: Dictionary Data
        :rtype: dict
        """
        try:
            with open(Filename, 'r') as f:
                dictionaryData = pkl.load(f)
        except FileNotFoundError:
            return "ERROR_FIND"
        except TypeError:
            # noinspection PyBroadException
            try:
                with open(Filename, 'rb') as f:
                    dictionaryData = pkl.load(f, encoding='bytes')
                    dictionaryData = Locomotion.convertFromPy27_Dict(dictionaryData)
            except Exception:
                return "ERROR_READ"

        return dictionaryData

    def fillFolderDictionary(self):
        """
        Function to index subfolders containing behavioral data

        **Requires**
            | self.folder_dictionary['behavior_folder']

        **Constructs**
            | self.folder_dictionary['behavioral_exports']
            | self.folder_dictionary['deep_lab_cut_data']
            | self.folder_dictionary['raw_behavioral_data']
            | self.folder_dictionary['processed_data']
            | self.folder_dictionary['analog_burrow_data']

        """
        self.folder_dictionary['behavioral_exports'] = CollectedDataFolder(
            self.folder_dictionary.get('behavior_folder') +
            "\\BehavioralExports")
        self.folder_dictionary['deep_lab_cut_data'] = CollectedDataFolder(
            self.folder_dictionary.get('behavior_folder') +
            "\\DeepLabCutData")
        self.folder_dictionary['raw_behavioral_data'] = CollectedDataFolder(
            self.folder_dictionary.get('behavior_folder') +
            "\\RawBehavioralData")
        self.folder_dictionary['processed_data'] = self.folder_dictionary.get('behavior_folder') + \
                                                   "\\ProcessedData"
        self.folder_dictionary['analog_burrow_data'] = self.folder_dictionary.get('behavior_folder') + \
                                                       "\\AnalogBurrowData"

    def generateFileID(self, SaveType):
        """
        Generate a file ID for a particular sort of data

        :param SaveType: Analog, Digital, State, or Dictionary
        :type SaveType: str
        :return: Filename containing respective data
        :rtype: str
        """

        # Generate file extension
        if SaveType == 'Analog':
            _save_type = 'analog.npy'
        elif SaveType == 'Digital':
            _save_type = 'digital.npy'
        elif SaveType == 'State':
            _save_type = 'state.npy'
        elif SaveType == 'Behavior Config':
            _save_type = 'behavior_config'
        else:
            return print("Unrecognized Behavioral Data Type")

        filename = self.folder_dictionary['raw_behavioral_data'].path + "\\" + _save_type
        return filename

    def loadBehavioralData(self):
        """
        Master function that loads the following data -> analog, digital, state, behavior config, hardware config
        """
        # Analog
        _analog_file = self.generateFileID('Analog')
        _analog_data = Locomotion.loadAnalogData(_analog_file)
        if type(_analog_data) == str and _analog_data == "ERROR":
            return print("Could not find analog data!")

        # State
        _state_file = self.generateFileID('State')
        _state_data = Locomotion.loadStateData(_state_file)
        if _state_data[0] == "ERROR":  # 0 because it's an array of strings so ambiguous str comparison
            return print("Could not find state data!")

        # Behavior Config
        # _behavior_config_file = self.generateFileID('Behavior Config')
        # _behavior_config_data = Locomotion.loadDictionaryData(_behavior_config_file)
        # if _behavior_config_data == "ERROR_FIND":
        #     return print("Could not find behavior config data!")
        # elif _behavior_config_data == "ERROR_READ":
        #     return print("Could not read behavior config data!")

        # Hardware Config
        # _hardware_config_file = self.generateFileID('Hardware Config')
        # _hardware_config_data = Locomotion.loadDictionaryData(_hardware_config_file)
        # if _hardware_config_data == "ERROR_FIND":
        #     return print("Could not find behavior config data!")
        # elif _hardware_config_data == "ERROR_READ":
        #    return print("Could not read behavior config data!")

        # noinspection PyTypeChecker
        self.data['Habituation'] = OrganizeBehavior('Habituation', _analog_data, _state_data)
        # noinspection PyTypeChecker
        self.data['Locomotion'] = OrganizeBehavior('Locomotion', _analog_data, _state_data)

    @staticmethod
    def convertFromPy27_Dict(Dict):
        """
        Convert a dictionary pickled in Python 2.7 to a Python 3 dictionary

        :param Dict: Dictionary to be converted
        :type Dict: dict
        :return: Converted Dictionary
        :rtype: dict
        """
        _allkeys = list(Dict.keys())
        new_dict = dict()

        for _key in range(len(_allkeys)):
            new_dict[_allkeys[_key].decode('utf-8')] = Dict.get(_allkeys[_key])

        return new_dict


class OrganizeBehavior:
    """
    Super-class for organized behavioral data by experimental stage

    **Class Methods**
        | **cls.indexData** : Function indexes the frames of some sort of task-relevant experimental state
    """

    def __init__(self, State, AnalogData, StateData, **kwargs):
        self.locomotion_channel = kwargs.get('Locomotion', 1)
        self.buffer_size = kwargs.get('BufferSize', 100)

        idx = OrganizeBehavior.indexData(State, StateData, self.buffer_size)

        self.locomotion = AnalogData[self.locomotion_channel, idx[0]:idx[1]].copy()

    @classmethod
    def indexData(cls, State, StateData, BufferSize):
        """
        Function indexes the frames of some sort of task-relevant experimental state within a specified trial

        :param State: The experimental state
        :type State: str
        :param StateData: The numpy array containing strings defining the state at any given time
        :param BufferSize: The size of the DAQ's buffer during recording (Thus x samples are considered one state)
        :type BufferSize: int
        :return: A numpy array indexing the state/trial specific samples
        """
        _idx = np.where(StateData == State)[0]
        idx = tuple([_idx[0] * BufferSize, _idx[-1] * BufferSize])
        return idx


class StatsModule:
    """
    Container for all stats-based stuff
    """

    def __init__(self):
        return

    @staticmethod
    def cumulative_displacement(AnalogLocomotionSignal):
        """
        Quantify the cumulative displacement of the animal

        That is, the cumulative sum of the absolute value of the first derivative

        :param AnalogLocomotionSignal:
        :return: Cumulative Displacement
        :rtype: float
        """
        return np.cumsum(np.abs(np.diff(AnalogLocomotionSignal)))

    @staticmethod
    def total_displacement(AnalogLocomotionSignal):
        return StatsModule.cumulative_displacement(AnalogLocomotionSignal)[-1]


class VisualsModule:
    """
    Container for visuals/plotting stuff
    """

    def __init__(self):
        return

    @staticmethod
    def plot_cumulative_displacement(cumulative_displacement, **kwargs):
        _unit_time = kwargs.get('time_unit', 1/1000)

        _time_vec = np.arange(0, cumulative_displacement.shape[0]*_unit_time, _unit_time)
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.plot(_time_vec, cumulative_displacement)


