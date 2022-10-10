import numpy as np
import pandas as pd
import pickle as pkl
from ExperimentManagement.ExperimentHierarchy import BehavioralStage, CollectedDataFolder
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt


class LickTraining(BehavioralStage):
    """
        Instance Factory for LickTraining Data

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
        self.data_frame = pd.DataFrame()
        self.multi_index = pd.MultiIndex
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
    def loadDigitalData(cls, Filename):
        try:
            digitalData = np.load(Filename)
        except FileNotFoundError:
            return "ERROR"

        return digitalData

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
                    dictionaryData = LickTraining.convertFromPy27_Dict(dictionaryData)
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
        elif SaveType == 'Config':
            _save_type = 'config'
        elif SaveType == 'Stats':
            _save_type = 'stats'
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
        _analog_data = LickTraining.loadAnalogData(_analog_file)
        if type(_analog_data) == str and _analog_data == "ERROR":
            return print("Could not find analog data!")

        # Digital
        _digital_file = self.generateFileID('Digital')
        _digital_data = LickTraining.loadDigitalData(_digital_file)
        if type(_digital_data) == str and _digital_data == "ERROR":
            return print("Could not find digital data!")

        # State
        _state_file = self.generateFileID('State')
        _state_data = LickTraining.loadStateData(_state_file)
        if _state_data[0] == "ERROR":  # 0 because it's an array of strings so ambiguous str comparison
            return print("Could not find state data!")

        # Behavior Config
        _config_file = self.generateFileID('Behavior Config')
        _config_data = LickTraining.loadDictionaryData(_config_file)
        if _config_data == "ERROR_FIND":
            return print("Could not find behavior config data!")
        elif _config_data == "ERROR_READ":
            return print("Could not read behavior config data!")


        _time_vector_1000Hz = np.around(np.arange(0, _analog_data.shape[1] * (1 / 1000), 1 / 1000, dtype=np.float64), decimals=3)
        _time_vector_10Hz = np.around(np.arange(0, _state_data.__len__() * (1 / 10), 1 / 10, dtype=np.float64), decimals=3)

        AnalogData = pd.DataFrame(_analog_data.T, index=_time_vector_1000Hz,
                                  columns=["Image Sync", "Motor Position", "Force", "Dummy"])
        AnalogData.index.name = "Time (s)"

        DigitalData = pd.DataFrame(_digital_data.T, index=_time_vector_1000Hz,
                                   columns=["Gate Trigger", "Sucrose Reward", "Water Reward", "Sucrose Lick", "Water Lick"])
        DigitalData.index.name = "Time (s)"

        StateData = pd.Series(_state_data.astype(int), index=_time_vector_10Hz)
        StateData.index.name = "Time (s)"
        StateData.name = "Trial"
        StateData = StateData.reindex(index=_time_vector_1000Hz)
        StateData.ffill(inplace=True)

        SyncData = AnalogData.join(DigitalData)
        SyncData = SyncData.join(StateData)
        SyncData.ffill(inplace=True)

        self.data_frame = SyncData.copy(deep=True)
        self.multi_index = pd.MultiIndex.from_arrays([self.data_frame.index.values.copy(),
                                         self.data_frame["Trial"].values.copy()])
        self.data_frame.reset_index(drop=False, inplace=True)
        self.data_frame.set_index(index=self.multi_index, drop=False, inplace=True)

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