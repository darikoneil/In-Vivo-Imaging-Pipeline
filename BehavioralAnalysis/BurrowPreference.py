import numpy as np
import pickle as pkl
from ExperimentManagement.ExperimentHierarchy import BehavioralStage, CollectedDataFolder
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt


class BurrowPreference(BehavioralStage):
    """
    Instance Factory for BurrowPreference Data

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
    def loadDigitalData(cls, Filename):
        """
        Loads Digital Data from a burrow behavioral session
        :param Filename: Numpy file containing digital data
        :type Filename: str
        :return: Digital Data
        """
        # Note that we flip the bit to convert the data such that 1 == gate triggered
        try:
            digitalData = np.load(Filename)
        except FileNotFoundError:
            return "ERROR"

        digitalData = digitalData.__abs__()-1
        # noinspection PyUnresolvedReferences
        digitalData[np.where(digitalData == 255)] = 1
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
                    dictionaryData = BurrowPreference.convertFromPy27_Dict(dictionaryData)
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
        self.folder_dictionary['behavioral_exports'] = CollectedDataFolder(self.folder_dictionary.get('behavior_folder') +
                                                                           "\\BehavioralExports")
        self.folder_dictionary['deep_lab_cut_data'] = CollectedDataFolder(self.folder_dictionary.get('behavior_folder') +
                                                                          "\\DeepLabCutData")
        self.folder_dictionary['raw_behavioral_data'] = CollectedDataFolder(self.folder_dictionary.get('behavior_folder') +
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
        elif SaveType == 'Hardware Config':
            _save_type = 'hardware_config'
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
        _analog_data = BurrowPreference.loadAnalogData(_analog_file)
        if type(_analog_data) == str and _analog_data == "ERROR":
            return print("Could not find analog data!")

        # Digital
        _digital_file = self.generateFileID('Digital')
        _digital_data = BurrowPreference.loadDigitalData(_digital_file)
        if type(_digital_data) == str and _digital_data == "ERROR":
            return print("Could not find digital data!")

        # State
        _state_file = self.generateFileID('State')
        _state_data = BurrowPreference.loadStateData(_state_file)
        if _state_data[0] == "ERROR": # 0 because it's an array of strings so ambiguous str comparison
            return print("Could not find state data!")

        # Behavior Config
       # _behavior_config_file = self.generateFileID('Behavior Config')
       # _behavior_config_data = BurrowPreference.loadDictionaryData(_behavior_config_file)
       # if _behavior_config_data == "ERROR_FIND":
       #     return print("Could not find behavior config data!")
       # elif _behavior_config_data == "ERROR_READ":
       #     return print("Could not read behavior config data!")

        # Hardware Config
       # _hardware_config_file = self.generateFileID('Hardware Config')
       # _hardware_config_data = BurrowPreference.loadDictionaryData(_hardware_config_file)
       # if _hardware_config_data == "ERROR_FIND":
       #     return print("Could not find behavior config data!")
       # elif _hardware_config_data == "ERROR_READ":
        #    return print("Could not read behavior config data!")

        # noinspection PyTypeChecker
        self.data['Habituation'] = OrganizeBehavior('Habituation', _analog_data, _digital_data, _state_data)
        # noinspection PyTypeChecker
        self.data['Release'] = OrganizeBehavior('Release', _analog_data, _digital_data, _state_data)
        # noinspection PyTypeChecker
        self.data['Retract'] = OrganizeBehavior('Retract', _analog_data, _digital_data, _state_data)
        # noinspection PyTypeChecker
        self.data['PreferenceTest'] = OrganizeBehavior('PreferenceTest', _analog_data, _digital_data, _state_data)

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
    def __init__(self, State, AnalogData, DigitalData, StateData, **kwargs):
        self.gate_channel = kwargs.get('GateChannel', 1)
        self.buffer_size = kwargs.get('BufferSize', 100)

        idx = OrganizeBehavior.indexData(State, StateData, self.buffer_size)

        if DigitalData.shape.__len__() == 1:
            self.gateData = DigitalData[idx[0]:idx[1]]
        elif DigitalData.shape.__len__() > 1:
            self.gateData = DigitalData[self.gate_channel, idx[0]:idx[1]]

    @classmethod
    def indexData(cls, State, StateData, BufferSize):
        """
        Function indexes the frames of some sort of task-relevant experimental state within a specified trial

        :param State: The experimental state
        :type State: str
        :param StateData: The numpy array containing strings defining the state at any given time
        :param BufferSize: The size of the DAQ's buffer during recording (Thus x samples are considered one state)
        :type BufferSize: int
        :param Trial: The specified trial to index
        :type Trial: int
        :return: A numpy array indexing the state/trial specific samples
        """
        _idx = np.where(StateData == State)[0]
        idx = tuple([_idx[0]*BufferSize, _idx[-1]*BufferSize])
        return idx


class StatsModule:
    """
    Container for all stats-based stuff
    """
    def __init__(self):
        return

    @staticmethod
    def calculate_percent_in_burrow(gateData):
        """
        Calculate the percentage of time spent in the burrow

        :param gateData: a numpy array of the gate / trigger burrow data
        :return: A tuple containing the percent of time in (0 index) and out (1 index) of the burrow\
        :rtype: tuple
        """

        return tuple([100*(np.sum(gateData)/gateData.shape[0]), 100*(gateData.shape[0]-np.sum(gateData))/gateData.shape[0]])


class VisualsModule:
    """
    Container for visuals/plotting stuff
    """
    def __init__(self):
        return

    @staticmethod
    def pie_chart_comparison(Group1Percentiles, Group2Percentiles, **kwargs):
        _cmap = kwargs.get("cmap", "Spectral_r")
        _radius = kwargs.get("radius", 1)
        _reduction = kwargs.get("reduction", 0.3)
        _inner_radius = kwargs.get("_inner_radius", _radius - _reduction)
        _edge_color = kwargs.get("edge_color", "w")
        _wedge_width = kwargs.get("wedge_width", _reduction)
        _title = kwargs.get("title", " ")
        _sub_1 = kwargs.get("sub_1", " ")
        _sub_2 = kwargs.get("sub_2", " ")
        _match_colors = kwargs.get("matching_colors", True)
        _angles = kwargs.get("angles", tuple([0, 0]))

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)

        cmap = plt.colormaps[_cmap]

        if _match_colors:
            oc = cmap(np.arange(0, 1, 0.25))

            ax1.pie(Group1Percentiles, radius=_radius, colors=oc[[0, 1], :].copy(),
                    startangle=_angles[0], wedgeprops=dict(width=_wedge_width,
                                                           edgecolor=_edge_color))

            ax1.pie(Group2Percentiles, radius=_inner_radius, colors=oc[[0, 1], :].copy(),
                    startangle=_angles[1], wedgeprops=dict(width=_wedge_width,
                                                           edgecolor=_edge_color))

        else:
            return print("Not Yet Implemented")

        ax1.set(aspect="equal", title=_title)
        ax1.text(0, 0, _sub_1, horizontalalignment='center', verticalalignment='center')
        ax1.text(0.85, 0.85, _sub_2, horizontalalignment='center', verticalalignment='center')
        ax1.legend(['Explore', 'Hide'], loc=3)
        return fig1, ax1
