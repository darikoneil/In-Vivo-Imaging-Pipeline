import numpy as np
import pickle as pkl
import pathlib
import pandas as pd
from tqdm.auto import tqdm
from AnalysisModules.ExperimentHierarchy import BehavioralStage, CollectedDataFolder
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns


class FearConditioning(BehavioralStage):
    """
    Instance Factory for Fear Conditioning Data

    **Self Methods**
        | **self.generateFileID** : Generate a file ID for a particular sort of data
        | **self.fillFolderDictionary** : Function to index subfolders containing behavioral data

    **Class Methods**
        | **cls.loadAnalogData** : Loads Analog Data from a burrow behavioral session
        | **cls.loadDigitalData** : Loads Digital Data from a burrow behavioral session
        | **cls.loadStateData** : Loads State Data from a burrow behavioral session
        | **cls.loadDictionaryData** : Loads Dictionary Data from a burrow behavioral session
        | **cls.loadBehavioralData** : Loads Behavioral Data from a burrow behavioral session

    **Static Methods**
        | **convertFromPy27_Array** : Convert a numpy array of strings in byte-form to numpy array of strings in string-form
        | **convertFromPy27_Dict**  : Convert a dictionary pickled in Python 2.7 to a Python 3 dictionary
    """
    def __init__(self, Meta, Stage, **kwargs):
        super().__init__(Meta, Stage)
        self.fillFolderDictionary()

        # PROTECT ME
        _trials_per_stim = kwargs.get('TrialsPerStim', 5)
        _num_stim = kwargs.get('NumStim', 2)
        _num_trials = _trials_per_stim * _num_stim
        _stage = Stage

        # PROTECTED
        self.__trials_per_stim = _trials_per_stim
        self.__num_stim = _num_stim
        self.__num_trials = _num_trials
        self.__stage_id = _stage

        # self.habituation_data = None
        # self.pretrial_data = None
        # self.iti_data = None
        # self.trial_data = None
        # self.trial_dictionary = dict()

        self.data = dict()
        self.data_frame = pd.DataFrame()
        self.state_casted_index = dict()
        self.multi_index = pd.MultiIndex()
        self.trial_parameters = dict()

    @property
    def stage_id(self):
        return self._FearConditioning__stage_id

    @property
    def num_trials(self):
        return self._FearConditioning__num_trials

    @property
    def trials_per_stim(self):
        return self._FearConditioning__trials_per_stim

    @property
    def num_stim(self):
        return self._FearConditioning__num_stim

    @classmethod
    def identifyTrialValence(cls, csIndexFile):
        _csIndex = np.genfromtxt(csIndexFile, int, delimiter=",")
        PlusTrials = np.where(_csIndex == 0)
        MinusTrials = np.where(_csIndex == 1)
        return PlusTrials[0], MinusTrials[0]

    @classmethod
    def reorganizeData(cls, NeuralActivity, TrialFeatures, ImFreq, **kwargs):
        _iti_time = kwargs.get('ITILength', 90)
        _retract_time = kwargs.get('RetractLength', 5)
        _release_time = kwargs.get('ReleaseLength', 5)
        _iti_pre_inc = kwargs.get('ITIIncludedBeforeTrial', 30)
        _iti_post_inc = kwargs.get('ITIIncludedAfterTrial', 30)
        _pre_time = kwargs.get('PreLength', 15)
        _cs_time = kwargs.get('CSLength', 15)
        _trace_time = kwargs.get('TraceLength', 10)
        _ucs_time = kwargs.get('UCSLength', 5.5)
        _response_time = kwargs.get('ResponseLength', 10)

        try:
            if len(NeuralActivity.shape) > 2:
                raise TypeError
            _num_neurons, _num_frames = NeuralActivity.shape
            if _num_neurons > _num_frames:
                print("Data contains more neurons than frames. Check to ensure your data is in the form neurons x frames.")
        except TypeError:
            print("Neural activity and trial features must be in matrix form")
            return
        except AttributeError:
            print("Data must be numpy array")
            return

        _iti_time *= ImFreq
        _retract_time *= ImFreq
        _release_time *= ImFreq
        _iti_pre_inc *= ImFreq
        _iti_post_inc *= ImFreq
        _pre_time *= ImFreq
        _cs_time *= ImFreq
        _trace_time *= ImFreq
        _ucs_time *= ImFreq
        _response_time *= ImFreq

        # Round to Integers
        _iti_time = int(round(_iti_time))
        _retract_time = int(round(_retract_time))
        _release_time = int(round(_release_time))
        _iti_pre_inc = int(round(_iti_pre_inc))
        _iti_post_inc = int(round(_iti_post_inc))
        _pre_time = int(round(_pre_time))
        _cs_time = int(round(_cs_time))
        _trace_time = int(round(_trace_time))
        _ucs_time = int(round(_ucs_time))
        _response_time = int(round(_response_time))

        _num_features = TrialFeatures.shape[0]
        _trial_indicator = np.sum(TrialFeatures[4:6, :], axis=0)
        _trial_indicator[_trial_indicator > 1] = 1
        _trial_frames = np.where(_trial_indicator == 1)[0]
        _diff_trial_frames = np.diff(_trial_frames)
        _end_trial_idx = np.append(np.where(_diff_trial_frames > 1)[0], _trial_frames.shape[0]-1)
        _start_trial_idx = np.append(np.array([0], dtype=np.int64), _end_trial_idx[0:-1]+1)

        _num_trials = len(_start_trial_idx)
        _trial_length = _end_trial_idx[0]+1
        _before_trial_start_frames = _pre_time + _release_time + _iti_pre_inc
        _after_trial_end_frames = _retract_time + _iti_post_inc
        _total_frames_per_trial = _before_trial_start_frames + _trial_length + _after_trial_end_frames

        NeuralActivity_TrialOrg = np.full((_num_trials, _num_neurons, _total_frames_per_trial), 0, dtype=np.float64)
        FeatureData_TrialOrg = np.full((_num_trials, _num_features, _total_frames_per_trial), 0, dtype=np.float64)

        for _trial in range(_num_trials):
            _start_idx = _trial_frames[_start_trial_idx[_trial]] - _before_trial_start_frames
            _end_idx = _trial_frames[_end_trial_idx[_trial]] + _after_trial_end_frames+1 # one to handle indexing
            NeuralActivity_TrialOrg[_trial, :, :] = NeuralActivity[:, _start_idx:_end_idx]
            FeatureData_TrialOrg[_trial, :, :] = TrialFeatures[:, _start_idx:_end_idx]

        FeatureIndex = dict()
        FeatureIndex['ITI_PRE'] = (0, _iti_pre_inc)
        FeatureIndex['RELEASE'] = (FeatureIndex['ITI_PRE'][1], FeatureIndex['ITI_PRE'][1]+_release_time)
        FeatureIndex['PRE'] = (FeatureIndex['RELEASE'][1], FeatureIndex['RELEASE'][1]+_pre_time)
        FeatureIndex['TRIAL'] = (FeatureIndex['PRE'][1], FeatureIndex['PRE'][1]+_trial_length)
        FeatureIndex['RETRACT'] = (FeatureIndex['TRIAL'][1], FeatureIndex['TRIAL'][1]+_retract_time)
        FeatureIndex['ITI_POST'] = (FeatureIndex['RETRACT'][1], FeatureIndex['RETRACT'][1]+_iti_post_inc)

        # Sub-Trial Index
        FeatureIndex['CS'] = (FeatureIndex['PRE'][1], FeatureIndex['PRE'][1]+_cs_time)
        FeatureIndex['TRACE'] = (FeatureIndex['CS'][1], FeatureIndex['CS'][1]+_trace_time)
        FeatureIndex['RESPONSE'] = (FeatureIndex['TRACE'][1], FeatureIndex['TRACE'][1]+_response_time)
        FeatureIndex['UCS'] = (FeatureIndex['TRACE'][1], FeatureIndex['TRACE'][1]+_ucs_time)
        # NOTE: DOUBLE CHECK UCS TIME

        return NeuralActivity_TrialOrg, FeatureIndex, FeatureData_TrialOrg

    @classmethod
    def generateTimeBins(cls):
        return

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

        stateData = cls.convertFromPy27_Array(stateData)
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
                    dictionaryData = FearConditioning.convertFromPy27_Dict(dictionaryData)
            except Exception:
                return "ERROR_READ"



        return dictionaryData

    @classmethod
    def OrganizeBehavioralData(cls, AnalogData, DigitalData, StateData, Trial, Dictionary, **kwargs):
        _pre_trial_data = OrganizePreTrial('PreTrial', AnalogData, DigitalData, StateData, Trial, Dictionary)
        _iti_data = OrganizeITI('InterTrial',  AnalogData, DigitalData, StateData, Trial, Dictionary)
        _trial_data = OrganizeTrial('Trial',  AnalogData, DigitalData, StateData, Trial, Dictionary)
        dictionary_update = {'pre_trial_data': _pre_trial_data,
                             'iti_data': _iti_data,
                             'trial_data': _trial_data,
                             }
        return dictionary_update

    def loadBehavioralData(self, **kwargs):
        """
        Master function that loads the following data -> analog, digital, state, dictionary

        """
        _use_pandas = kwargs.get("use_pandas", True)
        # Analog
        _analog_file = self.generateFileID('Analog')
        _analog_data = FearConditioning.loadAnalogData(_analog_file)
        if type(_analog_data) == str and _analog_data == "ERROR":
            return print("Could not find analog data!")

        # Digital
        _digital_file = self.generateFileID('Digital')
        _digital_data = FearConditioning.loadDigitalData(_digital_file)
        if type(_digital_data) == str and _digital_data == "ERROR":
            return print("Could not find digital data!")

        # State
        _state_file = self.generateFileID('State')
        _state_data = FearConditioning.loadStateData(_state_file)
        if _state_data[0] == "ERROR": # 0 because it's an array of strings so ambiguous str comparison
            return print("Could not find state data!")

        # Dictionary
        _dictionary_file = self.generateFileID('Dictionary')
        _dictionary_data = FearConditioning.loadDictionaryData(_dictionary_file)
        self.trial_parameters = _dictionary_data.copy() # For Safety

        if _dictionary_data == "ERROR_FIND":
            return print("Could not find dictionary data!")
        elif _dictionary_data == "ERROR_READ":
            return print("Could not read dictionary data!")

        _dlc = DeepLabModule(self.folder_dictionary['deep_lab_cut_data'], self.folder_dictionary['behavioral_exports'])

        # noinspection PyTypeChecker
        # Segregated Organization
        if _use_pandas:
            # noinspection PyTypeChecker
            # Copies for safety > No-copies for speed (avoid pandas gotchas)
            self.data_frame, self.state_casted_index, self.multi_index = \
                MethodsForPandasOrganization.ExportPandasDataFrame(
                    _analog_data.copy(), _digital_data.copy(), _state_data.copy())
            # merge cs index with data frame
            self.data_frame = \
                MethodsForPandasOrganization.merge_cs_index_into_dataframe(
                    self.data_frame.copy(), np.array(self.trial_parameters.get("stimulusTypes"), dtype=np.float64))
            # merge deeplabcut with data frame
        else:
            for _trial in range(self.num_trials): # Adjust for Zero Indexing
                # noinspection PyTypeChecker
                self.data['Trial ' + str(_trial+1)] = TrialData(FearConditioning.OrganizeBehavioralData(
                    _analog_data, _digital_data, _state_data, _trial, _dictionary_data))
            self.importDeepLabCutData()

    def importDeepLabCutData(self):
        """
        Not-Pandas importing
        """
        DLM = DeepLabModule(self.folder_dictionary['deep_lab_cut_data'], self.folder_dictionary['behavioral_exports'])
        for _trial in range(self.num_trials):
            # Trial
            self.data["".join(["Trial ", str(_trial+1)])].trial_data.dlc_raw = \
                DLM.SegregateTrials(DLM.trial_data, _trial+1)
            self.data["".join(["Trial ", str(_trial+1)])].trial_data.dlc_physical = \
                DLM.ConductPostProcessing(self.data["".join(["Trial ", str(_trial+1)])].trial_data.dlc_raw)
            # Pre Trial
            self.data["".join(["Trial ", str(_trial+1)])].pre_trial_data.dlc_raw = \
                DLM.SegregateTrials(DLM.pre_trial_data, _trial+1)
            self.data["".join(["Trial ", str(_trial+1)])].pre_trial_data.dlc_physical = \
                DLM.ConductPostProcessing(self.data["".join(["Trial ", str(_trial+1)])].pre_trial_data.dlc_raw)

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
            _save_type = 'AnalogData.npy'
        elif SaveType == 'Digital':
            _save_type = 'DigitalData.npy'
        elif SaveType == 'State':
            _save_type = 'StateHistory.npy'
        elif SaveType == 'Dictionary':
            _save_type = 'StimulusInfo.pkl'
        else:
            return print("Unrecognized Behavioral Data Type")

        filename = self.folder_dictionary['raw_behavioral_data'].path + "\\" + \
                   self.stage_id + "_" + self.mouse_id + "_" + str(self.num_trials) + "_of_" +\
                   str(self.num_trials) + "_" + _save_type

        return filename

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

    @staticmethod
    def convertFromPy27_Array(Array):
        """
        Convert a numpy array of strings in byte-form to numpy array of strings in string-form

        :param Array: An array of byte strings (e.g., b'Setup')
        :return: decoded_array
        """
        decoded_array = list()
        for i in range(Array.shape[0]):
            decoded_array.append("".join([chr(_) for _ in Array[i]]))
        decoded_array = np.array(decoded_array)
        return decoded_array

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


class TrialData:
    def __init__(self, Dictionary):
        self.iti_data = None
        self.pre_trial_data = None
        self.trial_data = None

        # noinspection PyTypeChecker
        for _key in self.__dict__.keys():
            if _key in Dictionary.keys():
                self.__dict__[_key] = Dictionary[_key]


class OrganizeBehavior:
    """
    Super-class for organized behavioral data by experimental stage and trial

    **Class Methods**
        | **cls.indexData** : Function indexes the frames of some sort of task-relevant experimental state within a specified trial
    """
    def __init__(self, State, AnalogData, DigitalData, StateData, Trial, **kwargs):
        self.imaging_sync_channel = kwargs.get('ImagingSyncChannel', 0)
        self.prop_channel = kwargs.get('PropChannel', 1)
        self.force_channel = kwargs.get('ForceChannel', 2)
        self.gate_channel = kwargs.get('GateChannel', 1)
        self.buffer_size = kwargs.get('BufferSize', 100)

        _idx = OrganizeBehavior.indexData(State, StateData, self.buffer_size, Trial)
        self.imaging_sync = AnalogData[self.imaging_sync_channel, _idx[0]*self.buffer_size:_idx[-1]*self.buffer_size]
        self.prop_data = AnalogData[self.prop_channel, _idx[0]*self.buffer_size:_idx[-1]*self.buffer_size]
        self.force_data = AnalogData[self.force_channel, _idx[0]*self.buffer_size:_idx[-1]*self.buffer_size]

        if DigitalData.shape.__len__() == 1:
            self.gateData = DigitalData[_idx[0]*self.buffer_size:_idx[-1]*self.buffer_size]
        elif DigitalData.shape.__len__() > 1:
            self.gateData = DigitalData[self.gate_channel, _idx[0]*self.buffer_size:_idx[-1]*self.buffer_size]

    @classmethod
    def indexData(cls, State, StateData, BufferSize, Trial):
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
        _derivative_idx = np.diff(_idx)
        # Add beginning
        _trial_edge_idx = np.append(np.array([0], dtype=np.int64), np.where(_derivative_idx > 1)[0]+1)
        # Add end
        _trial_edge_idx = np.append(_trial_edge_idx, _idx.shape[0])
        return _idx[_trial_edge_idx[Trial]:_trial_edge_idx[Trial+1]]


class OrganizeTrial(OrganizeBehavior):
    def __init__(self, State, AnalogData, DigitalData, StateData, Trial, Dictionary, **kwargs):
        # noinspection PyArgumentList
        super().__init__(State, AnalogData, DigitalData, StateData, Trial, **kwargs)

        # do stuff unique to this class
        self.trialStartTime = Dictionary["trialStart"]  # float
        self.trialEndTime = Dictionary["trialEnd"]  # float
        self.trialStartFrames = int(0)  # int
        self.trialEndFrames = int(self.trialEndTime[0] - self.trialStartTime[0])  # int
        self.csStartTime = Dictionary["csStart"]  # float
        self.csEndTime = Dictionary["csEnd"]  # float
        self.csStartFrames = int(self.csStartTime[0]-self.trialStartTime[0])  # int
        self.csEndFrames = int(self.csEndTime[0]-self.trialStartTime[0])  # int
        self.ucsStartTime = Dictionary["ucsStart"]  # float
        self.ucsEndTime = Dictionary["ucsEnd"]  # float
        self.ucsStartFrames = int(self.ucsStartTime[0] - self.trialStartTime[0])  # int
        self.ucsEndFrames = int(self.ucsEndTime[0] - self.trialStartTime[0])  # int
        self.csType = Dictionary["stimulusTypes"][Trial-1]


class OrganizeITI(OrganizeBehavior):
    def __init__(self, State, AnalogData, DigitalData, StateData, Trial, Dictionary, **kwargs):
        # noinspection PyArgumentList
        super().__init__(State, AnalogData, DigitalData, StateData, Trial, **kwargs)
        self.ITIStartTime = Dictionary["interStart"]
        self.ITIEndTime = Dictionary["interEnd"]
        self.ITIStartFrames = 0
        self.ITIEndFrames = int(self.ITIEndTime[0] - self.ITIStartTime[0])


class OrganizePreTrial(OrganizeBehavior):
    def __init__(self, State, AnalogData, DigitalData, StateData, Trial, Dictionary, **kwargs):
        # noinspection PyArgumentList
        super().__init__(State, AnalogData, DigitalData, StateData, Trial, **kwargs)
        self.preStartTime = Dictionary["preStart"]
        self.preEndTime = Dictionary["preEnd"]
        self.preStartFrames = 0
        self.preEndFrames = int(self.preEndTime[0] - self.preStartTime[0])


class OrganizeHabituation(OrganizeBehavior):
    def __init__(self, State, AnalogData, DigitalData, StateData, Trial, Dictionary, **kwargs):
        # noinspection PyArgumentList
        super().__init__(State, AnalogData, DigitalData, StateData, Trial, **kwargs)
        self.habStartTime = Dictionary["habStart"]
        self.habEndTime = Dictionary["habEnd"]
        self.habStartFrames = 0
        self.habEndFrames = int(self.habEndTime[0] - self.habStartTime[0])


class MethodsForPandasOrganization:
    """
    Simply a container for methods related to organization of behavioral data through a pandas dataframe
    """
    def __init__(self):
        return

    @classmethod
    def ExportPandasDataFrame(cls, AnalogData, DigitalData, StateData, **kwargs):
        _imaging_sync_channel = kwargs.get("imaging_sync_channel", 0)
        _motor_position_channel = kwargs.get("motor_position_channel", 1)
        _force_channel = kwargs.get("force_channel", 2)

        # Generate Indices describing different data acquisitions
        _time_vector_1000Hz = np.around(np.arange(0, AnalogData.shape[1] * (1 / 1000), 1 / 1000, dtype=np.float64), decimals=3)
        _time_vector_10Hz = np.around(np.arange(0, StateData.__len__() * (1 / 10), 1 / 10, dtype=np.float64), decimals=3)

        # Cast State Index to Integers for simplicity & avoiding gotcha's with pandas
        _integer_state_index, StateCastedDict = MethodsForPandasOrganization.castStateIntoFloat64(StateData)

        StateIndex = MethodsForPandasOrganization.match_data_to_new_index(_integer_state_index, _time_vector_10Hz,
                                                                                         _time_vector_1000Hz,
                                                                                         is_string=False)

        # noinspection PyTypeChecker
        TrialIndex = MethodsForPandasOrganization.nest_all_stages_under_trials(StateIndex.values,
                                                                               _time_vector_1000Hz,
                                                                               StateCastedDict)

        # Time = pd.Series(_time_vector_1000Hz, index=_time_vector_1000Hz) # Don't know why added index=var

        # MultiIndex = pd.MultiIndex.from_arrays([StateIndex, TrialIndex, Time], names=["State Index", "Trial Index", "Time"])

        try:
            OrganizedData = pd.DataFrame(None, index=_time_vector_1000Hz, dtype=np.float64)
            MultiIndex = pd.MultiIndex.from_arrays([StateIndex.values, TrialIndex.values, _time_vector_1000Hz])
            OrganizedData.index.name = "Time (s)"

            assert(StateIndex.values.dtype == np.float64)
            OrganizedData['State Integer'] = StateIndex

            assert(TrialIndex.values.dtype == np.float64)
            OrganizedData['Trial Set'] = TrialIndex

            assert(AnalogData.dtype == np.float64)

            # Check orientation of data
            if AnalogData.shape[0] < AnalogData.shape[1]:
                AnalogData = AnalogData.T

            OrganizedData['Imaging Sync'] = pd.Series(AnalogData[:, _imaging_sync_channel].copy(),
                                                      index=_time_vector_1000Hz,
                                                      dtype=np.float64)
            OrganizedData['Motor Position'] = pd.Series(AnalogData[:, _motor_position_channel].copy(),
                                                        index=_time_vector_1000Hz)
            OrganizedData['Force'] = pd.Series(AnalogData[:, _force_channel].copy(),
                                               index=_time_vector_1000Hz)

            # No Need To Type Check B/C Type-Casting
            OrganizedData['Gate'] = pd.Series(DigitalData.copy().astype(np.float64), index=_time_vector_1000Hz)
            return OrganizedData, StateCastedDict, MultiIndex

        except AssertionError:
            print("Bug: Incorrect Type detected")
            return


    @staticmethod
    def merge_cs_index_into_dataframe(DataFrame, CSIndex):
        # 0  = CS+, 1 = CS-, ..., Unique CS + 1 = Nil
        _nil_id = np.unique(CSIndex).__len__() # Equivalent to the number cs id + 1 when considering zero-indexing
        _nan_id = _nil_id + 1
        _trial_set = DataFrame['Trial Set'].values
        _cs_column = np.full(_trial_set.__len__(), _nan_id, dtype=np.float64)

        for _cs in range(np.unique(CSIndex).__len__()):
            _cs_column[np.searchsorted(_trial_set, np.where(CSIndex == _cs)[0]+1)] = _cs
            # + 1 -> Adjust because trial set is not zero indexed
        # Enter Nil Start, Maintain NaNs for forward fill
        _cs_column[0] = _nil_id

        _cs_series = pd.Series(_cs_column)
        _cs_series[_cs_series == _nan_id] = np.nan
        _cs_series.ffill(inplace=True)
        DataFrame["CS"] = _cs_column

        return DataFrame

    @staticmethod
    def match_data_to_new_index(OriginalVector, OriginalIndex, NewIndex, **kwargs):
        """
        Function to match data to a new index

        :param OriginalVector:
        :param OriginalIndex:
        :param NewIndex:
        :return: NewVector
        """
        _is_string = kwargs.get("is_string", False)
        _forward_fill = kwargs.get("forward_fill", True)
        if _is_string:
            new_vector = np.full(NewIndex.__len__(), '', dtype="<U21")
        else:
            new_vector = np.full(NewIndex.__len__(), 69, dtype=OriginalVector.dtype)

        for _sample in tqdm(

                range(OriginalIndex.__len__()),
                total=OriginalIndex.__len__(),
                desc="Generating New Index",
                disable=False
        ):
            _timestamp = OriginalIndex[_sample]
            idx = np.where(NewIndex == _timestamp)[0][0]
            new_vector[idx] = OriginalVector[_sample]


        if _is_string:
            new_vector = pd.Series(new_vector, index=NewIndex, dtype="string")
            new_vector[new_vector == ''] = np.nan
        else:
            new_vector = pd.Series(new_vector, index=NewIndex, dtype=new_vector.dtype)
            new_vector[new_vector == 69] = np.nan

        if _forward_fill:
            new_vector.ffill(inplace=True)
        else:
            new_vector.bfill(inplace=True)

        return new_vector

    @staticmethod
    def nest_all_stages_under_trials(StateData, Index, StateCastedDict):
        nested_trial_index = np.full(Index.__len__(), 69, dtype=np.float64)
        _trial_idx = np.where(StateData == StateCastedDict.get("Trial"))[0]
        _deriv_idx = np.diff(_trial_idx)
        _trailing_edge = _trial_idx[np.where(_deriv_idx > 1)[0]]
        _trailing_edge = np.append(_trailing_edge, _trial_idx[-1])
        _habituation_trailing_edge = np.where(StateData == StateCastedDict.get("Habituation"))[0][-1]

        nested_trial_index[_habituation_trailing_edge] = 0
        nested_trial_index[-1] = _trailing_edge.__len__()+1

        for _edge in range(_trailing_edge.shape[0]):
            nested_trial_index[_trailing_edge[_edge]] = _edge+1

        nested_trial_index = pd.Series(nested_trial_index, index=Index, dtype=np.float64)
        nested_trial_index[nested_trial_index == 69] = np.nan
        nested_trial_index.bfill(inplace=True)

        return nested_trial_index

    @staticmethod
    def MergeDLCIntoDataFrame(DataFrame, DLC, **kwargs):
        _individual_series = []
        _num_trials = kwargs.get("num_trials", np.max(DLC.trial_data.index))
        fps = kwargs.get("fps", 30)

        for i in range(_num_trials):
            if i == 0:
                # Do this to skip the "Habituation" Index
                continue
            # Trials
            _trial = DataFrame.xs(("Trial", i), level=(0, 1), drop_level=False)
            _dlc = DLC.SegregateTrials(DLC.trial_data, i)
            _num_frames = _dlc['X1'].__len__()
            _old_dlc_index = np.around(
                np.linspace(_trial['Seconds'].values[0], _trial['Seconds'].values[-1], _num_frames), decimals=3)
            _new_dlc_index = np.around(_trial['Seconds'].values, decimals=3)
            _individual_series.append(
                pd.Series(MethodsForPandasOrganization.match_data_to_new_index(_dlc['X1'].values, _old_dlc_index,
                                                                               _new_dlc_index), index=_new_dlc_index))
            # Pre Trials
            _pre_trial = DataFrame.xs(("PreTrial", i), level=(0, 1), drop_level=False)
            _dlc = DLC.SegregateTrials(DLC.trial_data, i)
            _old_dlc_index = np.around(np.arange(0, _dlc['X1'].__len__() * (1 / fps), (1 / fps)), decimals=3)
            _adjusted_dlc_index = np.around(_old_dlc_index + _trial['Seconds'].values[0], decimals=3)
            _new_dlc_index = _trial['Seconds'].values
            _individual_series.append(
                pd.Series(MethodsForPandasOrganization.match_data_to_new_index(_dlc['X1'].values, _adjusted_dlc_index,
                                                                               _new_dlc_index), index=_new_dlc_index))
            return _individual_series

    @staticmethod
    def castStateIntoFloat64(StateData):
        StateCastedDict = dict()
        IntegerStateIndex = np.full(StateData.shape[0], 0, dtype=np.float64)
        _unique_states = np.unique(StateData)

        for _unique_value in range(_unique_states.shape[0]):
            StateCastedDict[_unique_states[_unique_value]] = _unique_value
            IntegerStateIndex[np.where(StateData == _unique_states[_unique_value])[0]] = _unique_value

        return IntegerStateIndex, StateCastedDict

    @staticmethod
    def safe_extract(OldFrame, Commands, *args, **kwargs):
        _export_time_as_index = kwargs.get("export_time", False)
        _drop_time_as_index = kwargs.get("drop_time", False)

        NewFrame = OldFrame.copy(deep=True) # Initialize with deep copy for safety

        # Check
        if _export_time_as_index and (NewFrame.index.name != "Time (s)" or "Time (s)" not in NewFrame.names):
            raise KeyError("To export time as an index we need to start with a time index or time column")

        # if _drop_time_as_index and (NewFrame.index.name != "Time (s)"):
        #    raise AssertionError("Warning: time is not the current index so it cannot be dropped")

        # Reset Index & Determine Whether Dropping Time
        NewFrame.reset_index(drop=_drop_time_as_index, inplace=True)

        if args:
            if 1 < args.__len__() < 3:
                _index_tuple = args[0]
                _levels_scalar_or_tuple = args[1]
                NewFrame = NewFrame.xs(args[0], args[1], drop_level=False)
            elif args.__len__() == 3:
                _index_tuple = args[0]
                _levels_scalar_or_tuple = args[1]
                _drop_level = args[2]
                NewFrame = NewFrame.xs(args[0], args[1], drop_level=_drop_level)
        else:
            _index_name = Commands[0]
            _subset_name = Commands[1]
            NewFrame.set_index(_index_name, drop=False, inplace=True)
            NewFrame.sort_index(inplace=True)
            NewFrame = NewFrame.loc[_subset_name].copy(deep=True)

        if _export_time_as_index:
            NewFrame.reset_index(drop=True, inplace=True)
            NewFrame.sort_index(inplace=True)
            NewFrame.set_index("Time (s)", drop=True, inplace=True)

        return NewFrame.copy(deep=True) # Return a deep copy for safety



class DeepLabModule:
    """
    Module for deep lab cut data in the form of a pandas dataframe

    **Class Methods**
        | **cls.loadData** : Function loads the deep lab cut data
    """
    def __init__(self, DataFolderDLC, DataFolderBehavioralExports):
        self.pre_trial_data, self.trial_data = \
            self.loadData(DataFolderDLC, DataFolderBehavioralExports)
        return

    @classmethod
    def loadData(cls, DataFolderDLC, DataFolderBehavioralExports):
        # noinspection GrazieInspection
        """
                Load DeepLabCut Data

                :param DataFolderDLC: Collected Data Folder object for deep lab cut folder
                :param DataFolderBehavioralExports:  Collected Data Folder object for behavioral exports folder
                :type DataFolderDLC: object
                :type DataFolderBehavioralExports: object
                :returns: pre_trial_data, trial_data
                """
        # extract paths
        PathDLC = DataFolderDLC.path
        PathBE = DataFolderBehavioralExports.path
        # Find dlc .csv's
        _dlc_csv_files = []
        for i in DataFolderDLC.files:
            if pathlib.Path("".join([PathDLC, "\\", i])).suffix == ".csv":
                _dlc_csv_files.append(i)
        # Locate Pre Trial csv
        _pre_trial_csv = DataFolderDLC.fileLocator(_dlc_csv_files, "PRETRIALSDLC")
        # Locate Trial csv
        _trial_csv = DataFolderDLC.fileLocator(_dlc_csv_files, "TRIALSDLC")

        # Find behavioral exports .csv's
        _be_csv_files = []
        for i in DataFolderBehavioralExports.files:
            if pathlib.Path("".join([PathBE, "\\", i])).suffix == ".csv":
                _be_csv_files.append(i)
        # Pre Trial Frame IDs
        _pre_trial_frame_ids_csv = DataFolderBehavioralExports.fileLocator(_be_csv_files, "preTrial")
        # Trial Frame IDs
        _trial_frame_ids_csv = DataFolderBehavioralExports.fileLocator(_be_csv_files, "frameIDs")

        # load data
        pre_trial = pd.read_csv("".join([PathDLC, "\\", _pre_trial_csv]), header=2)
        trial = pd.read_csv("".join([PathDLC, "\\", _trial_csv]), header=2)

        # load ids
        _pre_trial_frame_ids = np.genfromtxt("".join([PathBE, "\\", _pre_trial_frame_ids_csv]), delimiter=",").astype(int)
        _trial_frame_ids = np.genfromtxt("".join([PathBE, "\\", _trial_frame_ids_csv]), delimiter=",").astype(int)

        # attach ids
        pre_trial.index = _pre_trial_frame_ids
        pre_trial.index.name = "Trial"
        pre_trial.rename(columns={"coords": "Frame", "x": "X1", "y": "Y1", "likelihood": "likelihood1",
                                  "x.1": "X2", "y.1": "Y2", "likelihood.1": "likelihood2"}, inplace=True)
        trial.index = _trial_frame_ids
        trial.index.name = "Trial"
        trial.rename(columns={"coords": "Frame", "x": "X1", "y": "Y1", "likelihood": "likelihood1",
                                  "x.1": "X2", "y.1": "Y2", "likelihood.1": "likelihood2"}, inplace=True)

        return pre_trial, trial

    @classmethod
    def ConductPostProcessing(cls, RawData, **kwargs):
        _old_min = kwargs.get("old min", 0)
        _old_max = kwargs.get("old max", 800)
        _markers_idx = kwargs.get("marker_index", tuple([1, 2, 4, 5]))
        dlc_physical = RawData.copy()

        for _marker in _markers_idx:
            dlc_physical[dlc_physical.columns[_marker]] = cls.ConvertToPhysicalUnits(dlc_physical[dlc_physical.columns[_marker]].values,
                                                                                     _old_min, _old_max)
        return dlc_physical

    @classmethod
    def SegregateTrials(cls, Data, TrialNumber):
        """
        Returns the data for that particular trial

        :param Data: Pandas Dataframe where the index is the trial number
        :param TrialNumber: the desired trial number
        :return: Pandas dataframe containing only that trial
        """
        return Data.loc[TrialNumber]

    @classmethod
    def ConvertToPhysicalUnits(cls, fPositions, oldMin, oldMax, **kwargs):
        """
        Converts the burrow positions to physical units (mm)

        :param fPositions:
        :param oldMin:
        :param oldMax:
        :return:
        """
        _new_max = kwargs.get("new_max", 140)
        _new_min = kwargs.get("new_min", 0)

        _old_range = oldMax-oldMin
        _new_range = _new_max - _new_min

        return (((fPositions-oldMin)*_new_range)/_old_range)+_new_min


class BurrowPlots:
    """
    Class Containing Methods for Visualizing Burrow Behavior
    """
    def __init__(self):
        return

    @staticmethod
    def plotTrial(StageData, Trial):
        _fig = plt.figure(figsize=(12, 6))
        _ax = _fig.add_subplot(111)
        _ax.set_title("Trial " + str(Trial))
        _ax.set_xlabel("Seconds")
        _ax.set_ylabel("State")
        burrowX = np.linspace(0, 35, 350)
        base = np.full((350,), 0, dtype=int)
        CS = np.full((350,), 0, dtype=int)
        CS[0:150] = 1
        _ax.fill_between(burrowX, CS, base, linewidth=2, alpha=0.5, facecolor=[1, 1, 1], edgecolor=[0.65, 0.65, 0.65], label="CS+", hatch="///")
        _ax.plot(burrowX, StageData['Trial ' + str(Trial)].trial_data.gateData, linewidth=2, color="#ff4e4b", label="Burrow Gate")
        _ax.plot([25, 25], [0, 1], linewidth=2, linestyle='dashed', color="black", alpha=0.75, label="Expected UCS")
        plt.figlegend(loc='lower left')
        plt.ylim(-0.2, 1.2)


def temp_sort(DataFrame, index_name, subset):
    DataFrame.set_index(index_name, drop=False, inplace=True)
    DataFrame.sort_index(inplace=True)
    return DataFrame.loc[subset].copy(deep=True)
