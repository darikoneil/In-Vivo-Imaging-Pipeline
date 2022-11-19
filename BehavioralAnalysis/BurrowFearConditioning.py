from __future__ import annotations
import numpy as np
import pickle as pkl
import pathlib
# import pandas as pd
import pandas as pd
from tqdm.auto import tqdm
from typing import Tuple, List, Optional, Union
import ExperimentManagement.ExperimentHierarchy
from ExperimentManagement.ExperimentHierarchy import BehavioralStage, CollectedDataFolder
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from MigrationTools.Converters import convertFromPy27_Array, convertFromPy27_Dict

# Some of this stuff needs some serious refactoring & cleaning out old code
# Really all the behaviors do
# To-Do refactor out the indexing that can be done by native functions
# To-Do refactor out the old behavioral organization such that only the pandas remains for easier maintenance
# Yo some dis ugly


class FearConditioning(BehavioralStage):
    """
    Instance Factory for Fear Conditioning Data

    See BehavioralStage for more information

    **Keyword Arguments**
        | *TrialsPerStim* : Number of trials per stimulus (int, default 5)
        | *NumStim* : Number of stimuli (int, default 2)

    **Self Methods**
        | *self.generateFileID* : Generate a file ID for a particular sort of data
        | *self.fillFolderDictionary* : Function to index subfolders containing behavioral data

    **Class Methods**
        | *cls.loadAnalogData* : Loads Analog Data from a burrow behavioral session
        | *cls.loadDigitalData* : Loads Digital Data from a burrow behavioral session
        | *cls.loadStateData* : Loads State Data from a burrow behavioral session
        | *cls.loadDictionaryData* : Loads Dictionary Data from a burrow behavioral session
        | *cls.loadBehavioralData* : Loads Behavioral Data from a burrow behavioral session

    """
    def __init__(self, Meta: Tuple[str, str], Stage: str, **kwargs):


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

        self.multi_index = None
        self.trial_parameters = None

    @property
    def stage_id(self) -> str:
        return self._FearConditioning__stage_id

    @property
    def num_trials(self) -> int:
        return self._FearConditioning__num_trials

    @property
    def trials_per_stim(self) -> int:
        return self._FearConditioning__trials_per_stim

    @property
    def num_stim(self) -> int:
        return self._FearConditioning__num_stim

    def loadBaseBehavior(self) -> Self:
        """
        Loads the basic behavioral data: analog, dictionary, digital, state, and CS identities

        :rtype: Any
        """

        print("Loading Base Data...")
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
        try:
            self.trial_parameters = _dictionary_data.copy() # For Safety
        except AttributeError:
            print(_dictionary_data)

        if _dictionary_data == "ERROR_FIND":
            return print("Could not find dictionary data!")
        elif _dictionary_data == "ERROR_READ":
            return print("Could not read dictionary data!")

        # Form Pandas DataFrame
        self.data, self.state_index, self.multi_index = self.organize_base_data(_analog_data, _digital_data,
                                                                                _state_data)

        # Add CS information
        self.data = self.merge_cs_index_into_dataframe(self.data, np.array(self.trial_parameters.get("stimulusTypes")))

        print("Finished.")

    def load_dlc_data(self, *args: Tuple[int, int]) -> Self:
        """
        This function loads deep lab cut data

        :param args: Optional input indicating min/max of video actuator range
        :type args: Tuple[int, int]
        :rtype: Any
        """
        if args:
            _old_min = args[0]
            _old_max = args[1]
        else:
            _old_min = 0
            _old_max = 800

        _dlc = DeepLabModule(self.folder_dictionary['deep_lab_cut_data'], self.folder_dictionary['behavioral_exports'])
        _dlc.trial_data = DeepLabModule.ConvertFullDataFrameToPhysicalUnits(_dlc.trial_data, _old_min, _old_max,
                                                                            ("X1", "X2"))
        _dlc.pre_trial_data = DeepLabModule.ConvertFullDataFrameToPhysicalUnits(_dlc.pre_trial_data, _old_min, _old_max,
                                                                                ("X1", "X2"))
        _dlc.trial_data = DeepLabModule.ConvertToMeanZero(_dlc.trial_data, ("Y1", "Y2"))
        _dlc.pre_trial_data = DeepLabModule.ConvertToMeanZero(_dlc.pre_trial_data, ("Y1", "Y2"))

        # Check Efficacy
        try:
            assert (np.min(_dlc.trial_data["likelihood1"].values) > 0.95)
            assert (np.min(_dlc.trial_data["likelihood2"].values) > 0.95)
            assert (np.min(_dlc.pre_trial_data["likelihood1"].values) > 0.95)
            assert (np.min(_dlc.pre_trial_data["likelihood2"].values) > 0.95)
        except AssertionError:
            print("Deep Lab Cut model label insufficient. Re-train")
            return

        self.data = MethodsForPandasOrganization.merge_dlc_data(self.data, _dlc, self.multi_index,
                                                                self.state_index)

    def load_bruker_data(self, Parameters) -> Self:
        """
        This function loads bruker data

        :rtype: Any
        """
        _analog_recordings = self.loadBrukerAnalogRecordings()
        if self.validate_bruker_recordings_completion(_analog_recordings, self.num_trials)[0]:
            self.data = self.sync_bruker_recordings(self.data.copy(deep=True),
                                                          _analog_recordings, self.meta, self.state_index,
                                                          ("State Integer", " TrialIndicator"), Parameters)
        else:
            print('Not Yet Implemented')

    def loadBehavioralData(self, **kwargs) -> Self:
        """
        Master function that loads the following data -> analog, digital, state, dictionary

        """
        _old_min = kwargs.get("min", 0)
        _old_max = kwargs.get("max", 800)

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
        try:
            self.trial_parameters = _dictionary_data.copy() # For Safety
        except AttributeError:
            print(_dictionary_data)

        if _dictionary_data == "ERROR_FIND":
            return print("Could not find dictionary data!")
        elif _dictionary_data == "ERROR_READ":
            return print("Could not read dictionary data!")

        _dlc = DeepLabModule(self.folder_dictionary['deep_lab_cut_data'], self.folder_dictionary['behavioral_exports'])
        _dlc.trial_data = DeepLabModule.ConvertFullDataFrameToPhysicalUnits(_dlc.trial_data, _old_min, _old_max,
                                                                                                    ("X1", "X2"))
        _dlc.pre_trial_data = DeepLabModule.ConvertFullDataFrameToPhysicalUnits(_dlc.pre_trial_data, _old_min, _old_max,
                                                                                                            ("X1", "X2"))
        _dlc.trial_data = DeepLabModule.ConvertToMeanZero(_dlc.trial_data, ("Y1", "Y2"))
        _dlc.pre_trial_data = DeepLabModule.ConvertToMeanZero(_dlc.pre_trial_data, ("Y1", "Y2"))

        # Check Efficacy
        try:
            assert(np.min(_dlc.trial_data["likelihood1"].values) > 0.95)
            assert (np.min(_dlc.trial_data["likelihood2"].values) > 0.95)
            assert (np.min(_dlc.pre_trial_data["likelihood1"].values) > 0.95)
            assert (np.min(_dlc.pre_trial_data["likelihood2"].values) > 0.95)
        except AssertionError:
            print("Deep Lab Cut model label insufficient. Re-train")
            return

            # noinspection PyTypeChecker
            # Copies for safety > No-copies for speed (avoid pandas gotchas)
        # noinspection PyTypeChecker
        self.data, self.state_index, self.multi_index = \
            MethodsForPandasOrganization.ExportPandasDataFrame(
                _analog_data.copy(), _digital_data.copy(), _state_data.copy())
        # merge cs index with data frame
        self.data = \
            MethodsForPandasOrganization.merge_cs_index_into_dataframe(
                self.data.copy(), np.array(self.trial_parameters.get("stimulusTypes"), dtype=np.float64))
        # merge deeplabcut with data frame
        self.data = MethodsForPandasOrganization.merge_dlc_data(self.data, _dlc, self.multi_index,
                                                                            self.state_index)
        # merge imaging data
        _analog_recordings = self.loadBrukerAnalogRecordings()
        if self.validate_bruker_recordings_completion(_analog_recordings, self.num_trials)[0]:
            self.data = self.sync_bruker_recordings(self.data.copy(deep=True),
                                                          _analog_recordings, self.meta, self.state_index,
                                                          ("State Integer", " TrialIndicator"))
            self.data = self.sync_downsampled_images(self.data.copy(deep=True), self.meta)
        else:
            try:
                self.mergeAdditionalBruker(_analog_recordings)
                # noinspection PyArgumentList
                self.data = self.sync_downsampled_images(self.data.copy(deep=True), self.meta,
                                                          two_files=True,
                                                          second_meta=self.loadAdditionalBrukerMetaData(str(2)))
            except KeyError:
                print("Only one of multiple bruker datasets loaded")

    def generateFileID(self, SaveType: str) -> Union[str, None]:
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

    def fillFolderDictionary(self) -> Self:
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

    def mergeAdditionalBruker(self, AnalogRecordings: pd.DataFrame) -> Self:
        """
        Merge a second bruker dataset's analog recordings

        :rtype: None
        """
        _tag = 2
        _, _detected_trials_1 = self.validate_bruker_recordings_completion(AnalogRecordings, self.num_trials)
        _frames_1 = self.index_trial_subset_for_bruker_sync(self.data, _detected_trials_1,
                                                            self.num_trials, "Start")
        _data_frame_1 = self.data.iloc[_frames_1[0]:_frames_1[-1]].copy(deep=True)
        _data_frame_1 = self.sync_bruker_recordings(_data_frame_1, AnalogRecordings, self.meta,
                                                    self.state_index,  ("State Integer", " TrialIndicator"))
        _num_frames_in_first_set = self.meta.imaging_metadata.get("relativeTimes").__len__()
        _analog_recordings_2 = self.loadAdditionalBrukerAnalogRecordings(str(2))
        _meta_data_2 = self.loadAdditionalBrukerMetaData(str(2))
        _, _detected_trials_2 = self.validate_bruker_recordings_completion(_analog_recordings_2, self.num_trials)
        _frames_2 = self.index_trial_subset_for_bruker_sync(self.data, _detected_trials_2,
                                                            self.num_trials, "End")
        _data_frame_2 = self.data.iloc[_frames_2[0]:_frames_2[-1]].copy(deep=True)
        _data_frame_2 = self.sync_bruker_recordings(_data_frame_2, _analog_recordings_2, _meta_data_2,
                                                    self.state_index, ("State Integer", " TrialIndicator"))
        _data_frame_2[["Imaging Frame", "[FILLED] Imaging Frame"]] = \
            _data_frame_2[["Imaging Frame", "[FILLED] Imaging Frame"]] + _num_frames_in_first_set
        _data_frame_concat = pd.concat([_data_frame_1, _data_frame_2])
        _data_frame_concat = _data_frame_concat[[" TrialIndicator", " UCSIndicator",
                                                 "Imaging Frame", "[FILLED] Imaging Frame"]]
        self.data = self.data.join(_data_frame_concat)

    @classmethod
    def loadAnalogData(cls, Filename: str) -> np.ndarray:
        """
        Loads Analog Data from a burrow behavioral session

        :param Filename: Numpy file containing analog data
        :type Filename: str
        :return: Analog Data
        :rtype: Any
        """
        try:
            analogData = np.load(Filename)
        except FileNotFoundError:
            return "ERROR"

        return analogData

    @classmethod
    def loadDigitalData(cls, Filename: str) -> np.ndarray:
        """
        Loads Digital Data from a burrow behavioral session

        :param Filename: Numpy file containing digital data
        :type Filename: str
        :return: Digital Data
        :rtype: Any
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
    def loadStateData(cls, Filename: str) -> np.ndarray:
        """
        Loads State Data from a burrow behavioral session

        :param Filename: Numpy file containing state data
        :type Filename: str
        :return: State Data
        :rtype: Any
        """
        try:
            stateData = np.load(Filename)
        except FileNotFoundError:
            return "ERROR"

        stateData = convertFromPy27_Array(stateData)
        return stateData

    @classmethod
    def loadDictionaryData(cls, Filename: str) -> dict:
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
                    dictionaryData = convertFromPy27_Dict(dictionaryData)
            except Exception:
                return "ERROR_READ"



        return dictionaryData

    @staticmethod
    def merge_cs_index_into_dataframe(DataFrame: pd.DataFrame, CSIndex: np.ndarray) -> pd.DataFrame:
        """
        Merged CS identities into dataframe

        :param DataFrame: Data
        :type DataFrame: pd.DataFrame
        :param CSIndex: CS Index
        type CSIndex: Any
        :return: DataFrame with CS identities
        :rtype: pd.DataFrame
        """
        # 0  = CS+, 1 = CS-, ..., Unique CS + 1 = Nil
        _nil_id = np.unique(CSIndex).__len__()  # Equivalent to the number cs id + 1 when considering zero-indexing
        _nan_id = _nil_id + 1
        _trial_set = DataFrame['Trial Set'].values
        _cs_column = np.full(_trial_set.__len__(), _nan_id, dtype=np.float64)

        for _cs in range(np.unique(CSIndex).__len__()):
            _cs_column[np.searchsorted(_trial_set, np.where(CSIndex == _cs)[0] + 1)] = _cs
            # + 1 -> Adjust because trial set is not zero indexed
        # Enter Nil Start, Maintain NaNs for forward fill
        _cs_column[0] = _nil_id

        _cs_series = pd.Series(_cs_column, index=DataFrame.index.values.copy())
        _cs_series[_cs_series == _nan_id] = np.nan
        _cs_series.ffill(inplace=True)
        _cs_series.name = "CS"

        DataFrame = DataFrame.join(_cs_series, on="Time (s)")

        return DataFrame

    @staticmethod
    def validate_bruker_recordings_labels(AnalogRecordings: pd.DataFrame, NumTrials: int) -> pd.DataFrame:
        """
        Validate correct labeling of indicator vector for syncing data

        :param AnalogRecordings: Bruker analog dataset
        :param NumTrials: number of trials
        :return: AnalogRecordings properly labeled
        :rtype: pd.Dataframe
        """

        try:
            assert(np.where(np.diff(AnalogRecordings[" TrialIndicator"].values) > 1)[0].__len__() <= NumTrials)
            return AnalogRecordings
        except AssertionError:
            print(" Analog Channel Labels are Swapped")
            AnalogRecordings = AnalogRecordings.rename(
                columns={" TrialIndicator": " UCSIndicator", " UCSIndicator": " TrialIndicator"})
            return AnalogRecordings

    @staticmethod
    def validate_bruker_recordings_completion(AnalogRecordings: pd.DataFrame, NumTrials: int) -> Tuple[bool, int]:
        """
        Determines whether bruker analog dataset contains all trials

        :param AnalogRecordings: Bruker analog dataset
        :type AnalogRecordings: pd.DataFrame
        :param NumTrials: number of trials
        :type NumTrials: int
        :return: True if dataset completed and number of detected trials
        :rtype: tuple[bool, int]
        """

        detected_trials = np.where(np.diff(AnalogRecordings[" TrialIndicator"].values) > 1)[0].__len__()
        try:
            assert(detected_trials == NumTrials)
            return True, detected_trials
        except AssertionError:
            return False, detected_trials

    @staticmethod
    def index_trial_subset_for_bruker_sync(DataFrame: pd.DataFrame, Trial: int,
                                           NumTrials: int, Direction: str) -> Union[pd.DataFrame, None]:
        """
        Subsets the trials to sync with bruker data

        :param DataFrame: Data
        :param Trial: specific trial
        :param NumTrials: total trials
        :param Direction: Whether indexing from start or end
        :return: a subset of the dataframe
        :rtype: pd.DataFrame
        """
        if Direction == "Start":
            return np.where(DataFrame["Trial Set"].values <= Trial+1)[0]
        elif Direction == "End":
            return np.where(DataFrame["Trial Set"].values >= NumTrials - Trial + 1)[0]
        else:
            print("Please specify the direction as Start or End")
            return

    @staticmethod
    def check_sync_plot(DataFrame: pd.DataFrame) -> None:
        """
        Visualized syncing of the data

        :param DataFrame: The data
        :return: Plots in matplotlib
        :rtype: None
        """
        fig1 = plt.figure(1)

        ax1 = fig1.add_subplot(311)
        ax1.title.set_text("Merge")
        ax1.plot(DataFrame.index.values, DataFrame["State Integer"].values, color="blue")
        ax1.plot(DataFrame.index.values, DataFrame[" TrialIndicator"].values, color="orange")

        ax2 = fig1.add_subplot(312)
        ax2.title.set_text("State Integer")
        ax2.plot(DataFrame.index.values, DataFrame["State Integer"].values, color="blue")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("State")

        ax3 = fig1.add_subplot(313)
        ax3.title.set_text("Trial Indicator")
        ax3.plot(DataFrame.index.values, DataFrame[" TrialIndicator"].values, color="orange")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Trial Flag")

        fig1.tight_layout()

    # Possibly Deprecated

    @classmethod
    def _identifyTrialValence(cls, csIndexFile: str) -> Tuple[int, int]:
        """
        Returns the index for CS+ and CS- trials from a .csv indicator file

        :param csIndexFile: Path to cs Index File (.csv)
        :type csIndexFile: str
        :return: Tuple of CS+ and CS- trials
        :rtype: tuple[int, int]
        """
        assert(pathlib.Path(csIndexFile).suffix == ".csv")
        _csIndex = np.genfromtxt(csIndexFile, int, delimiter=",")
        PlusTrials = np.where(_csIndex == 0)
        MinusTrials = np.where(_csIndex == 1)
        return PlusTrials[0], MinusTrials[0]

    @classmethod
    def _reorganizeData(cls, NeuralActivity, TrialFeatures, ImFreq, **kwargs):
        print("Maybe Deprecated? Warning!")
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


class MethodsForPandasOrganization:
    """
    Simply a container for methods related to organization of behavioral data through a pandas dataframe
    """
    def __init__(self):
        return

    @classmethod
    def merge_dlc_data(cls, DataFrame, DLC, MultiIndex, StateCastDict, **kwargs):
        _individual_series = []
        _num_trials = kwargs.get("num_trials", np.max(DLC.trial_data.index))
        _fps = kwargs.get("fps", 30)

        # Assert MultiIndex & Columns Contain Time

        for i in tqdm(
                range(_num_trials),
                total=_num_trials,
                desc="Merging DLC Data... Trials...",
                disable=False,
                ):
            if i == 0:
                # Do this to skip the "Habituation" Index
                continue
            # Trials
            _trial = MethodsForPandasOrganization.safe_extract(DataFrame, None, (StateCastDict.get("Trial"), i),
                                                               (0, 1), False, multi_index=MultiIndex,
                                                               reset=True, export_time=True)
            if _trial.index.name != "Time (s)":
                raise KeyError("The exported index must be time!")
            _dlc = DLC.SegregateTrials(DLC.trial_data, i)
            _num_frames = _dlc['X1'].__len__()
            _old_dlc_index = np.around(
                np.linspace(_trial.index.values[0], _trial.index.values[-1], _num_frames), decimals=3)
            _new_dlc_index = np.around(_trial.index.values, decimals=3)

            # for each column add to new index
            _trial_data = pd.DataFrame(None, index=_new_dlc_index, dtype=np.float64)
            _trial_data.index.name = "Time (s)"
            for _column in _dlc.columns.values:
                _trial_data[_column] = MethodsForPandasOrganization.match_data_to_new_index(_dlc[_column].values,
                                                                                            _old_dlc_index,
                                                                                            _new_dlc_index,
                                                                                            feedback=False)
            _individual_series.append(_trial_data)

        for i in tqdm(
                range(_num_trials),
                total=_num_trials,
                desc="Merging DLC Data... Pre-Trials...",
                disable=False,
                ):
            if i == 0:
                # Do this to skip the "Habituation" Index
                continue
            # Trials
            _pre_trial = MethodsForPandasOrganization.safe_extract(DataFrame, None, (StateCastDict.get("PreTrial"), i),
                                                                   (0, 1), False, multi_index=MultiIndex,
                                                                    reset=True, export_time=True)
            if _pre_trial.index.name != "Time (s)":
                raise KeyError("The exported index must be time!")
            _dlc = DLC.SegregateTrials(DLC.pre_trial_data, i)
            _num_frames = _dlc['X1'].__len__()
            _old_dlc_index = np.floor(np.linspace(0, _pre_trial.index.values.__len__()-1, _num_frames))
            if _old_dlc_index.__len__() != np.unique(_old_dlc_index).__len__():
                raise AssertionError("You should probably fix this code now...")
            _new_dlc_index = np.arange(0, _pre_trial.index.values.__len__(), 1)

            # for each column add to new index
            _pre_trial_data = pd.DataFrame(None, index=_new_dlc_index, dtype=np.float64)
            # _pre_trial_data.index.name = "Time (s)"
            for _column in _dlc.columns.values:
                _pre_trial_data[_column] = MethodsForPandasOrganization.match_data_to_new_index(_dlc[_column].values,
                                                                                                _old_dlc_index,
                                                                                                _new_dlc_index,
                                                                                                feedback=False)
            _pre_trial_data.set_index(_pre_trial.index, drop=True, inplace=True)
            _individual_series.append(_pre_trial_data)


        _concat_dataframe = pd.concat(_individual_series)
        _concat_dataframe.reindex(index=DataFrame.index)

        DataFrame = DataFrame.join(_concat_dataframe)
        DataFrame["X1"].fillna(value=np.nanmax(DataFrame["X1"].values), inplace=True)
        DataFrame["X2"].fillna(value=np.nanmax(DataFrame["X2"].values), inplace=True)
        DataFrame["Y1"].fillna(0, inplace=True)
        DataFrame["Y2"].fillna(0, inplace=True)
        DataFrame["likelihood1"].fillna(1, inplace=True)
        DataFrame["likelihood2"].fillna(1, inplace=True)
        return DataFrame

    @classmethod
    def merge_bruker_data(cls, DataFrame, BrukerData, MultiIndex, StateCastDict, **kwargs):

        return

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
        _feedback = kwargs.get("feedback", True)

        if _is_string:
            new_vector = np.full(NewIndex.__len__(), '', dtype="<U21")
        else:
            new_vector = np.full(NewIndex.__len__(), 69, dtype=OriginalVector.dtype)

        if _feedback:
            for _sample in tqdm(
                    range(OriginalIndex.__len__()),
                    total=OriginalIndex.__len__(),
                    desc="Generating New Index",
                    disable=False
            ):
                _timestamp = OriginalIndex[_sample]
                idx = np.where(NewIndex == _timestamp)[0][0]
                new_vector[idx] = OriginalVector[_sample]
        else:
            for _sample in range(OriginalIndex.__len__()):
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
    def safe_extract(OldFrame, Commands, *args, **kwargs):
        """
        Function for safe extraction

        :param OldFrame: Original DataFrame
        :type OldFrame: pd.DataFrame
        :param Commands: Tuple where Tuple[0] is index name & Tuple[1] is subset name
        :type Commands: tuple or None
        :param args: index_tuple, levels_scalar_or_tuple, drop_level in that order
        :param kwargs:
        :return:
        """
        _export_time_as_index = kwargs.get("export_time", False)
        _drop_index = kwargs.get("drop_index", False)
        _reset_index = kwargs.get("reset", False)
        _multi_index = kwargs.get("multi_index", None)

        NewFrame = OldFrame.copy(deep=True) # Initialize with deep copy for safety

        # Check
        if _export_time_as_index and NewFrame.index.name != "Time (s)" and "Time (s)" not in NewFrame.columns.values:
            raise KeyError("To export time as an index we need to start with a time index or time column")

        if _reset_index and _drop_index and NewFrame.index.name not in NewFrame.columns.values:
            raise AssertionError("Warning: index is not duplicated in a column and would be forever lost!")

        # Reset Index & Determine Whether Dropping Time
        if _reset_index:
            NewFrame.reset_index(drop=_drop_index, inplace=True)

        if _multi_index is not None:
            NewFrame.set_index(_multi_index, drop=False, inplace=True)

        if args:
            if 1 < args.__len__() < 3:
                _index_tuple = args[0]
                _levels_scalar_or_tuple = args[1]

                NewFrame = NewFrame.xs(args[0], args[1], drop_level=False)
            elif args.__len__() == 3:
                _index_tuple = args[0]
                _levels_scalar_or_tuple = args[1]
                _drop_level = args[2]
                NewFrame = NewFrame.xs(_index_tuple, level=_levels_scalar_or_tuple, drop_level=_drop_level)
        else:
            _index_name = Commands[0]
            _subset_name = Commands[1]
            NewFrame.set_index(_index_name, drop=False, inplace=True)
            NewFrame.sort_index(inplace=True)
            NewFrame = NewFrame.loc[_subset_name].copy(deep=True)

        if _export_time_as_index:
            NewFrame.reset_index(drop=True, inplace=True)
            NewFrame.set_index("Time (s)", drop=True, inplace=True)
            NewFrame.sort_index(inplace=True)

        return NewFrame.copy(deep=True) # Return a deep copy for safety


class DeepLabModule:
    """
    Module for deep lab cut data in the form of a pandas dataframe

    **Class Methods**
        |
        | **cls.loadData** : Function loads the deep lab cut data
        |
    """
    def __init__(self, DataFolderDLC: CollectedDataFolder, DataFolderBehavioralExports: CollectedDataFolder):
        self.pre_trial_data, self.trial_data = \
            self.loadData(DataFolderDLC, DataFolderBehavioralExports)
        return

    @classmethod
    def loadData(cls, DataFolderDLC: CollectedDataFolder, DataFolderBehavioralExports: CollectedDataFolder) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # noinspection GrazieInspection
        """
                Load DeepLabCut Data

                :param DataFolderDLC: Collected Data Folder object for deep lab cut folder
                :param DataFolderBehavioralExports:  Collected Data Folder object for behavioral exports folder
                :type DataFolderDLC: object
                :type DataFolderBehavioralExports: object
                :returns: pre_trial_data, trial_data
                :rtype: tuple[pd.DataFrame, pd.DataFrame]
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
        return Data.copy(deep=True).loc[TrialNumber]

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

    @classmethod
    def ConvertFullDataFrameToPhysicalUnits(cls, DataFrame, oldMin, oldMax, idx, **kwargs):
        _new_max = kwargs.get("new_max", 140)
        _new_min = kwargs.get("new_min", 0)

        DataFrame = DataFrame.copy(deep=True)
        for i in idx:
            DataFrame[i] = DeepLabModule.ConvertToPhysicalUnits(DataFrame[i].values, oldMin, oldMax)
        return DataFrame

    @classmethod
    def ConvertToMeanZero(cls, DataFrame, idx):
        DataFrame = DataFrame.copy(deep=True)
        for i in idx:
            DataFrame[i] = DataFrame[i].values - np.mean(DataFrame[i].values)
        return DataFrame
