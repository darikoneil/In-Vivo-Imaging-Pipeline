from __future__ import annotations
import numpy as np
import pickle as pkl
import pathlib
import pandas as pd
from tqdm.auto import tqdm
from typing import Tuple, List, Optional, Union
import ExperimentManagement.ExperimentHierarchy
from ExperimentManagement.ExperimentHierarchy import BehavioralStage, CollectedDataFolder
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from MigrationTools.Converters import convertFromPy27_Array, convertFromPy27_Dict


class FearConditioning(BehavioralStage):
    """
    Instance Factory for Fear Conditioning Data

    See BehavioralStage for more information

    **Keyword Arguments**
        | *TrialsPerStim* : Number of trials per stimulus (int, default 5)
        | *NumStim* : Number of stimuli (int, default 2)

    **Self Methods**
        | *self.generate_file_id* : Generate a file ID for a particular sort of data
        | *self.fill_folder_dictionary* : Function to index subfolders containing behavioral data

    **Class Methods**
        | *cls.load_analog_data* : Loads Analog Data from a burrow behavioral session
        | *cls.load_digital_data* : Loads Digital Data from a burrow behavioral session
        | *cls.load_state_data* : Loads State Data from a burrow behavioral session
        | *cls.load_dictionary_data* : Loads Dictionary Data from a burrow behavioral session
        | *cls.loadBehavioralData* : Loads Behavioral Data from a burrow behavioral session

    """
    def __init__(self, Meta: Tuple[str, str], Stage: str):


        super().__init__(Meta, Stage)
        self.fill_folder_dictionary()

        # PROTECT ME
        _stage = Stage

        # Super Protected
        self.__stage_id = _stage

    @property
    def stage_id(self) -> str:
        return self._FearConditioning__stage_id

    @property
    def num_trials(self) -> int:
        try:
            return self.trial_parameters.get("stimulusTypes").__len__()
        except KeyError:
            return 0

    @property
    def unique_stim(self) -> List[Any]:
        try:
            return list(np.unique(np.array(self.trial_parameters.get("stimulusTypes"))))
        except KeyError:
            return list(None)

    @property
    def num_stim(self) -> int:
        return self.unique_stim.__len__()

    @property
    def trials_per_stim(self) -> int:
        try:
            _trials_per_stim = [stimulus for stimulus in self.trial_parameters.get("stimulusTypes")
                                if stimulus == self.unique_stim[0]].__len__()
            if self.num_stim > 1:
                for _unique in self.unique_stim:
                    # make sure same number of all stimuli
                    assert(_trials_per_stim == [stimulus for stimulus in self.trial_parameters.get("stimulusTypes")
                                        if stimulus == _unique].__len__())
            return _trials_per_stim
        except KeyError:
            return 0

    @property
    def trial_groups(self) -> Union[Tuple[Tuple], None]:
        try:
            _trial_sets = []
            for _unique in self.unique_stim:
                _trial_sets.append(tuple(np.where(np.array(self.trial_parameters.get("stimulusTypes")) == _unique)[0]))
            _trial_sets = tuple(_trial_sets)
            return _trial_sets
        except KeyError:
            _ = [0]
            return None

    def load_base_behavior(self) -> Self:
        """
        Loads the basic behavioral data: analog, dictionary, digital, state, and CS identities

        :rtype: Any
        """

        print("Loading Base Data...")
        # Analog
        _analog_file = self.generate_file_id('Analog')
        _analog_data = FearConditioning.load_analog_data(_analog_file)
        if type(_analog_data) == str and _analog_data == "ERROR":
            return print("Could not find analog data!")

        # Digital
        _digital_file = self.generate_file_id('Digital')
        _digital_data = FearConditioning.load_digital_data(_digital_file)
        if type(_digital_data) == str and _digital_data == "ERROR":
            return print("Could not find digital data!")

        # State
        _state_file = self.generate_file_id('State')
        _state_data = FearConditioning.load_state_data(_state_file)
        if _state_data[0] == "ERROR": # 0 because it's an array of strings so ambiguous str comparison
            return print("Could not find state data!")

        # Dictionary
        _dictionary_file = self.generate_file_id('Dictionary')
        _dictionary_data = FearConditioning.load_dictionary_data(_dictionary_file)
        try:
            self.trial_parameters = _dictionary_data.copy() # For Safety
        except AttributeError:
            print(_dictionary_data)

        if _dictionary_data == "ERROR_FIND":
            return print("Could not find dictionary data!")
        elif _dictionary_data == "ERROR_READ":
            return print("Could not read dictionary data!")

        # ingest trial dictionary

        # Form Pandas DataFrame
        self.data, self.state_index, self.multi_index = self.organize_base_data(_analog_data, _digital_data,
                                                                                _state_data)

        # Add CS information
        self.data = self.merge_cs_index_into_dataframe(self.data, np.array(self.trial_parameters.get("stimulusTypes")))

        print("Finished.")

    def load_dlc_data(self, *args: Optional[Tuple[int, int]]) -> Self:
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
        _dlc.trial_data = DeepLabModule.convert_full_dataframe_to_physical_units(_dlc.trial_data, _old_min, _old_max,
                                                                                 ("X1", "X2"))
        _dlc.pre_trial_data = DeepLabModule.convert_full_dataframe_to_physical_units(_dlc.pre_trial_data, _old_min, _old_max,
                                                                                     ("X1", "X2"))
        _dlc.trial_data = DeepLabModule.convert_to_mean_zero(_dlc.trial_data, ("Y1", "Y2"))
        _dlc.pre_trial_data = DeepLabModule.convert_to_mean_zero(_dlc.pre_trial_data, ("Y1", "Y2"))

        # Check Efficacy
        try:
            assert (np.min(_dlc.trial_data["likelihood1"].to_numpy()) > 0.95)
            assert (np.min(_dlc.trial_data["likelihood2"].to_numpy()) > 0.95)
            assert (np.min(_dlc.pre_trial_data["likelihood1"].to_numpy()) > 0.95)
            assert (np.min(_dlc.pre_trial_data["likelihood2"].to_numpy()) > 0.95)
        except AssertionError:
            print("Deep Lab Cut model label insufficient. Re-train")
            return

        self.data = DeepLabModule.merge_dlc_data(self.data, _dlc, self.multi_index,
                                                                self.state_index)

    def load_bruker_data(self, Parameters) -> Self:
        """
        This function loads bruker data

        :rtype: Any
        """
        _analog_recordings = self.load_bruker_analog_recordings()
        if self.validate_bruker_recordings_completion(_analog_recordings, self.num_trials)[0]:
            self.data = self.sync_bruker_recordings(self.data.copy(deep=True),
                                                          _analog_recordings, self.meta, self.state_index,
                                                          ("State Integer", " TrialIndicator"), Parameters)
        else:
            print('Not Yet Implemented')

    def generate_file_id(self, SaveType: str) -> Union[str, None]:
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

    def fill_folder_dictionary(self) -> Self:
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
    def load_analog_data(Filename: str) -> np.ndarray:
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

    @staticmethod
    def load_digital_data(Filename: str) -> np.ndarray:
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

    @staticmethod
    def load_state_data(Filename: str) -> np.ndarray:
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

    @staticmethod
    def load_dictionary_data(Filename: str) -> dict:
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
        _trial_set = DataFrame['Trial Set'].to_numpy()
        _cs_column = np.full(_trial_set.__len__(), _nan_id, dtype=np.float64)

        for _cs in range(np.unique(CSIndex).__len__()):
            _cs_column[np.searchsorted(_trial_set, np.where(CSIndex == _cs)[0] + 1)] = _cs
            # + 1 -> Adjust because trial set is not zero indexed
        # Enter Nil Start, Maintain NaNs for forward fill
        _cs_column[0] = _nil_id

        _cs_series = pd.Series(_cs_column, index=DataFrame.index.to_numpy().copy())
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
            assert(np.where(np.diff(AnalogRecordings[" TrialIndicator"].to_numpy()) > 1)[0].__len__() <= NumTrials)
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

        detected_trials = np.where(np.diff(AnalogRecordings[" TrialIndicator"].to_numpy()) > 1)[0].__len__()
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
            return np.where(DataFrame["Trial Set"].to_numpy() <= Trial+1)[0]
        elif Direction == "End":
            return np.where(DataFrame["Trial Set"].to_numpy() >= NumTrials - Trial + 1)[0]
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
        ax1.plot(DataFrame.index.to_numpy(), DataFrame["State Integer"].to_numpy(), color="blue")
        ax1.plot(DataFrame.index.to_numpy(), DataFrame[" TrialIndicator"].to_numpy(), color="orange")

        ax2 = fig1.add_subplot(312)
        ax2.title.set_text("State Integer")
        ax2.plot(DataFrame.index.to_numpy(), DataFrame["State Integer"].to_numpy(), color="blue")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("State")

        ax3 = fig1.add_subplot(313)
        ax3.title.set_text("Trial Indicator")
        ax3.plot(DataFrame.index.to_numpy(), DataFrame[" TrialIndicator"].to_numpy(), color="orange")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Trial Flag")

        fig1.tight_layout()


class MethodsForPandasOrganization:
    """
    Simply a container for methods related to organization of behavioral data through a pandas dataframe
    """
    def __init__(self):
        return

    @staticmethod
    def safe_extract(OldFrame, Commands, *args, **kwargs):
        """
        Function for safe extraction
        DAO 11/24/2022 what am I?

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
        if _export_time_as_index and NewFrame.index.name != "Time (s)" and "Time (s)" not in NewFrame.columns.to_numpy():
            raise KeyError("To export time as an index we need to start with a time index or time column")

        if _reset_index and _drop_index and NewFrame.index.name not in NewFrame.columns.to_numpy():
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
    Module for importing deeplabcut data
    """
    def __init__(self, DataFolderDLC: CollectedDataFolder, DataFolderBehavioralExports: CollectedDataFolder):
        self.pre_trial_data, self.trial_data = \
            self.load_data(DataFolderDLC, DataFolderBehavioralExports)
        return

    @classmethod
    def load_data(cls, DataFolderDLC: CollectedDataFolder, DataFolderBehavioralExports: CollectedDataFolder) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        # Use -1 idx because newer file always would have larger tag on dlc file
        # DLC
        try:
            _pre_trial_csv = [file for file in DataFolderDLC.find_all_ext("csv") if "PRETRIAL" in file][-1]
            _trial_csv = [file for file in DataFolderDLC.find_all_ext("csv") if ("PRETRIAL" not in file and "TRIAL" in file)][-1]
        except IndexError:
            print("could not find csv data")
            return tuple(None)
        # BehavioralExports
        try:
            _pre_trial_frame_ids_csv = \
                [file for file in DataFolderBehavioralExports.find_all_ext("csv") if "preTrial" in file][-1]
            _trial_frame_ids_csv = \
                [file for file in DataFolderBehavioralExports.find_all_ext("csv") if
                 ("preTrial" not in file and "frameIDs" in file)][-1]
        except IndexError:
            print("could not find csv data")
            return tuple(None)

        # load data
        pre_trial = pd.read_csv(_pre_trial_csv, header=2)
        trial = pd.read_csv(_trial_csv, header=2)

        # load ids
        _pre_trial_frame_ids = np.genfromtxt(_pre_trial_frame_ids_csv, delimiter=",").astype(int)
        _trial_frame_ids = np.genfromtxt(_trial_frame_ids_csv, delimiter=",").astype(int)

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
    def conduct_post_processing(cls, RawData, **kwargs):
        _old_min = kwargs.get("old min", 0)
        _old_max = kwargs.get("old max", 800)
        _markers_idx = kwargs.get("marker_index", tuple([1, 2, 4, 5]))
        dlc_physical = RawData.copy()

        for _marker in _markers_idx:
            dlc_physical[dlc_physical.columns[_marker]] = cls.convert_to_physical_units(dlc_physical[dlc_physical.columns[_marker]].to_numpy(),
                                                                                        _old_min, _old_max)
        return dlc_physical

    @classmethod
    def segregate_trials(cls, Data, TrialNumber):
        """
        Returns the data for that particular trial

        :param Data: Pandas Dataframe where the index is the trial number
        :param TrialNumber: the desired trial number
        :return: Pandas dataframe containing only that trial
        """
        return Data.copy(deep=True).loc[TrialNumber]

    @classmethod
    def convert_to_physical_units(cls, fPositions, oldMin, oldMax, **kwargs):
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
    def convert_full_dataframe_to_physical_units(cls, DataFrame, oldMin, oldMax, idx, **kwargs):
        _new_max = kwargs.get("new_max", 140)
        _new_min = kwargs.get("new_min", 0)

        DataFrame = DataFrame.copy(deep=True)
        for i in idx:
            DataFrame[i] = DeepLabModule.convert_to_physical_units(DataFrame[i].to_numpy(), oldMin, oldMax)
        return DataFrame

    @classmethod
    def convert_to_mean_zero(cls, DataFrame, idx):
        DataFrame = DataFrame.copy(deep=True)
        for i in idx:
            DataFrame[i] = DataFrame[i].to_numpy() - np.mean(DataFrame[i].to_numpy())
        return DataFrame

    @classmethod
    def match_data_to_new_index(cls, OriginalVector, OriginalIndex, NewIndex, **kwargs):
        """
        Function to match data to a new index. can be done better, written when learning pandas

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

    @classmethod
    def merge_dlc_data(cls, DataFrame, DLC, MultiIndex, StateCastDict, **kwargs):
        _individual_series = []
        _num_trials = kwargs.get("num_trials", np.max(DLC.trial_data.index))
        _fps = kwargs.get("FPS", 30)

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
            _dlc = DLC.segregate_trials(DLC.trial_data, i)
            _num_frames = _dlc['X1'].__len__()
            _old_dlc_index = np.around(
                np.linspace(_trial.index.to_numpy()[0], _trial.index.to_numpy()[-1], _num_frames), decimals=3)
            _new_dlc_index = np.around(_trial.index.to_numpy(), decimals=3)

            # for each column add to new index
            _trial_data = pd.DataFrame(None, index=_new_dlc_index, dtype=np.float64)
            _trial_data.index.name = "Time (s)"
            for _column in _dlc.columns.to_numpy():
                _trial_data[_column] = cls.match_data_to_new_index(_dlc[_column].to_numpy(),
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
            _dlc = DLC.segregate_trials(DLC.pre_trial_data, i)
            _num_frames = _dlc['X1'].__len__()
            _old_dlc_index = np.floor(np.linspace(0, _pre_trial.index.to_numpy().__len__()-1, _num_frames))
            if _old_dlc_index.__len__() != np.unique(_old_dlc_index).__len__():
                raise AssertionError("You should probably fix this code now...")
            _new_dlc_index = np.arange(0, _pre_trial.index.to_numpy().__len__(), 1)

            # for each column add to new index
            _pre_trial_data = pd.DataFrame(None, index=_new_dlc_index, dtype=np.float64)
            # _pre_trial_data.index.name = "Time (s)"
            for _column in _dlc.columns.to_numpy():
                _pre_trial_data[_column] = cls.match_data_to_new_index(_dlc[_column].to_numpy(),
                                                                                                _old_dlc_index,
                                                                                                _new_dlc_index,
                                                                                                feedback=False)
            _pre_trial_data.set_index(_pre_trial.index, drop=True, inplace=True)
            _individual_series.append(_pre_trial_data)


        _concat_dataframe = pd.concat(_individual_series)
        _concat_dataframe.reindex(index=DataFrame.index)

        DataFrame = DataFrame.join(_concat_dataframe)
        DataFrame["X1"].fillna(value=np.nanmax(DataFrame["X1"].to_numpy()), inplace=True)
        DataFrame["X2"].fillna(value=np.nanmax(DataFrame["X2"].to_numpy()), inplace=True)
        DataFrame["Y1"].fillna(0, inplace=True)
        DataFrame["Y2"].fillna(0, inplace=True)
        DataFrame["likelihood1"].fillna(1, inplace=True)
        DataFrame["likelihood2"].fillna(1, inplace=True)
        return DataFrame


def plot_burrow_coordinates(Coordinates):
    _fig = plt.figure()
    _subplots_number = Coordinates.__len__()*100 + 10 + 0
    for _coord in range(Coordinates.__len__()):
        _subplots_number += 1
        _ax = _fig.add_subplot(_subplots_number)
        _ax.plot(Coordinates[_coord])
