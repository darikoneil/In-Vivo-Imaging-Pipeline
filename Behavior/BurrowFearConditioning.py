from __future__ import annotations
import numpy as np
import pickle as pkl
import pathlib
import pandas as pd
from tqdm.auto import tqdm
from typing import Tuple, List, Optional, Union
from itertools import product
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import sys

import ExperimentManagement.ExperimentHierarchy
from ExperimentManagement.ExperimentHierarchy import BehavioralStage, CollectedDataFolder
from MigrationTools.Converters import convertFromPy27_Array, convertFromPy27_Dict
from Behavior.Utilities import extract_specific_data, lowpass_filter


class FearConditioning(BehavioralStage):
    """
    Instance Factory for Fear Conditioning Data

    See BehavioralStage for more information
    """
    def __init__(self, Meta: Tuple[str, str], Stage: str):

        super().__init__(Meta, Stage)

        self._fill_behavior_folder_dictionary()

        # noinspection PyBroadException
        # try:
        #    self.load_data()
        # except Exception:
        #    print(sys.exc_info())

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

    def load_data(self, ImagingParameters: Optional[Union[dict, list[dict]]] = None, *args: Optional[Tuple[int, int]],
                  **kwargs) -> Self:
        """
        Loads all data (Convenience Function)

        :param ImagingParameters: Parameters for some imaging dataset
        :param args: Optional input indicating min/max of video actuator range
        :type args: Tuple[int, int]
        :param kwargs: passed to internal functions taking kwargs
        :rtype: Any
        """

        if self.data is not None:
            self.data = pd.DataFrame()


        print("\nLoading All Data...\n")

        self._load_base_behavior()

        if args:
            # noinspection PyArgumentList
            self._load_dlc_data(*args, **kwargs)
        else:
            # noinspection PyArgumentList
            self._load_dlc_data(**kwargs)

        if ImagingParameters is not None:
            self._load_bruker_meta_data()

        if ImagingParameters is not None:
            if isinstance(ImagingParameters, dict):
                self.data = self._sync_bruker_recordings(self.data, self._load_bruker_analog_recordings(), self.meta,
                                                         self.state_index, ("State Integer", " TrialIndicator"), ImagingParameters)
                if ImagingParameters.get(("preprocessing", "grouped-z project bin size")):
                    self.data = self._sync_grouped_z_projected_images(self.data, self.meta, ImagingParameters)
            elif isinstance(ImagingParameters, list) and isinstance(ImagingParameters[-1], dict):
                self.data = self._sync_bruker_recordings(self.data, self._load_bruker_analog_recordings(), self.meta,
                                                         self.state_index, ("State Integer", " TrialIndicator"), ImagingParameters[0])
                for _sampling in range(1, ImagingParameters.__len__(), 1):
                    self.data = self._sync_grouped_z_projected_images(
                        self.data, self.meta, ImagingParameters[_sampling],
                        "".join(["Downsampled Imaging Frame Set ", str(_sampling)]))

        print("\nFinished loading all data.")

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

    def _load_base_behavior(self) -> Self:
        """
        Loads the basic behavioral data: analog, dictionary, digital, state, and CS identities

        :rtype: Any
        """
        _dictionary_file = \
            self.folder_dictionary.get("raw_behavioral_data").find_matching_files("StimulusInfo")[0]
        _dictionary_data = FearConditioning._load_dictionary_data(_dictionary_file)
        try:
            self.trial_parameters = _dictionary_data.copy() # For Safety
        except AttributeError:
            print(_dictionary_data)


        print("Loading Base Data...")
        # Analog
        _analog_file = self._generate_file_id('Analog')
        _analog_data = FearConditioning._load_analog_data(_analog_file)
        if type(_analog_data) == str and _analog_data == "ERROR":
            return print("Could not find analog data!")

        # Digital
        _digital_file = self._generate_file_id('Digital')
        _digital_data = FearConditioning._load_digital_data(_digital_file)
        if type(_digital_data) == str and _digital_data == "ERROR":
            return print("Could not find digital data!")

        # State
        _state_file = self._generate_file_id('State')
        _state_data = FearConditioning._load_state_data(_state_file)
        if _state_data[0] == "ERROR": # 0 because it's an array of strings so ambiguous str comparison
            return print("Could not find state data!")

        # Dictionary
        _dictionary_file = self._generate_file_id('Dictionary')
        _dictionary_data = FearConditioning._load_dictionary_data(_dictionary_file)
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
        self.data, self.state_index, self.multi_index = self._organize_base_data(_analog_data, _digital_data,
                                                                                 _state_data)

        # Add CS information
        self.data = self._merge_cs_index_into_dataframe(self.data, np.array(self.trial_parameters.get("stimulusTypes")))

        # Post-Processing
        self.data["Force"] = lowpass_filter(self.data["Force"].to_numpy(), 1000, 100)

        print("Finished.")

    def _load_dlc_data(self, *args: Optional[Tuple[int, int]], **kwargs: bool) -> Self:
        """
        This function loads deep lab cut data

        :param args: Optional input indicating min/max of video actuator range
        :type args: Tuple[int, int]
        :rtype: Any
        """
        print("\nLoading and Merging Deep Lab Cut Data...")

        if args:
            try:
                _old_min = args[0]
                _old_max = args[1]
            except IndexError:
                _old_min = args[0][0]
                _old_max = args[0][1]
        else:
            _old_min = 0
            _old_max = 800

        _dlc = DeepLabData(self.folder_dictionary['deep_lab_cut_data'], self.folder_dictionary['behavioral_exports'])

        _convert = kwargs.get("convert", True)

        if _convert:
            _dlc.trial_data = DeepLabData.convert_dataframe_to_physical_units(_dlc.trial_data, _old_min, _old_max,
                                                                              ("X1", "X2"))
            _dlc.pre_trial_data = DeepLabData.convert_dataframe_to_physical_units(_dlc.pre_trial_data, _old_min, _old_max,
                                                                                  ("X1", "X2"))
            _dlc.trial_data = DeepLabData.convert_to_mean_zero(_dlc.trial_data, ("Y1", "Y2"))
            _dlc.pre_trial_data = DeepLabData.convert_to_mean_zero(_dlc.pre_trial_data, ("Y1", "Y2"))

        # Check Efficacy
        try:
            assert (np.min(_dlc.trial_data["likelihood1"].to_numpy()) > 0.95)
            assert (np.min(_dlc.trial_data["likelihood2"].to_numpy()) > 0.95)
            assert (np.min(_dlc.pre_trial_data["likelihood1"].to_numpy()) > 0.95)
            assert (np.min(_dlc.pre_trial_data["likelihood2"].to_numpy()) > 0.95)
        except AssertionError:
            print("Deep Lab Cut model label insufficient. Re-train")
            return

        # noinspection PyArgumentList
        self.data = DeepLabData.merge_dlc_data(self.data, _dlc, self.state_index)

        print("\nFinished.")

    def _generate_file_id(self, SaveType: str) -> Union[str, None]:
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

    def _fill_behavior_folder_dictionary(self) -> Self:
        """
        Constructs behavior folder data folder structures

        :rtype: Any
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

    @staticmethod
    def _load_analog_data(Filename: str) -> np.ndarray:
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
    def _load_digital_data(Filename: str) -> np.ndarray:
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
    def _load_state_data(Filename: str) -> np.ndarray:
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
    def _load_dictionary_data(Filename: str) -> dict:
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
    def _merge_cs_index_into_dataframe(DataFrame: pd.DataFrame, CSIndex: np.ndarray) -> pd.DataFrame:
        """
        Merged CS identities into dataframe

        :param DataFrame: Data
        :type DataFrame: pd.DataFrame
        :param CSIndex: CS Index
        type CSIndex: Any
        :return: DataFrame with CS identities
        :rtype: pd.DataFrame
        """

        if isinstance(CSIndex, list):
            CSIndex = np.array(CSIndex)

        # 0  = CS+, 1 = CS-, ..., Unique CS + 1 = Nil
        _nil_id = np.unique(CSIndex).__len__()  # Equivalent to the number cs id + 1 when considering zero-indexing
        _nan_id = _nil_id + 1
        _trial_set = DataFrame['Trial Set'].to_numpy()
        _cs_column = np.full(_trial_set.__len__(), _nan_id, dtype=np.float64)

        for _cs in range(np.unique(CSIndex).__len__()):
            _cs_column[np.searchsorted(_trial_set, np.where(CSIndex == _cs)[0])] = _cs
            # + 1 -> Adjust because trial set is not zero indexed
            # Remove + 1 on 11/30/2022 given moving towards zero index with hab as -1

        # Enter Nil Start, Maintain NaNs for forward fill
        _cs_column[0] = _nil_id

        _cs_series = pd.Series(_cs_column, index=DataFrame.index.to_numpy().copy())
        _cs_series[_cs_series == _nan_id] = np.nan
        _cs_series.ffill(inplace=True)
        _cs_series.name = "CS"

        DataFrame = DataFrame.join(_cs_series, on="Time (s)")

        DataFrame = DataFrame.reindex(columns=sorted(DataFrame.columns))

        return DataFrame

    @staticmethod
    def _validate_bruker_recordings_labels(AnalogRecordings: pd.DataFrame, NumTrials: int) -> pd.DataFrame:
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
    def _validate_bruker_recordings_completion(AnalogRecordings: pd.DataFrame, NumTrials: int) -> Tuple[bool, int]:
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


class DeepLabData:
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

        # make sure it's zero-indexed
        if np.min(_pre_trial_frame_ids) == 1:
            _pre_trial_frame_ids -= 1

        if np.min(_trial_frame_ids) == 1:
            _trial_frame_ids -= 1

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
    def convert_dataframe_to_physical_units(cls, DataFrame: pd.DataFrame, oldMin: int, oldMax: int, idx: Union[str, Tuple[str]], **kwargs: int) -> pd.DataFrame:
        """
        Converts data range to physical range

        :param DataFrame: dlc data
        :type DataFrame: pd.DataFrame
        :param oldMin: value representing left-side
        :type oldMin: int
        :param oldMax: value representing right-side
        :type oldMax: int
        :param idx: Which columns to rescale
        :param idx: Union[str, Tuple[str]]
        :keyword new_min: value representing new left-side (int, default 0)
        :keyword new_max: value representing new right-side (int, default 140)
        :return: DataFrame with rescaled data
        :rtype: pd.DataFrame
        """

        _new_max = kwargs.get("new_max", 140)
        _new_min = kwargs.get("new_min", 0)

        _old_range = oldMax - oldMin
        _new_range = _new_max - _new_min

        def convert_to_physical_units(fPositions):
            """
            Converts the burrow positions to physical units (mm)

            :param fPositions:
            :return:
            """
            nonlocal _new_min
            nonlocal oldMin
            nonlocal _new_range
            nonlocal _old_range

            return (((fPositions - oldMin) * _new_range) / _old_range) + _new_min

        for i in idx:
            DataFrame[i] = convert_to_physical_units(DataFrame[i].to_numpy())

        return DataFrame

    @staticmethod
    def convert_to_mean_zero(DataFrame: pd.DataFrame, idx: Union[str, Tuple[str]]) -> pd.DataFrame:
        """
        Converts data range to mean zero

        :param DataFrame: dlc data
        :type DataFrame: pd.DataFrame
        :param idx: Which columns to rescale
        :param idx: Union[str, Tuple[str]]
        :return: DataFrame with rescaled data
        :rtype: pd.DataFrame
        """

        DataFrame = DataFrame.copy(deep=True)
        for i in idx:
            DataFrame[i] = DataFrame[i].to_numpy() - np.mean(DataFrame[i].to_numpy())
        return DataFrame

    @classmethod
    def merge_dlc_data(cls, DataFrame: pd.DataFrame, DLC: DeepLabData, StateCastDict: dict) -> pd.DataFrame:
        """
        Function to merge DLC data with some DataFrame

        :param DataFrame: Data to merge with
        :type DataFrame: pd.DataFrame
        :param DLC: Data to merge
        :type DLC: DeepLabData
        :param StateCastDict: dictionary relating the state integers with pre-trial and trial states
        :type StateCastDict: dict
        :return: the DataFrame with DLC data merged and time-matched
        :rtype: pd.DataFrame
        """

        # We need to make sure the index is time
        assert(DataFrame.index.name == "Time (s)")
        _num_trials = np.unique(DLC.trial_data.index).__len__()

        def matcher(Stage: str, TrialNumber: int) -> pd.DataFrame:
            nonlocal DataFrame
            nonlocal DLC
            nonlocal StateCastDict
            nonlocal _num_trials

            # get new index
            _new_index = extract_specific_data(DataFrame,
                                               (("State Integer", StateCastDict.get(Stage)),
                                                ("Trial Set", TrialNumber))).index

            # get old index & data
            if Stage == "Trial":
                _dlc_data = DLC.trial_data.copy(deep=True).loc[TrialNumber]
            else:
                _dlc_data = DLC.pre_trial_data.copy(deep=True).loc[TrialNumber]

            _num_frames = _dlc_data["X1"].__len__()
            _old_dlc_index = pd.Series(np.around(np.linspace(
                _new_index.to_numpy()[0], _new_index.to_numpy()[-1], _num_frames).astype(np.float64), decimals=3))
            _old_dlc_index.name = "Time (s)"

            # use new index
            _dlc_data.reset_index(drop=True, inplace=True)
            _dlc_data.set_index(_old_dlc_index, inplace=True)
            _dlc_data = _dlc_data.reindex(_new_index)
            _dlc_data.interpolate("nearest", inplace=True)
            return _dlc_data

        # Match them, collect, concat, then join
        _matched_dlc_data = [matcher("Trial", _trial) for _trial in range(_num_trials)]
        _matched_dlc_data.extend([matcher("PreTrial", _trial) for _trial in range(_num_trials)])
        _matched_dlc_data = pd.concat(_matched_dlc_data)
        _matched_dlc_data.reindex(index=DataFrame.index)
        DataFrame = DataFrame.join(_matched_dlc_data)

        # nan fill
        DataFrame["X1"].fillna(value=np.nanmax(DataFrame["X1"].to_numpy()), inplace=True)
        DataFrame["X2"].fillna(value=np.nanmax(DataFrame["X2"].to_numpy()), inplace=True)
        DataFrame["Y1"].fillna(0, inplace=True)
        DataFrame["Y2"].fillna(0, inplace=True)
        DataFrame["likelihood1"].fillna(1, inplace=True)
        DataFrame["likelihood2"].fillna(1, inplace=True)

        # sort for ease of use
        DataFrame = DataFrame.reindex(columns=sorted(DataFrame.columns))

        return DataFrame

    @staticmethod
    def calculate_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Function calculates the euclidean distance for each pair of points in (X, Y)
        :param X: A numpy array of X positions
        :param Y: A numpy array of Y positions
        :return: A numpy array containing the distance between each sequential pair of points
        :rtype: Any
        """
        _coordinates = tuple([tuple([_x, _y]) for _x, _y in zip(X, Y)])
        EucDist = [scipy.spatial.distance.euclidean(_coordinates[_coord+1], _coordinates[_coord])
                   for _coord in range(_coordinates.__len__()-1)]
        EucDist.insert(0, 0.0)
        return np.array(EucDist)


def plot_burrow_coordinates(Coordinates):
    _fig = plt.figure()
    _subplots_number = Coordinates.__len__()*100 + 10 + 0
    for _coord in range(Coordinates.__len__()):
        _subplots_number += 1
        _ax = _fig.add_subplot(_subplots_number)
        _ax.plot(Coordinates[_coord])


def plot_column_by_trial_type(BehavioralObject: FearConditioning, ColumnName: str,
                              *args: Tuple[str, Union[str, int, float, list]], **kwargs: str) -> plt.Figure:
    """
    This function plots some column organized by trial type

    :param BehavioralObject: The FearConditioning object
    :type BehavioralObject: Any
    :param ColumnName: Name of the column to be plotted
    :type ColumnName: str
    :argument args: Second tuple for data extraction
    :keyword cmap: string identifying desired colormap
    :return: figure
    :rtype: Any
    """

    _cmap_str = kwargs.get("cmap", "icefire")
    _colors = plt.cm.get_cmap(_cmap_str)
    # Hacky code incoming ->
    # noinspection PyProtectedMember
    _colors = _colors._resample(BehavioralObject.num_stim).colors

    fig = plt.figure()
    _cols = BehavioralObject.num_stim
    _rows = BehavioralObject.trials_per_stim
    _grid = matplotlib.gridspec.GridSpec(ncols=_cols, nrows=_rows, figure=fig)

    if args:
        _additional_key = args[0]
    else:
        _additional_key = None

    for _stim, _trial in product(range(BehavioralObject.num_stim),
                             range(BehavioralObject.trials_per_stim)):

        _keys = ("Trial Set", BehavioralObject.trial_groups[_stim][_trial])

        if _additional_key is not None:
            _keys = (
                _keys,
                _additional_key
            )

        _data = extract_specific_data(BehavioralObject.data, _keys)
        _ax = fig.add_subplot(_grid[_trial, _stim])
        _ax.plot(_data.index.to_numpy(), _data[ColumnName].to_numpy(), color=_colors[_stim])
        _ax.set_xlabel("Time (s)")
        _ax.set_ylabel(ColumnName)
        _ax.set_title("".join(["Stimulus ", str(BehavioralObject.unique_stim[_stim]), ", Trial ",
                               str(BehavioralObject.trial_groups[_stim][_trial])]))

    plt.tight_layout()
    return fig


def plot_trial(BehavioralObject: FearConditioning, ColumnNames: list[str],
               Trials: list[int], **kwargs: str) -> plt.Figure:

    _cmap_str = kwargs.get("cmap", "icefire")
    _colors = plt.cm.get_cmap(_cmap_str)
    # Hacky code incoming ->
    # noinspection PyProtectedMember
    _colors = _colors._resample(ColumnNames.__len__()*Trials.__len__()).colors

    fig = plt.figure()
    _subplot_id = int("".join([str(Trials.__len__()), str(1), str(0)]))
    for _trial in range(Trials.__len__()):
        _subplot_id += 1
        _ax = fig.add_subplot(_subplot_id)
        _data = extract_specific_data(BehavioralObject.data, (("State Integer", BehavioralObject.state_index.get("Trial")), ("Trial Set", Trials[_trial])))
        for _column in range(ColumnNames.__len__()):
            _ax.plot(_data.index.to_numpy()-_data.index.to_numpy()[0], _data[ColumnNames[_column]].to_numpy(),
                     color=_colors[_column+_trial], lw=3)
            _ax.set_xlabel("Time (s)")
            _ax.set_title("".join(["Trial: ", str(Trials[_trial])]))

    plt.tight_layout()
    return fig
