from __future__ import annotations
from typing import Union, Tuple, List, Optional
import os
from datetime import date, datetime
import pickle as pkl
import numpy as np
import pandas as pd
from IPython import get_ipython
import pathlib

import ExperimentManagement.BrukerMetaModule
from ExperimentManagement.BrukerMetaModule import BrukerMeta
from MigrationTools.Converters import renamed_load
import math


class ExperimentData:
    """
    Class for Organizing & Managing Experimental Data Across Sessions

    **Keyword Arguments**
        | *Logfile* : Path to existing log file (str, default None)
        | *Mouse* : Mouse ID (str, default None)
        | *Condition* : Experimental Condition (str, default None)
        | *Directory* : Directory for hierarchy (str, default None)
        | *Study* : Study (str, default None)
        | *StudyMouse* : Study ID (str, default None)

    **Class Methods**
        | *loadHierarchy* : Function that loads the entire experimental hierarchy
        | *getDate*: Function returns date
        | *getTime* : Function returns time
        | *checkPath* : Checks Path
        | *generateDirectoryHierarchy* : Generates the Directory Structure (The structured folders where data stored)
        | *generateHistology* :  Generates Histology Folder
        | *generateROIMatchingIndex* : Generate ROI Matching Folder
        | *generateStage* : Generate Behavioral Stage Folder
        | *generateBehavior* : Generate Behavioral Folder
        | *generateImaging* : Generate Imaging Folder
        | *generateSampFreq* : Generate Sample Frequency Folder Innards
        | *generateComputation* : Generate Computation Folder
        | *generateAnalysisTechnique* : Generate Analysis Technique

    **Static Methods**
        | *generateReadMe* : Generate a read me file

    **Self Methods**
        | *passMeta* : Passes directory/mouse id
        | *recordMod* : Record modification of experiment
        | *recordStageMod* : Record modification of experiment and behavioral stage
        | *saveHierarchy* : Saves Hierarchy to pickle
        | *createLogFile* : Creates log file
        | *startLog* : Starts Log
        | *checkLog* : Checks Log Status
        | *create* : This function generates the directory hierarchy in one step

    **Properties**
        | *mouse_id* : ID of Mouse
        | *log_file* : Log File Path
        | *experimental_condition* : Experiment condition of the mouse
        | *instance_data* : Date when this experimental hierarchy was created

    **Attributes**
        | *directory* : Experimental Hierarchy Directory
        | *study* : Study
        | *study_mouse* : ID of mouse in study
        | *modifications* : modifications made to this file

    """

    def __init__(self, **kwargs):
        """
        :keyword LogFile: Path to existing log file (str, default None)
        :keyword Mouse: Mouse ID (str, default None)
        :keyword Condition: Experimental Condition (str, default None)
        :keyword Directory: Directory for hierarchy (str, default None)
        :keyword Study: Study (str, default None)
        :keyword StudyMouse: Study ID (str, default None)
        """
        # Hidden
        self._mouse_id_assigned = False  # to make sure mouse id only set once
        self._log_file_assigned = False  # to make sure log file only set once
        self._experimental_condition_assigned = False  # to make set condition only once
        # Protected In Practice
        self._log_file = kwargs.get('LogFile', None)
        self._mouse_id = kwargs.get('Mouse', None)
        self._experimental_condition = kwargs.get('Condition', None)
        self.__instance_date = self.getDate()
        #
        self.directory = kwargs.get('Directory', None)
        self.study = kwargs.get('Study', None)
        self.study_mouse = kwargs.get('StudyMouse', None)
        self.modifications = [(self.getDate(), self.getTime())]
        self.stages = []

        if self.directory is not None and self.mouse_id is not None:
            if os.path.exists(self.directory):
                print("Data organization directory located...")
            else:
                print("Directory not found, spawning empty directory for data organization")
                os.makedirs(self.directory)

        # Create log file if one does not exist
        if self._log_file_assigned is False and self.mouse_id is not None:
            self.createLogFile()



        # start logging if log file exists
        if self._log_file_assigned:
            self.startLog()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.endLog()

    @property
    def experimental_condition(self) -> str:
        """
        Experiment condition of the mouse

        :rtype: str
        """
        return self._experimental_condition

    @experimental_condition.setter
    def experimental_condition(self, Condition: str) -> Self:
        if self._experimental_condition_assigned is False:
            self._experimental_condition = Condition
            self._experimental_condition_assigned = True
        else:
            print("Experimental condition can only be assigned ONCE.")

    @property
    def log_file(self) -> str:
        """
        Log File Path

        :rtype: str
        """
        return self._log_file

    @log_file.setter
    def log_file(self, LogFile: str) -> Self:
        if self._log_file_assigned is False:
            self._log_file = LogFile
            self._log_file_assigned = True
        else:
            print("Log file can only be assigned ONCE.")

    @property
    def mouse_id(self) -> str:
        """
        ID of Mouse

        :rtype: str
        """
        return self._mouse_id

    @mouse_id.setter
    def mouse_id(self, ID: str) -> Self:
        if self._mouse_id_assigned is False:
            self._mouse_id = ID
            self._mouse_id_assigned = True
        else:
            print("Mouse ID can only be set ONCE.")

    @property
    def instance_date(self) -> str:
        """
        Date when this experimental hierarchy was created

        :rtype: str
        """
        return self._ExperimentData__instance_date

    @classmethod
    def loadHierarchy(cls, ExperimentDirectory: str) -> ExperimentData:
        """
        Function that loads the entire experimental hierarchy

        :param ExperimentDirectory: Directory containing the experimental hierarchy pickle file
        :type ExperimentDirectory: str
        :return: ExperimentData
        :rtype: ExperimentManagement.ExperimentHierarchy.ExperimentData
        """
        print("Loading Experimental Hierarchy...")
        _input_file = ExperimentDirectory + "\\ExperimentalHierarchy"
        _input_pickle = open(_input_file, 'rb')
        _pickles = pkl.load(_input_pickle)
        _keys = _pickles.keys()
        MouseData = ExperimentData()
        for _key in _keys:
            MouseData.__dict__.update(_pickles)
        print("Finished.")

        if MouseData._log_file_assigned:
            MouseData.startLog()
        return MouseData

    @classmethod
    def getDate(cls):
        return date.isoformat(date.today())

    @classmethod
    def getTime(cls):
        return datetime.now().strftime("%H:%M:%S")

    @classmethod
    def checkPath(cls, Path: str) -> bool:
        return os.path.exists(Path)

    @classmethod
    def generateDirectoryHierarchy(cls, MouseDirectory: str, **kwargs) -> None:
        """
        Generates the Directory Structure (The structured folders where data stored)

         Keyword Arguments
        -----------------
        *Histology* : Generate histology Folder (bool, default True)
        *ROIMatchingIndex* : Generate ROI Matching Folder (bool, default True)
        *Stage* :Generate Behavioral Stage Folder(bool, default True)
        *LabNotebook* : Generate Lab Notebook Folder (bool, default True)

        :param MouseDirectory: Directory to generate folders in
        :type MouseDirectory: str
        :keyword Histology: Generate histology Folder
        :keyword ROIMatchingIndex: Generate ROI Matching Folder
        :keyword Stage: Generate Behavioral Stage Folder
        :keyword: LabNotebook: Generate Lab Notebook Folder
        :rtype: None
        """
        _gen_histology = kwargs.get('Histology', True)
        _gen_roi_matching_index = kwargs.get('ROIMatchingIndex', True)
        _gen_stage = kwargs.get('Stage', True)
        _gen_lab_notebook = kwargs.get('LabNotebook', True)

        if _gen_histology:
            cls.generateHistology(MouseDirectory)
        if _gen_roi_matching_index:
            cls.generateROIMatchingIndex(MouseDirectory)
        if _gen_stage:
            cls.generateStage(MouseDirectory)
        if _gen_lab_notebook:
            os.makedirs(MouseDirectory + "\\LabNotebook")

    @classmethod
    def generateHistology(cls, MouseDirectory: str, **kwargs) -> None:
        """
        Generates Histology Folder

        Keyword Arguments
        -----------------
        *Title* : Title of Histology Experiment (str, default None)

        :param MouseDirectory: Directory to generate folders in
        :type MouseDirectory: str
        :keyword Title: Title of Histology Experiment
        :rtype: None
        """

        _visual_hist_title = kwargs.get('Title', None)
        _base_hist_dir = MouseDirectory + "\\Histology"
        if _visual_hist_title is not None:
            _visual_hist_dir = _base_hist_dir + "\\" + _visual_hist_title
        else:
            _visual_hist_dir = _base_hist_dir + "\\Visualization"

        _read_me_file = _visual_hist_dir + "\\ReadMe.txt"

        os.makedirs(_base_hist_dir)
        os.makedirs(_visual_hist_dir)

        cls.generateReadMe(_read_me_file, "Read-Me for Associated Histological Data")

    @classmethod
    def generateROIMatchingIndex(cls, MouseDirectory: str) -> None:
        """
        Generate ROI Matching Folder

        :rtype: None
        """
        _roi_matching_index_dir = MouseDirectory + "\\ROIMatchingIndex"
        _roi_matching_index_read_me = _roi_matching_index_dir + "\\ReadMe.txt"

        os.makedirs(_roi_matching_index_dir)
        cls.generateReadMe(_roi_matching_index_read_me,
                           "Read-Me for Index of Longitudinally-Matched ROIs")

    @classmethod
    def generateStage(cls, MouseDirectory: str, **kwargs) -> None:
        """
        Generate Behavioral Stage Folder

        Keyword Arguments
        -----------------
        *Behavior* : Include Behavioral Folder (bool, default True)
        *Imaging* : Include Imaging Folder (bool, default True)
        *Computation* : Include Computation Folder (bool, default True)
        *Title* : Title of Behavioral Stage (str, default Stage)
        :rtype: None
        """
        _include_behavior = kwargs.get('Behavior', True)
        _include_imaging = kwargs.get('Imaging', True)
        _include_computation = kwargs.get('Computation', True)
        _stage_title = kwargs.get('Title', 'Stage')
        _stage_directory = MouseDirectory + "\\" + _stage_title

        if _include_behavior:
            cls.generateBehavior(_stage_directory)
        if _include_imaging:
            cls.generateImaging(_stage_directory)
        if _include_computation:
            cls.generateComputation(_stage_directory)

    @classmethod
    def generateBehavior(cls, StageDirectory: str) -> None:
        """
        Generate Behavioral Folder

        :param StageDirectory: Directory for folder creation
        :type StageDirectory: str
        :rtype: None
        """

        _base_behav_dir = StageDirectory + "\\Behavior"
        _raw_behavioral_data = _base_behav_dir + "\\RawBehavioralData"
        _behavioral_exports = _base_behav_dir + "\\BehavioralExports"
        _deep_lab_cut_data = _base_behav_dir + "\\DeepLabCutData"
        os.makedirs(_base_behav_dir)
        os.makedirs(_raw_behavioral_data)
        os.makedirs(_behavioral_exports)
        os.makedirs(_deep_lab_cut_data)

    @classmethod
    def generateImaging(cls, StageDirectory: str) -> None:
        """
        Generate Imaging Folder

        Keyword Arguments
        -----------------
        *SampleFrequency* : Image frequency (int, default 30)

        :param StageDirectory: Directory for folder creation
        :type StageDirectory: str
        :keyword SampleFrequency: Image frequency
        :rtype: None
        """

        _base_image_dir = StageDirectory + "\\Imaging"
        _sample_frequency_dir = _base_image_dir + "\\" + _sample_frequency_string
        _raw_imaging_data = _base_image_dir + "\\RawImagingData"
        _bruker_meta_data = _base_image_dir + "\\BrukerMetaData"
        os.makedirs(_base_image_dir)
        os.makedirs(_raw_imaging_data)
        os.makedirs(_bruker_meta_data)

    @classmethod
    def generateSampFreq(cls, SampFreqDirectory: str) -> None:
        """
        Generate Sample Frequency Folder Innards

        :param SampFreqDirectory: Directory for folder creation
        :type SampFreqDirectory: str
        :rtype: None
        """
        _suite2p = SampFreqDirectory + "\\suite2p"
        _fissa = SampFreqDirectory + "\\fissa"
        _roi_sorting = SampFreqDirectory + "\\sorting"
        _denoised = SampFreqDirectory + "\\denoised"
        _cascade = SampFreqDirectory + "\\cascade"
        _compiled = SampFreqDirectory + "\\compiled"
        os.makedirs(_suite2p)
        os.makedirs(_fissa)
        os.makedirs(_denoised)
        os.makedirs(_roi_sorting)
        os.makedirs(_cascade)
        os.makedirs(_compiled)
        cls.generateReadMe(_roi_sorting + "\\ReadMe.txt", "Read-Me for ROI Sorting")

    @classmethod
    def generateComputation(cls, StageDirectory: str, **kwargs) -> None:
        """
        Generate Computation Folder

        Keyword Arguments
        -----------------
        *Title* : Computational Analysis Title (str, default AnalysisTechnique)

        :param StageDirectory: Directory for folder creation
        :type StageDirectory: str
        :keyword Title: Computational Analysis Title
        :rtype: None
        """
        _analysis_title = kwargs.get('Title', 'AnalysisTechnique')
        _base_comp_dir = StageDirectory + "\\Computational"
        _neural_data_dir = _base_comp_dir + "\\NeuralData"
        _analysis_dir = _base_comp_dir + "\\" + _analysis_title
        os.makedirs(_base_comp_dir)
        os.makedirs(_neural_data_dir)

    @classmethod
    def generateAnalysisTechnique(cls, BaseCompDirectory: str, AnalysisTitle: str) -> None:
        """
        Generate Analysis Technique

        :param BaseCompDirectory: Base Directory for Computation
        :type BaseCompDirectory: str
        :param AnalysisTitle: Title for Analysis
        :type AnalysisTitle: str
        :rtype: None
        """
        _analysis_dir = BaseCompDirectory + "\\" + AnalysisTitle
        os.makedirs(_analysis_dir)
        cls.generateReadMe(_analysis_dir + "\\ReadMe.txt", "Read-Me for Analysis Technique")

    @staticmethod
    def generateReadMe(AbsoluteFilePath: str, Text: str) -> None:
        """
        Generate a read me file

        :param AbsoluteFilePath: File path
        :type AbsoluteFilePath: str
        :param Text: Text inside
        :type Text: str
        :rtype: None
        """
        with open(AbsoluteFilePath, 'w') as _read_me:
            _read_me.write(Text)
            _read_me.close()

    def passMeta(self) -> Tuple[str, str]:
        """
        Passes directory/mouse id

        :returns: directory/mouse id
        :rtype: tuple[str, str]
        """

        return self.directory, self.mouse_id

    def recordMod(self, *args: str) -> Self:
        """
        Record modification of experiment (Data, Time, *args)


        :param args: A string explaining the modification
        :type args: str
        :rtype: Any
        """
        # noinspection PyTypeChecker
        self.modifications.append((self.getDate(), self.getTime(), *args))

    def recordStageMod(self, StageKey: str, *args) -> Self:
        """
        Record modification of experiment (Data, Time, *args)

        :param StageKey: The key name for the stage
        :type StageKey: str
        :param args: A string explaining the modification
        :type args: str
        :rtype: Any
        """
        if args:
            self.recordMod(args[0])
        else:
            self.recordMod()
        try:
            self.__dict__.get(StageKey).recordMod()
        except KeyError:
            print("Unable to identify stage from provided key")

    def saveHierarchy(self) -> Self:
        """
        Saves Hierarchy to pickle

        :rtype: Any
        """

        print("Saving Experimental Hierarchy...")
        if hasattr('self', '_IP'):
            # noinspection PyAttributeOutsideInit
            self._IP = True
        else:
            # noinspection PyAttributeOutsideInit
            self._IP = False

        _output_file = self.directory + "\\" + "ExperimentalHierarchy"
        _output_pickle = open(_output_file, 'wb')

        _output_dict = dict()
        _keys = self.__dict__.keys()
        # noinspection PyTypeChecker

        for _key in _keys:
            if not isinstance(self.__dict__.get(_key), np.ndarray):
                _output_dict[_key] = self.__dict__.get(_key)

        pkl.dump(_output_dict, _output_pickle)
        _output_pickle.close()
        print("Finished.")

        if self._IP:
            # noinspection PyAttributeOutsideInit
            self._IP = get_ipython()

    def createLogFile(self) -> Self:
        """
        Creates log file

        :rtype: Any
        """
        if self.directory is not None:
            self.log_file = self.directory + "\\log_file.log"
            if not os.path.exists(self.directory):
                print("Could not locate directory, saving in base.")
                self.log_file = "".join([os.getcwd(), "\\log_file.log"])
        else:
            print("Could not locate directory, saving in base.")
            self.log_file = "".join([os.getcwd(), "\\log_file.log"])


        with open(self.log_file, "w") as _log:
            _log.write("")
        _log.close()

        print("Logging file assigned as :" + self.log_file)

    def create(self) -> Self:
        """
        This function generates the directory hierarchy in one step

        :rtype: Any
        """
        if self.directory is not None and self.checkPath(self.directory):
            self.generateDirectoryHierarchy(self.directory)
        else:
            print("Unable to create organized directory in specified path.")

    # noinspection All
    def startLog(self) -> Self:
        """
        Starts Log

        :rtype: Any
        """
        self._IP = get_ipython()
        _magic_arguments = '-o -r -t ' + self.log_file + ' append'
        self._IP.run_line_magic('logstart', _magic_arguments)
        print("Logging Initiated")

    # noinspection All
    def endLog(self) -> Self:
        """
        Ends Logging

        :rtype: Any
        """
        self._IP.run_line_magic('logstop', '')

    # noinspection All
    def checkLog(self) -> Self:  # noinspection All
        """
        Checks log status

        :rtype: Any
        """

        self._IP.run_line_magic('logstate', '')


class BehavioralStage:
    """
    Data Class for a generic stage or day of a behavioral task

    **Required Inputs**
        | *Meta* : Passed meta from experimental hierarchy (directory, mouse_id)
        | *Stage* : Title of Stage

    **Positional Arguments**
        | *SyncID* : Sync ID (ID of columns used for syncing data)

    **Self Methods**
        | *recordMod* : Records a modification made to the behavioral stage (Date & Time)
        | *createFolderDictionary* : Creates a dictionary of locations for specific files
        | *fillImagingDictionary* :  Generates folders and a dictionary of imaging files (Raw, Meta, Compiled)
        | *addImageSamplingFolder* : Generates a folder for containing imaging data of a specific sampling rate
        | *addImageProcessingFolder* : Generates a folder for containing processed imaging data

    **Properties**
        | *mouse_id* : Identifies which mouse this data belongs to
        | *instance_data* : Identifies when this behavioral stage was created

    **Attributes**
        | *modifications* : List of modifications made to this behavioral stage
        | *folder_dictionary* : A dictionary of relevant folders for this behavioral stage
        | *data* : Pandas dataframe of synced data
        | *meta* : bruker metadata
        | *state_index* : index of states
        | *sync_id* : key indicator for syncing data
    """

    def __init__(self, Meta: Tuple[str, str], Stage: str, *args: Tuple[str, str]):
        """
        :param Meta: Passed Meta from experimental hierarchy
        :type Meta: tuple[str, str]
        :param Stage: Title of Stage
        :type Stage: str
        :param args: Sync ID (ID of columns used for syncing data)
        :type args: Tuple[str, str]
        """
        # PROTECTED
        self.__mouse_id = Meta[1]
        self.__instance_date = ExperimentData.getDate()
        #
        self.modifications = [(ExperimentData.getDate(), ExperimentData.getTime())]
        self.folder_dictionary = dict()
        # self.data = pd.DataFrame
        self.data = None
        self.meta = None
        self.multi_index = None
        self.state_index = None
        self.trial_parameters = None

        if len(args) >= 1:
            self.sync_id = args[0]
            assert(isinstance(self.sync_id, tuple))
        else:
            self.sync_id = None

        _mouse_directory = Meta[0]
        self.createFolderDictionary(_mouse_directory, Stage)
        self.fillImagingFolderDictionary()

    @property
    def mouse_id(self) -> str:
        """
        ID of mouse

        :rtype: str
        """
        return self._BehavioralStage__mouse_id

    @property
    def instance_date(self) -> str:
        """
        Date created

        :rtype: str
        """
        return self._BehavioralStage__instance_date

    def recordMod(self) -> Self:
        """
        Records a modification made to the behavioral stage (Date & Time)

        :rtype: Any
        """
        self.modifications.append((ExperimentData.getDate(), ExperimentData.getTime()))

    def createFolderDictionary(self, MouseDirectory: str, Stage: str) -> Self:
        """
        Creates a dictionary of locations for specific files

        :param MouseDirectory: Directory containing mouse data (passed meta)
        :type MouseDirectory: str
        :param Stage: The Stage ID
        :type Stage: str
        :rtype: Any
        """
        _stage_directory = MouseDirectory + "\\" + Stage
        self.folder_dictionary = {
            'mouse_directory': MouseDirectory,
            'stage_directory': _stage_directory,
            'computation_folder': _stage_directory + "\\Computation",
            'imaging_folder': _stage_directory + "\\Imaging",
            'behavior_folder': _stage_directory + "\\Behavior",
        }

    def fillImagingFolderDictionary(self) -> Self:
        """
        Generates folders and a dictionary of imaging files (Raw, Meta, Compiled)

        :rtype: Any
        """
        # RAW
        _raw_data_folder = self.folder_dictionary['imaging_folder'] + "\\RawImagingData"
        try:
            os.makedirs(_raw_data_folder)
        except FileExistsError:
            print("Existing Raw Data Folder Detected")
        self.folder_dictionary['raw_imaging_data'] = CollectedDataFolder(_raw_data_folder)
        # META
        _bruker_meta_folder = self.folder_dictionary['imaging_folder'] + "\\BrukerMetaData"
        try:
            os.makedirs(_bruker_meta_folder)
        except FileExistsError:
            print("Existing Bruker Meta Data Folder Detected")
        self.folder_dictionary['bruker_meta_data'] = CollectedDataFolder(_bruker_meta_folder)

    def addImageSamplingFolder(self, SamplingRate: int) -> Self:
        """
        Generates a folder for containing imaging data of a specific sampling rate

        :param SamplingRate: Sampling Rate of Dataset in Hz
        :type SamplingRate: int
        :rtype: Any
        """
        SamplingRate = str(SamplingRate)  # Because we know I'll always forget and send an int anyway
        _folder_name = "".join([self.folder_dictionary['imaging_folder'], "\\", SamplingRate, "Hz"])
        _key_name = "".join(["imaging_", SamplingRate, "Hz"])
        try:
            os.makedirs(_folder_name)
        except FileExistsError:
            print("The sampling folder already exists. Adding to folder dictionary")
        # setattr(self, _attr_name, CollectedImagingFolder(_folder_name)) changing to be in folder dictionary
        self.folder_dictionary[_key_name] = CollectedImagingFolder(_folder_name)
        ExperimentData.generateSampFreq(_folder_name)
        self.update_folder_dictionary()

    def update_folder_dictionary(self) -> Self:
        """
        This function reindexes all folders in the folder dictionary

        :rtype: Any
        """

        # noinspection PyTypeChecker
        for _key in self.folder_dictionary.keys():
            if isinstance(self.folder_dictionary.get(_key), CollectedDataFolder):
                self.folder_dictionary.get(_key).reindex()
            elif isinstance(self.folder_dictionary.get(_key), CollectedImagingFolder):
                self.folder_dictionary.get(_key).reindex()

    def loadBrukerMetaData(self) -> Self:
        """
        Loads Bruker Meta Data

        :rtype: Any
        """
        self.folder_dictionary["bruker_meta_data"].reindex()
        _files = self.folder_dictionary["bruker_meta_data"].find_all_ext("xml")
        self.meta = BrukerMeta(_files[0], _files[2], _files[1])
        self.meta.import_meta_data()
        self.meta.creation_date = ExperimentData.getDate()

    def loadBrukerAnalogRecordings(self) -> pd.DataFrame:
        """
        Loads Bruker Analog Recordings

        :returns: Analog Recording
        :rtype: pd.DataFrame
        """

        self.folder_dictionary["bruker_meta_data"].reindex()
        _files = self.folder_dictionary["bruker_meta_data"].find_all_ext("csv")
        return self.load_bruker_analog_recordings(_files[-1])

    def load_base_behavior(self) -> Self:
        """
        Loads the basic behavioral data: analog, dictionary, digital, state, and CS identities

        :rtype: Any
        """

        print("Loading Base Data...")
        # Analog
        _analog_file = self.generateFileID('Analog')
        _analog_data = FearConditioning.load_analog_data(_analog_file)
        if type(_analog_data) == str and _analog_data == "ERROR":
            return print("Could not find analog data!")

        # Digital
        _digital_file = self.generateFileID('Digital')
        _digital_data = FearConditioning.load_digital_data(_digital_file)
        if type(_digital_data) == str and _digital_data == "ERROR":
            return print("Could not find digital data!")

        # State
        _state_file = self.generateFileID('State')
        _state_data = FearConditioning.load_state_data(_state_file)
        if _state_data[0] == "ERROR": # 0 because it's an array of strings so ambiguous str comparison
            return print("Could not find state data!")

        # Dictionary
        _dictionary_file = self.generateFileID('Dictionary')
        _dictionary_data = FearConditioning.load_dictionary_data(_dictionary_file)
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

    @staticmethod
    def organize_base_data(Analog: np.ndarray, Digital: np.ndarray, State: np.ndarray,
                           HardwareConfig: Optional[dict] = None) -> pd.DataFrame:
        """
        This function organizes analog, digital, and state data into a pandas dataframe

        :param Analog: Analog Data
        :type Analog: Any
        :param Digital: Digital Data
        :type Digital: Any
        :param State: State Data
        :type State: Any
        :param HardwareConfig: Dictionary indicating configuration of hardware channels
        :type HardwareConfig: dict
        :return: Data organized into a pandas dataframe
        :rtype: pd.DataFrame
        """

        # Parse Channel IDs
        if HardwareConfig is None:
            _analog_dictionary = {
                "Imaging Sync": 0,
                "Motor Position": 1,
                "Force": 3,
            }
            _digital_dictionary = {
                "Gate": 0
            }
        else:
            print("Not yet implemented, reverting to defaults")
            _analog_dictionary = {
                "Imaging Sync": 0,
                "Motor Position": 1,
                "Force": 3,
            }
            _digital_dictionary = {
                "Gate": 0
            }

        # Determine Sampling Rates
        if HardwareConfig is None:
            _data_sampling_rate = 1000
            _state_sampling_rate = 10
        else:
            print("Not yet implemented, reverting to defaults")
            _data_sampling_rate = 1000
            _state_sampling_rate = 10

        def nest_all_stages_under_trials(State_Data: np.ndarray, Index: np.ndarray, State_Casted_Dict: dict):
            nested_trial_index = np.full(Index.__len__(), 69, dtype=np.float64)
            _trial_idx = np.where(State_Data == State_Casted_Dict.get("Trial"))[0]
            _deriv_idx = np.diff(_trial_idx)
            _trailing_edge = _trial_idx[np.where(_deriv_idx > 1)[0]]
            _trailing_edge = np.append(_trailing_edge, _trial_idx[-1])
            _habituation_trailing_edge = np.where(State_Data == State_Casted_Dict.get("Habituation"))[0][-1]

            nested_trial_index[_habituation_trailing_edge] = 0
            nested_trial_index[-1] = _trailing_edge.__len__() + 1

            for _edge in range(_trailing_edge.shape[0]):
                nested_trial_index[_trailing_edge[_edge]] = _edge + 1

            nested_trial_index = pd.Series(nested_trial_index, index=Index, dtype=np.float64)
            nested_trial_index[nested_trial_index == 69] = np.nan
            nested_trial_index.bfill(inplace=True)
            nested_trial_index.name = "Trial Set"
            return nested_trial_index

        def construct_state_index_series(State_Data: np.ndarray, Time_Vector_Data: np.ndarray,
                                         Time_Vector_State: np.ndarray) -> Tuple[dict, pd.Series]:
            def cast_state_into_float64(State__Data: np.ndarray) -> Tuple[np.ndarray, dict]:
                State__Casted__Dict = dict()
                Integer_State_Index = np.full(State__Data.shape[0], 0, dtype=np.float64)
                _unique_states = np.unique(State__Data)

                for _unique_value in range(_unique_states.shape[0]):
                    State__Casted__Dict[_unique_states[_unique_value]] = _unique_value
                    Integer_State_Index[np.where(State__Data == _unique_states[_unique_value])[0]] = _unique_value

                return Integer_State_Index, State__Casted__Dict

            _integer_state_index, State_Casted_Dict = cast_state_into_float64(State_Data)
            State_Index = pd.Series(_integer_state_index, index=Time_Vector_State)
            State_Index.sort_index(inplace=True)
            State_Index = State_Index.reindex(Time_Vector_Data)
            State_Index.ffill(inplace=True)
            State_Index.name = "State Integer"
            return State_Casted_Dict, State_Index

        # Create Time Vectors
        _time_vector_data = np.around(np.arange(0, Analog.shape[1] * (1 / _data_sampling_rate), 1 / _data_sampling_rate,
                                                dtype=np.float64),
                                      decimals=3)
        _time_vector_state = np.around(np.arange(0, State.__len__() * (1 / _state_sampling_rate), 1 /
                                                 _state_sampling_rate, dtype=np.float64), decimals=3)

        # Construct State Index
        StateCastedDict, _state_index = construct_state_index_series(State, _time_vector_data, _time_vector_state)

        # Construct Nested Trial Index
        _trial_index = nest_all_stages_under_trials(_state_index.to_numpy(), _time_vector_data, StateCastedDict)

        # Create Multi-Index
        MultiIndex = pd.MultiIndex.from_arrays([_state_index.to_numpy(), _trial_index.to_numpy(), _time_vector_data])
        MultiIndex.names = ["State Integer", "Trial Set", "Time (s)"]

        # Initialize Organized Data Frame
        OrganizedData = pd.DataFrame(None, index=_time_vector_data, dtype=np.float64)
        OrganizedData.index.name = "Time (s)"

        # Type Check
        try:
            assert (_state_index.to_numpy().dtype == np.float64)
            assert (_trial_index.to_numpy().dtype == np.float64)
            assert (Analog.dtype == np.float64)
            assert (Digital.dtype == np.float64)
        except AssertionError:
            print("Bug. Incorrect Type detected. Forcing conversion...")
            try:
                _state_index.to_numpy().astype(np.float64)
                _trial_index.to_numpy().astype(np.float64)
                Analog.astype(np.float64)
                Digital.astype(np.float64)
                print("Successfully converted erroneous data types")
            except AttributeError:
                print("Failed automatic conversion")
                return OrganizedData

        # Add According to Index, Start with State Integer
        OrganizedData = OrganizedData.join(_state_index, on="Time (s)")
        OrganizedData = OrganizedData.join(_trial_index, on="Time (s)")

        # Add Analogs
        if Analog.shape[0] < Analog.shape[1]:  # orientation check
            Analog = Analog.T

        for _key in _analog_dictionary:
            _analog_series = pd.Series(Analog[:, _analog_dictionary.get(_key)], index=_time_vector_data,
                                       dtype=np.float64)
            _analog_series.name = _key
            OrganizedData = OrganizedData.join(_analog_series, on="Time (s)")

        # Add Digital
        if _digital_dictionary.keys().__len__() > 1:
            for _key in _digital_dictionary:
                _digital_series = pd.Series(Digital[:, _digital_dictionary.get(_key)], index=_time_vector_data,
                                            dtype=np.float64)
                _digital_series.name = _digital_dictionary.get(_key)
                OrganizedData = OrganizedData.join(_digital_series, on="Time (s)")
        else:
            _digital_series = pd.Series(Digital, index=_time_vector_data, dtype=np.float64)
            _digital_series.name = list(_digital_dictionary.keys())[0]
            OrganizedData = OrganizedData.join(_digital_series, on="Time (s)")

        return OrganizedData, StateCastedDict, MultiIndex

    @staticmethod
    def load_bruker_analog_recordings(File: str) -> pd.DataFrame:
        """
        Method to load bruker analog recordings from .csv

        :param File: filepath
        :type File: str
        :return: analog recordings
        :rtype: pd.DataFrame
        """
        assert(pathlib.Path(File).suffix == ".csv")
        return pd.read_csv(File)

    @staticmethod
    def sync_bruker_recordings(DataFrame: pd.DataFrame, AnalogRecordings: pd.DataFrame, MetaData: BrukerMeta,
                               StateCastedDict: dict,
                               SyncKey: Optional[Tuple[str, str]], Parameters: dict, **kwargs) -> pd.DataFrame:
        """

        :param DataFrame:
        :type DataFrame: pd.DataFrame
        :param AnalogRecordings:
        :type AnalogRecordings: pd.DataFrame
        :param MetaData:
        :type MetaData: BrukerMeta
        :param StateCastedDict:
        :type StateCastedDict: dict
        :param SyncKey:
        :type SyncKey: tuple or None
        :param Parameters: dictionary containing preprocessing and other parameters
        :type Parameters: dict
        :return: Sync Data
        :rtype: pd.DataFrame
        """

        _fill_method = kwargs.get("fill", "nearest")
        if SyncKey is None:
            SyncKey = ("State Integer", " TrialIndicator")
        # Sync by matching first peak of the sync_key columns
        _DF_signal = DataFrame[SyncKey[0]].to_numpy().copy()
        _AR_signal = AnalogRecordings[SyncKey[1]].to_numpy().copy()
        _AR_first_peak = np.where(np.diff(_AR_signal) > 1.0)[0][0] + 1
        assert (_AR_signal[_AR_first_peak] >= 3.3)

        _first_peak_diff = np.where(_DF_signal == StateCastedDict.get("Trial"))[0][0] - _AR_first_peak

        if _first_peak_diff > 0:
            print("NOT YET IMPLEMENTED")
            # noinspection PyTypeChecker
            return
            # noinspection PyUnreachableCode
            _AR_signal = np.pad(_AR_signal, (_first_peak_diff, 0), constant_values=0)
        elif _first_peak_diff < 0:
            _first_peak_diff *= -1
            _AR_signal = pd.DataFrame(AnalogRecordings.iloc[_first_peak_diff:, 1:].to_numpy(),
                                      index=np.around(
                                          (AnalogRecordings.index.to_numpy()[_first_peak_diff:] - _first_peak_diff) /
                                          int(MetaData.acquisition_rate), decimals=3) + DataFrame.index.to_numpy()[0])
            _AR_signal.columns = AnalogRecordings.columns[1:]

            _frames = pd.Series(np.arange(0, MetaData.imaging_metadata.get("relativeTimes").__len__(), 1),
                                index=np.around(MetaData.imaging_metadata.get("relativeTimes") -
                                                AnalogRecordings["Time(ms)"].to_numpy()[_first_peak_diff] / int(
                                    MetaData.acquisition_rate), decimals=3) + DataFrame.index.to_numpy()[0])
            _frames = _frames[_frames.index >= 0 + DataFrame.index.to_numpy()[0]].copy(deep=True)
            _frames.name = "Imaging Frame"
        else:
            print("Already Synced")

        # Here I just make sure shape-match
        if _DF_signal.shape[0] < _AR_signal.shape[0]:
            _AR_signal = _AR_signal.iloc[0:_DF_signal.shape[0], :]
            _frames = _frames[_frames.index <= DataFrame.index.to_numpy()[-1]].copy(deep=True)
            print("Cropped Bruker Signals")
        elif _DF_signal.shape[0] > _AR_signal.shape[0]:
            # _AR_signal.to_numpy() = np.pad(_AR_signal.to_numpy(), pad_width=((0, _DF_signal.shape[0] - _AR_signal.shape[0]), (0, 0)),
            # mode="constant", constant_values=0)

            _AR_signal = _AR_signal.reindex(DataFrame.index)
            # _frames = _frames.reindex(DataFrame.index) no need, happens below anyway

        # Merge/Export Analog
        # DataFrame = DataFrame.join(_AR_signal.copy(deep=True))
        DataFrame = DataFrame.join(_AR_signal)

        # Merge/Export Images
        DataFrame = DataFrame.join(_frames.copy(deep=True))
        # na filling column
        _frames = pd.DataFrame(_frames).copy(deep=True)
        _frames = _frames.reindex(DataFrame.index)
        if _fill_method == "backward":
            _frames.bfill(inplace=True)
        elif _fill_method == "forward":
            _frames.ffill(inplace=True)
        elif _fill_method == "nearest":
            _frames.interpolate(method="nearest", inplace=True)
        _frames = _frames.rename(columns={"Imaging Frame": "[FILLED] Imaging Frame"})

        DataFrame = DataFrame.join(_frames.copy(deep=True))

        # adjust for preprocessing
        # parse params
        _bin_size = Parameters.get(("preprocessing", "grouped-z project bin size"), 3)
        _artifact = Parameters.get(("preprocessing", "shuttle artifact length"), 1000)
        _chunk_size = Parameters.get(("preprocessing", "chunk size"), 7000)

        # parse meta
        _num_frames = MetaData.imaging_metadata.get("relativeTimes").__len__()

        # determine original frames
        if _num_frames % _chunk_size >= _artifact:
            _first_frame = _num_frames % _chunk_size
        else:
            _artifact_free_frames = _num_frames - _artifact
            _additional_crop = _artifact_free_frames % _chunk_size
            _first_frame = _additional_crop + _artifact


        return DataFrame

    @staticmethod
    def sync_grouped_z_projected_images(DataFrame: pd.DataFrame, MetaData: BrukerMeta, Parameters: dict) -> \
            pd.DataFrame:

        # parse params
        _bin_size = Parameters.get(("preprocessing", "grouped-z project bin size"))
        _artifact = Parameters.get(("preprocessing", "shuttle artifact length"))
        _chunk_size = Parameters.get(("preprocessing", "chunk size"))

        # parse meta
        _num_frames = MetaData.imaging_metadata.get("relativeTimes").__len__()

        # determine downsample frames
        _original_downsample_frames = np.arange(0, _num_frames, 3).__len__()
        if _original_downsample_frames % _chunk_size >= _artifact:
            _first_downsample_frame = _original_downsample_frames % _chunk_size
        else:
            _artifact_free_downsample_frames = _original_downsample_frames - _artifact
            _additional_crop_downsample = _artifact_free_downsample_frames % _chunk_size
            _first_downsample_frame = _artifact_free_downsample_frames + _additional_crop_downsample
        _projected_frames = np.arange(_first_downsample_frame, _num_frames, 3)
        _projected_first_frame = _projected_frames[0]*3
        _matching_frames = np.arange(_projected_first_frame, _num_frames, 3) + 1
        # make the center frame the timestamp instead of the first frame
        _time_stamp = np.full(_matching_frames.shape, -1, dtype=np.float64)
        for _frame in range(_matching_frames.__len__()):
            try:
                _time_stamp[_frame] = DataFrame.index.to_numpy()[np.where(DataFrame["Imaging Frame"].to_numpy() == _matching_frames[_frame])[0]]
            except ValueError:
                pass
        _true_idx = np.where(_time_stamp != -1)[0]
        _time_stamp = _time_stamp[np.where(_time_stamp != -1)[0]]

        _downsampled_frames = pd.Series(np.arange(0, _time_stamp.shape[0], 1),
                                        index=_time_stamp)
        _downsampled_frames.name = "Downsampled Imaging Frame"

        DataFrame = DataFrame.join(_downsampled_frames.copy(deep=True), on="Time (s)")

        _downsampled_frames.reindex(DataFrame.index)
        _downsampled_frames.name = "[FILLED] Downsampled Imaging Frame"
        _downsampled_frames.interpolate(method="nearest", inplace=True)
        DataFrame = DataFrame.join(_downsampled_frames, on="Time (s)")

        return DataFrame


class CollectedDataFolder:
    """
This is a class for managing a folder of unorganized data files

**Required Inputs**
    | *Path* : path to folder

**Self Methods**
        | *find_matching_files* : Finds all matching files
        | *reindex* : Function that indexed the files within folder again
        | *find_all_ext* :  Finds all files with specific extension
**Properties**
        | *instance_data* : Data created
        | *path* : path to folder
        | *files* : List of files in folder
    """

    def __init__(self, Path: str):
        # Protected In Practice
        self.__instance_date = ExperimentData.getDate()
        self._path = str()
        self._path_assigned = False

        # Properties
        self._files = []
        self.path = Path
        self.files = self.path

    @property
    def instance_date(self) -> str:
        """
        Date Created

        :rtype: str
        """
        return self._CollectedDataFolder__instance_date

    @property
    def path(self) -> str:
        """
        Path to folder

        :rtype: str
        """
        return self._path

    @path.setter
    def path(self, Path: str) -> Self:
        if self._path_assigned is False:
            self._path = Path
            self._path_assigned = True
        else:
            print("Path can only be assigned ONCE.")

    @property
    def files(self) -> List[str]:
        return self._files

    @files.setter
    def files(self, Path: str) -> Self:
        """
       function to quickly fill recursively

        :param Path: Directory to check
        :type Path: str
        :return: Any
        """

        self._files = [_file for _file in pathlib.Path(Path).rglob("*") if _file.is_file()]

    def reindex(self) -> Self:
        """
        Function that indexes the files within folder again
        """
        
        self.files = self.path

    def find_matching_files(self, Filename: str, Folder: Optional[str] = None) -> Union[Tuple[str], None]:
        """
        Finds all matching files

        :param Filename: Filename or  ID to search for
        :type Filename: str
        :param Folder: Specify folder filename in
        :type Folder: Any
        :return: Matching file/s
        :rtype: Any
        """
        # arg for specifying folder
        if Folder is not None:
            Filename = "".join([Folder, "\\", Filename])

        return [str(_path) for _path in self.files if Filename in str(_path)]

    def find_all_ext(self, ext: str) -> Union[List[str], None]:
        """
        Finds all files with specific extension

        :param ext: File extension
        :type ext: str
        :return: List of files
        :rtype: List[str]
        """
        # make sure appropriately formatted

        if "." not in ext:
            ext = "".join([".", ext])

        if "*" in ext:
            ext.replace("*", "")

        return [str(file) for file in self.files if file.suffix == ext]


class CollectedImagingFolder(CollectedDataFolder):
    """
    Class specifically for imaging folder, inherits collected data folder

    **Self Methods**
        | *load_fissa_exports* : loads fissa exported files
        | *load_cascade_exports* : loads cascade exported files
        | *load_suite2p* : loads suite2p exported files
        | *export_registration_to_denoised* : moves registration to new folder for namespace compatibility when skipping denoising step
        | *clean_up_motion_correction* : This function removes the reg_tif folder and registered.bin generated during motion correction.
        | *clean_up_compilation* : This function removes the compiled tif files
        | *add_notes* : Function adds notes

    """

    def __init__(self, Path: str):
        super().__init__(Path)
        self.parameters = dict()
        self.folders = None
        self.default_folders()

    def default_folders(self):
        self.folders = {
            "denoised": "".join([self.path, "\\denoised"]),
            "fissa": "".join([self.path, "\\fissa"]),
            "suite2p": "".join([self.path, "\\suite2p"]),
            "cascade": "".join([self.path, "\\cascade"]),
            "sorting": "".join([self.path, "\\sorting"]),
            "plane0": "".join([self.path, "\\suite2p\\plane0"]),
            "compiled": "".join([self.path, "\\compiled"])
        }

    def load_fissa_exports(self) -> Tuple[dict, dict, dict]:
        """
        This function loads the prepared and separated files exported from Fissa

        :return: Prepared, Separated, ProcessedTraces
        :rtype: tuple[dict, dict, dict]
        """

        def load_processed_traces(Filename) -> dict:

            def load_proc_traces(Filename_) -> dict:
                """
                Load Processed Traces from file

                :keyword load_path: Path containing processed traces
                :keyword absolute_path: Absolute filepath
                :rtype: dict
                """
                try:
                    print("Loading Processed Traces...")
                    _input_pickle = open(Filename_, 'rb')
                    ProcessedTraces_ = pkl.load(_input_pickle)
                    _input_pickle.close()
                    print("Finished Loading Processed Traces.")
                except RuntimeError:
                    print("Unable to load processed traces. Check supplied path.")
                    return dict()

                return ProcessedTraces_

            try:
                return load_proc_traces(Filename)
            except ModuleNotFoundError:
                print("Detected Deprecated Save. Migrating...")
                with open(Filename, "rb") as _file:
                    _ = renamed_load(_file)
                _file.close()
                with open(sFilename, "wb") as _file:
                    pkl.dump(_, _file)
                _file.close()
                # noinspection PyBroadException
                try:
                    return load_proc_traces(Filename)
                except Exception:
                    print("Migration Unsuccessful")
                    return dict()

        try:
            Prepared = np.load(self.find_matching_files("prepared")[0], allow_pickle=True)
        except FileNotFoundError:
            print("Could Not Locate Fissa Prepared File")
            Prepared = dict()

        try:
            Separated = np.load(self.find_matching_files("separated")[0], allow_pickle=True)
        except FileNotFoundError:
            print("Could Not Locate Fissa Separated File")
            Separated = dict()

        # noinspection PyBroadException
        try:
            ProcessedTraces = load_processed_traces(self.find_matching_files("ProcessedTraces")[0])
        except Exception:
            print("Could not locate processed traces file")
            ProcessedTraces = dict()

        return {**Prepared}, {**Separated}, {**ProcessedTraces}

    def load_cascade_exports(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        This function loads the Spike Times, Spike Prob, Discrete Approximation and ProcessedInferences files exported from Cascade

        :return: SpikeTimes, SpikeProb, DiscreteApproximation, Processed Inferences
        :rtype: tuple[Any, Any, Any, dict]
        """

        def load_processed_inferences(Filename) -> dict:

            def load_proc_inferences(Filename_) -> dict:
                """
                Load Processed Inferences from file

                :keyword load_path: Path containing processed inferences
                :keyword absolute_path: Absolute filepath
                :rtype: dict
                """
                try:
                    print("Loading Processed Inferences...")
                    _input_pickle = open(Filename_, 'rb')
                    ProcessedInferences_ = pkl.load(_input_pickle)
                    _input_pickle.close()
                    print("Finished Loading Processed Inferences.")
                except RuntimeError:
                    print("Unable to load processed inferences. Check supplied path.")
                    return dict()

                return ProcessedInferences_

            try:
                return load_proc_inferences(Filename)
            except ModuleNotFoundError:
                print("Detected Deprecated Save. Migrating...")
                with open(Filename, "rb") as _file:
                    _ = renamed_load(_file)
                _file.close()
                with open(Filename, "wb") as _file:
                    pkl.dump(_, _file)
                _file.close()
                # noinspection PyBroadException
                try:
                    return load_proc_inferences(Filename)
                except Exception:
                    print("Migration Unsuccessful")
                    return dict()

        try:
            SpikeTimes = np.load(self.find_matching_files("spike_times", "cascade")[0], allow_pickle=True)
        except FileNotFoundError:
            print("Could not locate Cascade spike times file.")
            SpikeTimes = None

        try:
            SpikeProb = np.load(self.find_matching_files("spike_prob", "cascade")[0], allow_pickle=True)
        except FileNotFoundError:
            print("Could not locate Cascade spike prob file.")
            SpikeProb = None

        try:
            DiscreteApproximation = np.load(self.find_matching_files("discrete_approximation", "cascade")[0], allow_pickle=True)
        except FileNotFoundError:
            print("Could not locate Cascade discrete approximation file.")
            DiscreteApproximation = None

        # noinspection PyBroadException
        try:
            ProcessedInferences = load_processed_inferences(self.find_matching_files("ProcessedInferences")[0])
        except Exception:
            print("Unable to locate processed inferences file")
            ProcessedInferences = dict()
        return SpikeTimes, SpikeProb, DiscreteApproximation, {**ProcessedInferences}

    def load_suite2p(self, *args: str):

        if args:
            _folder = args[0]
        else:
            _folder = "denoised"


        # Dynamic imports because \m/_(>.<)_\m/
        print("Loading Suite2p...")
        from ImagingAnalysis.Suite2PAnalysis import Suite2PModule
        suite2p_module = Suite2PModule(self.folders.get(_folder), self.path, file_type="binary")
        suite2p_module.load_files() # load the files
        suite2p_module.db = suite2p_module.ops # make sure db never overwrites ops
        print("Finished.")
        return suite2p_module

    def export_registration_to_denoised(self):
        """
        moves registration to new folder for namespace compatibility

        :return:
        """
        _images = np.reshape(np.fromfile(self.find_matching_files("registered_data.bin", "plane0")[0], dtype=np.int16), (-1, 512, 512))
        PreProcessing.saveRawBinary(_images, self.folders.get("denoised"))

    def clean_up_motion_correction(self) -> Self:
        """
        This function removes the reg_tif folder and registered.bin generated during motion correction.
         (You can avoid the creation of these in the first place by changing suite2p parameters)

        :rtype: Any
        """

        if self.find_matching_files("reg_tif").__len__() != 0:
            [pathlib.Path(_file).unlink() for _file in self.find_matching_files("reg_tif")]
        if self.find_matching_files("registered_data.bin").__len__() != 0 and self.find_matching_files(
                "binary_video", "suite2p//plane0").__len__() != 0:
            [pathlib.Path(_file).unlink() for _file in self.find_matching_files("data.bin")]
        if self.find_matching_files("data.bin").__len__() != 0 and self.find_matching_files(
                "binary_video", "suite2p//plane0").__len__() != 0:
            [pathlib.Path(_file).unlink() for _file in self.find_matching_files("data.bin")]

    def clean_up_compilation(self) -> Self:
        """
        This function removes the compiled tif files generated inside CompiledImagingData
        (You can avoid the creation of these in the first place by changing suite2p parameters)

        :rtype: Any
        """

        if self.find_matching_files("compiledVideo", "compiled").__len__() != 0:
            [pathlib.Path(_file).unlink() for _file in self.find_matching_files("compiledVideo", "compiled")]

    def add_notes(self, Step: str, KeyOrDict: Union[str, dict], Notes: Optional[Any] = None) -> Self:
        """
        Function adds notes indicating steps

        :param Step: Step of Analysis
        :param Step: str
        :param KeyOrDict: Either a Key or a dictionary containing multiple key-value (note) pairs
        :type KeyOrDict: Union[str, dict]
        :param Notes: If using key, then notes is the paired value
        :type Notes: Optional[Any]
        :rtype: Any
        """
        if isinstance(KeyOrDict, str) and Notes is not None:
            self.parameters[(Step, KeyOrDict)] = Notes
        elif isinstance(KeyOrDict, str) and Notes is None:
            self.parameters[(Step, KeyOrDict)] = Notes
            print("No value (note) provided to pair with key value. Added None")
        elif isinstance(KeyOrDict, dict):
            for _key in KeyOrDict:
                self.parameters[(Step, _key)] = KeyOrDict.get(_key)

    @property
    def current_stage(self) -> str:
        """
        Stage of Analysis

        :rtype: str
        """

        if self.find_matching_files("cascade").__len__() >= 3:
            return "Ready for Analysis"
        elif 1 < self.find_matching_files("cascade").__len__() < 3:
            return "Cascade: Discrete Inference"
        elif self.find_matching_files("fissa").__len__() >= 2:
            return "Cascade: Spike Probability"
        elif 1 <= self.find_matching_files("fissa").__len__() < 2:
            return "Fissa: Source-Separation"
        elif self.find_matching_files("spks.npy", "suite2p\\plane0").__len__() > 0:
            return "Fissa: Trace Extraction"
        elif self.find_matching_files("iscell.npy", "suite2p\\plane0").__len__() > 0:
            return "Suite2P: Spike Inference [Formality]"
        elif self.find_matching_files("F.npy", "suite2p\\plane0").__len__() > 0:
            return "Suite2P: Classify ROIs"
        elif self.find_matching_files("stat.npy", "suite2p\\plane0").__len__() > 0:
            return "Suite2P: Trace Extraction"
        elif self.find_matching_files("denoised").__len__() >= 1:
            return "Suite2P: ROI Detection"
        elif self.find_matching_files("suite2p").__len__() >= 2:
            return "DeepCAD: Denoising"
        else:
            return "Motion Correction"
