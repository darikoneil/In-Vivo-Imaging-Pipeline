import os
from datetime import date, datetime
import pickle as pkl
import numpy as np
import pandas as pd
from IPython import get_ipython
import pathlib
from ExperimentManagement.BrukerMetaModule import BrukerMeta
from MigrationTools.Converters import renamed_load
import math


class ExperimentData:
    """
    Class for Organizing & Managing Experimental Data Across Sessions

    **Class Methods**
        | **loadHierarchy** : Function that loads the entire experimental hierarchy
    """

    def __init__(self, **kwargs):
        # Hidden
        self._mouse_id_assigned = False # to make sure mouse id only set once
        self._log_file_assigned = False # to make sure log file only set once
        self._experimental_condition_assigned = False # to make set condition only once
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

        # Create log file if one does not exist
        if self._log_file_assigned is False and self.directory is not None:
            self.log_file = self.directory + "\\log_file.log"
            print("Logging file assigned as :" + self.log_file)

        # start logging if log file exists
        if self._log_file_assigned:
            self.startLog()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.endLog()

    @property
    def experimental_condition(self):
        """
        Experiment condition of the mouse

        :rtype: str
        """
        return self._experimental_condition

    @experimental_condition.setter
    def experimental_condition(self, Condition):
        if self._experimental_condition_assigned is False:
            self._experimental_condition = Condition
            self._experimental_condition_assigned = True
        else:
            print("Experimental condition can only be assigned ONCE.")

    @property
    def log_file(self):
        return self._log_file

    @log_file.setter
    def log_file(self, LogFile):
        if self._log_file_assigned is False:
            self._log_file = LogFile
            self._log_file_assigned = True
        else:
            print("Log file can only be assigned ONCE.")

    @property
    def mouse_id(self):
        return self._mouse_id

    @mouse_id.setter
    def mouse_id(self, ID):
        if self._mouse_id_assigned is False:
            self._mouse_id = ID
            self._mouse_id_assigned = True
        else:
            print("Mouse ID can only be set ONCE.")

    @property
    def instance_date(self):
        return self._ExperimentData__instance_date

    @classmethod
    def loadHierarchy(cls, ExperimentDirectory):
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
    def checkPath(cls, Path):
        return os.path.exists(Path)

    @classmethod
    def generateDirectoryHierarchy(cls, MouseDirectory, **kwargs):
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
    def generateHistology(cls, MouseDirectory, **kwargs):
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
    def generateROIMatchingIndex(cls, MouseDirectory):
        _roi_matching_index_dir = MouseDirectory + "\\ROIMatchingIndex"
        _roi_matching_index_read_me = _roi_matching_index_dir + "\\ReadMe.txt"

        os.makedirs(_roi_matching_index_dir)
        cls.generateReadMe(_roi_matching_index_read_me,
                           "Read-Me for Index of Longitudinally-Matched ROIs")

    @classmethod
    def generateStage(cls, MouseDirectory, **kwargs):
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
    def generateBehavior(cls, StageDirectory):
        _base_behav_dir = StageDirectory + "\\Behavior"
        _raw_behavioral_data = _base_behav_dir + "\\RawBehavioralData"
        _behavioral_exports = _base_behav_dir + "\\BehavioralExports"
        _deep_lab_cut_data = _base_behav_dir + "\\DeepLabCutData"
        _processed_data = _base_behav_dir + "\\ProcessedData"
        _analog_burrow_data = _base_behav_dir + "\\AnalogBurrowData"
        os.makedirs(_base_behav_dir)
        os.makedirs(_raw_behavioral_data)
        os.makedirs(_behavioral_exports)
        os.makedirs(_deep_lab_cut_data)
        os.makedirs(_processed_data)
        os.makedirs(_analog_burrow_data)

    @classmethod
    def generateImaging(cls, StageDirectory, **kwargs):
        _sample_frequency = kwargs.get('SampleFrequency', 30)
        _sample_frequency_string = str(_sample_frequency) + "Hz"
        _base_image_dir = StageDirectory + "\\Imaging"
        _sample_frequency_dir = _base_image_dir + "\\" + _sample_frequency_string
        _raw_imaging_data = _base_image_dir + "\\RawImagingData"
        _bruker_meta_data = _base_image_dir + "\\BrukerMetaData"
        os.makedirs(_base_image_dir)
        os.makedirs(_raw_imaging_data)
        os.makedirs(_bruker_meta_data)
        cls.generateSampFreq(_sample_frequency_dir)

    @classmethod
    def generateSampFreq(cls, SampFreqDirectory):
        _suite2p = SampFreqDirectory + "\\suite2p"
        _fissa = SampFreqDirectory + "\\fissa"
        _roi_sorting = SampFreqDirectory + "\\sorting"
        _denoised = SampFreqDirectory + "\\denoised"
        os.makedirs(_suite2p)
        os.makedirs(_fissa)
        os.makedirs(_denoised)
        os.makedirs(_roi_sorting)
        cls.generateReadMe(_roi_sorting+"\\ReadMe.txt", "Read-Me for ROI Sorting")

    @classmethod
    def generateComputation(cls, StageDirectory, **kwargs):
        _analysis_title = kwargs.get('Title', 'AnalysisTechnique')
        _base_comp_dir = StageDirectory + "\\Computational"
        _neural_data_dir = _base_comp_dir + "\\NeuralData"
        _analysis_dir = _base_comp_dir + "\\" + _analysis_title
        os.makedirs(_base_comp_dir)
        os.makedirs(_neural_data_dir)

    @classmethod
    def generateAnalysisTechnique(cls, BaseCompDirectory, AnalysisTitle):
        _analysis_dir = BaseCompDirectory+"\\"+AnalysisTitle
        os.makedirs(_analysis_dir)
        cls.generateReadMe(_analysis_dir+"\\ReadMe.txt", "Read-Me for Analysis Technique")

    @staticmethod
    def generateReadMe(AbsoluteFilePath, Text):
        with open(AbsoluteFilePath, 'w') as _read_me:
            _read_me.write(Text)
            _read_me.close()

    def passMeta(self):
        return self.directory, self.mouse_id

    def recordMod(self, *args):
        """
        Record modification of experiment (Data, Time, *args)

        **Modifies**
            | self.modifications

        :param args: A string explaining the modification
        :rtype: None
        """
        # noinspection PyTypeChecker
        self.modifications.append((self.getDate(), self.getTime(), *args))

    def saveHierarchy(self):
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

    def createLogFile(self):
        self.log_file = self.directory + "\\log_file.log"

    # noinspection All
    def startLog(self):
        self._IP = get_ipython()
        _magic_arguments = '-o -r -t ' + self.log_file + ' append'
        self._IP.run_line_magic('logstart', _magic_arguments)
        return print("Logging Initiated")

    # noinspection All
    def endLog(self):
        self._IP.run_line_magic('logstop', '')

    # noinspection All
    def checkLog(self): # noinspection All
        self._IP.run_line_magic('logstate', '')
        return


class BehavioralStage:
    """
    **Self Methods**
        | **recordMod** : Records a modification made to the behavioral stage (Date & Time)
        | **createFolderDictionary** : Creates a dictionary of locations for specific files
        | **fillImagingDictionary** :  Generates folders and a dictionary of imaging files (Raw, Meta, Compiled)
        | **addImageSamplingFolder** : Generates a folder for containing imaging data of a specific sampling rate
        | **addImageProcessingFolder** : Generates a folder for containing processed imaging data

    **Properties**
        | **mouse_id** : Identifies which mouse this data belongs to
        | **instance_data** : Identifies when this behavioral stage was created

    **Attributes**
        | **modifications** : List of modifications made to this behavioral stage
        | **folder_dictionary** : A dictionary of relevant folders for this behavioral stage
    """
    def __init__(self, Meta, Stage):
        # PROTECTED
        self.__mouse_id = Meta[1]
        self.__instance_date = ExperimentData.getDate()
        #
        self.modifications = [(ExperimentData.getDate(), ExperimentData.getTime())]
        self.folder_dictionary = dict()
        # self.data = pd.DataFrame
        self.meta_data = None

        _mouse_directory = Meta[0]
        self.createFolderDictionary(_mouse_directory, Stage)
        self.fillImagingFolderDictionary()

    @property
    def mouse_id(self):
        return self._BehavioralStage__mouse_id

    @property
    def instance_date(self):
        return self._BehavioralStage__instance_date

    def recordMod(self):
        """
        Records a modification made to the behavioral stage (Date & Time)
        """
        self.modifications.append((ExperimentData.getDate(), ExperimentData.getTime()))

    def createFolderDictionary(self, MouseDirectory, Stage):
        """
        Creates a dictionary of locations for specific files

        **Modifies**
            | self.folder_dictionary

        :param MouseDirectory: Directory containing mouse data (passed meta)
        :type MouseDirectory: str
        :param Stage: The Stage ID
        :type Stage: str

        """
        _stage_directory = MouseDirectory + "\\" + Stage
        self.folder_dictionary = {
            'mouse_directory': MouseDirectory,
            'stage_directory': _stage_directory,
            'computation_folder': _stage_directory + "\\Computation",
            'imaging_folder': _stage_directory + "\\Imaging",
            'behavior_folder': _stage_directory + "\\Behavior",
        }

    def fillImagingFolderDictionary(self):
        """
        Generates folders and a dictionary of imaging files (Raw, Meta, Compiled)

        **Requires**
            | self.folder_dictionary

        **Modifies**
            | self.folder_dictionary
        """
        # RAW
        _raw_data_folder = self.folder_dictionary['imaging_folder'] + "\\RawImagingData"
        try:
            os.makedirs(_raw_data_folder)
        except FileExistsError:
            print("Existing Raw Data Folder Detected")
        self.folder_dictionary['raw_imaging_data'] = _raw_data_folder
        # META
        _bruker_meta_folder = self.folder_dictionary['imaging_folder'] + "\\BrukerMetaData"
        try:
            os.makedirs(_bruker_meta_folder)
        except FileExistsError:
            print("Existing Bruker Meta Data Folder Detected")
        self.folder_dictionary['bruker_meta_data'] = CollectedDataFolder(_bruker_meta_folder)
        # COMPILED
        _compiled_imaging_data_folder = self.folder_dictionary['imaging_folder'] + "\\CompiledImagingData"
        try:
            os.makedirs(_compiled_imaging_data_folder)
        except FileExistsError:
            print("Existing Compiled Data Folder Detected")

        self.folder_dictionary['compiled_imaging_data_folder'] = CollectedDataFolder(_compiled_imaging_data_folder)

    def addImageSamplingFolder(self, SamplingRate):
        """
        Generates a folder for containing imaging data of a specific sampling rate

        :param SamplingRate: Sampling Rate of Dataset in Hz
        :type SamplingRate: int
        """
        SamplingRate = str(SamplingRate) # Because we know I'll always forget and send an int anyway
        _folder_name = self.folder_dictionary['imaging_folder'] + "\\" + SamplingRate + "Hz"
        try:
            os.makedirs(_folder_name)
        except FileExistsError:
            print("The sampling folder already exists. Adding to folder dictionary")
        self.folder_dictionary[SamplingRate + "Hz"] = CollectedImagingFolder(_folder_name)
        ExperimentData.generateSampFreq(_folder_name)

    def addImageProcessingFolder(self, Title):
        """
        Generates a folder for containing processed imaging data

        :param Title: The name of the folder
        :type Title:  str
        """
        _folder_name = self.folder_dictionary['imaging_folder'] + "\\" + Title
        try:
            os.makedirs(_folder_name)
        except FileExistsError:
            print("The image processing folder already exists. Adding to folder dictionary")
        self.folder_dictionary[Title] = CollectedDataFolder(_folder_name)

    def addImagingAnalysis(self, SamplingRate):
        try:
            self.folder_dictionary.get(str(SamplingRate) + "Hz")
        except KeyError:
            self.addImageProcessingFolder(SamplingRate)

        self.__dict__["imaging_" + str(SamplingRate) + "_Hz"] = {
            'suite2p': None,
            'fissa': None,
            'cascade': None,
        }

    def loadBrukerMetaData(self):
        self.folder_dictionary["bruker_meta_data"].reIndex()
        _files = self.folder_dictionary["bruker_meta_data"].find_all_ext("xml")
        self.meta_data = BrukerMeta(_files[0], _files[2], _files[1])
        self.meta_data.import_meta_data()
        self.meta_data.creation_date = ExperimentData.getDate()

    def loadBrukerAnalogRecordings(self):
        self.folder_dictionary["bruker_meta_data"].reIndex()
        _files = self.folder_dictionary["bruker_meta_data"].find_all_ext("csv")
        return self.load_bruker_analog_recordings(_files[-1])

    @staticmethod
    def load_bruker_analog_recordings(File):
        return pd.read_csv(File)

    @staticmethod
    def sync_bruker_recordings(DataFrame, AnalogRecordings, MetaData):
        # this represents first bruker time point
        _relative_zero = np.searchsorted(DataFrame["Imaging Sync"].values, 1.4)
        _frame_period = MetaData.imaging_metadata.get("framePeriod")
        _num_frames = MetaData.imaging_metadata.get("relativeTimes").__len__()
        _relative_times = np.around(np.array(MetaData.imaging_metadata.get("relativeTimes")), decimals=3)

        _crop_idx = DataFrame.shape[0]-_relative_zero
        _crop_time = np.around(AnalogRecordings["Time(ms)"].values[_crop_idx]*(1/1000), decimals=3)
        _cropped_analog = AnalogRecordings.iloc[0:_crop_idx].values
        _image_crop_idx = np.searchsorted(_relative_times, _crop_time)-1
        _cropped_frames = np.arange(0, _image_crop_idx, 1, dtype=np.float64)

        _analog_time = np.around(np.arange(0 + _relative_zero*(1/1000), _cropped_analog.shape[0] * (1 / 1000) +
                                           _relative_zero*(1/1000), 1 / 1000, dtype=np.float64), decimals=3)

        _image_time = np.around(np.arange(0 + _relative_zero*(1/1000), _cropped_frames.shape[0]*_frame_period +
                                          _relative_zero*(1/1000),
                                          _frame_period, dtype=np.float64), decimals=3)

        _analog_indexed = pd.DataFrame(_cropped_analog[:, 1:], index=_analog_time)
        _image_indexed = pd.Series(_cropped_frames, index=_image_time)
        _image_indexed.sort_index(inplace=True)
        _image_indexed = _image_indexed.reindex(_analog_time)
        _image_indexed.name = "Imaging Frame"
        _channel_names = MetaData.analog_channel_names
        for _name in range(_channel_names.__len__()):
            _channel_names[_name] = _channel_names[_name].replace(" ", "")
        _analog_indexed.columns = _channel_names
        DataFrame = DataFrame.join(_analog_indexed)
        DataFrame = DataFrame.join(_image_indexed.copy(deep=True))
        _image_indexed.bfill(inplace=True)
        _image_indexed.name = "[BFILL] Imaging Frame"
        DataFrame = DataFrame.join(_image_indexed.copy(deep=True))
        return DataFrame

    @staticmethod
    def sync_downsampled_images(DataFrame, AnalogRecordings, MetaData, **kwargs):
        _downsample_size = kwargs.get("downsample_multiplier", 3)
        _relative_zero = np.searchsorted(DataFrame["Imaging Sync"].values, 1.4)
        _frame_period = MetaData.imaging_metadata.get("framePeriod")
        _num_frames = MetaData.imaging_metadata.get("relativeTimes").__len__()
        _relative_times = np.around(np.array(MetaData.imaging_metadata.get("relativeTimes")), decimals=3)

        _crop_idx = DataFrame.shape[0]-_relative_zero
        _crop_time = np.around(AnalogRecordings["Time(ms)"].values[_crop_idx]*(1/1000), decimals=3)
        _cropped_analog = AnalogRecordings.iloc[0:_crop_idx].values
        _image_crop_idx = np.searchsorted(_relative_times, _crop_time)-1
        _cropped_frames = np.arange(0, _image_crop_idx, 1, dtype=np.float64)
        _analog_time = np.around(np.arange(0 + _relative_zero*(1/1000), _cropped_analog.shape[0] * (1 / 1000) +
                                           _relative_zero*(1/1000), 1 / 1000, dtype=np.float64), decimals=3)

        _image_time = np.around(np.arange(0 + _relative_zero*(1/1000), _cropped_frames.shape[0]*_frame_period +
                                          _relative_zero*(1/1000),
                                          _frame_period, dtype=np.float64), decimals=3)

        _analog_indexed = pd.DataFrame(_cropped_analog[:, 1:], index=_analog_time)
        _downsampled_frames = _cropped_frames[0:math.floor(_cropped_frames.shape[0]/_downsample_size)]
        _downsampled_time = np.around(np.arange(0 + _relative_zero*(1/1000), _downsampled_frames.shape[0]*(
                _frame_period/_downsample_size) + _relative_zero*(1/1000), _frame_period/_downsample_size,
                                                dtype=np.float64), decimals=3)
        _downsampled_indexed = pd.Series(_downsampled_frames, index=_downsampled_time)
        _downsampled_indexed.sort_index(inplace=True)
        _downsampled_indexed = _downsampled_indexed.reindex(_analog_time)
        _downsampled_indexed.name = "Downsampled Frame"
        DataFrame = DataFrame.join(_downsampled_indexed.copy(deep=True))
        _downsampled_indexed.bfill(inplace=True)
        _downsampled_indexed.name = "[BFILL] Downsampled Frame"
        DataFrame = DataFrame.join(_downsampled_indexed)
        return DataFrame



class CollectedDataFolder:
    """
    This is a class for managing a folder of unorganized data files

**Self Methods**
        | **searchInFolder** : Search THIS object for the *first* file which matches the description
        | **reIndex** : Function that indexed the files within folder again

**Static Methods**
        | **fileParts** : Function returns each identifier of a file and its extension
        | **fileLocator** : Find the file which matches the description


    """
    def __init__(self, Path):
        # Protected In Practice
        self.__instance_date = ExperimentData.getDate()
        self._path = str()
        self._path_assigned = False

        # Properties
        self._files = []
        self.path = Path
        self.files = self.path

    @property
    def instance_date(self):
        return self._CollectedDataFolder__instance_date

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, Path):
        if self._path_assigned is False:
            self._path = Path
            self._path_assigned = True
        else:
            print("Path can only be assigned ONCE.")

    @property
    def files(self):
        return self._files

    @files.setter
    def files(self, Path):
        self._files = os.listdir(Path)

    def reIndex(self):
        """
        Function that indexed the files within folder again
        """
        self.files = self.path

    def searchInFolder(self, ID):
        """
        Search THIS object for the *first* file which matches the description

        :param ID: The description
        :type ID: str
        :return: The absolute file path of the matching filename
        :rtype: str
        """
        return self.path + "\\" + CollectedDataFolder.fileLocator(self.files, ID)

    def find_all_ext(self, ext):
        _ext = "".join(["*.", ext])
        Files = list(pathlib.Path(self.path).glob(_ext))
        for i in range(Files.__len__()):
            Files[i] = Files[i].__str__()
        return Files

    @staticmethod
    def fileLocator(files, ID):
        """
        Find the *first* file which matches the description

        :param files: A list of files
        :type files: list
        :param ID: The description
        :type ID: str
        :return: The matching filename
        :rtype: str
        """
        for i in range(len(files)):
            for _id in CollectedDataFolder.fileParts(files[i]):
                if ID == _id:
                    return files[i]
        return "Nil"

    @staticmethod
    def fileParts(file):
        """
        Function returns each identifier of a file and its extension

        :param file: Filename to be parsed
        :type file: str
        :return: stage, animal, ...[unique file identifiers]..., extension
        :rtype: list
        """
        return file.split("_")[0:-1] + file.split("_")[-1].split(".")


class CollectedImagingFolder(CollectedDataFolder):
    """ this is a super class containing methods for"""
    def __init__(self, Path):
        super().__init__(Path)

    def load_fissa(self):
        return print("Not Yet Implemented")

    def load_cascade(self):
        return print("Not Yet Implemented")

    def load_suite2p(self):
        return

    def import_proc_traces(self):
        return self.load_proc_inferences(absolute_path=self.searchInFolder("ProcessedTraces"))

    def import_proc_inferences(self):
        try:
            return self.load_proc_inferences(absolute_path=self.searchInFolder("ProcessedInferences"))
        except ModuleNotFoundError:
            print("Detected Deprecated Save. Migrating...")
            with open(self.searchInFolder("ProcessedInferences"), "rb") as _file:
                _ = renamed_load(_file)
            _file.close()
            with open(self.searchInFolder("ProcessedInferences"), "wb") as _file:
                pkl.dump(_, _file)
            _file.close()
            # noinspection PyBroadException
            try:
                return self.load_proc_inferences(absolute_path=self.searchInFolder("ProcessedInferences"))
            except Exception:
                print("Migration Unsuccessful")
                return

    @staticmethod
    def load_proc_inferences(**kwargs):
        """
        Save Processed Inferences to file

        **Requires**
            | self.ProcessedInferences

        :keyword load_path: Path containing processed inferences
        :keyword absolute_path: Absolute filepath
        :rtype: object
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
            ProcessedInferences = pkl.load(_input_pickle)
            _input_pickle.close()
            print("Finished Loading Processed Traces.")

        except RuntimeError:
            print("Unable to load processed inferences. Check supplied path.")
            return

        return ProcessedInferences
