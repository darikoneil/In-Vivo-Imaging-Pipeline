<<<<<<< Updated upstream
import os
from datetime import date, datetime
import pickle as pkl
import numpy as np



class ExperimentData:
    def __init__(self, **kwargs):
        self.directory = kwargs.get('Directory', None)
        self.mouse_id = kwargs.get('Mouse', None)
        self.study = kwargs.get('Study', None)
        self.study_mouse = kwargs.get('StudyMouse', None)

        self.instance_date = self.getDate()
        self.modifications = [(self.getDate(), self.getTime())]
        self.stages = []

    @classmethod
    def loadHierarchy(cls, ExperimentDirectory):
        print("Incomplete Implementation")
        print("Loading Experimental Hierarchy..")
        _input_file = ExperimentDirectory + "\\ExperimentalHierarchy"
        _input_pickle = open(_input_file, 'rb')
        _pickles = pkl.load(_input_pickle)
        _keys = _pickles.keys()
        MouseData = ExperimentData()
        for _key in _keys:
            MouseData.__dict__.update(_pickles)
        print("Finished.")
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
        return self.directory, self.mouse_id, self.study, self.study_mouse

    def recordMod(self, *args):
        # noinspection PyTypeChecker
        self.modifications.append((self.getDate(), self.getTime(), *args))

    def saveHierarchy(self):
        print("Incomplete Implementation")
        print("Saving Experimental Hierarchy..")
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


class BehavioralStage:
    def __init__(self, Meta):
        self.mouse_directory = Meta[0]
        self.mouse_id = Meta[1]
        self.study = Meta[2]
        self.study_mouse = Meta[3]
        self.instance_date = ExperimentData.getDate()
        self.modifications = [(ExperimentData.getDate(), ExperimentData.getTime())]
        self.stage_directory = None
        self.computation_output_folder = None
        self.data_input_folder = None
        self.behavior_folder = None
        self.index_file = None
        self.features_file = None

    def recordMod(self):
        self.modifications.append((ExperimentData.getDate(), ExperimentData.getTime()))

    def setFolders(self):
        self.computation_output_folder = self.stage_directory + "\\Computation"
        self.data_input_folder = self.stage_directory + "\\Imaging"
        self.behavior_folder = self.stage_directory + "\\Behavior"
=======
import os
from datetime import date, datetime
import pickle as pkl
import numpy as np



class ExperimentData:
    """
    Class for Organizing & Managing Experimental Data Across Sessions

    Class Methods
    -------------
    **loadHierarchy** : Function that loads the entire experimental hierarchy
    """
    def __init__(self, **kwargs):
        # Hidden
        self._mouse_id_assigned = False # to make sure mouse id only set once
        # Protected In Practice
        self._mouse_id = kwargs.get('Mouse', None)
        self.__instance_date = self.getDate()
        #
        self.directory = kwargs.get('Directory', None)
        self.study = kwargs.get('Study', None)
        self.study_mouse = kwargs.get('StudyMouse', None)
        self.modifications = [(self.getDate(), self.getTime())]
        self.stages = []

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
        :rtype: AnalysisModules.ExperimentHierarchy.ExperimentData
        """
        print("Incomplete Implementation")
        print("Loading Experimental Hierarchy..")
        _input_file = ExperimentDirectory + "\\ExperimentalHierarchy"
        _input_pickle = open(_input_file, 'rb')
        _pickles = pkl.load(_input_pickle)
        _keys = _pickles.keys()
        MouseData = ExperimentData()
        for _key in _keys:
            MouseData.__dict__.update(_pickles)
        print("Finished.")
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
        print("Saving Experimental Hierarchy..")
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


class BehavioralStage:
    def __init__(self, Meta, Stage):
        # PROTECTED
        self.__mouse_id = Meta[1]
        self.__instance_date = ExperimentData.getDate()
        #
        self.mouse_directory = Meta[0]
        self.modifications = [(ExperimentData.getDate(), ExperimentData.getTime())]
        self.stage_directory = self.mouse_directory + "\\" + Stage
        self.folder_dictionary = dict()
        self.createFolderDictionary()

    @property
    def mouse_id(self):
        return self._BehavioralStage__mouse_id

    @property
    def instance_date(self):
        return self._BehavioralStage__instance_date

    def recordMod(self):
        self.modifications.append((ExperimentData.getDate(), ExperimentData.getTime()))

    def createFolderDictionary(self):
        """
        Creates a dictionary of locations for specific files

        **Requires**
            | self.stage_directory

        **Modifies**
            | self.folder_dictionary

        """

        self.folder_dictionary = {
            'computation_folder': self.stage_directory + "\\Computation",
            'imaging_folder': self.stage_directory + "\\Imaging",
            'behavior_folder': self.stage_directory + "\\Behavior",
        }


class CollectedDataFolder:
    """
    This is a class for managing a folder of unorganized

    Static Methods
    --------------
        | **fileParts** : Function returns each identifier of a file and its extension

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
        Function that indexed the files within again
        """
        self.files = self.path

    @staticmethod
    def fileLocator(IDs):
        locate = 1
        return locate

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
>>>>>>> Stashed changes
