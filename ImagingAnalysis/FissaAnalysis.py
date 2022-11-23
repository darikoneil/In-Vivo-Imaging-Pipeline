from __future__ import annotations

import os.path

import numpy as np
import pickle as pkl
import fissa
import pathlib
from typing import Tuple, List
from ImagingAnalysis.PreprocessingImages import PreProcessing


# /// /// Main Module /// ///
class FissaModule:
    """
    Fissa Module
    ------------
    A module for Fissa Signal Extraction & Source-Separation

    Self Methods
    ------------
    | **loadDataFolder** : Load a Suite2P folder (single-plane only) into Class
    |
    | **loadSuite2P_ROIs** : Loads Suite2P ROI Masks into Class
    |
    | **loadNeuronalIndex** : Loads Neuronal Index into Class
    |
    | **deriveNeuronalIndex** : Derives Neuronal Index from Class
    |
    | **loadFissaPrep** : Load saved preparation data into class
    |
    | **loadFissaSep** : Load saved separation data into class
    |
    | **initializeFissa** : Initialize Fissa
    |
    | **passPrepToFissa** : Passes Preparation Data to Use in Separation Process
    |
    | **saveFissaPrep** : Saves Fissa Prep
    |
    | **saveFissaSep** : Saves Fissa Sep
    |
    | **saveFissaAll** : Saves Sep & Prep
    |
    | **saveProcessedTraces** : Saves Processed Traces
    |
    | **loadProcessedTraces** : Loads Processed Traces
    |
    | **saveAll** : Saves All
    |
    | **extractTraces**
    |
    | **separateTraces**
    |
    | **passExperimentToPrep**
    |
    | **preserveOriginalExtractions**
    |
    | **preserveOriginalSourceSeparations**
    |
    | **pruneNonNeuronalROIs**

    """
    def __init__(self, **kwargs):

        # ///Parse Inputs///
        _data_folder = kwargs.get('data_folder', None)
        _output_folder = kwargs.get("output_folder", None)
        _index_file = kwargs.get('index_file', None)
        _video_folder = kwargs.get('video_folder', None)
        self.neuronal_index = kwargs.get('neuronal_index', None)
        self.frame_rate = kwargs.get('frame_rate', None)
        self.prep_file = kwargs.get('prep_file', None)
        self.sep_file = kwargs.get('sep_file', None)

        # /// Initialize Paths ///
        self.images = None
        self.ops = {}
        self.stat = None
        self.iscell = None
        self.s2p_rois = None
        self.output_folder = _output_folder
        self.sep_file = None
        self.prep_file = None
        self.index_path = None

        # /// Load Data (Suite2P & FISSA) ///
        if _data_folder is not None:
            if _video_folder is None:
                self.loadDataFolder(_data_folder)
            else:
                self.loadDataFolder(_data_folder, _video_folder)
            # Load Neuronal Index if specified or derive
            if _index_file is not None:
                self.index_path = _index_file
                self.loadNeuronalIndex()
            else:
                self.deriveNeuronalIndex()
        else:
            print("Initializing without data")

        # /// Pre-Allocation///
        self.experiment = SeparationModule() # Store Fissa Separation Data
        self.preparation = PreparationModule() # Store Fissa Preparation Data
        self.ProcessedTraces = ProcessedTracesModule()

    def loadDataFolder(self, _data_folder: str, *args) -> Self:
        """
        loadDataFolder
        --------------
        Load a Suite2P folder (single-plane only) into Class

        **Modifies**
            | self.images
            | self.ops
            | self.stat
            | self.iscell
            | self.s2p_rois
            | self.output_folder
            | self.preparation
            | self.experiment
            | self.prep_file
            | self.sep_file

        :param _data_folder: Suite2P folder with registered tiffs
        :type _data_folder: str
        :rtype: None
        """

        if len(args) > 0:
            _video_folder = args[0]
        else:
            _video_folder = _data_folder


        if os.path.exists("".join([_video_folder, "\\plane0\\reg_tif"])):
            self.images = _video_folder + '\\suite2p\\plane0\\reg_tif'
            # Images stored here
        else:
            print("\nLoading and Splitting Images\n")
            self.images = self.split_binary_images(PreProcessing.loadRawBinary("", "", _video_folder))

        try:
            self.ops = np.load((_data_folder + '\\suite2p\\plane0\\ops.npy'),
                                allow_pickle=True).item()
            if type(self.ops) is not dict:
                raise TypeError
        except TypeError:
            print("Suite2P Ops file in unexpected format")
        except RuntimeError:
            print("Could not locate Suite2P ops file")
        # Suite2P ops stored here & loaded

        if self.frame_rate is None and self.ops:
            try:
                # noinspection PyUnresolvedReferences
                self.frame_rate = self.ops["fs"]
            except KeyError:
                print("Could not derive frame rate from ops file")
        # Derive frame_rate if and only if it was not specified

        try:
            self.stat = np.load((_data_folder + '\\suite2p\\plane0\\stat.npy'),
                                allow_pickle=True)
            if type(self.stat) is not np.ndarray:
                raise TypeError
        except TypeError:
            print("Suite2P Stats file in unexpected format")
        except RuntimeError:
            print("Could not locate Suite2P stats file")
        # Suite2P stats stored here & loaded

        try:
            self.iscell = np.load((_data_folder + '\\suite2p\\plane0\\iscell.npy'),
                                    allow_pickle=True)
            if type(self.iscell) is not np.ndarray:
                raise TypeError
        except TypeError:
            print("Suite2P iscell file in unexpected format")
        except RuntimeError:
            print("Could not locate Suite2P iscell file")
        # Suite2P neuronal index stored here & loaded

        try:
            self.loadSuite2P_ROIs()
            # Load Suite2P ROI Masks
        except RuntimeError or IndexError or AttributeError:
            print("Could not load Suite2P ROIs")

        if self.output_folder is None:
            self.output_folder = _data_folder + '\\'
            # Where to Save Any New Exports

        if self.prep_file is not None:
            try:
                self.loadFissaPrep()
            except RuntimeError or AttributeError:
                print("Could not load FISSA prepared file")
        else:
            self.prep_file = self.output_folder + "\\prepared.npz"
            # Location of Fissa Preparation File

        if self.sep_file is not None:
            try:
                self.loadFissaSep()
            except RuntimeError or AttributeError:
                print("Could not load FISSA separation file")
        else:
            self.sep_file = self.output_folder + "\\separated.npz"
            # Location of Fissa Separation File

    def loadSuite2P_ROIs(self) -> Self:
        """
        Loads Suite2P ROI Masks into Class

        **Requires**
            | self.ops
            | self.cells
            | self.stat

        **Modifies**
            | self.s2p_rois

        :rtype: None
        """
        # Get image size
        Lx = self.ops["Lx"]
        Ly = self.ops["Ly"]

        # Get the cell ids
        ncells = len(self.stat)
        cell_ids = np.arange(ncells)  # assign each cell an ID, starting from 0.
        cell_ids = cell_ids[self.iscell[:, 0] == 1]  # only take the ROIs that are actually cells.
        num_rois = len(cell_ids)

        # Generate ROI masks in a format usable by FISSA (in this case, a list of masks)
        rois = [np.zeros((Ly, Lx), dtype=bool) for _ in range(num_rois)]

        for i, n in enumerate(cell_ids):
            # Variable i is the position in cell_ids, and n is the actual cell number
            ypix = self.stat[n]["ypix"][~self.stat[n]["overlap"]]
            xpix = self.stat[n]["xpix"][~self.stat[n]["overlap"]]
            rois[i][ypix, xpix] = 1
        self.s2p_rois = rois

    def loadNeuronalIndex(self) -> Self:
        """
        Loads Neuronal Index into Class

        **Requires**
        self.index_path

        **Modifies**
        self.neuronal_index

        :rtype: None
        """
        # Load File
        file_ext = pathlib.Path(self.index_path).suffix
        if file_ext == ".npy" or file_ext == ".npz":
            self.neuronal_index = np.load(self.index_path, allow_pickle=True)
            if len(self.neuronal_index.shape) == 2 and self.neuronal_index.shape[1] == 2:
                self.neuronal_index = self.neuronal_index[:, 0]
                # Keep the index only
        elif file_ext == ".csv":
            self.neuronal_index = np.genfromtxt(self.index_path, dtype=int)
        else:
            print("Neuronal index is unexpected file type.")
        try:
            if type(self.neuronal_index) is not np.ndarray:
                raise TypeError
            elif len(self.neuronal_index.shape) != 1:
                raise ValueError
        except ValueError:
            print("Neuronal index in unexpected organization")
        except TypeError:
            print("Neuronal index in unexpected type")
        if ".csv" in self.index_path:
            print("Adjusting from 1-to-0 indexing")

    def deriveNeuronalIndex(self) -> Self:
        """
        Derives Neuronal Index from Class

        **Requires**
            | self.iscell

        **Modifies**
            | self.neuronal_index

        :rtype: None
        """
        self.neuronal_index = self.iscell[:, 0].astype(dtype=np.int64, copy=True)
        self.neuronal_index = np.where(self.neuronal_index < 1, np.nan, self.neuronal_index)
        self.neuronal_index = self.neuronal_index*range(self.neuronal_index.shape[0])
        self.neuronal_index = self.neuronal_index[~np.isnan(self.neuronal_index)]

    def loadFissaPrep(self) -> Self:
        """
        Load saved preparation data into class

        **Requires**
            | self.prep_file

        **Modifies**
            | self.preparation

        :rtype: None
        """
        # Load Existing Prep File
        print('Loading Fissa Preparation...')
        _prep = np.load(self.prep_file, allow_pickle=True)
        self.preparation = PreparationModule(preparation=_prep)
        print('Finished Loading Fissa Prep')

    def loadFissaSep(self) -> Self:
        """
        Load saved separation data into class

        **Requires**
            | self.sep_file

        **Modifies**
            | self.experiment

        :rtype: None
        """

        # Load Existing Sep File
        print('Loading Fissa Separation...')
        _sep = np.load(self.sep_file, allow_pickle=True)
        self.experiment = SeparationModule(experiment=_sep)
        print('Finished Loading Fissa Sep')

    def initializeFissa(self) -> Self:
        """
        Initialize Fissa

        **Requires**
            | self.images
            | self.s2p_rois

        **Modifies**
            | self.experiment

        :rtype: None
        """
        self.experiment = fissa.Experiment(self.images, [self.s2p_rois[:len(self.s2p_rois)]])
        # noinspection PyProtectedMember
        self.experiment._adopt_default_parameters(only_preparation=False, force=False)
        print("Initialized Fissa")

    def passPrepToFissa(self) -> Self:
        """
        Passes Preparation Data to Use in Separation Process

        **Requires**
            | self.preparation

        **Modifies**
            | self.experiment
        :rtype: None
        """
        self.experiment.means = self.preparation.means
        self.experiment.roi_polys = self.preparation.roi_polys
        self.experiment.expansion = self.preparation.expansion
        self.experiment.nRegions = self.preparation.nRegions
        self.experiment.raw = self.preparation.raw

    def saveFissaPrep(self) -> Self:
        """
        Saves Fissa Prep

        **Requires**
            | self.preparation
            | self.experiment
            | self.output_folder

        :rtype: None
        """
        self.experiment.save_prep(destination=self.output_folder+"\\prepared.npz")
        print("Finished Saving Prep")

    def saveFissaSep(self) -> Self:
        """
        Saves Fissa Sep

        **Requires**
            | self.preparation
            | self.experiment
            | self.output_folder

        :rtype: None
        """
        self.experiment.save_separated(destination=self.output_folder+"\\separated.npz")

    def saveFissaAll(self) -> Self:
        """
        Saves Fissa Sep & Prep

        **Requires**
            | self.preparation
            | self.experiment
            | self.output_folder

        :rtype: None
        """
        self.saveFissaPrep()
        self.saveFissaSep()

    def saveProcessedTraces(self) -> Self:
        """
        Saves Processed Traces

        **Requires**
            self.ProcessedTraces
            self.output_folder

        :rtype:
        """
        print("Saving Processed Traces...")
        _output_file = self.output_folder + "\\ProcessedTraces"
        _output_pickle = open(_output_file, 'wb')
        pkl.dump(self.ProcessedTraces, _output_pickle)
        _output_pickle.close()
        print("Finished Saving Processed Traces.")

    def loadProcessedTraces(self) -> Self:
        """
        Loads Processed Traces

        **Requires**
            | self.output_folder
            | self.ProcessedTraces

        **Modifies**
            | self.ProcessedTraces

        :rtype: None
        """
        print("Loading Processed Traces...")
        _input_file = self.output_folder + "\\ProcessedTraces"
        _input = open(_input_file, 'rb')
        self.ProcessedTraces = pkl.load(_input)
        _input.close()
        print("Finished Loading Processed Traces.")

    def saveAll(self) -> Self:
        """
        Saves All

        **Requires**
            | self.output_folder
            | self.preparation
            | self.experiment
            | self.ProcessedTraces

        :rtype: None
        """
        self.saveFissaPrep()
        self.saveFissaSep()
        self.saveProcessedTraces()

    def extractTraces(self) -> Self:
        self.experiment.separation_prep()
        print("Passing traces between modules.")
        self.passExperimentToPrep()
        self.preserveOriginalExtractions()
        print("Finished module-passing.")
        print("Ready for post-processing or source-separation.")

    def separateTraces(self) -> Self:
        print('Passing prepared traces to fissa.')
        self.passPrepToFissa()
        print('Initiating fissa source-separation')
        self.experiment.separate()
        print("Passing traces between modules.")
        self.preserveOriginalSourceSeparations()
        print("Finished module-passing.")
        print("Ready for further analysis.")

    def passExperimentToPrep(self) -> Self:
        self.preparation.raw = self.experiment.raw
        self.preparation.roi_polys = self.experiment.roi_polys
        self.preparation.expansion = self.experiment.expansion
        self.preparation.nRegions = self.experiment.nRegions
        self.preparation.means = self.experiment.means

    def preserveOriginalExtractions(self) -> Self:
        self.ProcessedTraces.original_raw = self.preparation.raw

    def preserveOriginalSourceSeparations(self) -> Self:
        self.ProcessedTraces.original_result = self.experiment.result

    def pruneNonNeuronalROIs(self) -> Self:
        self.stat = self.stat[self.neuronal_index]
        self.s2p_rois = [self.s2p_rois[i] for i in self.neuronal_index]
        self.iscell = self.iscell[self.neuronal_index, :]

    @staticmethod
    def split_binary_images(BinaryVideo: np.ndarray) -> List[np.ndarray]:
        """
        This Function splits binary image into stacks for multiprocessing

        :param BinaryVideo: Binary Video in numpy array [Z x Y x X]
        :type BinaryVideo: Any
        :return: List of Binary Videos
        :rtype: list
        """

        def determine_split_size(ImageLength: int, SizeLimit: int) -> int:
            """
            Function determines chunk size for evenly sized stacks
            :param ImageLength: Number of Frames
            :type ImageLength: int
            :param SizeLimit: Chunk Limit
            :type SizeLimit: int
            :return: Chunk Size for Evenly Sized Stacks
            :rtype: int
            """

            if ImageLength % 2 == 0 and ImageLength / 2 <= SizeLimit:
                return 2
            chunk_size = 3
            while chunk_size * chunk_size <= ImageLength:
                if ImageLength % chunk_size == 0 and ImageLength / chunk_size <= SizeLimit:
                    return chunk_size
                chunk_size += 1

        _num_frames = BinaryVideo.shape[0]
        _chunk_size = determine_split_size(_num_frames, 8000)
        _idx = np.arange(0, _num_frames+1, _num_frames/_chunk_size)
        img_list = []
        for i in range(_idx.shape[0] - 1):
            img_list.append(BinaryVideo[_idx[i].astype(int):_idx[i + 1].astype(int), :, :])
        return img_list


# /// /// Container for Preparation Data /// ///
class PreparationModule:
    """
    Preparation Module
    """
    def __init__(self, **kwargs):
        _preparation = kwargs.get('preparation', None)

        self.roi_polys = None # ROI Polygons
        self.raw = None
        # RAW TRACES - ROI x TIFF <- SUB-MASK x FRAME
        self.expansion = None # Honestly, no idea
        self.nRegions = None # Number of Sub-Masks (i.e., neuropil masks)
        self.means = None # Honestly, no idea

        if _preparation is not None:
            self.roi_polys = _preparation['roi_polys']
            self.raw = _preparation['raw']
            self.expansion = _preparation['expansion']
            self.nRegions = _preparation['nRegions']
            self.means = _preparation['means']


# /// /// Container for Separation Data /// ///
class SeparationModule:
    """
    Separation Module
    """
    def __init__(self, **kwargs):
        _experiment = kwargs.get('experiment', None)

        self.alpha = None  # alpha parameter
        self.max_iter = None  # maximum iterations limit
        self.method = None  # method used for separation
        self.tol = None  # tolerance for separation
        self.sep = None  # separation traces
        self.info = None  # Honestly, no idea
        self.expansion = None  # Honestly, no idea
        self.max_tries = None  # Honestly, no idea
        self.mixmat = None  # Honestly, no idea
        self.result = None
        # SOURCE-SEPARATED TRACES - ROI x TIFF <- SUB-MASK x FRAME
        self.nRegions = None  # Number of Sub-Masks

        if _experiment is not None:
            self.alpha = _experiment['alpha'] # alpha parameter
            self.max_iter = _experiment['max_iter'] # maximum iterations limit
            self.method = _experiment['method'] # method used for separation
            self.tol = _experiment['tol'] # tolerance for separation
            self.sep = _experiment['sep'] # separation traces
            self.info = _experiment['info'] # Honestly, no idea
            self.expansion = _experiment['expansion'] # Honestly, no idea
            self.max_tries = _experiment['max_tries'] # Honestly, no idea
            self.mixmat = _experiment['mixmat'] # Honestly, no idea
            self.result = _experiment['result']
            # SOURCE-SEPARATED TRACES - ROI x TIFF <- SUB-MASK x FRAME
            self.nRegions = _experiment['nRegions'] # Number of Sub-Masks


# /// /// Container for Processed Traces /// ///
class ProcessedTracesModule:
    def __init__(self):
        """
        Simply a container for Processed Traces
        """
        # /// Pre-Allocation Raw Traces///
        self.original_raw = None # Original Raw Traces
        # ROI x TIFF < - SUB - MASK x FRAME
        self.smoothed_raw = None # Smoothed Raw Traces
        # ROI x TIFF < - SUB - MASK x FRAME
        self.merged_smoothed_raw = None # Smoothed Raw Traces
        # ROI x FRAME
        self.dFoF_raw = None # Fo/F of Raw Traces
        # ROI x TIFF < - SUB - MASK x FRAME
        self.merged_dFoF_raw = None # Fo/F of Raw Traces
        # ROI (Primary Mask Only) x FRAME
        self.neuronal_dFoF_raw = None # Fo/F of Raw Traces
        # Neuronal ROI (Primary Mask Only)

        # /// Pre-Allocation Result Traces
        self.original_result = None # Original Result Traces
        # ROI x TIFF < - SUB - MASK x FRAME
        self.smoothed_result = None  # Smoothed Result Traces
        # ROI x TIFF < - SUB - MASK x FRAME
        self.merged_smoothed_result = None # Smoothed Result Traces
        # ROI x FRAME
        self.dFoF_result = None # Fo/F of Result Traces
        # ROI x TIFF < - SUB - MASK x FRAME
        self.merged_dFoF_result = None # Fo/F of Result Traces
        # ROI (Primary Mask Only) x FRAME
        self.neuronal_dFoF_result = None # Fo/F of Result Traces
        # Neuronal ROI (Primary Mask Only)
        self.detrended_merged_dFoF_result = None # detrended
        # ROI x FRAME
        self.detrended_dFoF_result = None # detrended
        # ROI x TIFF < - SUB - MASK x FRAME
