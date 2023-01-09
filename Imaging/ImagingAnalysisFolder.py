from __future__ import annotations
from typing import Union, Tuple, Optional
import pickle as pkl
import numpy as np
import pathlib
from Imaging.IO import save_raw_binary, determine_bruker_folder_contents
from MigrationTools.Converters import renamed_load
from Management.Organization import Data


class ImagingAnalysis(Data):
    """
    :class:`Data Folder <Management.Organization.Data>` specifically for imaging analysis folders.

    **Required Inputs**
        | *Path* : absolute filepath for data folder
    **Self Methods**
        | *load_fissa_exports* : loads fissa exported files
        | *load_cascade_exports* : loads cascade exported files
        | *load_suite2p* : loads suite2p exported files
        | *export_registration_to_denoised* : moves registration to new folder for namespace compatibility when skipping denoising step
        | *clean_up_motion_correction* : This function removes the reg_tif folder and registered.bin generated during motion correction.
        | *clean_up_compilation* : This function removes the compiled tif files
        | *add_notes* : Function adds notes
    **Properties**
        | *files* : List of files in folder
        | *folders* : List of sub-folders in folder
        | *instance_data* : Data created
        | *path* : path to folder
    """

    def __init__(self, Path: str):
        super().__init__(Path)
        self.parameters = dict()
        # self.default_folders()

    @property
    def current_experiment_step(self) -> str:
        """
        Current stage in imaging analysis

            | 1. Compilation
            | 2. Pre-Processing
            | 3. Motion Correction `Suite2P <https://github.com/MouseLand/suite2p>`_
            | 4. Denoising (Optional) `DeepCAD <https://github.com/cabooster/DeepCAD>`_
            | 5. ROI Detection `CellPose <https://github.com/MouseLand/cellpose>`_
            | 6. Float-32 Trace Extraction `Suite2P <https://github.com/MouseLand/suite2p>`_
            | 7. ROI Classification `CellPose <https://github.com/MouseLand/cellpose>`_
            | 8. Spike Inference [Formality] `Suite2P <https://github.com/MouseLand/suite2p>`_
            | 9. Float-64 Trace Extraction `Fissa <https://github.com/rochefort-lab/fissa>`_
            | 10. Post-Processing
            | 11. Source-Separation `Fissa <https://github.com/rochefort-lab/fissa>`_
            | 12. Infer Spike Probability `Cascade <https://github.com/HelmchenLabSoftware/Cascade>`_
            | 13. Discrete Event Inference `Cascade <https://github.com/HelmchenLabSoftware/Cascade>`_
            | 14. Ready for Analysis

        :rtype: str
        """

        if self.find_matching_files("cascade").__len__() >= 3:
            return "Ready for Analysis"
        elif 1 < self.find_matching_files("cascade").__len__() < 3:
            return "Cascade: Discrete Inference"
        elif self.find_matching_files("fissa").__len__() >= 3:
            return "Cascade: Spike Probability"
        elif 2 <= self.find_matching_files("fissa").__len__() < 3:
            return "Fissa: Source-Separation"
        elif self.find_matching_files("fissa").__len__() == 1:
            return "Post-Processing"
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
        elif self.find_matching_files("meta", "compiled").__len__() > 0:
            return "Suite2P: Motion Correction"
        elif self.find_matching_files("compiled", "compiled").__len__() >= 1:
            return "Pre-Processing"
        else:
            return "Compilation"

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

    def export_registration_to_denoised(self):
        """
        moves registration to new folder for namespace compatibility

        :return:
        """
        _images = np.reshape(np.fromfile(self.find_matching_files("registered_data.bin", "plane0")[0], dtype=np.int16), (-1, 512, 512))
        save_raw_binary(_images, self.folders.get("denoised"))

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
                with open(Filename, "wb") as _file:
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
            print("Could Not Locate Fissa Prepared Filename")
            Prepared = dict()

        try:
            Separated = np.load(self.find_matching_files("separated")[0], allow_pickle=True)
        except FileNotFoundError:
            print("Could Not Locate Fissa Separated Filename")
            Separated = dict()

        # noinspection PyBroadException
        try:
            ProcessedTraces = load_processed_traces(self.find_matching_files("ProcessedTraces")[0])
        except Exception:
            print("Could not locate processed traces file")
            ProcessedTraces = dict()

        if isinstance(ProcessedTraces, dict):
            return {**Prepared}, {**Separated}, {**ProcessedTraces}
        else:
            return {**Prepared}, {**Separated}, {**ProcessedTraces.__dict__}

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

        if isinstance(ProcessedInferences, dict):
            return SpikeTimes, SpikeProb, DiscreteApproximation, ProcessedInferences
        else:
            return SpikeTimes, SpikeProb, DiscreteApproximation, {**ProcessedInferences.__dict__}

    def load_suite2p(self, *args: str):

        if args:
            _folder = args[0]
        else:
            _folder = "denoised"


        # Dynamic imports because \m/_(>.<)_\m/
        print("Loading Suite2p...")
        from Imaging.ToolWrappers.Suite2PModule import Suite2PAnalysis
        suite2p_module = Suite2PAnalysis(self.folders.get(_folder), self.path, file_type="binary")
        suite2p_module.load_files() # load the files
        suite2p_module.db = suite2p_module.ops # make sure db never overwrites ops
        print("Finished.")
        return suite2p_module
