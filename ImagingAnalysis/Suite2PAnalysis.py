import numpy as np
import suite2p
from suite2p import gui
from natsort import natsorted
import os
import glob
from ExperimentManagement.ExperimentHierarchy import ExperimentData


class Suite2PModule:
    """
    Helper Module for Suite2P Analysis
    """
    def __init__(self, File_Directory, Output_Directory, **kwargs):

        _meta_file = kwargs.get("meta_file", None)
        _meta = kwargs.get("meta", None)
        _video_file = kwargs.get("video", None)
        self.file_type = kwargs.get("file_type", "tiff")
        _ops = kwargs.get("ops", None)

        # Initialized
        self.db = {
            'data_path': File_Directory,
            'save_path0': Output_Directory,
            "save_path": "".join([Output_Directory, "\\suite2p\\plane0"])
        }
        os.makedirs(self.db.get("save_path"))
        self.iscell = None
        self.stat = None

        # Protected
        self.__instance_date = ExperimentData.getDate()

        # Check File Types, only support binary of tiff
        if self.file_type == "tiff" or self.file_type == "tif" or self.file_type == ".tiff" or self.file_type == ".tif":
            self.db = {**self.db, **{"tiff_list": Suite2PModule.make_list_tiffs(File_Directory)[0]}}
        elif self.file_type == "binary":
            if _meta is None and _meta_file is None:
                AssertionError("Using binary is impossible with shape and endianness")
            elif _meta is None and _meta_file is not None:
                _meta = self.load_binary_meta(_meta_file)
            _db = {
                "input_format": "binary",
                "raw_file": _video_file,
                "Lx": _meta[2],
                "Ly": _meta[1],
                "nframes": _meta[0],
            }
            self.db = {**self.db, **_db}

        # Default Ops
        self.ops = {**suite2p.default_ops(), **self.my_default_ops()}

        # Passed Ops
        if _ops is not None:
            self.db = {**self.db, **_ops}

        # Final Overwrite
        self.ops = {**self.ops, **self.db}

    @property
    def instance_date(self):
        return self._Suite2PModule__instance_date

    @property
    def cell_index_path(self):
        return "".join([self.ops.get('save_path'), "\\iscell.npy"])

    @property
    def stat_file_path(self):
        return "".join([self.ops.get('save_path'), "\\stat.npy"])

    @property
    def reg_tiff_path(self):
        return "".join([self.ops.get('save_path'), "\\reg_tif"])

    @property
    def reg_binary_path(self):
        return self.ops.get("f_reg")

    def run(self):
        self.ops.update(suite2p.run_s2p(self.ops, self.db))

    def load_files(self):
        self.iscell = np.load(self.cell_index_path, allow_pickle=True)
        self.stat = np.load(self.stat_file_path, allow_pickle=True)

    def save_files(self):
        np.save(self.cell_index_path, self.iscell, allow_pickle=True)
        np.save(self.stat_file_path, self.stat, allow_pickle=True)

    def openGUI(self):
        gui.run(self.stat_file_path)

    def _motionCorrect(self):
        """
        DEPRECATED

        :return:
        """
        self.ops['roidetect'] = False
        self.ops['spikedetect'] = False
        self.ops['neuropil_extract'] = False
        self.ops.update(suite2p.run_s2p(self.ops, self.db))

    def motionCorrect(self):
        """
        Function simply runs suite2p motion correction
        Note that it assumes you have already converted video to binary
        """
        # Ingest ops (parameters)
        self.ops = {**self.ops, **self.db}
        # Set Registered Data Filename
        self.ops["reg_file"] = "".join([self.ops.get("save_path0"), "\\suite2p\\plane0\\registered_data.bin"])
        # Read in raw tif corresponding to our example tif
        f_raw = suite2p.io.BinaryRWFile(Ly=self.ops.get("Ly"), Lx=self.ops.get("Lx"), filename=self.ops.get("raw_file"))
        f_reg = suite2p.io.BinaryRWFile(Ly=self.ops.get("Ly"), Lx=self.ops.get("Lx"), filename="registered_data.bin")

        refImg, rmin, rmax, meanImg, rigid_offsets, \
        nonrigid_offsets, zest, meanImg_chan2, badframes, \
        yrange, xrange = suite2p.registration_wrapper(f_reg, f_raw=f_raw, f_reg_chan2=None,
                                                      f_raw_chan2=None, refImg=None,
                                                      align_by_chan2=False, ops=self.ops)
        self.ops = {**self.ops,
                        **{
                            "refImg": refImg,
                            "rmin": rmin,
                            "rmax": rmax,
                            "meanImg": meanImg,
                            "rigid_offsets": rigid_offsets,
                            "nonrigid_offsets": nonrigid_offsets,
                            "zest": zest,
                            "meanImg_chan2": meanImg_chan2,
                            "badframes": badframes,
                            "yrange": yrange,
                            "xrange": xrange,
                        }
                    }

    # noinspection PyMethodMayBeStatic
    def replaceTraces(self, NewTraces):
        return "Not Yet Implemented"

    @staticmethod
    def remove_small_neurons(cells, stats):
        _num_rois = stats.shape[0]
        diameters = Suite2PModule.find_diameters(stats)

        _rem_idx = np.where(diameters < 10)[0]
        cells[_rem_idx, 0] = 0
        for _roi in range(_num_rois):
            stats[_roi]['diameter'] = diameters[_roi]
        return cells, stats

    @staticmethod
    def make_list_tiffs(Directory):
        _pathPlusExt = os.path.join(Directory, "*.tif")
        files = []
        files.extend(glob.glob(_pathPlusExt))
        files = natsorted(set(files))
        if len(files) > 0:
            first_tiffs = np.zeros((len(files),), np.bool)
            first_tiffs[0] = True
        else:
            first_tiffs = np.zeros(0, np.bool)

        return files, first_tiffs

    @staticmethod
    def find_diameters(stats):
        _num_rois = stats.shape[0]
        diameters = np.zeros(_num_rois)
        for _roi in range(_num_rois):
            diameters[_roi] = stats[_roi].get('radius') * 2
        return diameters

    @staticmethod
    def load_binary_meta(File):
        """
        Loads meta file for binary video

        :param File: The meta file (.txt ext)
        :type File: str
        :return: A tuple containing the number of frames, x pixels, and y pixels
        :rtype: tuple[int, int, int, str]
        """
        _num_frames, _x_pixels, _y_pixels, _type = np.genfromtxt(File, delimiter=",", dtype="str")
        return tuple([int(_num_frames), int(_x_pixels), int(_y_pixels), _type])

    @staticmethod
    def my_default_ops():
        """
        Returns my default settings

        :return:  ops
        :rtype: dict
        """
        ops = {
                'fs': 30,
                'keep_movie_raw': True,
                'nimg_init': 1000,
                'batch_size': 7000,
                'reg_tif': True,
                'nonrigid': False,
                'denoise': True,
                'anatomical_only': True,
                'diameter': 14,
                'tau': 1.5,
                'spikedetect': True,
                'neuropil_extract': True,
                'roidetect': True,
            }
        return ops

    @staticmethod
    def exportCroppedCorrection():
        return print("Exported Cropped Motion-Corrected Video")
