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
    def __init__(self, Directory):
        # Protected
        self.__instance_date = ExperimentData.getDate()

        _files = Suite2PModule.make_list_tiffs(Directory)[0]
        self.db = {
            'data_path': Directory,
            'save_path0': Directory,
            'tiff_list': _files
        }
        _ops = {
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
        self.ops = {**suite2p.default_ops(), **_ops}
        self.iscell = None
        self.stat = None


    @property
    def instance_date(self):
        return self._Suite2PModule__instance_date

    @property
    def cell_index_path(self):
        return self.ops.get('save_path') + "\\" + "iscell.npy"

    @property
    def stat_file_path(self):
        return self.ops.get('save_path') + "\\" + "stat.npy"

    @property
    def reg_tiff_path(self):
        return self.ops.get('save_path') + "\\" + "reg_tif"

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

    def motionCorrect(self):
        self.ops['roidetect'] = False
        self.ops['spikedetect'] = False
        self.ops['neuropil_extract'] = False
        self.ops.update(suite2p.run_s2p(self.ops, self.db))

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
    def convertToBinary(TiffStack, OutputDirectory):
        """
        Converts a TiffStack to Binary

        :param TiffStack: Numpy Array
        :param OutputDirectory: Where to save binary
        :type OutputDirectory: str
        :return: TiffStack -- Binary Array
        """
        print("Writing tiffstack to binary...")
        TiffStack.tofile(OutputDirectory)
        print("Finished.")

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




