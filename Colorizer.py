import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm
from ImagingAnalysis.PreprocessingImages import PreProcessing
import sklearn

Video = PreProcessing.loadRawBinary("", "", "D:\\EM0122\\Encoding\\Imaging\\30Hz\\denoised")[0:1000]
Stat = np.load("D:\\EM0122\\Encoding\\Imaging\\30Hz\\suite2p\\plane0\\stat.npy", allow_pickle=True)


Images = Video
del Video
Stats = Stat
del Stat


# noinspection PyShadowingNames
def colorize(Images, Stats, *args):
    _num_frames, _y_pixels, _x_pixels = Images.shape
    Images = Images.astype(np.uint8)
    ColorImage = np.full((_num_frames, _y_pixels, _x_pixels, 3), 0, dtype=np.float64)
    for _dim in range(3):
        ColorImage[:, :, :, _dim] = Images[:, :, :]
    for _dim in range(3):
        ColorImage[:, :, :, _dim] = rescale(ColorImage[:, :, :, _dim], (0, 255), (0, 1))

    # _cm = plt.get_cmap("Blues")
    if len(args) >= 1:
        _cell_index = args[0]
    else:
        _cell_index = Stats.shape[0]

    for _roi in tqdm(
        range(_cell_index.__len__()),
        total=_cell_index.__len__(),
        desc="Colorizing",
        disable=False,
    ):
        _temp_color_mat = np.zeros_like(Images)
        _y = Stats[_cell_index[_roi]].get("ypix")
        _x = Stats[_cell_index[_roi]].get("xpix")
        _temp_color_mat[:, _y, _x] = Images[:, _y, _x]
        _temp_color_mat[:, :, :] = rescale(_temp_color_mat[:, :, :], (0, 1), (0, 255))

    return ColorImage


def rescale(vector, current_range, desired_range):
    return desired_range[0] + ((vector-current_range[0])*(desired_range[1]-desired_range[0]))/(current_range[1]-current_range[0])


ColorImage = colorize(Images, Stats, [0, 10, 267])

f = plt.figure(figsize=(8, 8))
a = f.add_subplot(111)
a.imshow(ColorImage[0, :, :, :])
