from typing import Tuple, Optional
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm
from ImagingAnalysis.PreprocessingImages import PreProcessing
import sklearn
import imageio as iio


Video = PreProcessing.loadRawBinary("", "", "D:\\EM0122\\Encoding\\Imaging\\30Hz\\denoised")[0:1000]
Stat = np.load("D:\\EM0122\\Encoding\\Imaging\\30Hz\\suite2p\\plane0\\stat.npy", allow_pickle=True)
Images = Video
del Video
Stats = Stat
del Stat


# noinspection PyShadowingNames
def colorize(Images: np.ndarray, Stats: np.ndarray, *args: Optional[Tuple[np.ndarray, str]]) -> np.ndarray:
    """
    This function maps sets or subsets of ROIs onto unique color maps and optionally exports as a video

    :param Images:
    :type Images: Any
    :param Stats:
    :type Stats: Any
    :return: Colorized Image
    :rtype: Any
    """

    _num_frames, _y_pixels, _x_pixels = Images.shape
    OriginalImage = Images.copy()
    #_mean = np.mean(OriginalImage)
    if len(args) >= 1:
        _cell_index = args[0]
    else:
        _cell_index = Stats.shape[0]

    if len(args) == 2:
        _filename = args[1]
    else:
        _filename = None

    for _roi in tqdm(
        range(_cell_index.__len__()),
        total=_cell_index.__len__(),
        desc="Colorizing",
        disable=False,
    ):
        _y = Stats[_cell_index[_roi]].get("ypix")
        _x = Stats[_cell_index[_roi]].get("xpix")
        Images[:, _y, _x] = 0

    # Images = Images.astype(np.uint8)
    ColorImage = np.full((_num_frames, _y_pixels, _x_pixels, 3), 0, dtype=np.float64)
    for _dim in range(3):
        ColorImage[:, :, :, _dim] = Images[:, :, :]
    for _dim in range(3):
        ColorImage[:, :, :, _dim] = rescale(ColorImage[:, :, :, _dim], (0, 65536), (0, 1))

    BlankC = np.zeros_like(ColorImage)
    CM = plt.cm.get_cmap("Blues")
    for _roi in tqdm(
        range(_cell_index.__len__()),
        total=_cell_index.__len__(),
        desc="Colorizing",
        disable=False,
    ):
        _y = Stats[_cell_index[_roi]].get("ypix")
        _x = Stats[_cell_index[_roi]].get("xpix")
        # _min = np.mean(OriginalImage[:, _y, _x]).min()
        _mean = np.mean(OriginalImage[:, _y, _x])
        BlankO = np.full(OriginalImage.shape, int(_mean), dtype=np.float64)
        BlankO[:, _y, _x] = OriginalImage[:, _y, _x]
        NilIdx = np.where(BlankO == int(_mean))

        _temp = CM(BlankO)
        _temp[NilIdx[0], NilIdx[1], NilIdx[2], :] = 0
        BlankC[:, _y, _x, :] = _temp[:, _y, _x, 0:3]

    ColorImage += BlankC
    if _filename is not None:
        print("Writing")
        iio.mimwrite(_filename, ColorImage, fps=30, quality=10, macro_block_size=4)
        print("DONE")
    return ColorImage


def rescale(vector: np.ndarray, current_range: Tuple[float, float], desired_range: Tuple[float, float]) -> np.ndarray:
    return desired_range[0] + ((vector-current_range[0])*(desired_range[1]-desired_range[0]))/(current_range[1]-current_range[0])


ColorImage = colorize(Images, Stats, [10, 267, 315], "C:\\Users\\YUSTE\\Desktop\\Test.mp4")

Fig = plt.figure(figsize=(8, 8))
Ax = Fig.add_subplot(111)
Ax.imshow(ColorImage[0, :, :, :])
