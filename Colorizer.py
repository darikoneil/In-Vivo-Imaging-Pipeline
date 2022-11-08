import os
from typing import Tuple, Optional, List, Union
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm
from ImagingAnalysis.PreprocessingImages import PreProcessing
import sklearn
import imageio as iio
import cv2


# Video = PreProcessing.loadRawBinary("", "", "D:\\EM0122\\Encoding\\Imaging\\30Hz\\denoised")[0:1000]
# Stat = np.load("D:\\EM0122\\Encoding\\Imaging\\30Hz\\suite2p\\plane0\\stat.npy", allow_pickle=True)
# IsCell = np.load("D:\\EM0122\\Encoding\\Imaging\\30Hz\\suite2p\\plane0\\iscell.npy", allow_pickle=True)
Video = PreProcessing.loadTiff(
     "H:\\DEM_Excitatory_Study\\DEM2\\Retrieval\\Imaging\\30Hz\\Denoised\\DEM2_RET_DN_stx_03.tif", 7000)
Stat = np.load(
    "H:\\DEM_Excitatory_Study\\DEM2\\Retrieval\\Imaging\\30Hz\\Suite2P\\plane0\\stat.npy", allow_pickle=True)
IsCell = np.load(
    "H:\\DEM_Excitatory_Study\\DEM2\\Retrieval\\Imaging\\30Hz\\Suite2P\\plane0\\iscell.npy", allow_pickle=True)
neurons = np.where(IsCell[:, 0] == 1)[0].astype(int)

Images = Video
del Video
Stats = Stat
del Stat


# noinspection PyShadowingNames
def _colorize(Images: np.ndarray, Stats: np.ndarray, *args: Optional[Tuple[np.ndarray, str]]) -> np.ndarray:
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

    #_mean = np.mean(OriginalImage)
    if len(args) >= 1:
        if args[0] is not None:
            _cell_index = args[0]
        else:
            _cell_index = np.arange(0, Stats.shape[0], 1)
    else:
        _cell_index = Stats.shape[0]

    if len(args) == 2:
        _filename = args[1]
    else:
        _filename = None

    ColorImage = np.full((_num_frames, _y_pixels, _x_pixels, 3), 0, dtype=np.float64)
    for _dim in range(3):
        ColorImage[:, :, :, _dim] = Images[:, :, :]
    for _dim in range(3):
        ColorImage[:, :, :, _dim] = rescale(ColorImage[:, :, :, _dim], (0, 65536), (0, 1))

    for _roi in tqdm(
        range(_cell_index.__len__()),
        total=_cell_index.__len__(),
        desc="Colorizing",
        disable=False,
    ):
        _y = Stats[_cell_index[_roi]].get("ypix")
        _x = Stats[_cell_index[_roi]].get("xpix")
        ColorImage[:, _y, _x, :] = 0

    CM = plt.cm.get_cmap("jet")
    ScaledImage = CM(Images)
    BlankC = np.zeros_like(ScaledImage)[:, :, :, 0:3]


    for _roi in tqdm(
        range(_cell_index.__len__()),
        total=_cell_index.__len__(),
        desc="Colorizing",
        disable=False,
    ):
        _y = Stats[_cell_index[_roi]].get("ypix")
        _x = Stats[_cell_index[_roi]].get("xpix")
        BlankC[:, _y, _x, :] = ScaledImage[:, _y, _x, 0:3]

    ColorImage += BlankC
    if _filename is not None:
        print("Writing")
        iio.mimwrite(_filename, ColorImage, fps=30, quality=10, macro_block_size=4)
        print("DONE")
    return ColorImage


def rescale(vector: np.ndarray, current_range: Tuple[float, float], desired_range: Tuple[float, float]) -> np.ndarray:
    return desired_range[0] + ((vector-current_range[0])*(desired_range[1]-desired_range[0]))/(current_range[1]-current_range[0])


def convert_grayscale_to_color(Image: np.ndarray) -> np.ndarray:
    """
    Converts Image to Grayscale

    :param Image: Image to be converted
    :type Image: Any
    :return: Color-Grayscale Image
    :rtype: Any
    """
    ColorGrayScaleImage = np.full((*Image.shape, 3), 0, dtype=Image.dtype)
    for _dim in range(3):
        ColorGrayScaleImage[:, :, :, _dim] = Image

    return np.uint8(normalize_image(ColorGrayScaleImage) * 255)


def normalize_image(Image: np.ndarray) -> np.ndarray:
    """
    Normalizes an image for color-mapping

    :param Image: Image to be normalized
    :type Image: Any
    :return: Normalized Image
    :rtype: Any
    """

    _image = Image.astype(np.float32)
    _image -= _image.min()
    _image /= _image.max()
    return _image


# noinspection PyUnusedLocal,PyShadowingNames
def write_video(Video: np.ndarray, Filename: str) -> None:
    """
    Function writes video to .mp4

    :param Video: Video to be written
    :type Video: Any
    :param Filename: Filename  (Or Complete File Path)
    :type Filename: str
    :rtype: None
    """

    if "\\" not in Filename:
        Filename = "".join([os.getcwd(), "\\", Filename])

    if Video.dtype.type != np.uint8:
        Video = Video.astype(np.uint8)

    print("\nWriting Video...\n")
    iio.mimwrite(Filename, Video, fps=30, quality=10, macro_block_size=4)
    print("Finished.")


def overlay_colorized_rois(Background: np.ndarray, ColorizedVideo: np.ndarray, *args: Optional[float]) -> np.ndarray:
    """
   This function overlays colorized videos onto background video

    :param Background: Background Video in Grayscale
    :type Background: Any
    :param ColorizedVideo: Colorized Overlays In Colormap Space + Alpha Channel
    :type ColorizedVideo: Any
    :param args: Alpha for Background
    :type args: float
    :return: Merged Video
    :rtype: Any
    """

    # noinspection PyShadowingNames
    def overlay_colorized_rois_frame(BackgroundFrame: np.ndarray, ColorizedVideoFrame: np.ndarray, Alpha: float, Beta: float) -> np.ndarray:
        """
        This function merged each frame and is looped through

        :param BackgroundFrame: Single Frame of Background
        :type BackgroundFrame: Any
        :param ColorizedVideoFrame: Single Frame of Color
        :type ColorizedVideoFrame: Any
        :param Alpha: Background Alpha
        :type Alpha: float
        :param Beta: Overlay Alpha
        :type Beta: float
        :return: Single Merged Frame
        :rtype: Any
        """

        return cv2.cvtColor(cv2.addWeighted(cv2.cvtColor(BackgroundFrame, cv2.COLOR_RGB2BGR), Alpha,
                        cv2.cvtColor(ColorizedVideoFrame, cv2.COLOR_RGBA2BGR), Beta, 0.0), cv2.COLOR_BGR2RGB)

    if len(args) >= 1:
        _alpha = 1
        _beta = 1 - _alpha
    else:
        _alpha = 0.5
        _beta = 0.5

    for _frame in tqdm(
        range(Background.shape[0]),
        total=Background.shape[0],
        desc="Overlaying",
        disable=False
    ):
        Background[_frame, :, :] = overlay_colorized_rois_frame(Background[_frame, :, :],
                                                                ColorizedVideo[_frame, :, :, :], _alpha, _beta)

    return Background


def colorize_rois(Images: np.ndarray, Stats: np.ndarray, ROIs: Optional[List[int]] = None, *args: Optional[plt.cm.colors.Colormap]) \
        -> np.ndarray:
    """
    Generates a colorized roi overlay video

    :param Images: Images To Extract ROI Overlay
    :type Images: Any
    :param Stats: Suite2P Stats
    :type Stats: Any
    :param ROIs: Subset of ROIs
    :type ROIs: list[int]|None
    :return: Colorized ROIs
    :rtype: Any
    """
    def flatten(list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]

    if len(args) >= 1:
        cmap = args[0]
    else:
        cmap = "binary"

    if ROIs is None:
        ROIs = np.arange(0, Stats.shape[0]+1, 1)

    ColorImage = colorize(Images, cmap)
    _y = []
    _x = []
    for _roi in ROIs:
        _y.append(Stats[_roi].get("ypix")[Stats[_roi].get("soma_crop")])
        _x.append(Stats[_roi].get("xpix")[Stats[_roi].get("soma_crop")])

    ColorizedROIs = np.zeros_like(ColorImage)
    ColorizedROIs[:, flatten(_y), flatten(_x), :] = \
        ColorImage[:, flatten(_y), flatten(_x), :]
    # ColorizedROIs[ColorizedROIs[:, :, :, 3] == 255] = 190
    return ColorizedROIs


def colorize(Images: np.ndarray, cmap: Union[plt.cm.colors.Colormap, str]) -> np.ndarray:
    """
    Colorizes an Image

    :param Images: Image to be colorized
    :type Images: Any
    :param cmap: Matplotlib colormap [Object or str]
    :type: Any
    :return: Colorized Image
    :rtype: Any
    """
    if isinstance(cmap, str):
        return np.uint8(plt.cm.get_cmap(cmap)(normalize_image(Images))*255)
    else:
        return np.uint8(cmap(normalize_image(Images)) * 255)


# noinspection PyBroadException
def generate_custom_map(Colors: List[str]) -> plt.cm.colors.Colormap:
    """
    Generates a custom linearized colormap

    :param Colors: List of colors included
    :type Colors: list[str]
    :return: Colormap
    :rtype: Any
    """
    try:
        return matplotlib.colors.LinearSegmentedColormap.from_list("", Colors)
    except Exception:
        print("Could not identify colors. Returning jet!")
        return plt.cm.jet


def rescale_images(Images: np.ndarray, LowCut: float, HighCut: float) -> np.ndarray:
    """
    Rescale Images within percentiles

    :param Images: Images to be rescaled
    :type Images: Any
    :param LowCut: Low Percentile Cutoff
    :type LowCut: float
    :param HighCut: High Percentile Cutoff
    :type HighCut: float
    :return: Rescaled Images
    :rtype: Any
    """

    assert(0.0 <= LowCut < HighCut <= 100.0)

    _num_frames, _y_pixels, _x_pixels = Images.shape
    _linearized_image = Images.flatten()
    _linearized_image = rescale(_linearized_image, (np.percentile(_linearized_image, LowCut),
                                                    np.percentile(_linearized_image, HighCut)), (0, 255))
    _linearized_image = np.reshape(_linearized_image, (_num_frames, _y_pixels, _x_pixels))
    _linearized_image[_linearized_image <= 0] = 0
    _linearized_image[_linearized_image >= 255] = 255

    return _linearized_image


def generate_alpha_map(Images: np.ndarray) -> np.ndarray:
    return np.uint8(normalize_image(Images)*255)/255


# noinspection PyRedeclaration
ImagesB = rescale_images(Images, 1.0, 99.75)
Images = rescale_images(Images, 1.0, 99.75)
ImagesB = ImagesB[0:1000, :, :]
Images = Images[0:1000, :, :]
custom_map = generate_custom_map([(0, 0, 0), (0, 0, 0), (0.0745, 0.6235, 1)])
Background = convert_grayscale_to_color(ImagesB)
ColorizedVideo = colorize_rois(Images, Stats, neurons, custom_map)
NewVideo = overlay_colorized_rois(Background, ColorizedVideo)
write_video(NewVideo, "C:\\Users\\YUSTE\\Desktop\\Test10.mp4")

