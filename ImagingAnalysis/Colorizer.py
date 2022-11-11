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


def colorize_video(Images: np.ndarray, Stats: np.ndarray, ROIs: Optional[List[int]] = None,
                   Cutoffs: Optional[Tuple[float, float, float, float]] = None, **kwargs) -> np.ndarray:
    """
    This function generates a video (i.e., numpy array [Z x Y x X])
    in which the ROIs or subsets of ROIs utilize a different colormap

     **Keyword Arguments**
    -----------------
     *cmap* : str
        Colormap to use on ROIs
            Default -> None
    *colors* : Tuple of Tupled Floats
        colors which will be used to generate custom colormap
            Default -> None
                Example -> ((0, 0, 0), (0.074, 0.624, 1.000), (0.074, 0.624, 1.000))
    *background* : bool
        boolean indicating whether to overlay on a blank image or the background of input image
            Default -> True
    *white_background* : bool
        boolean indicating whether to use a white or black background
            Default -> False
    *write* : bool
        boolean indicating whether to write video to file
            Default -> False
    *filename* : str
        file path for saving video
            Default -> None

    :param Images: The images to be colorized
    :type Images: Any
    :param Stats: Suite2P Stats file
    :type Stats: Any
    :param ROIs: A List of ROIs
    :type ROIs: list[int]
    :param Cutoffs: Percentile cutoffs for rescaling data. Data below or above these cutoffs
    will be replaced by the smallest or largest value in the data type
    :type Cutoffs: tuple[float]
    :keyword: cmap Colormap to use on ROIs (str, default None)
    :keyword colors: colors which will be used to generate custom colormap (default None, overrides cmap is not None)
    (Tuple of Tuples of Floats, RGB, ranged 0.0-1.0)(default None)
    :keyword background: boolean indicating whether to overlay on a blank image or the background of input image
    (default True)
    :keyword white_background: Boolean indicating whether to use a white or black background (default False,
    requires background = False)
    :keyword write: boolean indicating whether to write video to file (default False)
    :keyword filename: str file path for saving video(default None, which saves to current directory)
    :return: Colorized Video
    :rtype: Any
    """

    _cmap = kwargs.get("cmap", None)
    _colors = kwargs.get("colors", None)
    _include_background = kwargs.get("background", True)
    _white_background = kwargs.get("white_background", False)
    _write = kwargs.get("write", False)
    _filename = kwargs.get("filename", None)

    if ROIs is None:
        ROIs = np.array(range(Stats.shape[0]))

    if Cutoffs is None:
        Cutoffs = tuple([1.0, 99.75, 1.0, 99.75])
    elif len(Cutoffs) == 2:
        Cutoffs = tuple([*Cutoffs, *Cutoffs])

    if _colors is not None and _cmap is None:
        _cmap = generate_custom_map(_colors)
    elif _colors is None and _cmap is not None:
        try:
            _cmap = plt.cm.get_cmap(_cmap)
        except ValueError:
            print("".join(["Unable to locate ", _cmap, " colormap. Reverting to jet."]))
            _cmap = plt.cm.get_cmap("jet")
    elif _colors is not None and _cmap is not None:
        print("".join(["Cmap specified as ", _cmap, " but colors additional specified. Reverting to jet."]))
        _cmap = plt.cm.get_cmap("jet")
    else:
        _cmap = plt.cm.get_cmap("jet")

    _color_image = rescale_images(Images, Cutoffs[2], Cutoffs[3])
    ColorizedVideo = colorize_rois(_color_image, Stats, ROIs, _cmap)

    _background_image = rescale_images(Images, Cutoffs[0], Cutoffs[1])
    _background_image = convert_grayscale_to_color(_background_image)

    # dumb but works
    if not _include_background:
        _background_image[:, :, :, :] = 0
        if _white_background:
            _background_image[:, :, :, :] = 255

    # not include alpha right now... debug
    ColorizedVideo = overlay_colorized_rois(_background_image.copy(), ColorizedVideo)
    ColorizedVideo = merge_background(_background_image, ColorizedVideo, generate_pixel_pairs(Stats, ROIs))

    if _write and _filename is not None:
        write_video(ColorizedVideo, _filename)
    elif _write and _filename is None:
        write_video(ColorizedVideo, "".join([os.getcwd(), "\\colorized_video.mp4"]))

    return ColorizedVideo


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
        ROIs = np.array(range(Stats.shape[0]))

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


def colorize_complete_image(Images: np.ndarray, cmap: Union[plt.cm.colors.Colormap, str]) -> np.ndarray:
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


def merge_background(Background: np.ndarray, NewVideo: np.ndarray, PixelPairs: Tuple[Tuple[int, int]]) -> np.ndarray:
    """
    Merges background video and new video at each specified pixel pair

    :param Background: Background video
    :type Background: Any
    :param NewVideo: Video to merge with
    :type NewVideo: Any
    :param PixelPairs: Pairs of pixels at which merging will occur
    :type PixelPairs: tuple[tuple[int,int]]
    :return: Merged Image
    :rtype: Any
    """
    for _pair in PixelPairs:
        Background[:, _pair[0], _pair[1], :] = NewVideo[:, _pair[0], _pair[1], :]
    return Background


def generate_pixel_pairs(Stats: np.ndarray, ROIs: List[int]) -> Tuple[Tuple[int, int]]:
    """
    Generates a tuple containing a list of each pixel pair from every ROI

    :param Stats: Suite2P Stats
    :type Stats: Any
    :param ROIs: List of ROIs
    :type ROIs: list[int]
    :return: List of each pixel for every ROI
    :rtype: tuple[tuple[int, int]]
    """
    def flatten(list_of_zips):
        return tuple([item for zips in list_of_zips for item in zips])

    PixelPairs = []
    for _roi in ROIs:
        PixelPairs.append(zip(list(Stats[_roi].get("ypix")[Stats[_roi].get("soma_crop")]),
                     list(Stats[_roi].get("xpix")[Stats[_roi].get("soma_crop")])))
    PixelPairs = flatten(PixelPairs)
    # noinspection PyTypeChecker
    return PixelPairs


def generate_custom_map(Colors: List[str]) -> plt.cm.colors.Colormap:
    """
    Generates a custom linearized colormap

    :param Colors: List of colors included
    :type Colors: list[str]
    :return: Colormap
    :rtype: Any
    """
    # noinspection PyBroadException
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


    def rescale(vector: np.ndarray, current_range: Tuple[float, float],
                desired_range: Tuple[float, float]) -> np.ndarray:
        return desired_range[0] + ((vector - current_range[0]) * (desired_range[1] - desired_range[0])) / (
                    current_range[1] - current_range[0])

    assert(0.0 <= LowCut < HighCut <= 100.0)

    _num_frames, _y_pixels, _x_pixels = Images.shape
    _linearized_image = Images.flatten()
    _linearized_image = rescale(_linearized_image, (np.percentile(_linearized_image, LowCut),
                                                    np.percentile(_linearized_image, HighCut)), (0, 255))
    _linearized_image = np.reshape(_linearized_image, (_num_frames, _y_pixels, _x_pixels))
    _linearized_image[_linearized_image <= 0] = 0
    _linearized_image[_linearized_image >= 255] = 255

    return _linearized_image


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
