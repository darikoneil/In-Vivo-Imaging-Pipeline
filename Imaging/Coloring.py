import os
from typing import Tuple, Optional, List, Union
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm
import sklearn
import cv2
from Imaging.IO import save_video


class ColorImages:
    def __init__(self, Images: np.ndarray, Stats: np.ndarray, Cells: np.ndarray):
        self.images = Images
        self.stats = Stats
        self.cells = Cells
        
        # background
        self._background = np.ndarray
        self._background_cutoffs = (0, 100)
        self._background_style = "True"
        self.background_style = "True" # make first background

        # Subsets to Color
        self._neuron_subsets = []
        self._colormaps = []
        self._overlays = []
        self.cutoffs = (0, 100)

        return

    @property
    def total_rois(self):
        return self.cells.shape[0]

    @property
    def neuronal_ids(self):
        return [_neuron for _neuron in range(self.cells.shape[0]) if self.cells[_neuron, 0] == 1]

    @property
    def num_neurons(self):
        return np.sum(self.cells[:, 0])

    @property
    def num_frames(self):
        return self.images.shape[0]

    @property
    def ypix(self):
        return self.images.shape[1]

    @property
    def xpix(self):
        return self.images.shape[2]

    @property
    def background(self):
        return self._background

    @property
    def background_style(self):
        return self._background_style

    @background_style.setter
    def background_style(self, Option):
        self._background_style = Option
        # reset background with new style
        self._background = generate_background(self.images, self._background_style, self._background_cutoffs)

    @property
    def background_cutoffs(self):
        return self._background_cutoffs

    @background_cutoffs.setter
    def background_cutoffs(self, Cutoffs):
        self._background_cutoffs = Cutoffs
        # reset background with new cutoffs
        self._background = generate_background(self.images, self._background_style, self._background_cutoffs)

    @property
    def neuron_subsets(self):
        return self._neuron_subsets

    @neuron_subsets.setter
    def neuron_subsets(self, Inputs):

        def generate_colors():
            nonlocal Inputs

            # use mean as bottom
            _bottom = np.mean(self.background)/255
            # use specific colors
            if isinstance(Inputs[1], tuple):
                _colors = [(_bottom, _bottom, _bottom), (_bottom, _bottom, _bottom), (Inputs[1]), (Inputs[1])]
                _cmap = generate_custom_map(_colors)
            # use specific cmap
            if isinstance(Inputs[1], str):
                try:
                    _cmap = plt.cm.get_cmap(Inputs[1])
                except ValueError:
                    print("".join(["Unable to locate ", Inputs[1], " colormap. Reverting to jet."]))
                    _cmap = plt.cm.get_cmap("jet")
            # when in doubt
            else:
                _cmap = plt.cm.get_cmap("jet")

            return _cmap

        def generate_overlays():
            _scaled_image = rescale_images(self.images, *self.cutoffs)
            _color_rois = colorize_rois(self.images.copy(), self.stats, self.neuron_subsets[-1], self._colormaps[-1])
            _base_overlay = overlay_colorized_rois(self.background.copy(), _color_rois)
            return merge_background(self.background.copy(), _base_overlay, generate_pixel_pairs(self.stats, self.neuron_subsets[-1]))

        self._neuron_subsets.append(Inputs[0])
        self._colormaps.append(generate_colors())
        self._overlays.append(generate_overlays())

    @property
    def overlays(self):
        return self._overlays

    @property
    def color_video(self):
        color_video = self.background.copy()
        for _subset in range(self.neuron_subsets.__len__()):
            color_video = merge_background(color_video, self.overlays[_subset],
                                           generate_pixel_pairs(self.stats, self.neuron_subsets[_subset]))
        return color_video

    def preview_background(self) -> plt.Figure:
        _frames_idx = np.arange(0, self.num_frames, 1)
        _subplot_ids = 230

        def plot():
            nonlocal _frames_idx
            nonlocal Ax

            np.random.shuffle(_frames_idx)
            Ax.imshow(self.background[_frames_idx[0], :, :])
            Ax.set_xticks([])
            Ax.set_yticks([])
            Ax.set_title("".join(["Frame: ", str(_frames_idx[0])]), fontsize=10)

        Fig = plt.figure()
        Fig.suptitle("Background Frame Samples", fontsize=16)
        for _subplot in range(6):
            _subplot_ids += 1
            Ax = Fig.add_subplot(_subplot_ids)
            plot()

        Fig.tight_layout()
        Fig.tight_layout() # Don't know why but do it

        return Fig

    def preview_color(self, idx):
        _frames_idx = np.arange(0, self.num_frames, 1)
        _subplot_ids = 230

        _overlay = self.overlays[idx]

        def plot():
            nonlocal _frames_idx
            nonlocal Ax

            np.random.shuffle(_frames_idx)
            Ax.imshow(_overlay[_frames_idx[0], :, :, :])
            Ax.set_xticks([])
            Ax.set_yticks([])
            Ax.set_title("".join(["Frame: ", str(_frames_idx[0])]), fontsize=10)

        Fig = plt.figure()
        Fig.suptitle("Frame Samples", fontsize=16)
        for _subplot in range(6):
            _subplot_ids += 1
            Ax = Fig.add_subplot(_subplot_ids)
            plot()

        Fig.tight_layout()
        Fig.tight_layout()  # Don't know why but do it

        return Fig



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


def generate_background(Images: np.ndarray, Option: str = "True",
                         Cutoffs: Tuple[float, float] = (0, 100)) -> np.ndarray:
    if Option == "Black":
        _background_image = np.zeros(Images.shape, dtype=np.uint8)
    elif Option == "White":
        _background_image = np.full(Images.shape, 255, dtype=np.uint8)
    elif Option == "True":
        _background_image = rescale_images(Images, Cutoffs[0], Cutoffs[1])
        _background_image = convert_grayscale_to_color(_background_image)
    else:  # For now edge cases -> True Option
        _background_image = rescale_images(Images, Cutoffs[0], Cutoffs[1])
        _background_image = convert_grayscale_to_color(_background_image)

    return _background_image


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

    def points_in_circle(radius, x0=0, y0=0, ):
        x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
        y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
        x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)
        # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
        for x, y in zip(x_[x], y_[y]):
            yield x, y

    if len(args) >= 1:
        cmap = args[0]
    else:
        cmap = "binary"

    if ROIs is None:
        ROIs = np.array(range(Stats.shape[0]))

    ColorImage = colorize_complete_image(Images, cmap)
    # _y = []
    # _x = []
    # for _roi in ROIs:
    #    _y.append(Stats[_roi].get("ypix")[Stats[_roi].get("soma_crop")])
    #    _x.append(Stats[_roi].get("xpix")[Stats[_roi].get("soma_crop")])
    PixelPairs = []
    for _roi in ROIs:
        # PixelPairs.append(zip(list(Stats[_roi].get("ypix")[Stats[_roi].get("soma_crop")]),
        # list(Stats[_roi].get("xpix")[Stats[_roi].get("soma_crop")])))
        # PixelPairs.append(zip(list(Stats[_roi].get("ypix")),
        # list(Stats[_roi].get("xpix"))))
        PixelPairs.append([points for points in points_in_circle(Stats[_roi].get("radius"),
                                                                     *Stats[_roi].get("med"))])
    PixelPairs = flatten(PixelPairs)
    _yx = [list(_tuple) for _tuple in list(zip(*PixelPairs))]
    _y = np.array(_yx[0])
    _y[_y < 0] = 0
    _y[_y >= ColorImage.shape[1]] = ColorImage.shape[1]-1
    _x = np.array(_yx[1])
    _x[_x < 0] = 0
    _x[_x >= ColorImage.shape[2]] = ColorImage.shape[2]-1

    ColorizedROIs = np.zeros_like(ColorImage)
    ColorizedROIs[:, _y, _x, :] = \
        ColorImage[:, _y, _x, :]
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


def overlay_colorized_rois(Background: np.ndarray, ColorizedVideo: np.ndarray, *args: Optional[float]) -> np.ndarray:
    """
   This function overlays colorized videos onto background video

    :param Background: Background Images in Grayscale
    :type Background: Any
    :param ColorizedVideo: Colorized Overlays In Colormap Space + Alpha Channel
    :type ColorizedVideo: Any
    :param args: Alpha for Background
    :type args: float
    :return: Merged Images
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

    def points_in_circle(radius, x0=0, y0=0, ):
        x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
        y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
        x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)
        # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
        for x, y in zip(x_[x], y_[y]):
            yield x, y

    PixelPairs = []
    for _roi in ROIs:
        # PixelPairs.append(zip(list(Stats[_roi].get("ypix")[Stats[_roi].get("soma_crop")]),
        # list(Stats[_roi].get("xpix")[Stats[_roi].get("soma_crop")])))
        # PixelPairs.append(zip(list(Stats[_roi].get("ypix")),
        # list(Stats[_roi].get("xpix"))))
        PixelPairs.append([points for points in points_in_circle(Stats[_roi].get("radius"),
                                                                     *Stats[_roi].get("med"))])
    PixelPairs = flatten(PixelPairs)
    # noinspection PyTypeChecker
    return PixelPairs


def merge_background(Background: np.ndarray, NewVideo: np.ndarray, PixelPairs: Tuple[Tuple[int, int]]) -> np.ndarray:
    """
    Merges background video and new video at each specified pixel pair

    :param Background: Background video
    :type Background: Any
    :param NewVideo: Images to merge with
    :type NewVideo: Any
    :param PixelPairs: Pairs of pixels at which merging will occur
    :type PixelPairs: tuple[tuple[int,int]]
    :return: Merged Image
    :rtype: Any
    """
    _y = Background.shape[1]
    _x = Background.shape[2]

    for _pair in PixelPairs:
        if _pair[0] < 0:
            _yy = 0
        elif _pair[0] >= _y:
            _yy = _y-1
        else:
            _yy = _pair[0]

        if _pair[1] < 0:
            _xx = 0
        elif _pair[1] >= _x:
            _xx = _x-1
        else:
            _xx = _pair[1]

        Background[:, _yy, _xx, :] = NewVideo[:, _yy, _xx, :]
    return Background
