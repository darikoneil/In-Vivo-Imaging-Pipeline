from __future__ import annotations
import numpy as np
import os
from PIL import Image
from tqdm.auto import tqdm
import scipy.ndimage
import math
import cupy
import cupyx.scipy.ndimage
import tifffile
import skimage.measure
import animatplot as amp
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Callable, List, Tuple, Sequence


class PreProcessing:
    """
    PreProcessing
    -------------
    Class containing a variety of static methods for preprocessing images

    **Static Methods**
        | *load_bruker_tiffs* : Load a sequence of tiff files from a directory.
        | *repackage_bruker_tiffs* :  Repackages a sequence of tiff files within a directory to a smaller sequence of tiff stacks.
        | *filter_tiff* : Denoise a tiff stack using a multidimensional median filter
        | *fast_filter_tiff* : GPU-parallelized multidimensional median filter
        | *save_tiff* : Save a numpy array to a single tiff file
        | *load_tiff* : Load a single tiff file
        | *load_all_tiffs* : Load a sequence of tiff stacks
        | *load_raw_binary* : Loads a raw binary file
        | *save_raw_binary* : This function saves a tiff stack as a binary file
        | *removeShutterArtifact* : Function to remove the shuttle artifacts present at the initial imaging frames
        | *blockwise_fast_filter_tiff* : Blockwise, GPU-parallelized multidimensional median filter
        | *save_tiff_stack* : Save a numpy array to a sequence of tiff stacks
        | *grouped_z_project* : Utilize grouped z-project to downsample data
        | *view_image* :  Visualize a numpy array [Z x Y x X] as a video
        | *loadBinaryMeta* : Loads meta file for binary video
        | *loadMappedBinary*: Loads a raw binary file in the workspace without loading into memory
    """

    def __init__(self):
        return

    @staticmethod
    def load_bruker_tiffs(VideoDirectory: str) -> np.ndarray:
        """
        load_bruker_tiffs
        ---------------
        Load a sequence of tiff files from a directory.

        Designed to compile the outputs of a certain imaging utility
        that exports recordings such that each frame is saved as a single tiff.

        :param VideoDirectory: Directory containing a sequence of single frame tiff files
        :type VideoDirectory: str
        :return: complete_image:  All tiff files in the directory compiled into a single array (Z x Y x X, uint16)
        :rtype: Any
        """
        _fnames = os.listdir(VideoDirectory)
        _num_frames = len(_fnames)
        complete_image = np.full((_num_frames, 512, 512), 0, dtype=np.uint16)

        for _fname in tqdm(
                range(_num_frames),
                total=_num_frames,
                desc="Loading Image...",
                disable=False,
        ):
            complete_image[_fname, :, :] = np.asarray(Image.open(VideoDirectory + "\\" + _fnames[_fname]))
        return complete_image

    @staticmethod
    def repackage_bruker_tiffs(VideoDirectory: str, OutputDirectory: str) -> None:
        """

        Repackages a sequence of tiff files within a directory to a smaller sequence
        of tiff stacks. Designed to compile the outputs of a certain imaging utility
        that exports recordings such that each frame is saved as a single tiff.

        :param VideoDirectory: Directory containing a sequence of single frame tiff files
        :type VideoDirectory: str
        :param OutputDirectory: Empty directory where tiff stacks will be saved
        :type OutputDirectory: str
        :rtype: None

        """
        print("Repackaging...")
        _fnames = os.listdir(VideoDirectory)
        _num_frames = len(_fnames)
        # noinspection PyTypeChecker
        _chunks = math.ceil(_num_frames/7000)

        c_idx = 1
        _offset = int()
        for _chunk in range(0, _num_frames, 7000):

            _start_idx = _chunk
            _offset = _start_idx
            _end_idx = _chunk + 7000
            _chunk_frames = _end_idx-_start_idx
            # If this is the last chunk which may not contain a full 7000 frames...
            if _end_idx > _num_frames:
                _end_idx = _num_frames
                _chunk_frames = _end_idx - _start_idx
                _end_idx += 1

            image_chunk = np.full((_chunk_frames, 512, 512), 0, dtype=np.uint16)
            for _fname in tqdm(
                range(_chunk_frames),
                total=_chunk_frames,
                desc="Loading Images...",
                disable=False,
            ):
                image_chunk[_fname, :, :] = np.asarray(Image.open(VideoDirectory + "\\" + _fnames[_fname+_offset]))
            if c_idx < 10:
                PreProcessing.save_tiff(image_chunk, OutputDirectory + "\\" + "compiledVideo_0" + str(c_idx) + "_of_" + str(_chunks) + ".tif")
            else:
                PreProcessing.save_tiff(image_chunk, OutputDirectory + "\\" + "compiledVideo_" + str(c_idx) + "_of_" + str(_chunks) + ".tif")
            c_idx += 1
        return print("Finished Repackaging Bruker Tiffs")

    @staticmethod
    def filter_tiff(Tiff: np.ndarray, **kwargs) -> np.ndarray:
        """
        Denoise a tiff stack using a multidimensional median filter

        This function simply calls scipy.ndimage.median_filter

        Keyword Arguments
        -----------------
        *Footprint* : numpy array [z pixels, y pixels, x pixels]
            Mask indicating the footprint of the median filter
                Default -> 3 x 3 x 3
                    Example -> np.ones((3, 3, 3))

        :param Tiff: Tiff stack to be filtered [Z x Y x X]
        :type Tiff: Any
        :return: filtered_tiff [Z x Y x X]
        :rtype: Any
        :keyword Footprint: Mask of the median filter
        """
        _footprint = kwargs.get('Footprint', np.ones((3, 3, 3)))

        print("Filtering...")
        filtered_tiff = scipy.ndimage.median_filter(Tiff, footprint=_footprint)
        print("Finished")
        return filtered_tiff

    @staticmethod
    def fast_filter_tiff(Tiff: np.ndarray, **kwargs) -> np.ndarray:
        """
        GPU-parallelized multidimensional median filter

        Keyword Arguments
        -----------------
        *Footprint* : numpy array [z pixels, y pixels, x pixels]
            Mask indicating the footprint of the median filter
                Default -> 3 x 3 x 3
                    Example -> np.ones((3, 3, 3))

        :param Tiff: Tiff stack to be filtered [Z x Y x X]
        :type Tiff: Any
        :return: filtered_tiff [Z x Y x X]
        :rtype: Any
        :keyword Footprint: Mask of the median filter
        """
        _footprint = kwargs.get('Footprint', np.ones((3, 3, 3)))
        _converted_tiff = cupy.asarray(Tiff)
        filtered_tiff = cupyx.scipy.ndimage.median_filter(_converted_tiff, footprint=_footprint)
        return filtered_tiff

    @staticmethod
    def save_tiff(Tiff: np.ndarray, fname: str) -> None:
        """
        Save a numpy array to a single tiff file

        :param Tiff: numpy array [frames, y pixels, x pixels]
        :type Tiff: Any
        :param fname: filename
        :type fname: str
        :rtype: None
        """
        with tifffile.TiffWriter(fname) as tif:
            for frame in np.floor(Tiff).astype(np.int16):
                tif.save(frame)

    @staticmethod
    def load_tiff(fname: str, num_frames: int) -> np.ndarray:
        """
        Load a single tiff file

        :param fname: filename
        :param num_frames: number of frames
        :type fname: str
        :type num_frames: int
        :return: numpy array [Z x Y x X]
        :rtype: Any
        """
        return tifffile.imread(fname, key=range(0, num_frames, 1))

    @staticmethod
    def load_all_tiffs(VideoDirectory: str) -> np.ndarray:
        """
        Load a sequence of tiff stacks

        :param VideoDirectory: Directory containing a sequence of tiff stacks
        :type VideoDirectory: str
        :return: complete_image numpy array [Z x Y x X]
        :rtype: Any
        """
        _fnames = os.listdir(VideoDirectory)
        x_pix, y_pix = tifffile.TiffFile(VideoDirectory + "\\" + _fnames[0]).pages[0].shape
        _num_frames = [] # initialize
        [_num_frames.append(len(tifffile.TiffFile(VideoDirectory + "\\" + _fname).pages)) for _fname in _fnames]
        _total_frames = sum(_num_frames)
        complete_image = np.full((_total_frames, x_pix, y_pix), 0, dtype=np.int16) # not sure if correct x,y order
        _last_frame = 0

        for _fname in tqdm(
                range(len(_fnames)),
                total=len(_fnames),
                desc="Loading Images...",
                disable=False,
        ):
            complete_image[_last_frame:_last_frame+_num_frames[_fname], :, :] = \
                PreProcessing.load_tiff(VideoDirectory + "\\" + _fnames[_fname], _num_frames[_fname])
            _last_frame += _num_frames[_fname]

        return complete_image

    @staticmethod
    def load_raw_binary(fname: str, meta_file: str, *args) -> np.ndarray:
        """
        Loads a raw binary file

        Enter the path to autofill (assumes fname & meta are Path + binary_video, video_meta.txt)

        :param fname: filename for binary video
        :type fname: str
        :param meta_file: filename for meta file
        :type meta_file: str
        :param args: Path
        :type args: str
        :return: numpy array [Z x Y x X]
        :rtype: Any
        """
        if len(args) == 1:
            fname = "".join([args[0], "\\binary_video"])
            meta_file = "".join([args[0], "\\video_meta.txt"])

        _num_frames, _y_pixels, _x_pixels, _type = np.genfromtxt(meta_file, delimiter=",", dtype="str")
        _num_frames = int(_num_frames)
        _x_pixels = int(_x_pixels)
        _y_pixels = int(_y_pixels)
        return np.reshape(np.fromfile(fname, dtype=_type), (_num_frames, _y_pixels, _x_pixels))

    @staticmethod
    def save_raw_binary(Video: np.ndarray, VideoDirectory: str) -> None:
        """
        This function saves a tiff stack as a binary file

        :param Video: Video to be saved [Z x Y x X]
        :type Video: np.ndarray
        :param VideoDirectory: Directory to save video in
        :type VideoDirectory: str
        :rtype: None
        """
        print("Saving video as a binary...")
        _meta_file = "".join([VideoDirectory, "\\video_meta.txt"])
        _video_file = "".join([VideoDirectory, "\\binary_video"])

        try:
            with open(_meta_file, 'w') as f:
                f.writelines([str(Video.shape[0]), ",", str(Video.shape[1]), ",",
                            str(Video.shape[2]), ",", str(Video.dtype)])
        except FileNotFoundError:
            _meta_path = _meta_file.replace("\\video_meta.txt", "")
            os.makedirs(_meta_path)
            with open(_meta_file, 'w') as f:
                f.writelines([str(Video.shape[0]), ",", str(Video.shape[1]), ",",
                            str(Video.shape[2]), ",", str(Video.dtype)])

        Video.tofile(_video_file)
        print("Finished saving video as a binary.")

    @staticmethod
    def remove_shuttle_artifact(Video: np.ndarray, **kwargs) -> np.ndarray:
        """
        Function to remove the shuttle artifacts present at the initial imaging frames

        :param Video: Video array with shape Z x Y x X
        :type Video: Any
        :param kwargs:
        :keyword artifact_length: number of frames considered artifact (int)
        :keyword chunk_size: number of frames per chunk_size (makes divisible by value) (int)
        :return: Video
        :rtype: Any
        """
        if Video.shape[0] <= Video.shape[1] or Video.shape[0] <= Video.shape[2]:
            AssertionError("Video must be in the form Z x X x Y")

        _shuttle_artifact_length = kwargs.get("artifact_length", 1000)
        _chunk_size = kwargs.get("chunk_size", 7000)
        _num_frames = Video.shape[0]
        _crop_idx = _num_frames % _chunk_size
        if _crop_idx >= _shuttle_artifact_length:
            return Video[_crop_idx:, :, :]
        else:
            _num_frames -= _shuttle_artifact_length
            _crop_idx = _num_frames % _chunk_size
            return Video[_shuttle_artifact_length+_crop_idx:, :, :]

    @staticmethod
    def blockwise_fast_filter_tiff(TiffStack: np.ndarray, **kwargs) -> np.ndarray:
        """
        GPU-parallelized multidimensional median filter performed in overlapping blocks.

        Designed for use on arrays larger than the available memory capacity.

        Keyword Arguments
        -----------------
        *Footprint* : numpy array [z pixels, y pixels, x pixels]
            Mask indicating the footprint of the median filter
                Default -> 3 x 3 x 3
                    Example -> np.ones((3, 3, 3))
        *BlockSize* : int
            Integer indicating the size of each block. Must fit within memory.
                Default -> 21000
        *BlockBufferRegion* : int
            Integer indicating the size of the overlapping region between blocks
                Default -> 500

        :param TiffStack: Tiff stack to be filtered
        :type TiffStack: Any
        :keyword Footprint:  Mask indicating the footprint of the median filter (np.ndarray)
        :keyword BlockSize:   Integer indicating the size of each block. Must fit within memory. (int)
        :keyword BlockBufferRegion: Integer indicating the size of the overlapping region between blocks (int)
        :return: TiffStack: numpy array [Z x Y x X]
        :rtype: Any
        """
        _block_size = kwargs.get('BlockSize', int(21000))
        _block_buffer_region = kwargs.get('BlockBufferRegion', int(500))
        _footprint = kwargs.get('Footprint', np.ones((3, 3, 3)))
        _total_frames = TiffStack.shape[0]
        _blocks = range(0, _total_frames, _block_size)
        _num_blocks = len(_blocks)
        _remainder = np.full((_block_buffer_region, TiffStack.shape[1], TiffStack.shape[2]), 0, dtype=np.int16)

        for _block in tqdm(
                range(_num_blocks),
                total=_num_blocks,
                desc="Filtering Images...",
                disable=False,
        ):
            if _block == 0:
                _remainder = TiffStack[_blocks[_block + 1] - 500:_blocks[_block + 1], :, :].copy()
                TiffStack[0:_blocks[_block + 1], :, :] = cupy.asnumpy(PreProcessing.fast_filter_tiff(cupy.asarray(
                    TiffStack[0:_blocks[_block + 1], :, :]), Footprint=_footprint))
            elif _block == _num_blocks - 1:

                TiffStack[_blocks[_block]:_total_frames, :, :] = \
                    cupy.asnumpy(PreProcessing.fast_filter_tiff(
                        cupy.asarray(np.append(_remainder, TiffStack[_blocks[_block]:_total_frames, :, :],
                                               axis=0)), Footprint=_footprint))[_block_buffer_region:, :, :]

                # TiffStack[_blocks[_block]:_total_frames, :, :] = PreProcessing.fast_filter_tiff(np.append(_remainder, TiffStack[_blocks[_block]:_total_frames, :, :],
                # axis=0))[_block_buffer_region:, :, :])
            else:
                _remainder_new = TiffStack[_blocks[_block + 1] - 500:_blocks[_block + 1], :, :].copy()
                TiffStack[_blocks[_block]:_blocks[_block + 1], :, :] = \
                    cupy.asnumpy(PreProcessing.fast_filter_tiff(
                        cupy.asarray(np.append(_remainder, TiffStack[_blocks[_block]:_blocks[_block + 1], :, :],
                                               axis=0)), Footprint=_footprint))[_block_buffer_region:_block_size+_block_buffer_region, :, :]
                _remainder = _remainder_new.copy()

        return TiffStack

    @staticmethod
    def save_tiff_stack(TiffStack: str, OutputDirectory: str) -> None:
        """
        Save a numpy array to a sequence of tiff stacks

        :param TiffStack: A numpy array containing a tiff stack [Z x Y x X]
        :type TiffStack: Any
        :param OutputDirectory: A directory to save the sequence of tiff stacks in
        :type OutputDirectory: str
        :rtype: None
        """
        _num_frames = TiffStack.shape[0]

        _chunks = math.ceil(_num_frames / 7000)

        c_idx = 1
        for _chunk in range(0, _num_frames, 7000):

            _start_idx = _chunk
            _end_idx = _chunk + 7000
            if _end_idx > _num_frames:
                _end_idx = _num_frames + 1

            if c_idx < 10:
                PreProcessing.save_tiff(TiffStack[_start_idx:_end_idx, :, :],
                                        OutputDirectory + "\\" + "Video_0" + str(c_idx) + "_of_" + str(
                                           _chunks) + ".tif")
            else:
                PreProcessing.save_tiff(TiffStack[_start_idx:_end_idx, :, :],
                                        OutputDirectory + "\\" + "Video_" + str(c_idx) + "_of_" + str(
                                           _chunks) + ".tif")
            c_idx += 1

        return print("Finished Saving Tiffs")

    @staticmethod
    def grouped_z_project(TiffStack: np.ndarray, BinSize: int, DownsampleFunction: Callable[[np.ndarray], np.ndarray]) \
            -> np.ndarray:
        """
        Utilize grouped z-project to downsample data

        :param TiffStack: A numpy array containing a tiff stack [Z x Y x X]
        :type TiffStack: Any
        :param BinSize:  Size of each bin passed to downsampling function
        :type BinSize: int
        :param DownsampleFunction: Downsampling function
        :type DownsampleFunction: Any
        :return: downsampled_image [Z x Y x X]
        :rtype: Any
        """
        print("Downsampling data...")
        downsampled_image = skimage.measure.block_reduce(TiffStack, block_size=BinSize,
                                                            func=DownsampleFunction)
        print("Finished.")
        return downsampled_image

    @staticmethod
    def view_image(Video: np.ndarray, fps: float, **kwargs) -> \
            List[object]:
        """
        Visualize a numpy array [Z x Y x X] as a video

        :param Video: A numpy array [Z x Y x X]
        :type Video: Any
        :param fps: Frames Per Second
        :type fps: float
        :keyword cmap: colormap (str)
        :keyword interpolation: interpolation method (str)
        :keyword SpeedUp: fps multiplier (int)
        :keyword  Vmin: minimum value of colormap (int)
        :keyword Vmax: maximum value of colormap (int)
        :return: Figure Animation
        :rtype: list[matplotlib.pyplot.figure, matplotlib.pyplot.axes, matplotlib.pyplot.axes,
        matplotlib.pyplot.axes, Any, Any]
        """
        _cmap = kwargs.get('cmap', "binary_r")
        _interp = kwargs.get('interpolation', "none")
        _fps_multi = kwargs.get('SpeedUp', 1)
        _vmin = kwargs.get('Vmin', 0)
        _vmax = kwargs.get('Vmax', 32000)

        _new_fps = _fps_multi*fps
        TiffStack = Video.astype(np.uint16)
        frames = TiffStack.shape[0]
        _start = 0
        _stop = (1/fps)*frames
        _step = 1/fps
        _time_stamps = np.arange(_start, _stop, _step)
        _timeline = amp.Timeline(_time_stamps, units="s", fps=_new_fps)

        fig1 = plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((30, 30), (0, 0), rowspan=28, colspan=28, xticks=[], yticks=[])
        ax2 = plt.subplot2grid((30, 30), (29, 0), colspan=21)
        ax3 = plt.subplot2grid((30, 30), (29, 25), colspan=3)

        block = amp.blocks.Imshow(TiffStack, ax1, cmap=_cmap, vmin=_vmin, vmax=_vmax, interpolation=_interp)
        anim = amp.Animation([block], timeline=_timeline)
        anim.timeline_slider(text='Time', ax=ax2, color="#139fff")
        anim.toggle(ax=ax3)
        plt.show()
        return [fig1, ax1, ax2, ax3, block, anim]

    # noinspection PyTypeChecker
    @staticmethod
    def load_binary_meta(File: str) -> Tuple[int, int, int, str]:
        """
        Loads meta file for binary video

        :param File: The meta file (.txt ext)
        :type File: str
        :return: A tuple containing the number of frames, y pixels, and x pixels [Z x Y x X]
        :rtype: tuple[int, int, int, str]
        """
        _num_frames, _y_pixels, _x_pixels, _type = np.genfromtxt(File, delimiter=",", dtype="str")
        return tuple([int(_num_frames), int(_y_pixels), int(_x_pixels), str(_type)])

    @staticmethod
    def load_mapped_binary(fname: str, meta_file: str, *args: str, **kwargs: str) -> np.memmap:
        """
        Loads a raw binary file in the workspace without loading into memory

        Enter the path to autofill (assumes fname & meta are Path + binary_video, video_meta.txt)

        **Keyword Arguments**
            | *mode* : pass mode to numpy.memmap (str, default = "r")
        :param fname: filename for binary video
        :type fname: str
        :param meta_file: filename for meta file
        :type meta_file: str
        :param args: Path
        :type args: str
        :return: memmap(numpy) array [Z x Y x X]
        :rtype: Any
        """
        if len(args) == 1:
            fname = "".join([args[0], "\\binary_video"])
            meta_file = "".join([args[0], "\\video_meta.txt"])

        _mode = kwargs.get("mode", "r")

        _num_frames, _y_pixels, _x_pixels, _type = np.genfromtxt(meta_file, delimiter=",", dtype="str")
        _num_frames = int(_num_frames)
        _x_pixels = int(_x_pixels)
        _y_pixels = int(_y_pixels)
        return np.memmap(fname, dtype=_type, shape=(_num_frames, _y_pixels, _x_pixels), mode=_mode)
