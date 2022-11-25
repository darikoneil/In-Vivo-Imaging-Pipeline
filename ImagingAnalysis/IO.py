from __future__ import annotations
import numpy as np
import os
from PIL import Image
from tqdm.auto import tqdm
import tifffile
from typing import Callable, List, Tuple, Sequence, Optional, Union
import math


def load_bruker_tiffs(ImageDirectory: str) -> np.ndarray:
    """
    Load a sequence of tiff files from a directory.

    Designed to compile the outputs of a certain imaging utility
    that exports recordings such that each frame is saved as a single tiff.

    :param ImageDirectory: Directory containing a sequence of single frame tiff files
    :type ImageDirectory: str
    :return: complete_image:  All tiff files in the directory compiled into a single array (Z x Y x X, uint16)
    :rtype: Any
     """
    _fnames = os.listdir(ImageDirectory)
    _num_frames = len(_fnames)
    complete_image = np.full((_num_frames, 512, 512), 0, dtype=np.uint16)

    for _fname in tqdm(
            range(_num_frames),
            total=_num_frames,
            desc="Loading Image...",
            disable=False,
    ):
        complete_image[_fname, :, :] = np.asarray(Image.open(ImageDirectory + "\\" + _fnames[_fname]))
    return complete_image


def repackage_bruker_tiffs(ImageDirectory: str, OutputDirectory: str) -> None:
    """
    Repackages a sequence of tiff files within a directory to a smaller sequence
    of tiff stacks.

    Designed to compile the outputs of a certain imaging utility
    that exports recordings such that each frame is saved as a single tiff.

    :param ImageDirectory: Directory containing a sequence of single frame tiff files
    :type ImageDirectory: str
    :param OutputDirectory: Empty directory where tiff stacks will be saved
    :type OutputDirectory: str
    :rtype: None
    """
    print("Repackaging...")
    _fnames = os.listdir(ImageDirectory)
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
            image_chunk[_fname, :, :] = np.asarray(Image.open(ImageDirectory + "\\" + _fnames[_fname + _offset]))
        if c_idx < 10:
            save_tiff(image_chunk, OutputDirectory + "\\" + "compiledVideo_0" + str(c_idx) + "_of_" + str(_chunks) + ".tif")
        else:
            save_tiff(image_chunk, OutputDirectory + "\\" + "compiledVideo_" + str(c_idx) + "_of_" + str(_chunks) + ".tif")
        c_idx += 1
    return print("Finished Repackaging Bruker Tiffs")


def load_single_tiff(Filename: str, NumFrames: int) -> np.ndarray:
    """
    Load a single tiff file

    :param Filename: filename
    :param NumFrames: number of frames
    :type Filename: str
    :type NumFrames: int
    :return: numpy array [Z x Y x X]
    :rtype: Any
    """
    return tifffile.imread(Filename, key=range(0, NumFrames, 1))


def load_all_tiffs(ImageDirectory: str) -> np.ndarray:
    """
    Load a sequence of tiff stacks

    :param ImageDirectory: Directory containing a sequence of tiff stacks
    :type ImageDirectory: str
    :return: complete_image numpy array [Z x Y x X] as int16
    :rtype: Any
    """
    _fnames = os.listdir(ImageDirectory)
    y_pix, x_pix = tifffile.TiffFile(ImageDirectory + "\\" + _fnames[0]).pages[0].shape
    _num_frames = [] # initialize
    [_num_frames.append(len(tifffile.TiffFile(ImageDirectory + "\\" + _fname).pages)) for _fname in _fnames]
    _total_frames = sum(_num_frames)
    complete_image = np.full((_total_frames, y_pix, x_pix), 0, dtype=np.int16) # not sure if correct x,y order
    _last_frame = 0

    for _fname in tqdm(
            range(len(_fnames)),
            total=len(_fnames),
            desc="Loading Images...",
            disable=False,
    ):
        complete_image[_last_frame:_last_frame+_num_frames[_fname], :, :] = \
            load_single_tiff(ImageDirectory + "\\" + _fnames[_fname], _num_frames[_fname])
        _last_frame += _num_frames[_fname]

    return complete_image


def load_raw_binary(Filename: Union[str, None], MetaFile: Union[str, None], *args: Optional[str]) -> np.ndarray:
    """
    Loads a raw binary file

    Enter the path to autofill (assumes Filename & meta are path + binary_video, video_meta.txt)

    :param Filename: filename for binary video
    :type Filename: str
    :param MetaFile: filename for meta file
    :type MetaFile: str
    :param args: path to a directory containing Filename and MetaFile
    :type args: str
    :return: numpy array [Z x Y x X]
    :rtype: Any
    """
    if len(args) == 1:
        Filename = "".join([args[0], "\\binary_video"])
        MetaFile = "".join([args[0], "\\video_meta.txt"])

    _num_frames, _y_pixels, _x_pixels, _type = np.genfromtxt(MetaFile, delimiter=",", dtype="str")
    _num_frames = int(_num_frames)
    _x_pixels = int(_x_pixels)
    _y_pixels = int(_y_pixels)
    return np.reshape(np.fromfile(Filename, dtype=_type), (_num_frames, _y_pixels, _x_pixels))


def load_binary_meta(Filename: str) -> Tuple[int, int, int, str]:
    """
    Loads meta file for binary video

    :param Filename: The meta file (.txt ext)
    :type Filename: str
    :return: A tuple containing the number of frames, y pixels, and x pixels [Z x Y x X]
    :rtype: tuple[int, int, int, str]
    """
    _num_frames, _y_pixels, _x_pixels, _type = np.genfromtxt(Filename, delimiter=",", dtype="str")
    return int(_num_frames), int(_y_pixels), int(_x_pixels), str(_type)


def load_mapped_binary(Filename: str, MetaFile: str, *args: Optional[str], **kwargs: str) -> np.memmap:
    """
    Loads a raw binary file in the workspace without loading into memory

    Enter the path to autofill (assumes Filename & meta are path + binary_video, video_meta.txt)

    :param Filename: filename for binary video
    :type Filename: str
    :param MetaFile: filename for meta file
    :type MetaFile: str
    :param args: Path
    :type args: str
    :keyword mode: pass mode to numpy.memmap (str, default = "r")
    :return: memmap(numpy) array [Z x Y x X]
    :rtype: Any
    """
    if len(args) == 1:
        Filename = "".join([args[0], "\\binary_video"])
        MetaFile = "".join([args[0], "\\video_meta.txt"])

    _mode = kwargs.get("mode", "r")

    _num_frames, _y_pixels, _x_pixels, _type = np.genfromtxt(MetaFile, delimiter=",", dtype="str")
    _num_frames = int(_num_frames)
    _x_pixels = int(_x_pixels)
    _y_pixels = int(_y_pixels)
    return np.memmap(Filename, dtype=_type, shape=(_num_frames, _y_pixels, _x_pixels), mode=_mode)


def save_single_tiff(Images: np.ndarray, Filename: str) -> None:
    """
    Save a numpy array to a single tiff file as type int16

    :param Images: numpy array [frames, y pixels, x pixels]
    :type Images: Any
    :param Filename: filename
    :type Filename: str
    :rtype: None
    """
    with tifffile.TiffWriter(Filename) as tif:
        for frame in np.floor(Images).astype(np.int16):
            tif.save(frame)


def save_tiff_stack(Images: str, OutputDirectory: str) -> None:
    """
    Save a numpy array to a sequence of tiff stacks

    :param Images: A numpy array containing a tiff stack [Z x Y x X]
    :type Images: Any
    :param OutputDirectory: A directory to save the sequence of tiff stacks in int16
    :type OutputDirectory: str
    :rtype: None
    """
    _num_frames = Images.shape[0]

    _chunks = math.ceil(_num_frames / 7000)

    c_idx = 1
    for _chunk in range(0, _num_frames, 7000):

        _start_idx = _chunk
        _end_idx = _chunk + 7000
        if _end_idx > _num_frames:
            _end_idx = _num_frames + 1

        if c_idx < 10:
            save_tiff(Images[_start_idx:_end_idx, :, :],
                                    OutputDirectory + "\\" + "Video_0" + str(c_idx) + "_of_" + str(
                                       _chunks) + ".tif")
        else:
            save_tiff(Images[_start_idx:_end_idx, :, :],
                                    OutputDirectory + "\\" + "Video_" + str(c_idx) + "_of_" + str(
                                       _chunks) + ".tif")
        c_idx += 1

    return print("Finished Saving Tiffs")


def save_raw_binary(Images: np.ndarray, ImageDirectory: str) -> None:
    """
    This function saves a tiff stack as a binary file

    :param Images: Images to be saved [Z x Y x X]
    :type Images: np.ndarray
    :param ImageDirectory: Directory to save images in
    :type ImageDirectory: str
    :rtype: None
    """
    print("Saving images as a binary file...")
    _meta_file = "".join([ImageDirectory, "\\video_meta.txt"])
    _video_file = "".join([ImageDirectory, "\\binary_video"])

    try:
        with open(_meta_file, 'w') as f:
            f.writelines([str(Images.shape[0]), ",", str(Images.shape[1]), ",",
                          str(Images.shape[2]), ",", str(Images.dtype)])
    except FileNotFoundError:
        _meta_path = _meta_file.replace("\\video_meta.txt", "")
        os.makedirs(_meta_path)
        with open(_meta_file, 'w') as f:
            f.writelines([str(Images.shape[0]), ",", str(Images.shape[1]), ",",
                          str(Images.shape[2]), ",", str(Images.dtype)])

    Images.tofile(_video_file)
    print("Finished saving images as a binary file.")
