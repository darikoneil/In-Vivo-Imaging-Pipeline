from __future__ import annotations

import itertools

import numpy as np
import os
from PIL import Image
from tqdm.auto import tqdm
from tqdm import tqdm as tq
import tifffile
from typing import Callable, List, Tuple, Sequence, Optional, Union
import math
import pathlib
from prettytable import PrettyTable


def _pretty_print_bruker_command(Channels, Planes, Frames, Height, Width) -> None:
    """
    Function simply prints the bruker folder contents detected

    :param Channels: Number of Channels
    :type Channels: int
    :param Planes: Number of Planes
    :type Planes: int
    :param Frames: Number of Frames
    :type Frames: int
    :param Height: Height of Image (Y Pixels)
    :type Height: int
    :param Width:  Width of Image (X Pixels)
    :type Width:
    :rtype: None
    """
    _table = PrettyTable()
    _table.header = False
    _table.add_row(["Total Images Detected", Channels * Planes * Frames])
    _table.add_row(["Channels", Channels])
    _table.add_row(["Planes", Planes])
    _table.add_row(["Frames", Frames])
    _table.add_row(["Height", Height])
    _table.add_row(["Width", Width])
    print("\n")
    print(_table)


def determine_bruker_folder_contents(ImageDirectory: str) -> Tuple[int, int, int, int, int]:
    """
    Function determine contents of the bruker folder

    :param ImageDirectory: Directory containing bruker imaging data
    :type ImageDirectory: str
    :rtype: tuple
    """
    _directory = pathlib.Path(ImageDirectory)
    _files = [_file for _file in _directory.rglob("*.tif") if _file.is_file()]
    # preallocate

    def parser(Tag_: str, SplitStrs: list) -> str:
        return [_tag for _tag in SplitStrs if Tag_ in _tag]

    def find_unique_substrings(Tag: str) -> int:
        nonlocal _files
        _hits = [parser(Tag, str(_file).split("_")) for _file in _files]
        _hits = [_hit for _nested_hit in _hits for _hit in _nested_hit]
        return list(dict.fromkeys(_hits)).__len__()

    def find_unique_strings_given_substring(StaticTag: str) -> int:
        nonlocal _files
        return [parser(StaticTag, str(_file).split("_")) for _file in _files].__len__()

    def find_channels() -> int:
        return find_unique_substrings("Ch")

    def is_multiplane() -> bool:
        if find_unique_substrings("Cycle") > 1:
            return True
        else:
            return False

    def find_planes() -> int:
        nonlocal channels

        if is_multiplane():
            return find_unique_strings_given_substring("Cycle00001")//channels
        else:
            return 1

    def find_frames() -> int:
        nonlocal channels
        if is_multiplane():
            return find_unique_strings_given_substring("000001.ome")//channels
        else:
            return find_unique_strings_given_substring("Cycle00001")//channels

    def find_dimensions() -> Tuple[int, int]:
        nonlocal ImageDirectory
        nonlocal _files
        return np.asarray(Image.open("".join(str(_files[0])))).shape

    channels = find_channels()

    # noinspection PyTypeChecker
    return (channels, find_planes(), find_frames(), *find_dimensions())


def load_bruker_tiffs(ImageDirectory: str) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """
    Load a sequence of tiff files from a directory.

    Designed to compile the outputs of a certain imaging utility
    that exports recordings such that each frame is saved as a single tiff.

    :param ImageDirectory: Directory containing a sequence of single frame tiff files
    :type ImageDirectory: str
    :return: complete_image:  All tiff files in the directory compiled into a single array (Z x Y x X, uint16)
    :rtype: Any
     """

    _channels, _planes, _frames, _y_pixels, _x_pixels = determine_bruker_folder_contents(ImageDirectory)
    _pretty_print_bruker_command(_channels, _planes, _frames, _y_pixels, _x_pixels)

    def find_files(Tag: Union[str, list[str]]):
        nonlocal ImageDirectory
        nonlocal _files

        def check_file_contents(Tag_: str, File_: pathlib.WindowsPath) -> bool:
            if Tag_ in str(File_).split("_"):
                return True
            else:
                return False

        # reset, rather maintain code as is then make a new temporary variable since this
        # is basically instant
        _files = [_file for _file in pathlib.Path(ImageDirectory).rglob("*.tif")]

        # now find
        if isinstance(Tag, list):
            Tag = "".join([Tag[0], Tag[1]])
            _files = [_file for _file in _files if Tag in str(_file.stem)]
        else:
            _files = [_file for _file in _files if check_file_contents(Tag, _file)]

    def load_images():
        nonlocal _files
        nonlocal _frames
        nonlocal _y_pixels
        nonlocal _x_pixels
        nonlocal _tqdm_desc

        def load_image():
            nonlocal ImageDirectory
            nonlocal _files
            nonlocal _file
            return np.asarray(Image.open(str(_files[_file])))

        # Bruker is usually saved to .tif in 0-65536 (uint16) even though recording says 8192 (uint13)
        complete_image = np.full((_frames, _y_pixels, _x_pixels), 0, dtype=np.uint16)
        for _file in tqdm(
                range(_frames),
                total=_frames,
                desc=_tqdm_desc,
                disable=False,
        ):
            complete_image[_file, :, :] = load_image()

        return complete_image

    _files = [_file for _file in pathlib.Path(ImageDirectory).rglob("*.tif")]

    if _channels == 1 and _planes == 1:
        _tqdm_desc = "Loading Images..."
        return load_images()
    elif _channels > 1 and _planes == 1:
        images = []
        for _channel in range(_channels):
            _tqdm_desc = "".join(["Loading Channel ", str(_channel+1), " Images..."])
            _tag = "".join(["Ch", str(_channel+1)])
            find_files(_tag)
            images.append(load_images())
        return images
    elif _channels == 1 and _planes > 1:
        images = []
        _base_tag = "00000"
        for _plane in range(_planes):
            _tqdm_desc = "".join(["Loading Plane ", str(_plane), " Images..."])
            _tag = "".join([_base_tag, str(_plane+1), ".ome"])
            find_files(_tag)
            images.append(load_images())
        return images
    elif _channels > 1 and _planes > 1:
        images = []
        _base_tag = "00000"
        for _channel, _plane in itertools.combinations(range(_channels), range(_planes)):
            _tqdm_desc = "".join(["Loading Plane ", str(_plane),
                                  " Channel ", str(_channel), " Images..."])
            _tag = ["".join(["Ch", str(_channel)]), "".join([_base_tag, str(_plane+1), ".ome"])]
            find_files(_tag)
            images.append(load_images())
        return images


def repackage_bruker_tiffs(ImageDirectory: str, OutputDirectory: str, *args: Tuple[int, int]) -> None:
    """
    Repackages a sequence of tiff files within a directory to a smaller sequence
    of tiff stacks.
    Designed to compile the outputs of a certain imaging utility
    that exports recordings such that each frame is saved as a single tiff.
    :param ImageDirectory: Directory containing a sequence of single frame tiff files
    :type ImageDirectory: str
    :param OutputDirectory: Empty directory where tiff stacks will be saved
    :type OutputDirectory: str
    :param args: optional argument to indicate the repackaging of a specific channel and/or plane
    :type args: Tuple[int, int]
    :rtype: None
    """

    def load_image():
        nonlocal ImageDirectory
        nonlocal _files
        nonlocal _file
        nonlocal _offset
        return np.asarray(Image.open(str(_files[_file + _offset])))

    def find_files(Tag: Union[str, list[str]]):
        nonlocal ImageDirectory
        nonlocal _files

        def check_file_contents(Tag_: str, File_: pathlib.WindowsPath) -> bool:
            if Tag_ in str(File_).split("_"):
                return True
            else:
                return False

        # reset, rather maintain code as is then make a new temporary variable since this
        # is basically instant
        _files = [_file for _file in pathlib.Path(ImageDirectory).rglob("*.tif")]

        # now find
        if isinstance(Tag, list):
            Tag = "".join([Tag[0], "_", Tag[1]])
            _files = [_file for _file in _files if Tag in str(_file.stem)]
        else:
            _files = [_file for _file in _files if check_file_contents(Tag, _file)]

    _files = [_file for _file in pathlib.Path(ImageDirectory).rglob("*.tif")]
    _channels, _planes, _frames, _y, _x = determine_bruker_folder_contents(ImageDirectory)
    _pretty_print_bruker_command(_channels, _planes, _frames, _y, _x)

    if args:
        if _channels > 1 and _planes > 1 and args.__len__() >= 2:
            _base_tag = "00000"
            _tag = ["".join(["Ch", str(args[0])]), "".join([_base_tag, str(args[1]), ".ome"])]
            find_files(_tag)
        elif _channels == 1 and _planes > 1 and len(args) == 1:
            _base_tag = "00000"
            _tag = "".join([_base_tag, str(args[0]), ".ome"])
            find_files(_tag)
        elif _channels == 1 and _planes > 1 and len(args) >= 2:
            _base_tag = "00000"
            _tag = "".join([_base_tag, str(args[1]), ".ome"])
            find_files(_tag)
        elif _channels > 1 and _planes == 1:
            _tag = "".join(["Ch", str(args[0])])
            find_files(_tag)
        else:
            pass
    else:
        try:
            assert(_channels == 1)
        except AssertionError:
            print("Folder contains multiple channels")
            return
        try:
            assert(_planes == 1)
        except AssertionError:
            print("Folder contains multiple planes")
            return

    # noinspection PyTypeChecker
    _chunks = math.ceil(_frames/7000)
    c_idx = 1
    _offset = int()

    _pbar = tq(total=_frames)
    _pbar.set_description("Repackaging Bruker Tiffs...")

    for _chunk in range(0, _frames, 7000):

        _start_idx = _chunk
        _offset = _start_idx
        _end_idx = _chunk + 7000
        _chunk_frames = _end_idx-_start_idx
        # If this is the last chunk which may not contain a full 7000 frames...
        if _end_idx > _frames:
            _end_idx = _frames
            _chunk_frames = _end_idx - _start_idx
            _end_idx += 1

        image_chunk = np.full((_chunk_frames, _y, _x), 0, dtype=np.uint16)

        for _file in range(_chunk_frames):
            image_chunk[_file, :, :] = load_image()
            _pbar.update(1)

        if c_idx < 10:
            save_single_tiff(image_chunk, OutputDirectory + "\\" + "compiledVideo_0" + str(c_idx) + "_of_" + str(_chunks) + ".tif")
        else:
            save_single_tiff(image_chunk, OutputDirectory + "\\" + "compiledVideo_" + str(c_idx) + "_of_" + str(_chunks) + ".tif")
        c_idx += 1
    return


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
    complete_image = np.full((_total_frames, y_pix, x_pix), 0, dtype=np.int16)
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


def save_single_tiff(Images: np.ndarray, Filename: str, Type: Optional[np.dtype] = np.int16) -> None:
    """
    Save a numpy array to a single tiff file as type int16

    :param Images: numpy array [frames, y pixels, x pixels]
    :type Images: Any
    :param Filename: filename
    :type Filename: str
    :param Type: type for saving
    :type Type: Optional[Any]
    :rtype: None
    """
    with tifffile.TiffWriter(Filename) as tif:
        for frame in np.floor(Images).astype(Type):
            tif.save(frame)


def save_tiff_stack(Images: str, OutputDirectory: str, Type: Optional[np.dtype] = np.int16) -> None:
    """
    Save a numpy array to a sequence of tiff stacks

    :param Images: A numpy array containing a tiff stack [Z x Y x X]
    :type Images: Any
    :param OutputDirectory: A directory to save the sequence of tiff stacks in int16
    :type OutputDirectory: str
    :param Type: type for saving
    :type Type: Optional[Any]
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
            save_single_tiff(Images[_start_idx:_end_idx, :, :],
                                    OutputDirectory + "\\" + "Video_0" + str(c_idx) + "_of_" + str(
                                       _chunks) + ".tif")
        else:
            save_single_tiff(Images[_start_idx:_end_idx, :, :],
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
