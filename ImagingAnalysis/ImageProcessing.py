from __future__ import annotations
from typing import Callable, List, Tuple, Sequence, Optional, Union
import numpy as np
from tqdm.auto import tqdm
import scipy.ndimage
import skimage.measure
import math

try:
    import cupy
    import cupyx.scipy.ndimage
except ModuleNotFoundError:
    print("CuPy required for fast implementation of multidimensional filters.")



def filter_images(Images: np.ndarray, Footprint: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Denoise a tiff stack using a multidimensional median filter

    This function simply calls scipy.ndimage.median_filter

    Footprint is of the form np.ones((Z pixels, Y pixels, X pixels)) with the origin in the center

    :param Images: Images stack to be filtered [Z x Y x X]
    :type Images: Any
    :param Footprint: Mask of the median filter (Optional, Default 3 x 3 x 3)
    :type Footprint: Any
    :return: filtered images [Z x Y x X]
    :rtype: Any
    """
    if Footprint is None:
        Footprint = np.ones((3, 3, 3))

    for _dim in Footprint.shape:
        assert(_dim % 2 != 0) # Must be uneven

    if Images.shape[0] <= Images.shape[1] or Images.shape[0] <= Images.shape[2]:
        AssertionError("Images must be in the form Z x Y x X")

    return scipy.ndimage.median_filter(Images, footprint=Footprint)


def fast_filter_images(Images: np.ndarray, Footprint: Optional[np.ndarray] = None) -> np.ndarray:
    """
    GPU-parallelized multidimensional median filter

    Footprint is of the form np.ones((Z pixels, Y pixels, X pixels)) with the origin in the center

    Requires CuPy

    :param Images: Image stack to be filtered [Z x Y x X]
    :type Images: Any
    :param Footprint: Mask of the median filter (Optional, Default 3 x 3 x 3)
    :type Footprint: Any
    :return: filtered_image [Z x Y x X]
    :rtype: Any
    """
    if Footprint is None:
        Footprint = np.ones((3, 3, 3))

    for _dim in Footprint.shape:
        assert(_dim % 2 != 0) # assert odd

    if Images.shape[0] <= Images.shape[1] or Images.shape[0] <= Images.shape[2]:
        AssertionError("Images must be in the form Z x Y x X")

    return cupyx.scipy.ndimage.median_filter(cupy.asarray(Images), footprint=Footprint)


def remove_shuttle_artifact(Images: np.ndarray, **kwargs: int) -> np.ndarray:
    """
    Function to remove the shuttle artifacts present at the initial imaging frames

    :param Images: Images array with shape Z x Y x X
    :type Images: Any
    :param kwargs:
    :keyword artifact_length: number of frames considered artifact (int)
    :keyword chunk_size: number of frames per chunk_size (makes divisible by value) (int)
    :return: Images
    :rtype: Any
    """
    if Images.shape[0] <= Images.shape[1] or Images.shape[0] <= Images.shape[2]:
        AssertionError("Images must be in the form Z x Y x X")

    _shuttle_artifact_length = kwargs.get("artifact_length", 1000)
    _chunk_size = kwargs.get("chunk_size", 7000)
    _num_frames = Images.shape[0]
    _crop_idx = _num_frames % _chunk_size
    if _crop_idx >= _shuttle_artifact_length:
        return Images[_crop_idx:, :, :]
    else:
        _num_frames -= _shuttle_artifact_length
        _crop_idx = _num_frames % _chunk_size
        return Images[_shuttle_artifact_length + _crop_idx:, :, :]


def blockwise_fast_filter_tiff(Images: np.ndarray, Footprint: Optional[str] = None, **kwargs: int) -> np.ndarray:
    """
    GPU-parallelized multidimensional median filter performed in overlapping blocks.

    Designed for use on arrays larger than the available memory capacity.

    Footprint is of the form np.ones((Z pixels, Y pixels, X pixels)) with the origin in the center

    Requires CuPy

    :param Images: Images stack to be filtered
    :type Images: Any
    :param Footprint: Mask of the median filter (Optional, Default 3 x 3 x 3)
    :type Footprint: Any
    :keyword block_size:   Integer indicating the size of each block. Must fit within memory. (int, default 21000)
    :keyword block_buffer_region: Integer indicating the size of the overlapping region between blocks (int, default 500)
    :return: Images: numpy array [Z x Y x X]
    :rtype: Any
    """
    _block_size = kwargs.get('block_size', int(21000))
    _block_buffer_region = kwargs.get('block_buffer_region', int(500))

    if Footprint is None:
        Footprint = np.ones((3, 3, 3))

    for _dim in Footprint.shape:
        assert(_dim % 2 != 0) # Assert odd


    if Images.shape[0] <= Images.shape[1] or Images.shape[0] <= Images.shape[2]:
        AssertionError("Images must be in the form Z x Y x X")

    _total_frames = Images.shape[0]
    _blocks = range(0, _total_frames, _block_size)
    _num_blocks = len(_blocks)
    _remainder = np.full((_block_buffer_region, Images.shape[1], Images.shape[2]), 0, dtype=np.int16)

    for _block in tqdm(
            range(_num_blocks),
            total=_num_blocks,
            desc="Filtering Images...",
            disable=False,
    ):
        if _block == 0:
            _remainder = Images[_blocks[_block + 1] - 500:_blocks[_block + 1], :, :].copy()
            Images[0:_blocks[_block + 1], :, :] = cupy.asnumpy(fast_filter_images(cupy.asarray(
                Images[0:_blocks[_block + 1], :, :]), Footprint))
        elif _block == _num_blocks - 1:

            Images[_blocks[_block]:_total_frames, :, :] = \
                cupy.asnumpy(fast_filter_images(
                    cupy.asarray(np.append(_remainder, Images[_blocks[_block]:_total_frames, :, :],
                                           axis=0)), Footprint))[_block_buffer_region:, :, :]
        else:
            _remainder_new = Images[_blocks[_block + 1] - 500:_blocks[_block + 1], :, :].copy()
            Images[_blocks[_block]:_blocks[_block + 1], :, :] = \
                cupy.asnumpy(fast_filter_images(
                    cupy.asarray(np.append(_remainder, Images[_blocks[_block]:_blocks[_block + 1], :, :],
                                           axis=0)), Footprint))[_block_buffer_region:_block_size+_block_buffer_region, :, :]
            _remainder = _remainder_new.copy()

    return Images


def grouped_z_project(Images: np.ndarray, BinSize: Union[Tuple[int, int, int], int], DownsampleFunction: Callable[[np.ndarray], np.ndarray]) \
            -> np.ndarray:
    """
    Utilize grouped z-project to downsample data

    Downsample example function -> np.mean

    :param Images: A numpy array containing a tiff stack [Z x Y x X]
    :type Images: Any
    :param BinSize:  Size of each bin passed to downsampling function
    :type BinSize: Union[tuple, int]
    :param DownsampleFunction: Downsampling function
    :type DownsampleFunction: Any
    :return: downsampled image [Z x Y x X]
    :rtype: Any
    """
    return skimage.measure.block_reduce(Images, block_size=BinSize,
                                                     func=DownsampleFunction).astype(Images.dtype)
    # cast back down from float64

