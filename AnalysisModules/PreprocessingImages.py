import numpy as np
import os
from PIL import Image
from tqdm.auto import tqdm
import scipy.ndimage
import math
import cupy
import cupyx.scipy.ndimage
import tifffile


class PreProcessing:
    def __init__(self):
        return

    @staticmethod
    def loadBrukerTiffs(VideoDirectory):
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
    def repackageBrukerTiffs(VideoDirectory, OutputDirectory):
        print("Repackaging...")
        _fnames = os.listdir(VideoDirectory)
        _num_frames = len(_fnames)
        # noinspection PyTypeChecker
        _chunks = math.ceil(_num_frames/7000)

        c_idx = 1
        for _chunk in range(0, _num_frames, 7000):

            _start_idx = _chunk
            _end_idx = _chunk + 7000
            if _end_idx > _num_frames:
                _end_idx = _num_frames+1
            _chunk_frames = _end_idx-_start_idx
            image_chunk = np.full((_chunk_frames, 512, 512), 0, dtype=np.uint16)
            for _fname in tqdm(
                range(_chunk_frames),
                total=_chunk_frames,
                desc="Loading Images...",
                disable=False,
            ):

                image_chunk[_fname, :, :] = np.asarray(Image.open(VideoDirectory + "\\" + _fnames[_fname]))
            if c_idx < 10:
                PreProcessing.saveTiff(image_chunk, OutputDirectory + "\\" + "compiledVideo_0" + str(c_idx) + "_of_" + str(_chunks) + ".tif")
            else:
                PreProcessing.saveTiff(image_chunk, OutputDirectory + "\\" + "compiledVideo_" + str(c_idx) + "_of_" + str(_chunks) + ".tif")
            c_idx += 1
        return print("Finished Repackaging Bruker Tiffs")

    @staticmethod
    def filterTiff(Tiff, **kwargs):
        _kernel = kwargs.get('Kernel', np.ones((3, 3, 3)))

        print("Filtering...")
        filtered_tiff = scipy.ndimage.median_filter(Tiff, footprint=_kernel)
        print("Finished")
        return filtered_tiff

    @staticmethod
    def fastFilterTiff(Tiff, **kwargs):
        _kernel = kwargs.get('Kernel', np.ones((3, 3, 3)))
        _converted_tiff = cupy.asarray(Tiff)
        filtered_tiff = cupyx.scipy.ndimage.median_filter(_converted_tiff, footprint=_kernel)
        return filtered_tiff

    @staticmethod
    def saveTiff(Tiff, fname):
        with tifffile.TiffWriter(fname) as tif:
            for frame in np.floor(Tiff).astype(np.int16):
                tif.save(frame)

    @staticmethod
    def loadTiff(fname, num_frames):
        return tifffile.imread(fname, key=range(0, num_frames, 1))

    @staticmethod
    def loadAllTiffs(VideoDirectory):
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
                PreProcessing.loadTiff(VideoDirectory + "\\" + _fnames[_fname], _num_frames[_fname])
            _last_frame += _num_frames[_fname]

        return complete_image

    @staticmethod
    def blockwiseFastFilterTiff(TiffStack, **kwargs):
        _block_size = kwargs.get('BlockSize', 21000)
        _block_buffer_region = kwargs.get('BlockBufferRegion', 500)
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
                TiffStack[0:_blocks[_block + 1], :, :] = cupy.asnumpy(PreProcessing.fastFilterTiff(cupy.asarray(
                    TiffStack[0:_blocks[_block + 1], :, :])))
            elif _block == _num_blocks - 1:

                TiffStack[_blocks[_block]:_total_frames, :, :] = \
                    cupy.asnumpy(PreProcessing.fastFilterTiff(
                        cupy.asarray(np.append(_remainder, TiffStack[_blocks[_block]:_total_frames, :, :],
                                               axis=0))))[_block_buffer_region:, :, :]

                # TiffStack[_blocks[_block]:_total_frames, :, :] = PreProcessing.fastFilterTiff(np.append(_remainder, TiffStack[_blocks[_block]:_total_frames, :, :],
                # axis=0))[_block_buffer_region:, :, :])
            else:
                _remainder_new = TiffStack[_blocks[_block + 1] - 500:_blocks[_block + 1], :, :].copy()
                TiffStack[_blocks[_block]:_blocks[_block + 1], :, :] = \
                    cupy.asnumpy(PreProcessing.fastFilterTiff(
                        cupy.asarray(np.append(_remainder, TiffStack[_blocks[_block]:_blocks[_block + 1], :, :],
                                               axis=0))))[_block_buffer_region:_block_size+_block_buffer_region, :, :]
                _remainder = _remainder_new.copy()

        return TiffStack

    @staticmethod
    def saveTiffStack(TiffStack, OutputDirectory):
        _num_frames = TiffStack.shape[0]

        _chunks = math.ceil(_num_frames / 7000)

        c_idx = 1
        for _chunk in range(0, _num_frames, 7000):

            _start_idx = _chunk
            _end_idx = _chunk + 7000
            if _end_idx > _num_frames:
                _end_idx = _num_frames + 1

            if c_idx < 10:
                PreProcessing.saveTiff(TiffStack[_start_idx:_end_idx, :, :],
                                       OutputDirectory + "\\" + "Video_0" + str(c_idx) + "_of_" + str(
                                           _chunks) + ".tif")
            else:
                PreProcessing.saveTiff(TiffStack[_start_idx:_end_idx, :, :],
                                       OutputDirectory + "\\" + "Video_" + str(c_idx) + "_of_" + str(
                                           _chunks) + ".tif")
            c_idx += 1

        return print("Finished Saving Tiffs")
