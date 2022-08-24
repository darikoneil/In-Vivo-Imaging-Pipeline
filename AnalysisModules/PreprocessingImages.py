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
    """
    PreProcessing
    -------------
    Class containing a variety of static methods for preprocessing images

    Static Methods
    --------------
    **loadBrukerTiffs** : Load a sequence of tiff files from a directory.

    **repackageBrukerTiffs** :  Repackages a sequence of tiff files within a directory to a smaller sequence of tiff stacks.

    **filterTiff** : Denoise a tiff stack using a multidimensional median filter

    **fastFilterTiff** : GPU-parallelized multidimensional median filter


    **saveTiff** : Save an array to a single tiff file

    **loadTiff** : Load a single tiff file

    **loadAllTiffs** : Load a sequence of tiff stacks

    **blockwiseFastFilterTiff** : Blockwise, GPU-parallelized multidimensional median filter

    **saveTiffStack** : Save an array to a sequence of tiff stacks
    """
    def __init__(self):
        return

    @staticmethod
    def loadBrukerTiffs(VideoDirectory):
        """
        loadBrukerTiffs
        ---------------
        Load a sequence of tiff files from a directory.

        Designed to compile the outputs of a certain imaging utility
        that exports recordings such that each frame is saved as a single tiff.

        Inputs
        ------
        *VideoDirectory* : string
            Directory containing a sequence of single frame tiff files

        Outputs
        -------
        *complete_image* : numpy array [frames, x pixels, y pixels]
            All tiff files in the directory compiled into a single array

        See Also
        --------
        *loadTiff* : Load a tiff file
        *loadAllTiffs* : Load a sequence of tiff stacks`

        Example
        -------
        complete_image = loadBrukerTiffs("D:\\MyVideoDirectory")

        :param VideoDirectory: Directory containing a sequence of single frame tiff files
        :type VideoDirectory: str
        :return: complete_image:  All tiff files in the directory compiled into a single array
        :rtype: np.uint16
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
    def repackageBrukerTiffs(VideoDirectory, OutputDirectory):
        """
        repackageBrukerTiffs
        --------------------
        Repackages a sequence of tiff files within a directory to a smaller sequence
        of tiff stacks.

        Designed to compile the outputs of a certain imaging utility
        that exports recordings such that each frame is saved as a single tiff.

        Inputs
        ------
        *VideoDirectory* : string
            Directory containing a sequence of single frame tiff files

        *OutputDirectory* : string
            Empty directory where tiff stacks will be saved

        See Also
        --------
        *loadBrukerTiffs* : Load a sequence of tiff files from a directory.

        Example
        -------
        repackageBrukerTiffs(
            "D:\\MyVideoDirectory", "D:\\NewVideoDirectory"
                )

        :param VideoDirectory: Directory containing a sequence of single frame tiff files
        :type VideoDirectory: str
        :param OutputDirectory: Empty directory where tiff stacks will be saved
        :type OutputDirectory: str

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
                image_chunk[_fname, :, :] = np.asarray(Image.open(VideoDirectory + "\\" + _fnames[_fname+_offset]))
            if c_idx < 10:
                PreProcessing.saveTiff(image_chunk, OutputDirectory + "\\" + "compiledVideo_0" + str(c_idx) + "_of_" + str(_chunks) + ".tif")
            else:
                PreProcessing.saveTiff(image_chunk, OutputDirectory + "\\" + "compiledVideo_" + str(c_idx) + "_of_" + str(_chunks) + ".tif")
            c_idx += 1
        return print("Finished Repackaging Bruker Tiffs")

    @staticmethod
    def filterTiff(Tiff, **kwargs):
        """
        filterTiff
        ----------
        Denoise a tiff stack using a multidimensional median filter

        This function simply calls scipy.ndimage.median_filter

        Inputs
        ------
        *Tiff* : numpy array [frames, x pixels, y pixels]
            Tiff stack to be filtered

        Keyword Arguments
        -----------------
        *Footprint* : numpy array [z pixels, x pixels, y pixels]
            Mask indicating the footprint of the median filter
                Default -> 3 x 3 x 3
                    Example -> np.ones((3, 3, 3))

        Outputs
        -------
        *filtered_tiff* : numpy array [frames, x pixels, y pixels]

        See Also
        --------
        *fastFilterTiff* : GPU-parallelized multidimensional median filter
        *blockwiseFastFilterTiff* : Blockwise, GPU-parallelized multidimensional median filter

        :param Tiff: Tiff stack to be filtered
        :return: filtered_tiff
        :keyword Footprint: Mask of the median filter
        :type Tiff: np.int16
        :rtype: np.int16
        """
        _footprint = kwargs.get('Footprint', np.ones((3, 3, 3)))

        print("Filtering...")
        filtered_tiff = scipy.ndimage.median_filter(Tiff, footprint=_footprint)
        print("Finished")
        return filtered_tiff

    @staticmethod
    def fastFilterTiff(Tiff, **kwargs):
        """
        fastFilterTiff
        --------------
        GPU-parallelized multidimensional median filter

        Inputs
        ------
        *Tiff* : numpy array [frames, x pixels, y pixels]
            Tiff stack to be filtered

        Keyword Arguments
        -----------------
        *Footprint* : numpy array [z pixels, x pixels, y pixels]
            Mask indicating the footprint of the median filter
                Default -> 3 x 3 x 3
                    Example -> np.ones((3, 3, 3))

        Outputs
        -------
        *filtered_tiff* : numpy array [frames, x pixels, y pixels]

        See Also
        --------
        *filterTiff* : Denoise a tiff stack using multidimensional median filter
        *blockwiseFastFilterTiff* : Blockwise, GPU-parallelized multidimensional median filter

        :param Tiff: Tiff stack to be filtered
        :type Tiff: np.int16
        :return: filtered_tiff
        :keyword Footprint: Mask of the median filter
        """
        _footprint = kwargs.get('Footprint', np.ones((3, 3, 3)))
        _converted_tiff = cupy.asarray(Tiff)
        filtered_tiff = cupyx.scipy.ndimage.median_filter(_converted_tiff, footprint=_footprint)
        return filtered_tiff

    @staticmethod
    def saveTiff(Tiff, fname):
        with tifffile.TiffWriter(fname) as tif:
            for frame in np.floor(Tiff).astype(np.int16):
                tif.save(frame)

    @staticmethod
    def loadTiff(fname, num_frames):
        """
        hello
        -----
        goodbye
        *hi*

        :param fname: filename
        :param num_frames: number of frames
        :return: numpy array
        """
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
                TiffStack[0:_blocks[_block + 1], :, :] = cupy.asnumpy(PreProcessing.fastFilterTiff(cupy.asarray(
                    TiffStack[0:_blocks[_block + 1], :, :]), footprint=_footprint))
            elif _block == _num_blocks - 1:

                TiffStack[_blocks[_block]:_total_frames, :, :] = \
                    cupy.asnumpy(PreProcessing.fastFilterTiff(
                        cupy.asarray(np.append(_remainder, TiffStack[_blocks[_block]:_total_frames, :, :],
                                               axis=0)), footprint=_footprint))[_block_buffer_region:, :, :]

                # TiffStack[_blocks[_block]:_total_frames, :, :] = PreProcessing.fastFilterTiff(np.append(_remainder, TiffStack[_blocks[_block]:_total_frames, :, :],
                # axis=0))[_block_buffer_region:, :, :])
            else:
                _remainder_new = TiffStack[_blocks[_block + 1] - 500:_blocks[_block + 1], :, :].copy()
                TiffStack[_blocks[_block]:_blocks[_block + 1], :, :] = \
                    cupy.asnumpy(PreProcessing.fastFilterTiff(
                        cupy.asarray(np.append(_remainder, TiffStack[_blocks[_block]:_blocks[_block + 1], :, :],
                                               axis=0)), footprint=_footprint))[_block_buffer_region:_block_size+_block_buffer_region, :, :]
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
