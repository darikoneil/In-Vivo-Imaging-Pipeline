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

    **saveTiff** : Save a numpy array to a single tiff file

    **loadTiff** : Load a single tiff file

    **loadAllTiffs** : Load a sequence of tiff stacks

    **blockwiseFastFilterTiff** : Blockwise, GPU-parallelized multidimensional median filter

    **saveTiffStack** : Save a numpy array to a sequence of tiff stacks

    **groupedZProject** : Utilize grouped z-project to downsample data

    **Visualize** : a numpy array [frames, x pixels, y pixels] as a video
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

        Example
        -------
        filterTiff(MyTiff, Footprint=np.ones((3, 3, 3)))

        :param Tiff: Tiff stack to be filtered
        :return: filtered_tiff
        :keyword Footprint: Mask of the median filter
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

        Example
        -------
        fastFilterTiff(MyTiff, Footprint=np.ones((3, 3, 3)))

        :param Tiff: Tiff stack to be filtered
        :return: filtered_tiff
        :keyword Footprint: Mask of the median filter
        """
        _footprint = kwargs.get('Footprint', np.ones((3, 3, 3)))
        _converted_tiff = cupy.asarray(Tiff)
        filtered_tiff = cupyx.scipy.ndimage.median_filter(_converted_tiff, footprint=_footprint)
        return filtered_tiff

    @staticmethod
    def saveTiff(Tiff, fname):
        """
        saveTiff
        --------
        Save a numpy array to a single tiff file

        Inputs
        ------
        *Tiff* : numpy array [frames, x pixels, y pixels]
        *fname* : str
            filename

        See Also
        --------
        *saveTiffStack* : Save a numpy array to a sequence of tiff stacks

        Example
        -------
        saveTiff(MyTiff, "D:\\MyTiff.tif")

        :param Tiff: numpy array [frames, x pixels, y pixels]
        :param fname: filename
        :type fname: str
        """
        with tifffile.TiffWriter(fname) as tif:
            for frame in np.floor(Tiff).astype(np.int16):
                tif.save(frame)

    @staticmethod
    def loadTiff(fname, num_frames):
        """
        loadTiff
        --------
        Load a single tiff file

        Inputs
        ------
        *fname* : str
            filename
        *num_frames* : int
            num_frames

        Outputs
        -------
        numpy array

        See Also
        --------
        *loadAllTiffs* : Load a sequence of tiff stacks
        *loadBrukerTiffs : Load a sequence of tiff files from a directory

        Example
        -------
        loadTiff("D:\\MyTiff.tiff", 7000)

        :param fname: filename
        :param num_frames: number of frames
        :type fname: str
        :type num_frames: int
        :return: numpy array
        """
        return tifffile.imread(fname, key=range(0, num_frames, 1))

    @staticmethod
    def loadAllTiffs(VideoDirectory):
        """
        loadAllTiffs
        ------------
        Load a sequence of tiff stacks

        Inputs
        ------
        *VideoDirectory* : str
            Directory containing a sequence of tiff stacks

        Outputs
        -------
        *complete_image* : numpy array [frames, x pixels, y pixels]
            A numpy array containing a sequence of tiff stacks

        See Also
        --------
        *loadTiff* : Load a single tiff file
        *loadBrukerTiffs* : Load a sequence of tiff files from a directory.

        Example
        -------
        complete_image = loadAllTiffs("D:\\MyTiffDirectory")

        :param VideoDirectory: Directory containing a sequence of tiff stacks
        :return: complete_image numpy array [frames, x pixels, y pixels]
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
                PreProcessing.loadTiff(VideoDirectory + "\\" + _fnames[_fname], _num_frames[_fname])
            _last_frame += _num_frames[_fname]

        return complete_image

    @staticmethod
    def blockwiseFastFilterTiff(TiffStack, **kwargs):
        """
        blockwiseFastFilterTiff
        --------------
        GPU-parallelized multidimensional median filter performed in overlapping blocks.

        Designed for use on arrays larger than the available memory capacity.

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
        *BlockSize* : int
            Integer indicating the size of each block. Must fit within memory.
                Default -> 21000
        *BlockBufferRegion* : int
            Integer indicating the size of the overlapping region between blocks
                Default -> 500

        Outputs
        -------
        *filtered_tiff* : numpy array [frames, x pixels, y pixels]

        See Also
        --------
        *filterTiff* : Denoise a tiff stack using multidimensional median filter
        *fastFilterTiff* : GPU-parallelized multidimensional median filter

        Example
        -------
        blockwiseFastFilterTiff(MyTiff, Footprint=np.ones((3, 3, 3)), BlockSize=21000, BlockBufferRegion=500)

        :param TiffStack: Tiff stack to be filtered
        :param kwargs:
        :return: TiffStack: numpy array [frames, x pixels, y pixels]
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
        """
        saveTiffStack
        -------------
        Save a numpy array to a sequence of tiff stacks

        Inputs
        ------
        *TiffStack* : numpy array [frames, x pixels, y pixels]
            A numpy array containing a tiff stack
        *OutputDirectory* : str
            A directory to save the sequence of tiff stacks in

        See Also
        --------
        *saveTiff* : Save a numpy array to a single tiff file

        Example
        -------
        saveTiffStack(MyTiffStack, "D:\\MyTiffStackDirectory")

        :param TiffStack: A numpy array containing a tiff stack
        :param OutputDirectory: A directory to save the sequence of tiff stacks in
        :type OutputDirectory: str
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
                PreProcessing.saveTiff(TiffStack[_start_idx:_end_idx, :, :],
                                       OutputDirectory + "\\" + "Video_0" + str(c_idx) + "_of_" + str(
                                           _chunks) + ".tif")
            else:
                PreProcessing.saveTiff(TiffStack[_start_idx:_end_idx, :, :],
                                       OutputDirectory + "\\" + "Video_" + str(c_idx) + "_of_" + str(
                                           _chunks) + ".tif")
            c_idx += 1

        return print("Finished Saving Tiffs")

    @staticmethod
    def groupedZProject(TiffStack, BinSize, DownsampleFunction):
        """
        groupedZProject
        ---------------
        Utilize grouped z-project to downsample data

        Inputs
        ------
        *TiffStack* : numpy array [frames, x pixels, y pixels]
            A numpy array containing a tiff stack
        *BinSize* : integer or array of integers
            Size of each bin passed to downsampling function
        *DownsampleFunction* : function
            Downsampling function to run on each bin

        Outputs
        -------
        *TiffStack* : numpy array [frames, x pixels, y pixels]

        Example
        -------
        groupedZProject(MyTiffStack, (3, 1, 1), np.mean)

        :param TiffStack: A numpy array containing a tiff stack
        :param BinSize:  Size of each bin passed to downsampling function
        :param DownsampleFunction: Downsampling function
        :return: downsampled_image
        """
        downsampled_image = skimage.measure.block_reduce(TiffStack, block_size=BinSize,
                                                         func=DownsampleFunction)
        return downsampled_image

    @staticmethod
    def viewImage(TiffStack, fps, **kwargs):
        """
        viewImage
        ---------
        Visualize a numpy array [frames, x pixels, y pixels] as a video
        :param TiffStack: A numpy array containing a tiff stack [frames, x pixels, y pixels]
        :param fps: Frames Per Second
        :type fps: float
        :param kwargs:
        :return: Animation
        """
        _cmap = kwargs.get('cmap', "binary_r")
        _interp = kwargs.get('interpolation', "none")
        _fps_multi = kwargs.get('SpeedUp', 1)

        _new_fps = _fps_multi*fps

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

        block = amp.blocks.Imshow(TiffStack, ax1, cmap=_cmap, interpolation=_interp)
        anim = amp.Animation([block], timeline=_timeline)
        anim.timeline_slider(text='Time', ax=ax2, color="#139fff")
        anim.toggle(ax=ax3)
        plt.show()



