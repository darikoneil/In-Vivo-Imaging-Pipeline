import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
from ImagingAnalysis.network import Network_3D_Unet
import numpy as np
from ImagingAnalysis.utils import save_yaml, read_yaml
from ImagingAnalysis.data_process import test_preprocess_lessMemoryNoTail_feedImage, \
    testset, multibatch_test_save, singlebatch_test_save
from skimage import io
from ImagingAnalysis.PreprocessingImages import PreProcessing

# Make sure to  edit the forking pickler to use protocol 4 in multiprocessing library


class DenoisingModule:
    """
    Class for Denoising Calcium Imaging Data
    """

    def __init__(self, ModelName, DataFile, **kwargs):
        """
        :param ModelName: Name of denoising model
        :type ModelName: str
        :param DataFile: Name of data file
        :type DataFile: str
        :param kwargs: Keyword arguments
        """
        # noinspection PyTypeChecker
        self.opt = self.construct_parser().parse_args(self.parse_inputs(ModelName, DataFile, kwargs))
        self.denoiser = None

        if self.validate_environment():
            # Set CUDA device environmental variable (good idea)
            os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.GPU
        else:
            print("Warning: Unable to interface with CUDA. Debug PyTorch installation.")

    def runDenoising(self, *args):
        if len(args) >= 1:
            _verbose = True
        else:
            _verbose = False

        _model_path, _model_list = self.retrieve_models(self.opt)

        _training_params_file, _model_list = self.extract_training_params(_model_list)
        try:
            os.path.exists("".join([_model_path, "\\",  _training_params_file]))
            save_yaml(self.opt, "".join([self.opt.output_dir, "\\", _training_params_file]))
            if _verbose:
                print("\nLocated Training Parameters:")
                read_yaml(self.opt, "".join([_model_path, "\\", _training_params_file]))
        except FileNotFoundError:
            print("\nUnable to locate training parameters file")
            return

        if _verbose:
            print("\nLoading Images...")
        _img_list = self.retrieve_images(self.opt)
        if _verbose:
            print('\033[1;31mStacks for processing -----> \033[0m')
            print('Total number -----> ', len(_img_list))

        self.denoiser = Network_3D_Unet(in_channels=1, out_channels=1, f_maps=self.opt.fmap,
                                        final_sigmoid=True)

        if _verbose:
            print('\033[1;31mUsing {} GPU for testing -----> \033[0m'.format(torch.cuda.device_count()))

        self.denoiser = self.denoiser.cuda()
        self.denoiser = nn.DataParallel(self.denoiser, device_ids=range(torch.cuda.device_count()))
        # noinspection DuplicatedCode,PyUnusedLocal
        Tensor = torch.cuda.FloatTensor

        for _model_idx in range(len(_model_list)):
            _model = _model_list[_model_idx]
            if ".pth" in _model:
                # Instance
                _model_name = "".join([self.opt.pth_path, "//", self.opt.denoise_model,
                                       "\\", _model])
                if isinstance(self.denoiser, nn.DataParallel):
                    self.denoiser.module.load_state_dict(torch.load(_model_name))
                    self.denoiser.eval()
                else:
                    self.denoiser.load_state_dict(torch.load(_model_name))
                    self.denoiser.eval()
                self.denoiser.cuda()

                # Iterate
                for _image in range(len(_img_list)):
                    _name_list, _noise_img, _coordinate_list = \
                        test_preprocess_lessMemoryNoTail_feedImage(self.opt, np.array(_img_list[_image]))
                    _prev_time = time.time()
                    _time_start = time.time()
                    _denoise_img = np.zeros(_noise_img.shape)
                    _input_img = np.zeros(_noise_img.shape)

                    _test_data = testset(_name_list, _coordinate_list, _noise_img)
                    _testloader = DataLoader(_test_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=4)
                    for _iteration, (_noise_patch, _single_coordinate) in enumerate(_testloader):
                        _noise_patch = _noise_patch.cuda()
                        _real_A = _noise_patch
                        _real_A = Variable(_real_A)
                        _fake_B = self.denoiser(_real_A)
                        ################################################################################################################
                        # Determine approximate time left
                        _batches_done = _iteration
                        _batches_left = 1 * len(_testloader) - _batches_done
                        _time_left_seconds = int(_batches_left * (time.time() - _prev_time))
                        _time_left = datetime.timedelta(seconds=_time_left_seconds)
                        _prev_time = time.time()
                        ################################################################################################################
                        if _iteration % 1 == 0:
                            _time_end = time.time()
                            _time_cost = _time_end - _time_start  # datetime.timedelta(seconds= (time_end - time_start))
                            print(
                                '\r[Model %d/%d, %s] [Stack %d/%d, %s] [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                                % (
                                    _model_idx + 1,
                                    len(_model_list),
                                    _model_name,
                                    _image + 1,
                                    len(_img_list),
                                    "temp",
                                    _iteration + 1,
                                    len(_testloader),
                                    _time_cost,
                                    _time_left_seconds
                                ), end=' ')

                        if (_iteration + 1) % len(_testloader) == 0:
                            print('\n', end=' ')
                        ################################################################################################################
                        _output_image = np.squeeze(_fake_B.cpu().detach().numpy())
                        _raw_image = np.squeeze(_real_A.cpu().detach().numpy())
                        if _output_image.ndim == 3:
                            _turn = 1
                        else:
                            _turn = _output_image.shape[0]
                        # print(turn)
                        if _turn > 1:
                            for _id in range(_turn):
                                _aaaa, _bbbb, _stack_start_w, _stack_end_w, _stack_start_h, _stack_end_h, \
                                _stack_start_s, _stack_end_s = multibatch_test_save(
                                    _single_coordinate, _id, _output_image, _raw_image)
                                _denoise_img[_stack_start_s:_stack_end_s, _stack_start_h:_stack_end_h,
                                _stack_start_w:_stack_end_w] \
                                    = _aaaa * (np.sum(_bbbb) / np.sum(_aaaa)) ** 0.5
                                _input_img[_stack_start_s:_stack_end_s, _stack_start_h:_stack_end_h,
                                _stack_start_w:_stack_end_w] \
                                    = _bbbb
                        else:
                            _aaaa, _bbbb, _stack_start_w, _stack_end_w, _stack_start_h, _stack_end_h, \
                            _stack_start_s, _stack_end_s = singlebatch_test_save(
                                _single_coordinate, _output_image, _raw_image)
                            _denoise_img[_stack_start_s:_stack_end_s, _stack_start_h:_stack_end_h,
                            _stack_start_w:_stack_end_w] \
                                = _aaaa * (np.sum(_bbbb) / np.sum(_aaaa)) ** 0.5
                            _input_img[_stack_start_s:_stack_end_s, _stack_start_h:_stack_end_h,
                            _stack_start_w:_stack_end_w] \
                                = _bbbb

                    _output_img = _denoise_img.squeeze().astype(np.float32) * self.opt.normalize_factor
                    del _denoise_img
                    _output_img = _output_img - _output_img.min()
                    _output_img = _output_img / _output_img.max() * 65535
                    _output_img = np.clip(_output_img, 0, 65535).astype('uint16')
                    _output_img = _output_img - _output_img.min()
                    _output_img = _output_img.astype('int16')

                    _result_name = "".join([self.opt.output_dir, "\\", _model.replace(".pth", ""), _image])
                    Preprocessing.saveRawBinary(_output_img, _result_name)

    @classmethod
    def retrieve_images(cls, opt):
        if opt.image_type == "binary":
            _image_file = "".join([opt.datasets_path, "\\", opt.datasets_folder])
            _meta_file = "".join([opt.datasets_path, "\\video_meta.txt"])
            try:
                os.path.exists(_meta_file)
            except FileNotFoundError:
                print("Could not locate meta file")
                return
            _num_frames, _y_pixels, _x_pixels, _type = \
                np.genfromtxt(_meta_file, delimiter=",", dtype="str")
            _num_frames = int(_num_frames)
            _y_pixels = int(_y_pixels)
            _x_pixels = int(_x_pixels)


            _images = np.memmap(_image_file, mode="r", shape=(_num_frames, _y_pixels, _x_pixels), dtype=_type)
            if not cls.validate_size_limit(opt, _y_pixels, _x_pixels):
                AssertionError("Tensors too large for GPU")
                return

            _chunk_size = cls.calculate_chunk_size(_num_frames, opt.test_datasize)
            _indices = np.arange(0, _num_frames+1, _num_frames/_chunk_size)
            img_list = []
            for i in range(_indices.shape[0]-1):
                img_list.append(_images[_indices[i].astype(int):_indices[i+1].astype(int), :, :])

            return img_list

    @staticmethod
    def extract_training_params(model_list):
        _training_params_file = None

        for _model in range(len(model_list)):
            if ".yaml" in model_list[_model]:
                _training_params_file = model_list[_model]
                del model_list[_model]
        if _training_params_file is None:
            AssertionError("Unable to identify training parameters")
        else:
            return _training_params_file, model_list

    @staticmethod
    def retrieve_models(opt):
        model_path = "".join([opt.pth_path, "\\", opt.denoise_model])
        model_list = list(os.walk(model_path, topdown=False))[-1][-1]
        model_list.sort()
        return model_path, model_list

    @staticmethod
    def validate_environment():
        return torch.cuda.is_available()

    @staticmethod
    def parse_inputs(ModelName, DataFile, kwargs):
        """
        This function parses class inputs to merge with default deepcad options

        :param ModelName: Name of denoising model
        :type ModelName: str
        :param DataFile: Name of data file
        :type DataFile: str
        :param kwargs: keyword arguments to class
        :type kwargs: dict
        :return: A list of key-value pairs in string form
        :rtype: list[str]
        """
        parsed_inputs = []

        _model_path = kwargs.get("model_path", "".join([os.getcwd(), "\\pth"]))
        try:
            os.path.exists(_model_path)
        except FileNotFoundError:
            print("Could not locate model directory")
            return
        _model_path = "".join(["--pth_path=", _model_path])
        parsed_inputs.append(_model_path)

        if ModelName is None:
            AssertionError("No denoising model specified")
        else:
            try:
                _model_loc = "".join([_model_path, "\\", ModelName])
                os.path.exists(_model_loc)
            except FileNotFoundError:
                print("Could not locate specified denoising model")
                return
        _model = "".join(["--denoise_model=", ModelName])
        parsed_inputs.append(_model)

        _output_path = "".join(["--output_dir=", kwargs.get("output_path", os.getcwd())])
        try:
            os.path.exists(_output_path)
        except FileNotFoundError:
            print("Could not located specified output path")
            return
        parsed_inputs.append(_output_path)

        # note data path string variable name change
        _data_path = kwargs.get("data_path", os.getcwd())
        try:
            os.path.exists(_data_path)
        except FileNotFoundError:
            print("Could not locate specified data path")
        _data_path_string = "".join(["--datasets_path=", _data_path])
        parsed_inputs.append(_data_path_string)

        _test_data_size = "".join(["--test_datasize=", str(kwargs.get("length", 21000))])
        parsed_inputs.append(_test_data_size)

        if DataFile is None:
            AssertionError("No data file specified")
        try:
            _data_loc = "".join([_data_path, "\\", DataFile])
            os.path.exists(_data_loc)
        except FileNotFoundError:
            print("Specified data not located")
            return
        _data_file = "".join(["--datasets_folder=", DataFile])
        parsed_inputs.append(_data_file)

        _image_type = "".join(["--image_type=", kwargs.get("image_type", "binary")])
        parsed_inputs.append(_image_type)

        _vram = "".join(["--VRAM=", str(kwargs.get("vram", 24))])
        return parsed_inputs

    @staticmethod
    def construct_parser():
        # Let's construct their stuff
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
        parser.add_argument('--GPU', type=str, default='0', help="the index of GPU you will use for computation")
        parser.add_argument('--batch_size', type=int, default=1, help="batch size")
        parser.add_argument('--img_w', type=int, default=150, help="the width of image sequence")
        parser.add_argument('--img_h', type=int, default=150, help="the height of image sequence")
        parser.add_argument('--img_s', type=int, default=150, help="the slices of image sequence")
        parser.add_argument('--gap_w', type=int, default=60, help='the width of image gap')
        parser.add_argument('--gap_h', type=int, default=60, help='the height of image gap')
        parser.add_argument('--gap_s', type=int, default=60, help='the slices of image gap')
        parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')
        parser.add_argument("--b1", type=float, default=0.5, help="Adam: bata1")
        parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")
        parser.add_argument('--normalize_factor', type=int, default=1, help='normalize factor')
        parser.add_argument('--fmap', type=int, default=16, help='number of feature maps')
        parser.add_argument('--output_dir', type=str, default='./results', help="output directory")
        parser.add_argument('--datasets_path', type=str, default='datasets', help="dataset root path")
        parser.add_argument('--pth_path', type=str, default='pth', help="pth file root path")
        parser.add_argument('--datasets_folder', type=str, default='test',
                            help="A folder containing files to be tested")
        parser.add_argument('--denoise_model', type=str, default='train_20210401_1712',
                            help='A folder containing models to be tested')
        parser.add_argument('--test_datasize', type=int, default=300, help='dataset size to be tested')
        parser.add_argument('--train_datasets_size', type=int, default=100000, help='datasets size for training')

        # Let's add my stuff
        parser.add_argument('--image_type', type=str, default="binary", help="type/ext of image files")
        parser.add_argument('--VRAM', type=int, default=24, help="amount of VRAM in GPU")
        return parser

    @staticmethod
    def calculate_chunk_size(ImageLength, SizeLimit):
        """
        Function calculates the smallest divisor such that the chunk size
        is less than the VRAM of the GPU (SizeLimit)

        :param ImageLength: Number of imaging frames
        :type ImageLength: int
        :param SizeLimit: Amount of VRAM on GPU
        :type SizeLimit: int
        :return: chunk_size
        :rtype: int
        """
        if ImageLength % 2 == 0 and ImageLength/2 <= SizeLimit:
            return 2

        chunk_size = 3
        while chunk_size*chunk_size <= ImageLength:
            if ImageLength % chunk_size == 0 and ImageLength/chunk_size <= SizeLimit:
                return chunk_size
            chunk_size += 1

    @staticmethod
    def validate_size_limit(opt, XPixels, YPixels):
        _gpu_memory = opt.VRAM*1000000000
        _test_tensor = torch.FloatTensor(opt.test_datasize, XPixels, YPixels)
        if _gpu_memory*0.90 >= _test_tensor.element_size()*_test_tensor.nelement():
            return True
        else:
            return False
