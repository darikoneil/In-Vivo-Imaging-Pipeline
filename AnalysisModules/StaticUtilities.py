import ruamel.yaml as yaml
from cascade2p import cascade
import tensorflow as tf
from tensorflow.python.client import device_lib


def pullModels(ModelFolder):
    cascade.download_model('update_models', verbose=1, model_folder=ModelFolder)
    _available_models = ModelFolder + "\\available_models.yaml"
    yaml_file = open(_available_models)
    X = yaml.load(yaml_file, Loader=yaml.Loader)
    list_of_models = list(X.keys())
    print('\n List of available models: \n')
    for model in list_of_models:
        print(model)
    return list_of_models


def downloadModel(ModelName, ModelFolder):
    cascade.download_model(ModelName, verbose=1, model_folder=ModelFolder)


def setVerboseGPU():
    tf.debugging.set_log_device_placement(True)


def confirmGPU():
    print("Here are the local devices...")
    print(device_lib.list_local_devices())

    print("Is Tensorflow Built with CUDA...")
    tf.test.is_built_with_cuda()

    print("Is GPU Available for Use...")
    tf.test.is_gpu_available()

