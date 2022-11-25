from importlib import import_module
import sys


def test_external_neuro_packages() -> None:
    _num_exceptions = 0

    _libnames = [
        "suite2p",
        "cellpose",
        "fissa",
        "bayes_opt",
    ]

    for _libname in _libnames:
        # noinspection PyBroadException
        try:
            _lib = import_module(_libname)
        except Exception:
            _num_exceptions += 1
            print(sys.exc_info())
        else:
            globals()[_libname] = _lib

    assert(_num_exceptions <= 0)


def test_gpu_packages() -> None:

    _num_exceptions = 0

    _libnames = [
        "cupy",
        "cupyx",
    ]
    for _libname in _libnames:
        # noinspection PyBroadException
        try:
            _lib = import_module(_libname)
        except Exception:
            _num_exceptions += 1
            print(sys.exc_info())
        else:
            globals()[_libname] = _lib

    assert(_num_exceptions <= 0)