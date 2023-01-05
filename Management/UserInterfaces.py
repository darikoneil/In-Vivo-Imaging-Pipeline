import os.path
import tkinter as tk
from tkinter.filedialog import askdirectory
from shutil import copytree, copy2, rmtree
from tqdm import tqdm
import pathlib


def select_directory(**kwargs) -> str:
    # Make Root
    root = tk.Tk()

    # collect path & format
    # noinspection PyArgumentList
    path = askdirectory(**kwargs)
    path = str(pathlib.Path(path))

    # destroy root
    root.destroy()
    return path


def verbose_copying(src, dst) -> None:

    # Make sure src is a pathlib.Path object for rglob and dst is a str for copy2
    if isinstance(src, str):
        src = pathlib.Path(src)
    if isinstance(dst, pathlib.Path):
        dst = str(dst)

    # Make sure destination is not the source
    if src == dst:
        raise ValueError("Destination and source files are identical")

    _num_files = sum([1 for _file in src.rglob("*") if _file.is_file()])
    _pbar = tqdm(total=_num_files)
    _pbar.set_description("Copying files...")

    if os.path.exists(dst):
        rmtree(dst)

    def verbose_copy(_src, _dst):
        copy2(_src, _dst)
        _pbar.update(1)

    copytree(src, dst, copy_function=verbose_copy)
