import numpy as np
import os
import tifffile

dirname = "C:\\Users\\YUSTE\\Desktop\\AM001_PRE\\AM001_Pre_Imaging-002"

final = []

from tifffile import TiffFile
from tifffile import imread

A = os.listdir(dirname)
B = A[0]
_file = dirname + "\\" + B

exampleImage = TiffFile(_file)

for fname in os.listdir(dirname):
    tmp_image =