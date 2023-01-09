import os
import pytest
from shutil import rmtree
from Imaging.IO import determine_bruker_folder_contents, repackage_bruker_tiffs

FIXTURE_DIR = "".join([os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "\\TestingData"])

DATASET = pytest.mark.datafiles(
    "".join([FIXTURE_DIR, "\\TestData_4p_1c_256h_256w_6609f_26436t"]),
    keep_top_dir=False,
    on_duplicate="ignore",
)


@DATASET
def test_determine_bruker_folder_channels(datafiles):
    assert(determine_bruker_folder_contents(str(datafiles))[0] == 1)
    rmtree(datafiles)


@DATASET
def test_determine_bruker_folder_planes(datafiles):
    assert (determine_bruker_folder_contents(str(datafiles))[1] == 4)
    rmtree(datafiles)


@DATASET
def test_determine_bruker_folder_frames(datafiles):
    assert (determine_bruker_folder_contents(str(datafiles))[2] == 6609)
    rmtree(datafiles)


@DATASET
def test_determine_bruker_folder_height(datafiles):
    assert (determine_bruker_folder_contents(str(datafiles))[3] == 256)
    rmtree(datafiles)


@DATASET
def test_determine_bruker_folder_width(datafiles):
    assert (determine_bruker_folder_contents(str(datafiles))[4] == 256)
    rmtree(datafiles)


@DATASET
def test_repackage_bruker_tiffs_default(datafiles, tmp_path):
    input_folder = str(datafiles)
    output_folder = "".join([str(tmp_path), "\\output"])
    os.mkdir(output_folder)
    repackage_bruker_tiffs(input_folder, output_folder)
    rmtree(datafiles)


@DATASET
def test_repackage_bruker_tiffs_single_plane(datafiles, tmp_path):
    input_folder = str(datafiles)
    output_folder = "".join([str(tmp_path), "\\output"])
    os.mkdir(output_folder)
    repackage_bruker_tiffs(input_folder, output_folder, 0)
    rmtree(datafiles)
