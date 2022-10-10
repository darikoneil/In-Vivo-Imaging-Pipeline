import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns

ras_path = "I:\\EM0084_Ras_Licking"


def convert_to_csv(SyncData, NewFilename, RasPath):
    """
    Converts SyncData to csv

    :param SyncData: synchronized data
    :type SyncData: pd.DataFrame
    :param NewFilename: new filename
    :type NewFilename: str
    :param RasPath: folder with data
    :type RasPath: str
    :return: 1 if successful
    :rtype: int
    """
    if not isinstance(SyncData, pd.DataFrame):
        raise AssertionError("sync_data must be a pandas dataframe")

    if not isinstance(NewFilename, str):
        raise AssertionError("Must be a string which indicates the new filename")

    if not isinstance(RasPath, str):
        raise AssertionError("RasPath must be a string which indicates the data folder")

    SyncData.to_csv("".join([RasPath, "\\", NewFilename]))

    return 1


def load_video(RasPath):
    """
    Function loads video

    :param RasPath: path to folder
    :type RasPath: str
    :return: List containing buffer_ids, frame_ids, video_meta, video_frames
    :rtype: list
    """

    if not isinstance(RasPath, str):
        raise AssertionError("RasPath must be a string which indicates the data folder")

    buffer_ids = np.load("".join([RasPath, "\\_cam2__BufferIDs.npy"]))

    frame_ids = np.load("".join([RasPath, "\\_cam2__FramesIDS.npy"]))

    video_meta = np.genfromtxt("".join([RasPath, "\\_cam2__meta.txt"]), delimiter=",", dtype=int)

    video_frames = np.fromfile("".join([RasPath, "\\_cam2__Frame.npy"]), dtype=np.uint8)

    video_frames = video_frames.reshape(video_meta[0], video_meta[1], video_meta[2])

    return [buffer_ids, frame_ids, video_meta, video_frames]


def load_sync_data(RasPath):
    """
    Function loads sync data

    :param RasPath: folder with data
    :type RasPath: str
    :return: sync_data
    :rtype: pd.DataFrame
    """

    if not isinstance(RasPath, str):
        raise AttributeError("RasPath must be a string which indicates the data folder")

    file = "".join([RasPath, "\\EM0084_Licking_Training"])
    with open(file, "rb") as f:
        sync_data = pkl.load(f)
    return sync_data


def load_full(RasPath):
    """
    Loads all data

    :param RasPath: folder with data
    :type RasPath: str
    :return: all_data (Stats, Analog, Digital, State, buffer_ids, frame_ids, video_meta, video_frames, AnalogData)
    :rtype: list
    """
    if not isinstance(RasPath, str):
        raise AttributeError("RasPath must be a string which indicates the data folder")

    Analog = np.load("".join([RasPath, "\\analog.npy"]),
                     allow_pickle=True)

    Digital = np.load("".join([RasPath, "\\digital.npy"]),
                     allow_pickle=True)

    State = np.load("".join([RasPath, "\\state.npy"]),
                     allow_pickle=True)

    with open("".join([RasPath, "\\stats"]), "rb") as f:
        Stats = pkl.load(f)
    f.close()

    buffer_ids = np.load("".join([RasPath, "\\_cam2__BufferIDs.npy"]))

    frame_ids = np.load("".join([RasPath, "\\_cam2__FramesIDS.npy"]))

    video_meta = np.genfromtxt("".join([RasPath, "\\_cam2__meta.txt"]), delimiter=",", dtype=int)

    video_frames = np.fromfile("".join([RasPath, "\\_cam2__Frame.npy"]), dtype=np.uint8)

    video_frames = video_frames.reshape(video_meta[0], video_meta[1], video_meta[2])

    # with open("".join([RasPath, "\\config"]), "rb") as g:
    #    Config = pkl.load(g)
    # g.close()

    _time_vector_1000Hz = np.around(np.arange(0, Analog.shape[1] * (1 / 1000), 1 / 1000, dtype=np.float64), decimals=3)
    _time_vector_10Hz = np.around(np.arange(0, State.__len__() * (1 / 10), 1 / 10, dtype=np.float64), decimals=3)

    AnalogData = pd.DataFrame(Analog.T, index=_time_vector_1000Hz, columns=["Image Sync", "Motor Position", "Force", "Dummy"])
    AnalogData.index.name = "Time (s)"

    DigitalData = pd.DataFrame(Digital.T, index=_time_vector_1000Hz, columns=["Gate Trigger", "Sucrose Reward", "Water Reward", "Sucrose Lick", "Water Lick"])
    DigitalData.index.name = "Time (s)"

    StateData = pd.Series(State.astype(int), index=_time_vector_10Hz)
    StateData.index.name = "Time (s)"
    StateData.name = "Trial"
    StateData = StateData.reindex(index=_time_vector_1000Hz)
    StateData.ffill(inplace=True)

    AnalogData = AnalogData.join(DigitalData)
    AnalogData = AnalogData.join(StateData)
    AnalogData.ffill(inplace=True)

    return [Stats, Analog, Digital, State, buffer_ids, frame_ids, video_meta, video_frames, AnalogData]


def sync_video(SyncData, VideoData, **kwargs):
    """
    Synchronizes video with data

    :param SyncData: Synchronized Data
    :type SyncData: pd.DataFrame
    :param VideoData: video data
    :type: list
    :return: synchronized data containing video data
    :rtype: pd.DataFrame
    """
    _fps = kwargs.get("fps", 30)
    _time_vector_full = SyncData.index.values
    _num_trials = np.unique(SyncData["Trial"].values).__len__()



def plot_trial(SyncData, Trial, Keys, **kwargs):
    """
    Function plots a trial

    :param SyncData: synchronized data
    :type SyncData: pd.DataFrame
    :param Trial: desired trial
    :type Trial: int
    :param Keys: Desired columns to plot
    :type Keys: tuple
    :rtype: None
    """

    cmap_id = kwargs.get("cmap", "mako")
    _trial_data = np.where(SyncData["Trial"] == Trial)
    cmap = matplotlib.cm.get_cmap(cmap_id)
    _colors = cmap(np.arange(0, 1, Keys.__len__()))

    _subplot_constructor_int = int("".join([str(Keys.__len__()), str(11)]))
    fig1 = plt.figure(figsize=(12, 6))

    for _key_idx in range(Keys.__len__()):
        _ax = fig1.add_subplot(_subplot_constructor_int)
        _subplot_constructor_int += 1
        _ax.plot(SyncData.index.values[_trial_data], SyncData[Keys[_key_idx]].values[_trial_data], color=_colors[_key_idx])
        _ax.set_xlabel("Time (s)")
        _ax.set_ylabel(Keys[_key_idx])
        _ax.title.set_text("".join(["Trial ", str(Trial)]))


sync_data = load_sync_data(ras_path)
video_data = load_video(ras_path)




