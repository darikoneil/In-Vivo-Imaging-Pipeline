import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns


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
    _offset = np.around(VideoData[0][0]/10, decimals=3)
    _video_trial_data = []
    for _trial in range(_num_trials):
        _trial_frames_idx = np.where(VideoData[1] == _trial)[0]
        _true_index = np.where(SyncData["Trial"].values == _trial)
        _trial_time = SyncData.index.values[_true_index].copy()
        _start, _end = _trial_time[0], np.around(_trial_time[-1]-(1/30), decimals=3)
        if _trial == 0:
            _start += _offset
        _video_time = np.around(np.linspace(_start, _end, _trial_frames_idx.__len__()), decimals=3)
        _video_series = pd.Series(np.arange(_trial_frames_idx[0], _trial_frames_idx[-1]+1,
                                                     1), index=_video_time).reindex(_trial_time)
        _video_series.sort_index(inplace=True)
        _video_series.ffill(inplace=True)
        _video_trial_data.append(_video_series.copy(deep=True))

    _video_trial_data = pd.concat(_video_trial_data)
    _video_trial_data.sort_index(inplace=True)
    _video_trial_data.name = "Video Frame"

    SyncData = SyncData.join(_video_trial_data)
    SyncData.sort_index(inplace=True)
    SyncData.ffill(inplace=True)

    return SyncData


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

    _cmap_id = kwargs.get("cmap", "rainbow")
    _alpha = kwargs.get("alpha", 0.9)
    _face_color = kwargs.get("facecolor", None)
    _fig_color = kwargs.get("figcolor", None)

    _trial_data = np.where(SyncData["Trial"] == Trial)
    cmap = matplotlib.cm.get_cmap(_cmap_id)
    _colors = cmap(np.linspace(0, 1, Keys.__len__()))
    _colors[:, 3] = _alpha
    # if Keys.__len__() == 1:
    #    _subplot_constructor_int = 1
    #else:
    #    _subplot_constructor_int = int("".join([str(Keys.__len__()+2), str(11)]))

    _fig = plt.figure(figsize=(16, 8))
    _gs = _fig.add_gridspec(Keys.__len__()+2, 6)

    _custom_legend_handles = []
    for _key_idx in range(Keys.__len__()+1):
        if _key_idx < Keys.__len__():
            _ax = _fig.add_subplot(_gs[_key_idx, :])
            #_subplot_constructor_int += 1
            _ax.plot(SyncData.index.values[_trial_data], SyncData[Keys[_key_idx]].values[_trial_data],
                     color=_colors[_key_idx, :])
            _ax.set_xlabel("Time (s)")
            _ax.set_ylabel(Keys[_key_idx])
            if _face_color is not None:
                _ax.set_facecolor(_face_color)
            _custom_legend_handles.append(matplotlib.patches.Patch(facecolor=_colors[_key_idx], label=Keys[_key_idx]))
        else:
            _ax = _fig.add_subplot(_gs[Keys.__len__():Keys.__len__()+2, 0:5])
            # _subplot_constructor_int += 1
            for _key_idx_inner in range(Keys.__len__()):
                _ax.plot(SyncData.index.values[_trial_data], SyncData[Keys[_key_idx_inner]].values[_trial_data],
                     color=_colors[_key_idx_inner, :])
            _ax.set_xlabel("Time (s)")
            _ax.set_ylabel("Merged")
            if _face_color is not None:
                _ax.set_facecolor(_face_color)

    if _face_color is not None:
        _fig.patch.set_facecolor(_fig_color)
    _fig.suptitle("".join(["Trial ", str(Trial)]), fontsize=24)
    _fig.legend(handles=_custom_legend_handles, loc="lower right")
    _fig.set_tight_layout("tight")


def main(RasPath):
    sync_data = load_sync_data(ras_path)
    video_data = load_video(ras_path)
    sync_data = sync_video(sync_data, video_data)
    plot_trial(sync_data, 0,
               ("Water Lick", "Water Reward", "Sucrose Lick", "Sucrose Reward"))
               #facecolor="black", figcolor="black")


if __name__ == "__main__":
    ras_path = "I:\\EM0084_Ras_Licking"
    main(ras_path)
