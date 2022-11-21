from __future__ import annotations
from typing import Tuple, Union, Optional, List
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns
import numpy as np
import sys


def compareTraces(RawTraces: np.ndarray, SmoothTraces: np.ndarray, FrameRate: float, Frames: int) -> None:
    """
    Compare two sets of traces interactively

    :param RawTraces: Trace Set 1
    :type RawTraces: Any
    :param SmoothTraces: Trace Set 2
    :type SmoothTraces: Any
    :param FrameRate: FrameRate
    :type FrameRate: float
    :param Frames: Number of Frames
    :type Frames: int
    :rtype: None
    """

    x = np.arange(0, (Frames * (1 / FrameRate)), 1 / FrameRate, dtype=np.float64)

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Trace: Neuron 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal (a.u.)")
    ax1.plot(x, np.concatenate(RawTraces[0], axis=1)[0, :], color="#40cc8b", lw=3, alpha=0.95)
    ax1.plot(x, np.concatenate(SmoothTraces[0], axis=1)[0, :], color="#004c99", lw=3, alpha=0.95)

    fig.subplots_adjust(bottom=0.25, hspace=0.5)
    xmin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    xmin_slider = Slider(
        ax=xmin,
        label="X-Min",
        valmin=0,
        valmax=x[-1],
        valinit=0,
        valstep=(1 / FrameRate),
        color="#7840cc"
    )

    xmax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    xmax_slider = Slider(
        ax=xmax,
        label="X-Max",
        valmin=0,
        valmax=x[-1],
        valinit=x[-1],
        valstep=(1/FrameRate),
        color="#7840cc"
    )

    def update(val):
        ax1.set_xlim([xmin_slider.val, xmax_slider.val])
        fig.canvas.draw_idle()

    xmin_slider.on_changed(update)
    xmax_slider.on_changed(update)

    def on_key(event):
        sys.stdout.flush()
        # print("press", event.key)
        if event.key == "up":
            # print('press', event.key)
            n = int(ax1.title._text.split()[2]) - 1  # ZERO INDEXED
            if n < RawTraces.shape[0] - 1:
                n += 1  # new n
                ax1.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, np.concatenate(RawTraces[n], axis=1)[0, :], color="#40cc8b", lw=3, alpha=0.95)
                ax1.plot(x, np.concatenate(SmoothTraces[n], axis=1)[0, :], color="#004c99", lw=3, alpha=0.95)

                n += 1
                ax1.set_title("Trace: Neuron " + str(n))
                fig.canvas.draw()

        elif event.key == "down":
            # print('press', event.key)
            n = int(ax1.title._text.split()[2]) - 1  # ZERO INDEXED
            if n > 0:
                n -= 1  # new n
                ax1.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, np.concatenate(RawTraces[n], axis=1)[0, :], color="#40cc8b", lw=3, alpha=0.95)
                ax1.plot(x, np.concatenate(SmoothTraces[n], axis=1)[0, :], color="#004c99", lw=3, alpha=0.95)

                n += 1
                ax1.set_title("Trace: Neuron " + str(n))

                fig.canvas.draw()
        elif event.key == "right":
            print(event.key)
            if xmin_slider < xmin_slider.valmax:
                xmin_slider.val += xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()

        elif event.key == "left":
            print(event.key)
            if xmin_slider.val > xmin_slider.valmin:
                xmin_slider.val -= xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def compareTraces2(RawTraces: np.ndarray, SmoothTraces: np.ndarray, FrameRate: float, Frames: int) -> None:
    """
    Compare two sets of traces interactively

    :param RawTraces: Trace Set 1
    :type RawTraces: Any
    :param SmoothTraces: Trace Set 2
    :type SmoothTraces: Any
    :param FrameRate: FrameRate
    :type FrameRate: float
    :param Frames: Number of Frames
    :type Frames: int
    :rtype: None
    """

    x = np.arange(0, (Frames * (1 / FrameRate)), 1 / FrameRate, dtype=np.float64)

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Trace: Neuron 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal (a.u.)")
    ax1.plot(x, RawTraces[0, :], color="#40cc8b", lw=3, alpha=0.95)
    ax1.plot(x, np.concatenate(SmoothTraces[0], axis=1)[0, :], color="#004c99", lw=3, alpha=0.95)

    fig.subplots_adjust(bottom=0.25, hspace=0.5)
    xmin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    xmin_slider = Slider(
        ax=xmin,
        label="X-Min",
        valmin=0,
        valmax=x[-1],
        valinit=0,
        valstep=(1 / FrameRate),
        color="#7840cc"
    )

    xmax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    xmax_slider = Slider(
        ax=xmax,
        label="X-Max",
        valmin=0,
        valmax=x[-1],
        valinit=x[-1],
        valstep=(1/FrameRate),
        color="#7840cc"
    )

    def update(val):
        ax1.set_xlim([xmin_slider.val, xmax_slider.val])
        fig.canvas.draw_idle()

    xmin_slider.on_changed(update)
    xmax_slider.on_changed(update)

    def on_key(event):
        sys.stdout.flush()
        # print("press", event.key)
        if event.key == "up":
            # print('press', event.key)
            n = int(ax1.title._text.split()[2]) - 1  # ZERO INDEXED
            if n < RawTraces.shape[0] - 1:
                n += 1  # new n
                ax1.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, RawTraces[n, :], color="#40cc8b", lw=3, alpha=0.95)
                ax1.plot(x, np.concatenate(SmoothTraces[n], axis=1)[0, :], color="#004c99", lw=3, alpha=0.95)

                n += 1
                ax1.set_title("Trace: Neuron " + str(n))
                fig.canvas.draw()

        elif event.key == "down":
            # print('press', event.key)
            n = int(ax1.title._text.split()[2]) - 1  # ZERO INDEXED
            if n > 0:
                n -= 1  # new n
                ax1.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, RawTraces[n, :], color="#40cc8b", lw=3, alpha=0.95)
                ax1.plot(x, np.concatenate(SmoothTraces[n], axis=1)[0, :], color="#004c99", lw=3, alpha=0.95)

                n += 1
                ax1.set_title("Trace: Neuron " + str(n))

                fig.canvas.draw()
        elif event.key == "right":
            print(event.key)
            if xmin_slider < xmin_slider.valmax:
                xmin_slider.val += xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()

        elif event.key == "left":
            print(event.key)
            if xmin_slider.val > xmin_slider.valmin:
                xmin_slider.val -= xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def compareTraces3(RawTraces: np.ndarray, SmoothTraces: np.ndarray, FrameRate: float, Frames: int) -> None:
    """
    Compare two sets of traces interactively

    :param RawTraces: Trace Set 1
    :type RawTraces: Any
    :param SmoothTraces: Trace Set 2
    :type SmoothTraces: Any
    :param FrameRate: FrameRate
    :type FrameRate: float
    :param Frames: Number of Frames
    :type Frames: int
    :rtype: None
    """

    x = np.arange(0, (Frames * (1 / FrameRate)), 1 / FrameRate, dtype=np.float64)

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Trace: Neuron 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal (a.u.)")
    ax1.plot(x, RawTraces[0, :], color="#40cc8b", lw=3, alpha=0.95)
    ax1.plot(x, SmoothTraces[0, :], color="#004c99", lw=3, alpha=0.95)

    fig.subplots_adjust(bottom=0.25, hspace=0.5)
    xmin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    xmin_slider = Slider(
        ax=xmin,
        label="X-Min",
        valmin=0,
        valmax=x[-1],
        valinit=0,
        valstep=(1 / FrameRate),
        color="#7840cc"
    )

    xmax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    xmax_slider = Slider(
        ax=xmax,
        label="X-Max",
        valmin=0,
        valmax=x[-1],
        valinit=x[-1],
        valstep=(1/FrameRate),
        color="#7840cc"
    )

    def update(val):
        ax1.set_xlim([xmin_slider.val, xmax_slider.val])
        fig.canvas.draw_idle()

    xmin_slider.on_changed(update)
    xmax_slider.on_changed(update)

    def on_key(event):
        sys.stdout.flush()
        # print("press", event.key)
        if event.key == "up":
            # print('press', event.key)
            n = int(ax1.title._text.split()[2]) - 1  # ZERO INDEXED
            if n < RawTraces.shape[0] - 1:
                n += 1  # new n
                ax1.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, RawTraces[n, :], color="#40cc8b", lw=3, alpha=0.95)
                ax1.plot(x, SmoothTraces[n, :], color="#004c99", lw=3, alpha=0.95)

                n += 1
                ax1.set_title("Trace: Neuron " + str(n))
                fig.canvas.draw()

        elif event.key == "down":
            # print('press', event.key)
            n = int(ax1.title._text.split()[2]) - 1  # ZERO INDEXED
            if n > 0:
                n -= 1  # new n
                ax1.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, RawTraces[n, :], color="#40cc8b", lw=3, alpha=0.95)
                ax1.plot(x, SmoothTraces[n, :], color="#004c99", lw=3, alpha=0.95)

                n += 1
                ax1.set_title("Trace: Neuron " + str(n))

                fig.canvas.draw()
        elif event.key == "right":
            print(event.key)
            if xmin_slider < xmin_slider.valmax:
                xmin_slider.val += xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()

        elif event.key == "left":
            print(event.key)
            if xmin_slider.val > xmin_slider.valmin:
                xmin_slider.val -= xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def plotNeuralHeatMap(NeuralActivity: np.ndarray, FrameRate: float,
                      *args: Optional[Tuple[Union[List[int], Tuple[int, int]]]]) -> None:

    plt.rc("font", size=4)
    plt.rc("axes", titlesize=4)
    plt.rc("axes", labelsize=4)
    plt.rc("axes", linewidth=1)
    plt.rc("xtick", labelsize=4)
    plt.rc("ytick", labelsize=4)
    plt.rc("figure", titlesize=6)
    plt.rc("xtick.major", width=0.25)
    plt.rc("ytick.major", width=0.25)

    _figure = plt.figure(dpi=600)
    _axes = _figure.add_subplot(111)

    if args:
        NeuralSubset = args[0]
        if isinstance(NeuralSubset, np.ndarray):
            NeuralSubset = list(NeuralSubset)
        NeuralSubset = NeuralActivity[NeuralSubset, :]
        if len(args) == 2:
            TemporalSubset = args[1]
            NeuralSubset = NeuralSubset[:, TemporalSubset[0]:TemporalSubset[1]]
        else:
            TemporalSubset = tuple([0, NeuralSubset.shape[1]])
    else:
        NeuralSubset = NeuralActivity[:, :]
        TemporalSubset = tuple([0, NeuralSubset.shape[1]])

    _time_vector = np.arange(0, NeuralSubset.shape[1]/FrameRate, 1/FrameRate)

    _heatmap = sns.heatmap(NeuralSubset, cmap="Spectral_r", ax=_axes)
    _axes.set_yticks([0, NeuralSubset.shape[0]], labels=[0, NeuralSubset.shape[0]])
    _axes.set_xticks([])
    _axes.set_xlabel("Time (s)")
    _axes.set_ylabel("Neuron (#)")
    plt.tight_layout()
