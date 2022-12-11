from __future__ import annotations
from typing import Tuple, Union, Optional, List
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns
import numpy as np
import sys
import animatplot as amp
import scipy.ndimage


# compare traces need refactored and merged together


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


def interactive_traces(Traces: np.ndarray, FrameRate: float, **kwargs) -> None:
    _num_neurons, _num_frames = Traces.shape

    _line_width = kwargs.get("lw", 3)
    _alpha = kwargs.get("alpha", 0.95)

    x = np.arange(0, (_num_frames * (1 / FrameRate)), 1 / FrameRate, dtype=np.float64)

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Trace: Neuron 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal (a.u.)")
    ax1.plot(x, Traces[0, :], color="#40cc8b", lw=_line_width, alpha=_alpha)
    ax1.set_xlim([0, x[-1]])
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
        valstep=(1 / FrameRate),
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
            if n < Traces.shape[0] - 1:
                n += 1  # new n
                ax1.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, Traces[n, :], color="#40cc8b", lw=_line_width, alpha=_alpha)
                ax1.set_xlim([0, x[-1]])
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
                ax1.plot(x, Traces[n, :], color="#40cc8b", lw=_line_width, alpha=_alpha)

                n += 1
                ax1.set_title("Trace: Neuron " + str(n))
                ax1.set_xlim([0, x[-1]])
                fig.canvas.draw()

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


def view_image(Images: np.ndarray, FPS: float, **kwargs: Union[str, int]) -> \
            List[object]:
    """
    Visualize a numpy array [Z x Y x X] as a video

    :param Images: A numpy array [Z x Y x X]
    :type Images: Any
    :param FPS: Frames Per Second
    :type FPS: float
    :keyword cmap: colormap (str, default binary_r)
    :keyword interpolation: interpolation method (str, default none)
    :keyword SpeedUp: FPS multiplier (int, default 1)
    :keyword  Vmin: minimum value of colormap (int, default 0)
    :keyword Vmax: maximum value of colormap (int, default 32000)
    :return: Figure Animation
    :rtype: list[matplotlib.pyplot.figure, matplotlib.pyplot.axes, matplotlib.pyplot.axes,
    matplotlib.pyplot.axes, Any, Any]
    """
    _cmap = kwargs.get('cmap', "binary_r")
    _interp = kwargs.get('interpolation', "none")
    _fps_multi = kwargs.get('SpeedUp', 1)
    _vmin = kwargs.get('Vmin', 0)
    _vmax = kwargs.get('Vmax', 32000)

    _new_fps = _fps_multi * FPS
    TiffStack = Images.astype(np.uint16)
    frames = TiffStack.shape[0]
    _start = 0
    _stop = (1 / FPS) * frames
    _step = 1 / FPS
    _time_stamps = np.arange(_start, _stop, _step)
    _timeline = amp.Timeline(_time_stamps, units="s", fps=_new_fps)

    fig1 = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((30, 30), (0, 0), rowspan=28, colspan=28, xticks=[], yticks=[])
    ax2 = plt.subplot2grid((30, 30), (29, 0), colspan=21)
    ax3 = plt.subplot2grid((30, 30), (29, 25), colspan=3)

    block = amp.blocks.Imshow(TiffStack, ax1, cmap=_cmap, vmin=_vmin, vmax=_vmax, interpolation=_interp)
    anim = amp.Animation([block], timeline=_timeline)
    anim.timeline_slider(text='Time', ax=ax2, color="#139fff")
    anim.toggle(ax=ax3)
    plt.show()
    return [fig1, ax1, ax2, ax3, block, anim]


def plotROC(TPR, FPR, **kwargs):
    _auc = kwargs.get('AUC', None)
    _ax = kwargs.get('ax', None)
    _color = kwargs.get('color', "#139fff")
    _ls = kwargs.get('ls', "-")
    _alpha = kwargs.get('alpha', 0.95)

    if _ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(111)
        ax1.plot(TPR, FPR, color=_color, lw=1, alpha=0.95, ls=_ls)
        ax1.set_title("Receiver Operating Characteristic")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_xlim([-0.01, 1.01])
        ax1.set_ylim([-0.01, 1.01])

        # Baseline
        ax1.plot([0, 1], [0, 1], color="black", lw=1, alpha=_alpha, ls="--")
        plt.show()
    else:
        _ax.plot(TPR, FPR, color=_color, lw=3, ls=_ls, alpha=_alpha)
        _ax.set_xlim([-0.01, 1.01])
        _ax.set_ylim([-0.01, 1.01])


def plotNoise(Traces, FrameRate):
    # plt.figure(1)
    from cascade2p.utils import plot_noise_level_distribution
    _noise_levels = plot_noise_level_distribution(Traces, FrameRate)


def plotSpikeInference(SpikeProb, SpikeTimes, Traces, FrameRate, **kwargs):
    from cascade2p.utils import plot_dFF_traces
    NeuronalSubset = kwargs.get('NeuronalSubset', np.random.randint(Traces.shape[0], size=10))
    TemporalSubset = kwargs.get('TemporalSubset', [0, FrameRate * 30])
    plot_dFF_traces(Traces[:, int(TemporalSubset[0]):int(TemporalSubset[1])], NeuronalSubset, FrameRate,
                    spiking=SpikeProb, discrete_spikes=SpikeTimes)


def assessSpikeInference(SpikeProb, SpikeTimes, Traces, FrameRate):
    _frames = SpikeProb.shape[1]
    x = np.arange(0, ( _frames * (1 / FrameRate)), 1 / FrameRate, dtype=np.float64)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_title("Spike Inference: Neuron 1")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Δf/f0 (a.u.)")
    ax.plot(x, Traces[0, :], color="#139fff", lw=1, alpha=0.95)
    ax.plot(x, SpikeProb[0, :]-1, color="#ff8a00", lw=1, alpha=0.95)
    ax.plot(np.array([SpikeTimes[0], SpikeTimes[0]])/FrameRate+1/FrameRate, [-1.4, -1.2], 'k')
    ax.set_ylim([-1.5, 2])

    fig.subplots_adjust(bottom=0.25, hspace=0.5)
    xmin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    xmin_slider = Slider(
        ax=xmin,
        label="X-Min",
        valmin=0,
        valmax=x[-1],
        valinit=0,
        valstep=(1/FrameRate),
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
        ax.set_xlim([xmin_slider.val, xmax_slider.val])
        fig.canvas.draw_idle()

    xmin_slider.on_changed(update)
    xmax_slider.on_changed(update)

    def on_key(event):
        sys.stdout.flush()
        # print("press", event.key)
        if event.key == "up":
            # print('press', event.key)
            n = int(ax.title._text.split()[3]) - 1  # ZERO INDEXED
            if n < Traces.shape[0] - 1:
                n += 1  # new n
                ax.clear()

                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Δf/f0 (a.u.)")

                ax.plot(x, Traces[n, :], color="#139fff", lw=1, alpha=0.95)
                ax.plot(x, SpikeProb[n, :] - 1, color="#ff8a00", lw=1, alpha=0.95)
                ax.plot(np.array([SpikeTimes[n], SpikeTimes[n]]) / FrameRate + 1 / FrameRate, [-1.4, -1.2], 'k')
                ax.set_ylim([-1.5, 2])

                n += 1

                ax.set_title("Spike Inference: Neuron " + str(n))

                fig.canvas.draw()

        elif event.key == "down":
            # print('press', event.key)
            n = int(ax.title._text.split()[3]) - 1  # ZERO INDEXED
            if n > 0:
                n -= 1
                ax.clear()

                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Δf/f0 (a.u.)")

                ax.plot(x, Traces[n, :], color="#139fff", lw=1, alpha=0.95)
                ax.plot(x, SpikeProb[n, :] - 1, color="#ff8a00", lw=1, alpha=0.95)
                ax.plot(np.array([SpikeTimes[n], SpikeTimes[n]]) / FrameRate + 1 / FrameRate, [-1.4, -1.2], 'k')
                ax.set_ylim([-1.5, 2])

                n += 1

                ax.set_title("Spike Inference: Neuron " + str(n))

                fig.canvas.draw()
        elif event.key == "right":
            print(event.key)
            if xmin_slider < xmin_slider.valmax:
                xmin_slider.val += xmin_slider.valstep
                ax.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()
        elif event.key == "left":
            print(event.key)
            if xmin_slider.val > xmin_slider.valmin:
                xmin_slider.val -= xmin_slider.valstep
                ax.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def plotFiringRateMatrix(FiringRates, FrameRate, **kwargs):
    _neuronal_subset = kwargs.get('NeuronalSubset', None)
    _neuronal_index = kwargs.get('NeuronalIndex', None)
    _temporal_subset = kwargs.get('TemporalSubset', None)
    step_size = kwargs.get('StepSize', int(600))
    FrameRate = int(FrameRate)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    sns.heatmap(FiringRates, ax=ax1, cmap="Spectral_r")
    ax1.set_xticks((range(0, int(FiringRates.shape[1]), int(step_size))),
                   labels=(range(0, int(FiringRates.shape[1]/FrameRate),
                                 int(step_size/FrameRate))))
    ax1.set_yticks([0, int(FiringRates.shape[0])], labels=[0, int(FiringRates.shape[0])])
    ax1.set_title("Firing Rate Map")
    plt.show()
