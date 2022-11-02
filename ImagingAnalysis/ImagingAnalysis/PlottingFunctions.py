import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import sys


def compareRawResult(Module, Neuron, Limits):

    Raw = np.concatenate(Module.preparation['raw'][Neuron], axis=1)[0, :]
    Result = np.concatenate(Module.experiment['result'][Neuron], axis=1)[0, :]
    fig= plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.set_title("Raw")
    ax1.plot(Raw, color="blue")
    ax1.set_xlim(Limits)

    ax2.set_title("Result")
    ax2.plot(Result, color="red")
    ax2.set_xlim(Limits)


#def compareRawResult_DFOF(Module, Neuron, Limits):


def plotResult_DFOF(Module, Neuron, Limits):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Δf/f0 Result")
    ax1.plot(Module.neuronal_dFoF_results[Neuron, :], color="orange")
    ax1.set_xlim(Limits)


def interactivePlot1(RawTraces, SmoothTraces, FrameRate, Frames):
    x = np.arange(0, (Frames * (1 / FrameRate)), 1 / FrameRate, dtype=np.float64)

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Trace: Neuron 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal (a.u.)")
    ax1.plot(x, np.concatenate(RawTraces[0], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)
    ax1.plot(x, np.concatenate(SmoothTraces[0], axis=1)[0, :], color="#004c99", lw=1, alpha=0.95)

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
                ax1.plot(x, np.concatenate(RawTraces[n], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)
                ax1.plot(x, np.concatenate(SmoothTraces[n], axis=1)[0, :], color="#004c99", lw=1, alpha=0.95)

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
                ax1.plot(x, np.concatenate(RawTraces[n], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)
                ax1.plot(x, np.concatenate(SmoothTraces[n], axis=1)[0, :], color="#004c99", lw=1, alpha=0.95)

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


def interactivePlot2_Merged(RawTraces, MergedSmoothTraces, FrameRate, Frames):
    x = np.arange(0, (Frames * (1 / FrameRate)), 1 / FrameRate, dtype=np.float64)

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(211)
    ax1.set_title("Raw Trace: Neuron 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal (a.u.)")
    ax1.plot(x, np.concatenate(RawTraces[0], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)

    ax2 = fig.add_subplot(212)
    ax2.set_title("Smoothed Trace: Neuron 1")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Signal (a.u.)")
    ax2.plot(x, MergedSmoothTraces[0, :], color="#004c99", lw=1, alpha=0.95)

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
        ax2.set_xlim([xmin_slider.val, xmax_slider.val])
        fig.canvas.draw_idle()

    xmin_slider.on_changed(update)
    xmax_slider.on_changed(update)

    def on_key(event):
        sys.stdout.flush()
        # print("press", event.key)
        if event.key == "up":
            # print('press', event.key)
            n = int(ax1.title._text.split()[3]) - 1  # ZERO INDEXED
            if n < RawTraces.shape[0] - 1:
                n += 1  # new n
                ax1.clear()
                ax2.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, np.concatenate(RawTraces[n], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)

                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Signal (a.u.)")
                ax2.plot(x, MergedSmoothTraces[n, :], color="#004c99", lw=1, alpha=0.95)

                n += 1
                ax1.set_title("Raw Trace: Neuron " + str(n))
                ax2.set_title("Smoothed Trace: Neuron " + str(n))
                fig.canvas.draw()

        elif event.key == "down":
            # print('press', event.key)
            n = int(ax1.title._text.split()[3]) - 1  # ZERO INDEXED
            if n > 0:
                n -= 1  # new n
                ax1.clear()
                ax2.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, np.concatenate(RawTraces[n], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)

                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Signal (a.u.)")
                ax2.plot(x, MergedSmoothTraces[n, :], color="#004c99", lw=1, alpha=0.95)

                n += 1
                ax1.set_title("Raw Trace: Neuron " + str(n))
                ax2.set_title("Smooth Trace: Neuron " + str(n))

                fig.canvas.draw()
        elif event.key == "right":
            print(event.key)
            if xmin_slider < xmin_slider.valmax:
                xmin_slider.val += xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                ax2.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()

        elif event.key == "left":
            print(event.key)
            if xmin_slider.val > xmin_slider.valmin:
                xmin_slider.val -= xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                ax2.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def interactivePlot5(Traces1, Traces2, Traces3, Traces4, Traces5, FrameRate, Frames):
    x = np.arange(0, (Frames*(1/FrameRate)), 1/FrameRate, dtype=np.float64)

    fig = plt.figure(figsize=(16, 12))

    #ax1
    ax1 = fig.add_subplot(311)
    ax1.set_title("Raw Trace: Neuron 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal (a.u.)")
    ax1.plot(x, np.concatenate(Traces1[0], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)
    ax1.plot(x, Traces4[0, :], color="#004c99", lw=1, alpha=0.95)

    #ax2
    ax2 = fig.add_subplot(312)
    ax2.set_title("Source-Separated Trace: Neuron 1")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Signal (a.u.)")
    ax2.plot(x, np.concatenate(Traces2[0], axis=1)[0, :], color="#139fff", lw=1, alpha=0.95)
    ax2.plot(x, Traces5[0, :], color="#ff8a00", lw=1, alpha=0.95)

    #ax3
    ax3 = fig.add_subplot(313)
    ax3.set_title("Source-Separated Δf/f0: Neuron 1")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Signal (a.u.)")
    ax3.plot(x, np.concatenate(Traces3[0], axis=1)[0, :], color="#ff4e4b", lw=1, alpha=0.95)

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
        ax1.set_xlim([xmin_slider.val, xmax_slider.val])
        ax2.set_xlim([xmin_slider.val, xmax_slider.val])
        ax3.set_xlim([xmin_slider.val, xmax_slider.val])
        fig.canvas.draw_idle()

    xmin_slider.on_changed(update)
    xmax_slider.on_changed(update)

    def on_key(event):
        sys.stdout.flush()
        # print("press", event.key)
        if event.key == "up":
            # print('press', event.key)
            n = int(ax1.title._text.split()[3]) - 1  # ZERO INDEXED
            if n < Traces1.shape[0] - 1:
                n += 1  # new n
                ax1.clear()
                ax2.clear()
                ax3.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, np.concatenate(Traces1[n], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)
                ax1.plot(x, Traces4[n, :], color="#004c99", lw=1, alpha=0.95)

                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Signal (a.u.)")
                ax2.plot(x, np.concatenate(Traces2[n], axis=1)[0, :], color="#139fff", lw=1, alpha=0.95)
                ax2.plot(x, Traces5[n, :], color="#ff8a00", lw=1, alpha=0.95)

                ax3.set_xlabel("Time (s)")
                ax3.set_ylabel("Signal (a.u.)")
                ax3.plot(x, np.concatenate(Traces3[n], axis=1)[0, :], color="#ff4e4b", lw=1, alpha=0.95)

                n += 1
                ax1.set_title("Raw Trace: Neuron " + str(n))
                ax2.set_title("Source-Separated Trace: Neuron " + str(n))
                ax3.set_title("Source-Separated Δf/f0: Neuron " + str(n))

                fig.canvas.draw()
        elif event.key == "down":
            # print('press', event.key)
            n = int(ax1.title._text.split()[3]) - 1  # ZERO INDEXED
            if n > 0:
                n -= 1  # new n
                ax1.clear()
                ax2.clear()
                ax3.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, np.concatenate(Traces1[n], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)
                ax1.plot(x, Traces4[n, :], color="#004c99", lw=1, alpha=0.95)

                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Signal (a.u.)")
                ax2.plot(x, np.concatenate(Traces2[n], axis=1)[0, :], color="#139fff", lw=1, alpha=0.95)
                ax2.plot(x, Traces5[n, :], color="#ff8a00", lw=1, alpha=0.95)

                ax3.set_xlabel("Time (s)")
                ax3.set_ylabel("Signal (a.u.)")
                ax3.plot(x, np.concatenate(Traces3[n], axis=1)[0, :], color="#ff4e4b", lw=1, alpha=0.95)

                n += 1
                ax1.set_title("Raw Trace: Neuron " + str(n))
                ax2.set_title("Source-Separated Trace: Neuron " + str(n))
                ax3.set_title("Source-Separated Δf/f0: Neuron " + str(n))

                fig.canvas.draw()
        elif event.key == "right":
            print(event.key)
            if xmin_slider < xmin_slider.valmax:
                xmin_slider.val += xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                ax2.set_xlim([xmin_slider.val, xmax_slider.val])
                ax3.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()
        elif event.key == "left":
            print(event.key)
            if xmin_slider.val > xmin_slider.valmin:
                xmin_slider.val -= xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                ax2.set_xlim([xmin_slider.val, xmax_slider.val])
                ax3.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()


    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


#b = #139fff
#r = #ff4e4b
#p = #7840cc
#o = #ff8a00
#db = #004c99

def interactivePlot3(Traces1, Traces2, Traces3, FrameRate, Frames):
    x = np.arange(0, (Frames*(1/FrameRate)), 1/FrameRate, dtype=np.float64)

    fig = plt.figure(figsize=(16, 12))

    #ax1
    ax1 = fig.add_subplot(311)
    ax1.set_title("Raw Trace: Neuron 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal (a.u.)")
    ax1.plot(x, np.concatenate(Traces1[0], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)
    #ax1.plot(x, Traces4[0, :], color="#004c99", lw=1, alpha=0.95)

    #ax2
    ax2 = fig.add_subplot(312)
    ax2.set_title("Source-Separated Trace: Neuron 1")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Signal (a.u.)")
    ax2.plot(x, np.concatenate(Traces2[0], axis=1)[0, :], color="#139fff", lw=1, alpha=0.95)
    #ax2.plot(x, Traces5[0, :], color="#ff8a00", lw=1, alpha=0.95)

    #ax3
    ax3 = fig.add_subplot(313)
    ax3.set_title("Source-Separated Δf/f0: Neuron 1")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Signal (a.u.)")
    ax3.plot(x, np.concatenate(Traces3[0], axis=1)[0, :], color="#ff4e4b", lw=1, alpha=0.95)

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
        ax1.set_xlim([xmin_slider.val, xmax_slider.val])
        ax2.set_xlim([xmin_slider.val, xmax_slider.val])
        ax3.set_xlim([xmin_slider.val, xmax_slider.val])
        fig.canvas.draw_idle()

    xmin_slider.on_changed(update)
    xmax_slider.on_changed(update)

    def on_key(event):
        sys.stdout.flush()
        # print("press", event.key)
        if event.key == "up":
            # print('press', event.key)
            n = int(ax1.title._text.split()[3]) - 1  # ZERO INDEXED
            if n < Traces1.shape[0] - 1:
                n += 1  # new n
                ax1.clear()
                ax2.clear()
                ax3.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, np.concatenate(Traces1[n], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)

                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Signal (a.u.)")
                ax2.plot(x, np.concatenate(Traces2[n], axis=1)[0, :], color="#139fff", lw=1, alpha=0.95)

                ax3.set_xlabel("Time (s)")
                ax3.set_ylabel("Signal (a.u.)")
                ax3.plot(x, np.concatenate(Traces3[n], axis=1)[0, :], color="#ff4e4b", lw=1, alpha=0.95)

                n += 1
                ax1.set_title("Raw Trace: Neuron " + str(n))
                ax2.set_title("Source-Separated Trace: Neuron " + str(n))
                ax3.set_title("Source-Separated Δf/f0: Neuron " + str(n))

                fig.canvas.draw()
        elif event.key == "down":
            # print('press', event.key)
            n = int(ax1.title._text.split()[3]) - 1  # ZERO INDEXED
            if n > 0:
                n -= 1  # new n
                ax1.clear()
                ax2.clear()
                ax3.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, np.concatenate(Traces1[n], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)

                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Signal (a.u.)")
                ax2.plot(x, np.concatenate(Traces2[n], axis=1)[0, :], color="#139fff", lw=1, alpha=0.95)

                ax3.set_xlabel("Time (s)")
                ax3.set_ylabel("Signal (a.u.)")
                ax3.plot(x, np.concatenate(Traces3[n], axis=1)[0, :], color="#ff4e4b", lw=1, alpha=0.95)

                n += 1
                ax1.set_title("Raw Trace: Neuron " + str(n))
                ax2.set_title("Source-Separated Trace: Neuron " + str(n))
                ax3.set_title("Source-Separated Δf/f0: Neuron " + str(n))

                fig.canvas.draw()
        elif event.key == "right":
            print(event.key)
            if xmin_slider < xmin_slider.valmax:
                xmin_slider.val += xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                ax2.set_xlim([xmin_slider.val, xmax_slider.val])
                ax3.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()
        elif event.key == "left":
            print(event.key)
            if xmin_slider.val > xmin_slider.valmin:
                xmin_slider.val -= xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                ax2.set_xlim([xmin_slider.val, xmax_slider.val])
                ax3.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()


    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def compareSmoothing_TwoPlots(RawTraces, SmoothTraces, FrameRate, Frames):
    x = np.arange(0, (Frames * (1 / FrameRate)), 1 / FrameRate, dtype=np.float64)

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(211)
    ax1.set_title("Raw Trace: Neuron 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal (a.u.)")
    ax1.plot(x, np.concatenate(RawTraces[0], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)

    ax2 = fig.add_subplot(212)
    ax2.set_title("Smoothed Trace: Neuron 1")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Signal (a.u.)")
    ax2.plot(x, np.concatenate(SmoothTraces[0], axis=1)[0, :], color="#004c99", lw=1, alpha=0.95)

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
        ax2.set_xlim([xmin_slider.val, xmax_slider.val])
        fig.canvas.draw_idle()

    xmin_slider.on_changed(update)
    xmax_slider.on_changed(update)

    def on_key(event):
        sys.stdout.flush()
        # print("press", event.key)
        if event.key == "up":
            # print('press', event.key)
            n = int(ax1.title._text.split()[3]) - 1  # ZERO INDEXED
            if n < RawTraces.shape[0] - 1:
                n += 1  # new n
                ax1.clear()
                ax2.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, np.concatenate(RawTraces[n], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)

                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Signal (a.u.)")
                ax2.plot(x, np.concatenate(SmoothTraces[n], axis=1)[0, :], color="#004c99", lw=1, alpha=0.95)

                n += 1
                ax1.set_title("Raw Trace: Neuron " + str(n))
                ax2.set_title("Smoothed Trace: Neuron " + str(n))
                fig.canvas.draw()

        elif event.key == "down":
            # print('press', event.key)
            n = int(ax1.title._text.split()[3]) - 1  # ZERO INDEXED
            if n > 0:
                n -= 1  # new n
                ax1.clear()
                ax2.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, np.concatenate(RawTraces[n], axis=1)[0, :], color="#40cc8b", lw=1, alpha=0.95)

                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Signal (a.u.)")
                ax2.plot(x, np.concatenate(SmoothTraces[n], axis=1)[0, :], color="#004c99", lw=1, alpha=0.95)

                n += 1
                ax1.set_title("Raw Trace: Neuron " + str(n))
                ax2.set_title("Smoothed Trace: Neuron " + str(n))

                fig.canvas.draw()
        elif event.key == "right":
            print(event.key)
            if xmin_slider < xmin_slider.valmax:
                xmin_slider.val += xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                ax2.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()

        elif event.key == "left":
            print(event.key)
            if xmin_slider.val > xmin_slider.valmin:
                xmin_slider.val -= xmin_slider.valstep
                ax1.set_xlim([xmin_slider.val, xmax_slider.val])
                ax2.set_xlim([xmin_slider.val, xmax_slider.val])
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()