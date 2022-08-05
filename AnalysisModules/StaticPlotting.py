
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import sys
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution
import seaborn as sns


def plotNoise(Traces, FrameRate):
    # plt.figure(1)
    _noise_levels = plot_noise_level_distribution(Traces, FrameRate)


def plotTraces(self, **kwargs):
    NeuronalSubset = kwargs.get('NeuronalSubset', np.random.randint(self.traces.shape[0], size=10))
    TemporalSubset = kwargs.get('TemporalSubset', [0, self.frame_rate * 30])
    plot_dFF_traces(self.traces[:, TemporalSubset[0]:TemporalSubset[1]], NeuronalSubset, self.frame_rate)


def plotTraceComparisons(self, **kwargs):
    NeuronalSubset = kwargs.get('NeuronalSubset', np.random.randint(self.traces.shape[0], size=10))
    TemporalSubset = kwargs.get('TemporalSubset', [0, self.frame_rate * 30])
    plot_dFF_traces(self.traces[:, TemporalSubset[0]:TemporalSubset[1]], NeuronalSubset, self.frame_rate,
                    spiking=self.spike_prob)


def plotSpikeInference(SpikeProb, SpikeTimes, Traces, FrameRate, **kwargs):
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
