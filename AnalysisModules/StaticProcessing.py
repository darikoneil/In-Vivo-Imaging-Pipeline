import sys
import numpy as np
from tqdm.auto import tqdm
from fissa.deltaf import findBaselineF0
from obspy.signal.detrend import polynomial
import itertools


def calculate_dFoF(Traces, FrameRate, **kwargs):
    # /// Parse Inputs///
    _raw = kwargs.get('raw', None)
    _use_raw_f0 = kwargs.get('use_raw_f0', True)
    _across_tiffs = kwargs.get('across_tiffs', True)
    _merge_after = kwargs.get('merge_after', True)
    _offset = kwargs.get('offset', 0.0001)

    # /// Pre-Allocate ///
    dFoF = np.empty_like(Traces)
    # /// Initialize Feedback
    msg = "Calculating Δf/f0"
    if _across_tiffs:
        msg += "  (using single f0 across all tiffs"
    else:
        msg += " (using unique f0 for each tiff"
    if _use_raw_f0:
        msg += " using f0 derived from raw traces during calculations)"
    else:
        msg += ")"
    print(msg)
    sys.stdout.flush()
    desc = "Calculating {}f/f0".format("d" if sys.version_info < (3, 0) else "Δ")

    # /// Determine Format ///
    if Traces[0, 0].shape:
        _format = 0 # ROIs x TIFFS <- SUB-MASKS x FRAMES
    else:
        _format = 1 # ROIS (PRIMARY MASK ONLY x FRAMES)

    # Now Let's Calculate
    if _format == 0:
        Traces += _offset # Add the Offset
        _neurons = len(Traces)
        _tiffs = len(Traces[0])

        # Loop & Solve
        for _neuron in tqdm(
            range(_neurons),
            total=_neurons,
            desc=desc,
            disable=False,
        ):

            if _across_tiffs:
                _trace_conc = np.concatenate(Traces[_neuron], axis=1)
                _trace_f0 = findBaselineF0(_trace_conc, FrameRate, 1).T[:, None]
                if _use_raw_f0 and _raw is not None:
                    _raw_conc = np.concatenate(_raw[_neuron], axis=1)[0, :]
                    _raw_f0 = findBaselineF0(_raw_conc, FrameRate)
                    _trace_conc = (_trace_conc - _trace_f0) / _raw_f0
                else:
                    _trace_conc = (_trace_conc - _trace_f0) / _trace_f0

                # Store
                _curTiff = 0
                for _tiff in range(_tiffs):
                    _nextTiff = _curTiff + Traces[_neuron][_tiff].shape[1]
                    _signal = _trace_conc[:, _curTiff:_nextTiff]
                    dFoF[_neuron][_tiff] = _signal
                    _curTiff = _nextTiff

            else:
                for _tiff in range(_tiffs):
                    _trace_conc = Traces[_neuron][_tiff]
                    _trace_f0 = findBaselineF0(_trace_conc, FrameRate, 1).T[:, None]
                    _trace_f0[_trace_f0 < 0] = 0
                    if _use_raw_f0 and _raw is not None:
                        _raw_conc = _raw[_neuron][_tiff][0, :]
                        _raw_f0 = findBaselineF0(_raw_conc, FrameRate)
                        _trace_conc = (_trace_conc - _trace_f0) / _raw_f0
                    else:
                        _trace_conc = (_trace_conc - _trace_f0) / _trace_f0

                    dFoF[_neuron][_tiff] = _trace_conc
        if _merge_after:
            _numNeurons = dFoF.shape[0]
            _numTiffs = dFoF.shape[1]
            _firstTiffSize = dFoF[0,0].shape[1]
            _lastTiffSize = dFoF[0, _numTiffs-1].shape[1]
            _totalFrames = _firstTiffSize*(_numTiffs-1)+_lastTiffSize
            _unmerged_dFoF = dFoF.copy()
            dFoF = np.full((_numNeurons, _totalFrames), 0, dtype=np.float64)

            #Merge Here
            for _neuron in tqdm(
                range(_numNeurons),
                total=_numNeurons,
                desc="Merging Tiffs",
                disable=False,
            ):
                dFoF[_neuron, :] = np.concatenate(_unmerged_dFoF[_neuron], axis=1)[0, :]

        return dFoF

    elif _format == 1:
        print("NOT YET")


def mergeTraces(Traces, **kwargs):
    _component = kwargs.get('component', 0)

    [_neurons, _tiffs] = Traces.shape
    _frames = np.concatenate(Traces[0], axis=1)[0, :].shape[0]
    mergedTraces = np.full((_neurons, _frames), 0, dtype=np.float64)

    # Merge Here
    for _neuron in tqdm(
        range(_neurons),
        total=_neurons,
        desc="Merging Traces Across Tiffs",
        disable=False,
    ):
        mergedTraces[_neuron, :] = np.concatenate(Traces[_neuron], axis=1)[_component, :]

    return mergedTraces


def smoothTraces(Traces, **kwargs):
    _niter = kwargs.get('niter', 5)
    _kappa = kwargs.get('kappa', 100)
    _gamma = kwargs.get('gamma', 0.15)
    from AnalysisModules.smoothies import anisotropic_diffusion

    # Find Sizes
    _neurons = Traces.shape[0]
    _frames = np.concatenate(Traces[0], axis=1)[0, :].shape[0]
    smoothedTraces = np.full((_neurons, _frames), 0, dtype=np.float64)
    for _neuron in tqdm(
            range(_neurons),
            total=_neurons,
            desc="SMOOTHING",
            disable=False,
    ):
        smoothedTraces[_neuron, :] = anisotropic_diffusion(np.concatenate(Traces[_neuron],
                                                                          axis=1)[0, :],
                                                           niter=_niter, kappa=_kappa, gamma=_gamma)
    return smoothedTraces


def smoothTraces_TiffOrg(Traces, **kwargs):
    _niter = kwargs.get('niter', 5)
    _kappa = kwargs.get('kappa', 100)
    _gamma = kwargs.get('gamma', 0.15)

    from AnalysisModules.smoothies import anisotropic_diffusion

    # Find Sizes of Traces - Frames, Neurons, Tiffs, Components, Frames In Tiff
    _frames = np.concatenate(Traces[0], axis=1)[0, :].shape[0]
    [_neurons, _tiffs] = Traces.shape
    _components = Traces[0, 0].shape[0]
    _smoothedTracesByComponent = np.full((_neurons, _frames, _components), 0, dtype=np.float64)
    #smoothedTraces = Traces.copy()

    smoothedTraces = np.empty((_neurons, _tiffs), dtype=object)
    for _neuron, _tiff in itertools.product(range(_neurons), range(_tiffs)):
        smoothedTraces[_neuron, _tiff] = np.zeros(Traces[_neuron, _tiff].shape, dtype=np.float64)

    for _neuron in tqdm(
        range(_neurons),
        total=_neurons,
        desc="Smoothing",
        disable=False,
    ):
        for _component in range(_components):
            _smoothedTracesByComponent[_neuron, :, _component] = anisotropic_diffusion(
                np.concatenate(Traces[_neuron], axis=1)[_component, :], niter=_niter, kappa=_kappa,
                gamma=_gamma)

    for _neuron in tqdm(
        range(_neurons),
        total=_neurons,
        desc="Organizing By Tiff",
        disable=False,
    ):
        _currTiff = 0
        for _tiff in range(_tiffs):
            _nextTiff = _currTiff + Traces[_neuron, _tiff].shape[1]
            for _component in range(_components):
                _trace = _smoothedTracesByComponent[_neuron, _currTiff:_nextTiff, _component]
                smoothedTraces[_neuron, _tiff][_component, :] = _trace
            _currTiff = _nextTiff

    return smoothedTraces, _smoothedTracesByComponent


def pruneTracesByNeuronalIndex(Traces, NeuronalIndex):
    pruned_traces = Traces[NeuronalIndex, :]
    return pruned_traces


def detrendTraces(Traces, **kwargs):
    _order = kwargs.get('order', 4)
    _plot = kwargs.get('plot', False)
    [_neurons, _frames] = Traces.shape
    detrended_traces = Traces.copy()

    for _neuron in tqdm(
        range(_neurons),
        total=_neurons,
        desc="Detrending",
        disable=False,
    ):
        detrended_traces[_neuron, :] = polynomial(detrended_traces[_neuron, :], order=_order, plot=_plot)

    return detrended_traces


def detrendTraces_TiffOrg(Traces, **kwargs):
    _order = kwargs.get('order', 4)
    _plot = kwargs.get('plot', False)

    _frames = np.concatenate(Traces[0], axis=1)[0, :].shape[0]
    [_neurons, _tiffs] = Traces.shape
    _components = Traces[0, 0].shape[0]
    _mergedDetrendedTraces = np.full((_neurons, _frames, _components), 0, dtype=np.float64)
    detrendedTraces = Traces.copy()

    for _neuron in tqdm(
        range(_neurons),
        total=_neurons,
        desc="Detrending",
        disable=False,
    ):
        for _component in range(_components):
            _mergedDetrendedTraces[_neuron, :, _component] = polynomial(np.concatenate(Traces[_neuron],
                                                                                       axis=1)[_component, :].copy(),
                                                                                        order=_order, plot=_plot)

    for _neuron in tqdm(
            range(_neurons),
            total=_neurons,
            desc="Organizing By Tiff",
            disable=False,
    ):
        _currTiff = 0
        for _tiff in range(_tiffs):
            _nextTiff = _currTiff + Traces[_neuron, _tiff].shape[1]
            for _component in range(_components):
                _trace = _mergedDetrendedTraces[_neuron, _currTiff:_nextTiff, _component]
                detrendedTraces[_neuron, _tiff][_component, :] = _trace
            _currTiff = _nextTiff

    return detrendedTraces


def anisotropicDiffusion(Trace, **kwargs):

    # parse inputs
    numIterations = kwargs.get('numIterations', 50) # number of iterations
    K = kwargs.get('kappa', 100) # diffusivity conductance that controls the diffusion process
    gamma = kwargs.get('gamma', 0.15) # Step Size, must be <= 1/neighbors
    neighbors = kwargs.get('neighbors', 1) # number of neighboring points to consider

    # Define Diffusion Function Such To Avoid Circular Import
    def diffusionFunction(s, neighbors):
        # Perona-Malik 2
        # exp( - (s/K)^2 )
        # s is the gradient of the image
        # K is a diffusivity conductance that controls the diffusion process
        return np.exp(-(s / K) ** 2) / neighbors

    # Pre-Allocate & Format
    _smoothTrace = Trace.copy()
    s = np.zeros_like(_smoothTrace)
    neighbors = tuple([1.0]*neighbors)

    for n in range(numIterations):
        # Calculate Gradients
        s[slice(None, -1)] = np.diff(_smoothTrace, axis=0)

        # Update
        diffusion = [diffusionFunction(oneGradient, neighbors) * oneGradient for oneGradient in s]

        # Adjust Position
        diffusion[slice(None, -1)] = np.diff(diffusion, axis=0)

        # Update Trace with Diffusion (Constrain Rate)
        _smoothTrace += gamma * (np.sum(diffusion, axis=0))

    return _smoothTrace


def calculateFiringRate(SpikeProb, FrameRate):
    firing_rate = SpikeProb*FrameRate
    return firing_rate

