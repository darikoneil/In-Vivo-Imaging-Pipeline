import sys
import numpy as np
from tqdm.auto import tqdm
from fissa.deltaf import findBaselineF0
from obspy.signal.detrend import polynomial
import itertools
import scipy.ndimage


class Processing:
    """

    """

    def __init__(self):
        return

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def smoothTraces(Traces, **kwargs):
        _niter = kwargs.get('niter', 5)
        _kappa = kwargs.get('kappa', 100)
        _gamma = kwargs.get('gamma', 0.15)
        from ImagingAnalysis.smoothies import anisotropic_diffusion

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

    @staticmethod
    def smoothTraces_TiffOrg(Traces, **kwargs):
        _niter = kwargs.get('niter', 5)
        _kappa = kwargs.get('kappa', 100)
        _gamma = kwargs.get('gamma', 0.15)

        from ImagingAnalysis.smoothies import anisotropic_diffusion

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

    @staticmethod
    def pruneTracesByNeuronalIndex(Traces, NeuronalIndex):
        pruned_traces = Traces[NeuronalIndex, :]
        return pruned_traces

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def calculateFiringRate(SpikeProb, FrameRate):
        firing_rate = SpikeProb*FrameRate
        return firing_rate

    @staticmethod
    def generateCovarianceMatrix(NeuralActivity, ActivityMeasure, **kwargs):
        _num_neurons = kwargs.get('NumNeurons', NeuralActivity.shape[0])
        _num_frames = kwargs.get('NumFrames', NeuralActivity.shape[1])
        _bin_length = kwargs.get('BinLength', None)

        if ActivityMeasure == "Firing Rate" and _bin_length is None:
            covMatrix = np.cov(NeuralActivity)
            return covMatrix

    @staticmethod
    def pruneNaN(NeuralActivity, **kwargs):
        _feature_data = kwargs.get('FeatureData', None)
        _label_data = kwargs.get('LabelData', None)

        try:
            if len(NeuralActivity.shape) != 2:
                print("Culprit: Neural Activity")
                raise TypeError
            prunedNeuralActivity = np.delete(NeuralActivity, np.where(np.isnan(NeuralActivity)), axis=1)

            if _feature_data is not None:
                if len(_feature_data.shape) > 2:
                    print("Culprit: Feature Data")
                    raise TypeError
                if _feature_data.shape[-1] != NeuralActivity.shape[-1]:
                    print("Culprit: Feature Data")
                    raise ValueError
            prunedFeatureData = np.delete(_feature_data, np.where(np.isnan(NeuralActivity)),
                                          axis=len(_feature_data.shape)-1)

            if _label_data is not None:
                if len(_label_data.shape) > 2:
                    print("Culprit: Label Data")
                    raise TypeError
                if _label_data.shape[-1] != NeuralActivity.shape[-1]:
                    print("Culprit: Label Data")
                    raise ValueError
            prunedLabelData = np.delete(_label_data, np.where(np.isnan(NeuralActivity)),
                                        axis=len(_label_data.shape)-1)

            if _feature_data is not None and _label_data is not None:
                return prunedNeuralActivity, prunedFeatureData, prunedLabelData

            elif _feature_data is not None and _label_data is None:
                return prunedNeuralActivity, prunedFeatureData

            elif _feature_data is None and _label_data is not None:
                return prunedNeuralActivity, prunedLabelData

            elif _feature_data is None and _label_data is None:
                return prunedNeuralActivity

        except TypeError:
            print("Data must be in Matrix Form!")
        except AttributeError:
            print("Data must be in a numpy array!")
        except ValueError:
            print("The number of Feature or Label samples must match the number of Neural samples!")

    @staticmethod
    def normalizeSmoothFiringRates(FiringRates, Sigma):
        smoothedFiringRates = scipy.ndimage.gaussian_filter1d(FiringRates, Sigma, axis=1)
        #normFiringRates = sklearn.preprocessing.minmax_scale(smoothedFiringRates, axis=1, copy=True)
        normFiringRates = smoothedFiringRates/np.max(smoothedFiringRates, axis=0)
        return normFiringRates

    @staticmethod
    def generateSpikeMatrix(SpikeTimes, NumFrames):
        _num_neurons = SpikeTimes.shape[0]
        SpikeMatrix = np.full((_num_neurons, NumFrames), 0, dtype=np.int32)

        for _neuron in tqdm(
                range(_num_neurons),
                total=_num_neurons,
                desc="Formatting Spike Matrix",
                disable=False,
        ):
            for _spikes in SpikeTimes[_neuron]:
                SpikeMatrix[_neuron, _spikes] = 1

        return SpikeMatrix

    @staticmethod
    def trial_matrix_org(DataFrame, NeuralData):
        _trial_index = np.where(DataFrame[" TrialIndicator"] >= 3.3)[0]
        _trial_frame = DataFrame.iloc[_trial_index]
        _cut_to_images = _trial_frame[~_trial_frame["Downsampled Frame"].isnull()]
        _selected_frames = np.unique(_cut_to_images["Downsampled Frame"].values)
        _trial_start_indices = np.append(_selected_frames[0], _selected_frames[np.where(np.diff(_selected_frames) > 1)[0]])
        _trial_frames = []
        for i in range(_trial_start_indices.__len__()):
            _trial_frames.append(np.arange(_trial_start_indices[i], _trial_start_indices[i]+345, 1))
        _trial_frames = np.array(_trial_frames).ravel()
        OrgNeuralData = NeuralData[:, _trial_frames.astype(int)]
        return OrgNeuralData

    @staticmethod
    def generate_features(FramesPerTrial, NumTrials, TrialParameters):
        _trial_time = TrialParameters.get("trialEnd")[0] - TrialParameters.get("trialStart")[0]
        _fp = _trial_time / FramesPerTrial
        _time_stamps = np.arange(0, _trial_time, _fp)
        assert (_time_stamps.__len__() == FramesPerTrial)

        _plus_trials = [p for p in range(NumTrials) if TrialParameters.get("stimulusTypes")[p] == 0]
        _cs_time = TrialParameters.get("csEnd")[0] - TrialParameters.get("csStart")[0]
        _ucs_time = TrialParameters.get("ucsEnd")[_plus_trials[0]] - TrialParameters.get("ucsStart")[_plus_trials[0]]
        _trace_time = TrialParameters.get("ucsStart")[_plus_trials[0]] - TrialParameters.get("csEnd")[_plus_trials[0]]

        Features = np.full((NumTrials, 7, FramesPerTrial), 0, dtype=np.int)
        FeaturesLabels = ("Plus CS", "Minus CS", "Plus Trace", "Minus Trace", "Plus Trial", "Minus Trial", "UCS")

        # Plus Trials
        for _trial in _plus_trials:
            Features[_trial, 0, np.where(_time_stamps <= _cs_time)[0]] = 1
            Features[_trial, 2, np.where((_cs_time < _time_stamps) & (_time_stamps <= _cs_time + _trace_time))[0]] = 1
            Features[_trial, 4, :] = 1
            Features[_trial, 6, np.where(
                (_cs_time + _trace_time < _time_stamps) & (_time_stamps <= _cs_time + _trace_time + _ucs_time))[0]] = 1

        # Minus Trials
        _minus_trials = [x for x in range(NumTrials) if x not in _plus_trials]
        for _trial in _minus_trials:
            Features[_trial, 1, np.where(_time_stamps <= _cs_time)[0]] = 1
            Features[_trial, 3, np.where((_cs_time < _time_stamps) & (_cs_time <= _cs_time + _trace_time))[0]] = 1
            Features[_trial, 5, :] = 1

        return Features, FeaturesLabels
