from __future__ import annotations
from typing import Union, Tuple, List, Optional
import sys
import numpy as np
from tqdm.auto import tqdm
import itertools
import scipy.ndimage
import pandas as pd


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


def pruneTracesByNeuronalIndex(Traces, NeuronalIndex):
    pruned_traces = Traces[NeuronalIndex, :]
    return pruned_traces


def generateCovarianceMatrix(NeuralActivity, ActivityMeasure, **kwargs):
    _num_neurons = kwargs.get('NumNeurons', NeuralActivity.shape[0])
    _num_frames = kwargs.get('NumFrames', NeuralActivity.shape[1])
    _bin_length = kwargs.get('BinLength', None)

    if ActivityMeasure == "Firing Rate" and _bin_length is None:
        covMatrix = np.cov(NeuralActivity)
        return covMatrix


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
                                      axis=len(_feature_data.shape) - 1)

        if _label_data is not None:
            if len(_label_data.shape) > 2:
                print("Culprit: Label Data")
                raise TypeError
            if _label_data.shape[-1] != NeuralActivity.shape[-1]:
                print("Culprit: Label Data")
                raise ValueError
        prunedLabelData = np.delete(_label_data, np.where(np.isnan(NeuralActivity)),
                                    axis=len(_label_data.shape) - 1)

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


def trial_matrix_org(DataFrame, NeuralData):
    _trial_index = np.where(DataFrame[" TrialIndicator"] >= 3.3)[0]
    _trial_frame = DataFrame.iloc[_trial_index]
    _cut_to_images = _trial_frame[~_trial_frame["Downsampled Frame"].isnull()]
    _selected_frames = np.unique(_cut_to_images["Downsampled Frame"].values)
    _trial_start_indices = np.append(_selected_frames[0], _selected_frames[np.where(np.diff(_selected_frames) > 1)[0]])
    _trial_frames = []
    for i in range(_trial_start_indices.__len__()):
        _trial_frames.append(np.arange(_trial_start_indices[i], _trial_start_indices[i] + 345, 1))
    _trial_frames = np.array(_trial_frames).ravel()
    OrgNeuralData = NeuralData[:, _trial_frames.astype(int)]
    return OrgNeuralData


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

