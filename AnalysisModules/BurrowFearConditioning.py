import numpy as np
from AnalysisModules.ExperimentHierarchy import BehavioralStage
# Functions specific to burrow behaviors

# Trace Fear Burrow Behavior


class Retrieval(BehavioralStage):
    def __init__(self, Meta):
        super().__init__(Meta)
        self.stage_directory = self.mouse_directory + "\\Retrieval"
        self.setFolders()
        return


class Encoding(BehavioralStage):
    def __init__(self, Meta):
        super().__init__(Meta)
        self.stage_directory = self.mouse_directory + "\\Encoding"
        self.setFolders()
        return


class PreExposure(BehavioralStage):
    def __init__(self, Meta):
        super().__init__(Meta)
        self.stage_directory = self.mouse_directory + "\\PreExposure"
        self.setFolders()
        return


def identifyTrialValence(csIndexFile):
    _csIndex = np.genfromtxt(csIndexFile, int, delimiter=",")
    PlusTrials = np.where(_csIndex == 0)
    MinusTrials = np.where(_csIndex == 1)
    return PlusTrials, MinusTrials


def reorganizeData(NeuralActivity, TrialFeatures, ImFreq, **kwargs):
    _iti_time = kwargs.get('ITILength', 90)
    _retract_time = kwargs.get('RetractLength', 5)
    _release_time = kwargs.get('ReleaseLength', 5)
    _iti_pre_inc = kwargs.get('ITIIncludedBeforeTrial', 30)
    _iti_post_inc = kwargs.get('ITIIncludedAfterTrial', 30)
    _pre_time = kwargs.get('PreLength', 15)
    _cs_time = kwargs.get('CSLength', 15)
    _trace_time = kwargs.get('TraceLength', 10)
    _ucs_time = kwargs.get('UCSLength', 5.5)
    _response_time = kwargs.get('ResponseLength', 10)

    try:
        if len(NeuralActivity.shape) > 2:
            raise TypeError
        _num_neurons, _num_frames = NeuralActivity.shape
        if _num_neurons > _num_frames:
            print("Data contains more neurons than frames. Check to ensure your data is in the form neurons x frames.")
    except TypeError:
        print("Neural activity and trial features must be in matrix form")
        return
    except AttributeError:
        print("Data must be numpy array")
        return

    _iti_time *= ImFreq
    _retract_time *= ImFreq
    _release_time *= ImFreq
    _iti_pre_inc *= ImFreq
    _iti_post_inc *= ImFreq
    _pre_time *= ImFreq
    _cs_time *= ImFreq
    _trace_time *= ImFreq
    _ucs_time *= ImFreq
    _response_time *= ImFreq

    # Round to Integers
    _iti_time = int(round(_iti_time))
    _retract_time = int(round(_retract_time))
    _release_time = int(round(_release_time))
    _iti_pre_inc = int(round(_iti_pre_inc))
    _iti_post_inc = int(round(_iti_post_inc))
    _pre_time = int(round(_pre_time))
    _cs_time = int(round(_cs_time))
    _trace_time = int(round(_trace_time))
    _ucs_time = int(round(_ucs_time))
    _response_time = int(round(_response_time))

    _num_features = TrialFeatures.shape[0]
    _trial_indicator = np.sum(TrialFeatures[4:6, :], axis=0)
    _trial_indicator[_trial_indicator > 1] = 1
    _trial_frames = np.where(_trial_indicator == 1)[0]
    _diff_trial_frames = np.diff(_trial_frames)
    _end_trial_idx = np.append(np.where(_diff_trial_frames > 1)[0], _trial_frames.shape[0]-1)
    _start_trial_idx = np.append(np.array([0], dtype=np.int64), _end_trial_idx[0:-1]+1)

    _num_trials = len(_start_trial_idx)
    _trial_length = _end_trial_idx[0]+1
    _before_trial_start_frames = _pre_time + _release_time + _iti_pre_inc
    _after_trial_end_frames = _retract_time + _iti_post_inc
    _total_frames_per_trial = _before_trial_start_frames + _trial_length + _after_trial_end_frames

    NeuralActivity_TrialOrg = np.full((_num_trials, _num_neurons, _total_frames_per_trial), 0, dtype=np.float64)
    FeatureData_TrialOrg = np.full((_num_trials, _num_features, _total_frames_per_trial), 0, dtype=np.float64)

    for _trial in range(_num_trials):
        _start_idx = _trial_frames[_start_trial_idx[_trial]] - _before_trial_start_frames
        _end_idx = _trial_frames[_end_trial_idx[_trial]] + _after_trial_end_frames+1 # one to handle indexing
        NeuralActivity_TrialOrg[_trial, :, :] = NeuralActivity[:, _start_idx:_end_idx]
        FeatureData_TrialOrg[_trial, :, :] = TrialFeatures[:, _start_idx:_end_idx]

    FeatureIndex = dict()
    FeatureIndex['ITI_PRE'] = (0, _iti_pre_inc)
    FeatureIndex['RELEASE'] = (FeatureIndex['ITI_PRE'][1], FeatureIndex['ITI_PRE'][1]+_release_time)
    FeatureIndex['PRE'] = (FeatureIndex['RELEASE'][1], FeatureIndex['RELEASE'][1]+_pre_time)
    FeatureIndex['TRIAL'] = (FeatureIndex['PRE'][1], FeatureIndex['PRE'][1]+_trial_length)
    FeatureIndex['RETRACT'] = (FeatureIndex['TRIAL'][1], FeatureIndex['TRIAL'][1]+_retract_time)
    FeatureIndex['ITI_POST'] = (FeatureIndex['RETRACT'][1], FeatureIndex['RETRACT'][1]+_iti_post_inc)

    # Sub-Trial Index
    FeatureIndex['CS'] = (FeatureIndex['PRE'][1], FeatureIndex['PRE'][1]+_cs_time)
    FeatureIndex['TRACE'] = (FeatureIndex['CS'][1], FeatureIndex['CS'][1]+_trace_time)
    FeatureIndex['RESPONSE'] = (FeatureIndex['TRACE'][1], FeatureIndex['TRACE'][1]+_response_time)
    FeatureIndex['UCS'] = (FeatureIndex['TRACE'][1], FeatureIndex['TRACE'][1]+_ucs_time)
    # NOTE: DOUBLE CHECK UCS TIME

    return NeuralActivity_TrialOrg, FeatureIndex, FeatureData_TrialOrg


def shuffleTrials(NeuralActivityInTrialForm, **kwargs):
    print("Not Yet")
    _features = kwargs.get('FeatureData', None)
    _trial_subset = kwargs.get('TrialSubset', None)
    try:
        if len(NeuralActivityInTrialForm.shape) != 3:
            raise ValueError
        if _features is not None:
            if len(NeuralActivityInTrialForm.shape) != len(_features.shape):
                raise AssertionError
    except ValueError:
        print("Neural Activity must be in trial form")
        return
    except AssertionError:
        print("Neural and Feature Data must be in the same shape")
        return

    if _trial_subset is not None:
        _shuffle_index = _trial_subset.copy()
        np.random.shuffle(_shuffle_index)
    else:
        _shuffle_index = np.arange(NeuralActivityInTrialForm.shape[0])
        np.random.shuffle(_shuffle_index)

    if _features is not None:
        shuffled_features = _features[_shuffle_index, :, :]
        shuffled_neural_data = NeuralActivityInTrialForm[_shuffle_index, :, :]
        return shuffled_neural_data, shuffled_features
    else:
        shuffled_neural_data = NeuralActivityInTrialForm[_shuffle_index, :, :]
        return shuffled_neural_data


def shuffleFrames(NeuralActivityInMatrixForm):
    _shuffle_index = np.arange(NeuralActivityInMatrixForm.shape[1])
    np.random.shuffle(_shuffle_index)
    shuffled_neural_data = NeuralActivityInMatrixForm[:, _shuffle_index]
    return shuffled_neural_data


def shuffleEachNeuron(NeuralActivityInMatrixForm):
    _num_neurons, _num_frames = NeuralActivityInMatrixForm.shape
    shuffled_neural_data = np.zeros_like(NeuralActivityInMatrixForm).copy() # cuz paranoid
    _shuffle_index = np.arange(_num_frames)

    for _neuron in range(_num_neurons):
        np.random.shuffle(_shuffle_index)
        shuffled_neural_data[_neuron, :] = NeuralActivityInMatrixForm[_neuron, _shuffle_index]

    return shuffled_neural_data

