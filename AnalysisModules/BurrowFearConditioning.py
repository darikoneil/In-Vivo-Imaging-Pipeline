import numpy as np
import pathlib
import pickle as pkl
from AnalysisModules.ExperimentHierarchy import BehavioralStage
# Functions specific to burrow behaviors

# Trace Fear Burrow Behavior


class FearConditioning(BehavioralStage):
    def __init__(self, Meta, Stage, **kwargs):
        super().__init__(Meta)
        self.stage_directory = self.mouse_directory + "\\" + Stage
        self.setFolders()
        self.trials_per_stim = kwargs.get('TrialsPerStim', None)
        self.num_stim = kwargs.get('NumStim', 2)
        self.total_trials = self.trials_per_stim * self.num_stim
        self.stage_id = Stage
        self.habituation_data = None
        self.pretrial_data = None
        self.iti_data = None
        self.trial_data = None
        return

    @classmethod
    def identifyTrialValence(cls, csIndexFile):
        _csIndex = np.genfromtxt(csIndexFile, int, delimiter=",")
        PlusTrials = np.where(_csIndex == 0)
        MinusTrials = np.where(_csIndex == 1)
        return PlusTrials[0], MinusTrials[0]

    @classmethod
    def reorganizeData(cls, NeuralActivity, TrialFeatures, ImFreq, **kwargs):
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

    @classmethod
    def generateTimeBins(cls):
        return

    @classmethod
    def loadAnalogData(cls, Filename):
        analogData = np.load(Filename)
        return analogData

    @classmethod
    def loadDigitalData(cls, Filename):
        digitalData = np.load(Filename)
        digitalData = digitalData.__abs__()-1
        # noinspection PyUnresolvedReferences
        digitalData[np.where(digitalData == 255)] = 1
        return digitalData

    @classmethod
    def loadStateData(cls, Filename):
        stateData = np.load(Filename)
        return stateData

    @classmethod
    def loadDictionaryData(cls, Filename):
        with open(Filename, 'r') as f:
            dictionaryData = pkl.load(f)
        return dictionaryData

    @classmethod
    def OrganizeBehavioralData(cls, AnalogData, DigitalData, StateData, Trial, Dictionary, **kwargs):
        HabituationData = OrganizeHabituation('Habituation', AnalogData, DigitalData, StateData, Trial, Dictionary)
        PreTrialData = OrganizePreTrial('PreTrial', AnalogData, DigitalData, StateData, Trial, Dictionary)
        ITIData = OrganizeITI('InterTrial',  AnalogData, DigitalData, StateData, Trial, Dictionary)
        TrialData = OrganizeTrial('Trial',  AnalogData, DigitalData, StateData, Trial, Dictionary)
        return HabituationData, PreTrialData, ITIData, TrialData

    def loadBehavioralData(self):
        analogData = FearConditioning.loadAnalogData(self.generateFileID('Analog'))
        digitalData = FearConditioning.loadDigitalData(self.generateFileID('Digital'))
        stateData = FearConditioning.loadStateData(self.generateFileID('State'))
        dictionaryData = FearConditioning.loadDictionaryData(self.generateFileID('Dictionary'))
        # noinspection PyTypeChecker
        self.habituation_data, self.iti_data, self.pretrial_data, self.trial_data = FearConditioning.OrganizeBehavioralData(analogData, digitalData, stateData, self.total_trials, dictionaryData)

    def generateFileID(self, SaveType):
        if SaveType == 'Analog':
            _save_type = 'AnalogData.npy'
        elif SaveType == 'Digital':
            _save_type = 'DigitalData'
        elif SaveType == 'State':
            _save_type = 'State'
        elif SaveType == 'Dictionary':
            _save_type = 'StimulusInfo.pkl'
        else:
            return print("Unrecognized Behavioral Data Type")

        filename = self.stage_directory + "\\Behavior\\RawBehavioralData\\" + self.stage_id + "_" + self.mouse_id + "_" + str(self.total_trials) + "_of_" + str(self.total_trials) + "_" + _save_type
        return filename


class OrganizeBehavior:
    def __init__(self, State, AnalogData, DigitalData, StateData, Trial, **kwargs):
        self.imaging_sync_channel = kwargs.get('ImagingSyncChannel', 0)
        self.prop_channel = kwargs.get('PropChannel', 1)
        self.force_channel = kwargs.get('ForceChannel', 2)
        self.gate_channel = kwargs.get('GateChannel', 1)
        self.buffer_size = kwargs.get('BufferSize', 100)

        _idx = OrganizeBehavior.indexData(State, StateData, self.buffer_size, Trial)
        self.imaging_sync = AnalogData[self.imaging_sync_channel, _idx]
        self.prop_data = AnalogData[self.prop_channel, _idx]
        self.force_data = AnalogData[self.force_channel, _idx]

        if DigitalData.shape.__len__() == 1:
            self.gateData = DigitalData[_idx]
        elif DigitalData.shape.__len__() > 1:
            self.gateData = DigitalData[self.gate_channel, _idx]

    @classmethod
    def indexData(cls, State, StateData, BufferSize, Trial):
        _idx = np.where(StateData == State)
        _dIdx = np.diff(_idx[0])
        _fIdx = np.where(_dIdx > 1)
        # since current trial is also the last trial right now, the current index will always be the
        # last of the fidx to the end of idx
        # if _fIdx.__len__() > 0 and 1 < currenttrial < totaltrials:
        #   _indexIdx = range((_fIdx[currenttrial-2]+1), _fIdx[currenttrial-1])
        #  _idx = _idx[0][_indexIdx]*buffersize
        # elif _fIdx.__len__() > 0 and currenttrial == totaltrials:
        # #_indexIdx = range((_fIdx[currenttrial-2]+1), _idx[0].__len__())
        # #_idx = _idx[0][_indexIdx]*buffersize
        if _fIdx[0].__len__() > 0:
            _indexIdx = range((_fIdx[0][Trial-2]+1), _idx[0].__len__())
            _idx = _idx[0][_indexIdx]*BufferSize
        else:
            _idx = _idx[0]*BufferSize
        return _idx


class OrganizeTrial(OrganizeBehavior):
    def __init__(self, State, AnalogData, DigitalData, StateData, Trial, Dictionary, **kwargs):
        # noinspection PyArgumentList
        super().__init__(State, AnalogData, DigitalData, StateData, Trial, **kwargs)

        # do stuff unique to this class
        self.trialStartTime = Dictionary["trialStart"]  # float
        self.trialEndTime = Dictionary["trialEnd"]  # float
        self.trialStartFrames = int(0)  # int
        self.trialEndFrames = int(self.trialEndTime[0] - self.trialStartTime[0])  # int
        self.csStartTime = Dictionary["csStart"]  # float
        self.csEndTime = Dictionary["csEnd"]  # float
        self.csStartFrames = int(self.csStartTime[0]-self.trialStartTime[0])  # int
        self.csEndFrames = int(self.csEndTime[0]-self.trialStartTime[0])  # int
        self.ucsStartTime = Dictionary["ucsStart"]  # float
        self.ucsEndTime = Dictionary["ucsEnd"]  # float
        self.ucsStartFrames = int(self.ucsStartTime[0] - self.trialStartTime[0])  # int
        self.ucsEndFrames = int(self.ucsEndTime[0] - self.trialStartTime[0])  # int
        self.csType = Dictionary["stimulusTypes"][Trial-1]


class OrganizeITI(OrganizeBehavior):
    def __init__(self, State, AnalogData, DigitalData, StateData, Trial, Dictionary, **kwargs):
        # noinspection PyArgumentList
        super().__init__(State, AnalogData, DigitalData, StateData, Trial, **kwargs)
        self.ITIStartTime = Dictionary["interStart"]
        self.ITIEndTime = Dictionary["interEnd"]
        self.ITIStartFrames = 0
        self.ITIEndFrames = int(self.ITIEndTime[0] - self.ITIStartTime[0])


class OrganizePreTrial(OrganizeBehavior):
    def __init__(self, State, AnalogData, DigitalData, StateData, Trial, Dictionary, **kwargs):
        # noinspection PyArgumentList
        super().__init__(State, AnalogData, DigitalData, StateData, Trial, **kwargs)
        self.preStartTime = Dictionary["preStart"]
        self.preEndTime = Dictionary["preEnd"]
        self.preStartFrames = 0
        self.preEndFrames = int(self.preEndTime[0] - self.preStartTime[0])


class OrganizeHabituation(OrganizeBehavior):
    def __init__(self, State, AnalogData, DigitalData, StateData, Trial, Dictionary, **kwargs):
        # noinspection PyArgumentList
        super().__init__(State, AnalogData, DigitalData, StateData, Trial, **kwargs)
        self.habStartTime = Dictionary["habStart"]
        self.habEndTime = Dictionary["habEnd"]
        self.habStartFrames = 0
        self.habEndFrames = int(self.habEndTime[0] - self.habStartTime[0])
