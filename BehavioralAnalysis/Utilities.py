from __future__ import annotations
import numpy as np
from tqdm.auto import tqdm
from typing import Tuple, List, Optional, Union
import pandas as pd
import scipy.signal as signal
from BehavioralAnalysis.BurrowFearConditioning import FearConditioning


def extract_specific_data(DataFrame: pd.DataFrame,
                          KeyValuePairs: Union[Tuple[Tuple[str, Union[str, int, float, list]]],
                                               Tuple[str, Union[str, int, float, list]]],
                          **kwargs: bool) -> pd.DataFrame:
    """
    This Function extracts some specific portion of the behavior

    :param DataFrame: synced behavioral data
    :type DataFrame: Any
    :param KeyValuePairs: A tuple containing a column name in the data and expression for pattern matching. Can use tuple of tuples for multiple extracts. ORDER MATTERS.
    :type KeyValuePairs: Union[tuple[str, Union[str, int, float]], tuple[str, Union[str, int, float]]]
    :return: some subset of the dataset
    :rtype: pd.DataFrame
    :keyword keep_index: whether to keep original index on export (bool, default True)
    """
    _use_original_index = kwargs.get("keep_index", True)

    _dataframe = DataFrame.copy(deep=True) # copy for safety
    _original_index = _dataframe.index.name

    def do_extraction():
        nonlocal _dataframe
        nonlocal _key
        nonlocal _expression

        _dataframe.reset_index(drop=False, inplace=True)
        _dataframe.set_index(_key, drop=True, inplace=True)

        if isinstance(_expression, str):
            _eval_string = "".join(["_dataframe.index.to_numpy()", _expression])
            _dataframe = _dataframe.loc[eval(_eval_string)].copy(deep=True)
        elif isinstance(_expression, list):
            _dataframe = _dataframe.loc[_dataframe.index.isin(_expression)]
        else:
            _dataframe = _dataframe.loc[_expression].copy(deep=True)

    def set_original_index():
        nonlocal _dataframe
        nonlocal _original_index

        if _dataframe.index.name == "index":
            _dataframe.set_index(_original_index, drop=True, inplace=True)
        elif _dataframe.index.name == _original_index:
            pass
        else:
            _dataframe.reset_index(drop=False, inplace=True)
            _dataframe.set_index(_original_index, drop=True, inplace=True)

        _dataframe.sort_index(inplace=True)

    # ensure correct format
    if isinstance(KeyValuePairs[0], tuple):
        for _key, _expression in KeyValuePairs:
            do_extraction()
    elif isinstance(KeyValuePairs, tuple):
        _key, _expression = KeyValuePairs
        do_extraction()
    else:
        print("Incorrect KeyValuePairs Format!")
        return _dataframe

    if _use_original_index:
        set_original_index()

    _dataframe = _dataframe.reindex(columns=sorted(_dataframe.columns))

    return _dataframe


def lowpass_filter(Data: np.ndarray, SamplingFrequency: float,
                   Cutoff: float, Order: Optional[int] = None) -> np.ndarray:
    """
    Low pass filter (butter)

    :param Data: Data to be filtered
    :type Data: Any
    :param SamplingFrequency: Sampling frequency of data
    :type SamplingFrequency: float
    :param Cutoff: Cutoff Frequency for filter
    :type Cutoff: float
    :param Order: Optional Order of Filter
    :type Order: Optional[int]
    :return: Filtered Data
    :rtype: Any
    """

    if Order is None:
        Order = 2

    return signal.filtfilt(*signal.butter(Order, Cutoff/(0.5*SamplingFrequency)), Data)


def time_spent_in_burrow(BehavioralObject: FearConditioning, *args: int) -> Tuple[float]:
    """
    Calculates time spent in burrow via the gate signal

    :param BehavioralObject: FearConditioning Behavioral Stage Object
    :type BehavioralObject: Any
    :param args: Number of trials per stimulus to drop due to forced retraction
    :type args: int
    :return: Time spent in burrow (%) per stage
    :rtype: Tuple[float]
    """

    def extract_gate_data(_stim) -> np.ndarray:
        nonlocal BehavioralObject
        return extract_specific_data(BehavioralObject.data,
                                     (("State Integer", BehavioralObject.state_index.get("Trial")),
                                      ("Trial Set", list(BehavioralObject.trial_groups[_stim]))))["Gate"].to_numpy()

    def calculate_time_spent_in_burrow(_gate_data) -> float:
        return (np.sum(_gate_data)/_gate_data.shape[0])*100.0

    # extract
    _gate_by_stim = [extract_gate_data(_stim) for _stim in range(BehavioralObject.num_stim)]
    # calculate
    if args:
        _times = [calculate_time_spent_in_burrow(_gate_data) for _gate_data in _gate_by_stim]
        return tuple(_times/((BehavioralObject.trials_per_stim-args[0])/BehavioralObject.trials_per_stim))
    else:
        return tuple([calculate_time_spent_in_burrow(_gate_data) for _gate_data in _gate_by_stim])
