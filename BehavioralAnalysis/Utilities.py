from __future__ import annotations
import numpy as np
from tqdm.auto import tqdm
from typing import Tuple, List, Optional, Union
import pandas as pd



def extract_specific_data(DataFrame: pd.DataFrame,
                          KeyValuePairs: Union[Tuple[Tuple[str, Union[str, int, float]]],
                                               Tuple[str, Union[str, int, float]]],
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