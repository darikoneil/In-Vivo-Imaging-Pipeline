import numpy as np
from itertools import chain
from collections import OrderedDict


class USNAPModule:
    """
    U Schedule Neuroscience Analysis Pipelines Module - ported from batch
    """
    def __init__(self, **kwargs):
        _needs_logging = kwargs.get('NeedsLogging', False)
        _log_file = kwargs.get('LogFile', None)
        if _needs_logging and _log_file is not None:
            print("Logging at " + _log_file)
            self.startLog(_log_file)
        elif _needs_logging and _log_file is None:
            self.startLog()

        self.NAP = OrderedDict()
        self.DNA = None
        self.config = None
        return

    # noinspection All
    def startLog(self, *args):
        from IPython import get_ipython
        self._IP = get_ipython()
        if args:
            _magic_arguments = 'o -t -r ' + arg[0]
            self._IP.run_line_magic('logstart', _magic_arguments)
        else:
            self._IP.run_line_magic('logstart', '')

    def add_function(self, Function):
        """
        :param Function: function to run
        :type Function: function
        """
        _current_step = len(self.NAP.keys())
        self.NAP[_current_step] = Function

