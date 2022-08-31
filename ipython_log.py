# IPython log file

print('PyDev console: using IPython 7.34.0\n')

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\ProgramData\\Anaconda3\\envs\\Calcium-Imaging-Analysis-Pipeline', 'C:/ProgramData/Anaconda3/envs/Calcium-Imaging-Analysis-Pipeline'])
from IPython import get_ipython
IP = get_ipython()
IP.run_line_magic('logstart', '')
print("HI")
"PRINT"
IP.run_line_magic('logstop', '')
