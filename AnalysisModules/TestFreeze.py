import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

Movement = np.diff(np.genfromtxt("C:\\Users\\Yuste\\Desktop\\Movement.csv", dtype=np.float64, delimiter=","), axis=0)


intervals = pd.interval_range(0, Movement.shape[0], freq=30)
move = np.full((50, 10), 0, dtype=np.float64)
for _interval in range(len(intervals)):
    for _trial in range(Movement.shape[1]):
        move[_interval, _trial] = np.sum(
            Movement[int(intervals.values[_interval].left):int(intervals.values[_interval].right), _trial])

for _trial in range(Movement.shape[1]):
    move[49, _trial] = np.sum(Movement[int(intervals.values[-1].right):int(Movement.shape[0]+1), _trial])


Minus = [0, 2, 4, 6, 8]
Plus = [1, 3, 5, 7, 9]

trial_ids = np.full((50, 10), 0, dtype=int)
trial_valence = np.full((50, 10), 0, dtype=int)

trial_valence[:, Minus] = 1

for _trial in range(Movement.shape[1]):
    trial_ids[:, _trial] = _trial


stage_id = np.full((50, 10), 0, dtype=int)
stage_id[0:15, :] = 0 # PRE
stage_id[15:30, :] = 1 # CS
stage_id[30:40, :] = 2 # TRACE
stage_id[40:50, :] = 3 # RESPONSE

timing = np.full((50, 10), 0, dtype=int)
for _frame in range(50):
    timing[_frame, :] = _frame


DF = pd.DataFrame({'Trial': np.concatenate(trial_ids.T),
                   'CS': np.concatenate(trial_valence.T),
                   'Movement': np.concatenate(move.T),
                   'StageId': np.concatenate(stage_id.T),
                   'Time': np.concatenate(timing.T),
                   })

DF.set_index(['Trial', 'StageId', 'CS', 'Time'])

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(211)
PM = np.append(Plus, Minus)
s = sns.heatmap(data=np.abs(move[:, Plus]), ax=ax1)
ax2 = fig.add_subplot(212)
s2 = sns.heatmap(data=np.abs(move[:, Minus]), ax=ax2)
#s = sns.heatmap(data=DF)
