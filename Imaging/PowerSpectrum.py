import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

software_input = np.arange(0, 205, 5)
base_power = np.array([1.34, 1.55, 1.91, 2.73, 3.59, 4.837, 6.4, 8.1, 10.2, 12.45, 15, 16.5, 19.2, 22.2, 25.72, 29.5, 35.1, 37.5, 41.6, 46.2, 50.1, 55.4, 60.8, 66.2, 71.2, 77.3, 82.8, 88.2, 95, 100.1, 107, 112.4, 119, 125.3, 131.2, 137.9, 144.4, 148, 155.1, 161, 169])
_obj_power = base_power * 0.82
_sample_power = _obj_power * 0.99 * 0.9 * 0.99 * 0.95



fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_title("Prairie One Power Curves")
ax.set_xlabel("Software Power (a.u.)")
ax.set_ylabel("Measured Power (mW)")
spacing = 5
mL = MultipleLocator(spacing)
ax.set_xlim([0, 200])
ax.set_ylim([0, 175])

ax.xaxis.set_minor_locator(mL)
ax.yaxis.set_minor_locator(mL)
ax.plot(software_input, base_power, lw=3, color="blue", label="Rear Aperture")
ax.plot(software_input, _obj_power, lw=3, color="orange", label="10X Obj")
ax.plot(software_input, _sample_power, lw=3, color="red", label="Window")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
