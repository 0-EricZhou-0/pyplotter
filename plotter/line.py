

import matplotlib as mpl
import matplotlib.axes as pa
import matplotlib.pyplot as plt

from numpy.typing import ArrayLike
import numpy as np

# fig, ax = plt.figure()
def draw_line_chart(ax: pa.Axes, x_series: ArrayLike, y_series_list: ArrayLike | list[ArrayLike],
                    ravg_win=1, **kwargs):
    if type(y_series_list) is not list or (len(y_series_list) > 0 and type(y_series_list[0]) is not list):
        y_series_list = [y_series_list]
    for y_series in y_series_list:
        assert len(x_series) == len(y_series)
        if ravg_win != 1:
            x_series = np.convolve(x_series, np.ones(ravg_win) / ravg_win)
            y_series = np.convolve(y_series, np.ones(ravg_win) / ravg_win)
        ax.plot(x_series, y_series, **kwargs)
