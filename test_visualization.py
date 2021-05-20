import pytest

import visualization

import numpy as np

def test_plot_heatmap():
	data = np.array([[1,2],[3,4]])
	visualization.plot_heatmap(data)

def test_plot_both_value_heatmaps():
	data1 = np.array([[1,2],[3,4]])
	data2 = -data1
	visualization.plot_both_value_heatmaps(data1, data2, stepnum='null')