import pytest

import visualization

import numpy as np

def test_plot_heatmap():
	data = np.array([[1,2],[3,4]])
	visualization.plot_heatmap(data)