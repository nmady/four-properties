import visualization
import seaborn as sns

import numpy as np

data = np.zeros((11,11))

visualization.plot_final_heatmap(data, 
	cmap="bwr_r",
	linewidths=0.1,
	linecolor='lightgrey',
	vmin=None,vmax=None,
	target=None,
	spawn=(5,5),
	start=(10,5),
	agent=(10,5),
	display="Save", 
	savepostfix="figure2a")

