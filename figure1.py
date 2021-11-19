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
	spawn=None,
	start=(10,5),
	agent=None, 
	figsize=(2,2), 
	cbar=True,
	display="Save", 
	savepostfix="figure1a")

visualization.plot_final_heatmap(np.zeros((1,1)), 
	cmap="bwr_r",
	vmin=None,vmax=None,
	target=None,
	spawn=None,
	agent=(0,0),
	display="Save", 
	savepostfix="figure1b",
	xticklabels=False,
	yticklabels=False)

for column in range(1,10):
	data[1][column] = 1

visualization.plot_final_heatmap(data, 
	cmap=sns.cubehelix_palette(rot=0, dark=0.5, light=100, as_cmap=True),
	linewidths=0.1,
	linecolor='lightgrey',
	vmin=None,vmax=None,
	target=None,
	spawn=(5,5),
	start=(10,5),
	agent=None, 
	figsize=(2,2), 
	cbar=True,
	display="Save", 
	savepostfix="figure1c")