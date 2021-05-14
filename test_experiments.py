""" Note that these tests access the filesystem, which is not great practice,
	but certainly makes it easier to determine if all the files that should be 
	created actually are!

	Run with:
		pytest test_experiments.py
"""

import pytest
import experiments
import os

def test_animation_flag_on_file_exists(capfd):
	""" This test function determines if a new animation is created if flag on.
	"""
	os.system("rm ./output/_11_11_steps10_trial0.avi")
	os.system("touch ./output/_11_11_steps10_trial0.avi")

	os.system("python experiments.py --width=11 --height=11 --steps=10 --animation")
	captured = capfd.readouterr()

	assert captured.err == "File './output/_11_11_steps10_trial0.avi' already exists. Overwrite ? [y/N] Not overwriting - exiting\n"
	assert os.path.getsize("./output/_11_11_steps10_trial0.avi") == 0

def test_animation_flag_on_file_no_exists(capfd):
	""" This test function determines if a new animation is created if flag on
	"""

	os.system("rm ./output/_11_11_steps10_trial0.avi")

	os.system("python experiments.py --width=11 --height=11 --steps=10 --animation")
	captured = capfd.readouterr()

	assert os.path.exists("./output/_11_11_steps10_trial0.avi")
	assert captured.err == ""

def test_animation_flag_off():
	"""
	This test makes sure that no animation file is created if --no-animation 
	flag is raised.
	"""
	os.system("rm ./output/_11_11_steps10_trial0.avi")
	os.system("touch ./output/_11_11_steps10_trial0.avi")

	os.system("python experiments.py --width=11 --height=11 --steps=10 --no-animation")

	assert os.path.getsize("./output/_11_11_steps10_trial0.avi") == 0

def test_lineplot_flag_on_file_exists():
	""" This test function determines if a new lineplot is created if flag on.
	"""
	os.system("rm ./output/Value_over_Time_stackedLineplot_11_11_steps10.png")
	os.system("touch ./output/Value_over_Time_stackedLineplot_11_11_steps10.png")

	os.system("python experiments.py --width=11 --height=11 --steps=10 --lineplot")
	
	assert os.path.getsize("./output/Value_over_Time_stackedLineplot_11_11_steps10.png") > 0

def test_lineplot_flag_on_file_no_exists():
	""" This test function determines if a new lineplot is created if flag on
	"""

	os.system("rm ./output/Value_over_Time_stackedLineplot_11_11_steps10.png")

	os.system("python experiments.py --width=11 --height=11 --steps=10 --lineplot")

	assert os.path.exists("./output/Value_over_Time_stackedLineplot_11_11_steps10.png")

def test_lineplot_flag_off():
	"""
	This test makes sure that no lineplot file is created if --no-lineplot 
	flag is raised.
	"""
	os.system("rm ./output/Value_over_Time_stackedLineplot_11_11_steps10.png")
	os.system("touch ./output/Value_over_Time_stackedLineplot_11_11_steps10.png")

	os.system("python experiments.py --width=11 --height=11 --steps=10 --no-lineplot")

	assert os.path.getsize("./output/Value_over_Time_stackedLineplot_11_11_steps10.png") == 0