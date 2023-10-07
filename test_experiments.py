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
	os.system("rm ./output/_11_11_trials1_steps10_bookstore5_5_trial0.avi")
	os.system("touch ./output/_11_11_trials1_steps10_bookstore5_5_trial0.avi")

	os.system("python experiments.py --width=11 --height=11 --steps=10 --animation='both'")
	captured = capfd.readouterr()

	assert captured.err == "File './output/_11_11_trials1_steps10_bookstore5_5_trial0.avi' already exists. Overwrite? [y/N] Not overwriting - exiting\n"
	assert os.path.getsize("./output/_11_11_trials1_steps10_bookstore5_5_trial0.avi") == 0

def test_animation_flag_on_file_no_exists(capfd):
	""" This test function determines if a new animation is created if flag on
	"""

	os.system("rm ./output/_11_11_trials1_steps10_bookstore5_5_trial0.avi")

	os.system("python experiments.py --width=11 --height=11 --steps=10 --animation='both'")
	captured = capfd.readouterr()

	assert os.path.exists("./output/_11_11_trials1_steps10_bookstore5_5_trial0.avi")
	assert captured.err == ""

def test_animation_flag_off():
	"""
	This test makes sure that no animation file is created if --animation='none' 
	flag is raised.
	"""
	os.system("rm ./output/_11_11_trials1_steps10_bookstore5_5_trial0.avi")
	os.system("touch ./output/_11_11_trials1_steps10_bookstore5_5_trial0.avi")

	os.system("python experiments.py --width=11 --height=11 --steps=10 --animation='none'")

	assert os.path.getsize("./output/_11_11_trials1_steps10_bookstore5_5_trial0.avi") == 0

def test_lineplot_flag_on_file_exists():
	""" This test function determines if a new lineplot is created if flag on.
	"""
	os.system("rm ./output/Persistent_Value_over_Time_stackedLineplot__11_11_trials1_steps10_bookstore5_5.png")
	os.system("touch ./output/Persistent_Value_over_Time_stackedLineplot__11_11_trials1_steps10_bookstore5_5.png")

	os.system("python experiments.py --width=11 --height=11 --steps=10 --lineplot")
	
	assert os.path.getsize("./output/Persistent_Value_over_Time_stackedLineplot__11_11_trials1_steps10_bookstore5_5.png") > 0

def test_lineplot_flag_on_file_no_exists():
	""" This test function determines if a new lineplot is created if flag on
	"""

	os.system("rm ./output/Persistent_Value_over_Time_stackedLineplot__11_11_trials1_steps10_bookstore5_5.png")

	os.system("python experiments.py --width=11 --height=11 --steps=10 --lineplot")

	assert os.path.exists("./output/Persistent_Value_over_Time_stackedLineplot__11_11_trials1_steps10_bookstore5_5.png")

def test_lineplot_flag_off():
	"""
	This test makes sure that no lineplot file is created if --no-lineplot 
	flag is raised.
	"""
	os.system("rm ./output/Persistent_Value_over_Time_stackedLineplot__11_11_trials1_steps10_bookstore5_5.png")
	os.system("touch .Persistent_Value_over_Time_stackedLineplot__11_11_trials1_steps10_bookstore5_5.png")

	os.system("python experiments.py --width=11 --height=11 --steps=10 --no-lineplot")

	assert not os.path.exists("./output/Persistent_Value_over_Time_stackedLineplot__11_11_trials1_steps10_bookstore5_5.png")

def test_csv_output_no_directory_exists():
	"""
	This test ensures that if the --csv-output includes a directory that doesn't exist, 
	the program makes the directory and the inner file.
	"""

	# ensures the test directory doesn't 
	if os.path.exists('./this_is_a_test_dir/'):
		os.system("rm -r ./this_is_a_test_dir/")

	# run experiments
	os.system("python experiments.py --csv-output='./this_is_a_test_dir/test_csv.csv' --width=11 --height=11 --steps=10 --no-lineplot --animation='none'")

	# new csv should exist and no error should occur
	assert os.path.exists('./this_is_a_test_dir/test_csv.csv')

	#clean-up
	os.system("rm ./this_is_a_test_dir/test_csv.csv")
	os.system("rm -r ./this_is_a_test_dir/")