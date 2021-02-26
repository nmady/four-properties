import pytest

import agents

import numpy as np

def test_agent_init():
	"""Test to ensure a GridworldTDLearner initializes with typical arguments
	"""
	learner = agents.GridworldTDLearner((11,11))
	learner = agents.GridworldTDLearner((50,11))


def test_curious_init():
	learner = agents.CuriousTDLearner((11,11))
	learner = agents.CuriousTDLearner((50,11))

def one_state_update(learner):
	"""Test to ensure the TD-learning update seems to work in a very tiny world
	"""
	xt = (0,0)
	xtp1 = (0,0)
	R = 1
	learner.update(xt, xtp1, R, gamma=1)
	assert learner.w[0][0] == learner.alpha

def test_one_state_update():
	learner = agents.GridworldTDLearner((1,1))
	one_state_update(learner)


def test_curious_one_state_update():
	learner = agents.CuriousTDLearner((1,1))
	one_state_update(learner)

def test_one_state_vshort_update():
	"""Test to ensure the TD-learning update seems to work in a very tiny world
	"""
	learner = agents.CuriousTDLearner((1,1))
	xt = (0,0)
	xtp1 = (0,0)
	R = 1
	learner.update(xt, xtp1, R, gamma=1, vshort=np.array([[1.0]]))
	assert learner.w[0][0] == 2*learner.alpha