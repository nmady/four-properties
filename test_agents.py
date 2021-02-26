import pytest
import agents

def test_agent_init():
	"""Test to ensure a GridworldTDLearner initializes with typical arguments
	"""
	learner = agents.GridworldTDLearner((11,11))
	learner = agents.GridworldTDLearner((50,11))
