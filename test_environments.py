import pytest

import environments

import numpy as np

def test_world_init():
	"""Test to ensure a SimpleGridWorld initializes with typical arguments
	"""
	world = environments.SimpleGridWorld((11,11))
	world = environments.SimpleGridWorld((1,1))
	world = environments.SimpleGridWorld((50,11))


def test_world_get_next_state_walls():
	"""Ensure agent does not pass through SimpleGridWorld walls
	"""
	world = environments.SimpleGridWorld((11,11))

	#test go left from bottom left corner
	assert world.get_next_state((0,0),(-1,0)) == (0,0)

	#test go down from bottom left corner
	assert world.get_next_state((0,0),(0,-1)) == (0,0)

	#test go down and left diagonal from bottom left corner
	assert world.get_next_state((0,0),(-1,-1)) == (0,0)

	#test go right from top right corner
	assert world.get_next_state((10,10),(1,0)) == (10,10)

	#test go up from top right corner
	assert world.get_next_state((10,10),(0,1)) == (10,10)

	#test go right and up diagonal from top right corner
	assert world.get_next_state((10,10),(1,1)) == (10,10)


