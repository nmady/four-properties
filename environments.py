import numpy as np

from itertools import product

class SimpleGridWorld(object):
    """
    """

    def __init__(self, side_lengths):
        """

        self.actions (tuple of tuples): 
        """

        # Start position is always in the middle, in the second from bottom row
        self.start_pos = (side_lengths[0]-2, int(side_lengths[1]/2))

        self.pos = self.start_pos
        self.dimensions = side_lengths
        self.state_array = np.zeros(self.dimensions, dtype=np.bool_)
        self.visit_array = np.zeros(self.dimensions, dtype=np.int_)
        self.reward_array = np.zeros(self.dimensions)
        self.actions = tuple(a for a in product((-1,0,1),(-1,0,1)) if a!=(0,0))

    def get_next_state(self, state, action):
        # actions look like tuples that you add to the state to get the new position:
        #   [-1,0] is left, [1,0] is right, [0, -1] is probably up? [0,1] is probably down
        ##TODO: consider using np.clip
        return (min(max(state[0]+action[0],0), self.dimensions[0]-1), 
                min(max(state[1]+action[1],0), self.dimensions[1]-1))

    def visit(self, state):
        """
        """
        # Increment visit counter
        self.visit_array[state[0]][state[1]] += 1

        # Update state_array to show current position
        self.state_array.fill(0)
        self.state_array[state[0]][state[1]] = 1

    def get_reward(self):
        """ This gridworld doesn't provide any extrinsic reward!

        Returns:
            reward (int): always 0
        """
        return 0

    def value_iteration(self, reward_grid, initial_value_grid=None, gamma=0.9):
        """ Perform value iteration.

        Assumes that the reward that you get when you leave a square is given by
        that item in reward_grid.

        Simulates a "stay-here" action
        """

        assert(self.dimensions == reward_grid.shape)

        if initial_value_grid is not None:
            value_grid = np.copy(initial_value_grid)
        else:
            value_grid = np.zeros(self.dimensions)

        next_value_grid = np.zeros(self.dimensions)

        while True:
            for state in product(range(self.dimensions[0]), range(self.dimensions[1])):

                next_value_grid[state] = -np.inf

                for action in product((-1,0,1),(-1,0,1)):
                    next_state = self.get_next_state(state, action)
                    value = reward_grid[state] + gamma*value_grid[next_state]
                    if value > next_value_grid[state]:
                        next_value_grid[state] = value

            if np.allclose(value_grid, next_value_grid):
                return next_value_grid

            ''' next_value_grid gets obliterated anyways, so we can use the old 
            value_grid to avoid allocating a new array every time.'''
            value_grid, next_value_grid = next_value_grid, value_grid

