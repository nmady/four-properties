import numpy as np

from itertools import product

class SimpleGridWorld(object):
    """
    """

    def __init__(self, side_lengths, walls=None):
        """

        self.actions (tuple of tuples): 
        """



        # Default start position is always in the middle, in the bottom row
        self.start_pos = (side_lengths[0]-1, side_lengths[1]//2)

        self.pos = self.start_pos
        self.next_pos = None
        self.dimensions = side_lengths
        self.state_array = np.zeros(self.dimensions, dtype=np.bool_)
        self.visit_array = np.zeros(self.dimensions, dtype=np.int_)
        self.reward_array = np.zeros(self.dimensions)
        self.actions = tuple(a for a in product((-1,0,1),(-1,0,1)) if a!=(0,0))
        self.walls = set(walls) if walls is not None else set()
        self.verify_walls()

    def get_next_state(self, state, action):
        # actions look like tuples that you add to the state to get the new position:
        #   [-1,0] is left, [1,0] is right, [0, -1] is probably up? [0,1] is probably down
        # this function prevents movement beyond any wall, turning a transition into
        #   a wall into a no-op (left into the left wall means the agent doesn't move)
        # NEW: agent cannot move through any wall pairs
        next_state = (np.clip(state[0]+action[0], 0, self.dimensions[0]-1),
                np.clip(state[1]+action[1], 0, self.dimensions[1]-1))

        if (state, next_state) in self.walls or (next_state, state) in self.walls:
            next_state = state

        return next_state


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

        for i in range(0,max(self.dimensions[0], self.dimensions[1])):
            for state in product(range(self.dimensions[0]), range(self.dimensions[1])):

                next_value_grid[state] = -np.inf

                for action in self.actions + tuple([(0,0)]):
                    next_state = self.get_next_state(state, action)
                    value = reward_grid[state] + gamma*value_grid[next_state]
                    if value > next_value_grid[state]:
                        next_value_grid[state] = value

            ''' next_value_grid gets obliterated anyways, so we can use the old 
            value_grid to avoid allocating a new array every time.'''
            value_grid, next_value_grid = next_value_grid, value_grid

        return next_value_grid

    def verify_walls(self):
        for state_a, state_b in self.walls:
            if (state_a[0] < 0 or state_a[0] >= self.dimensions[0] or 
                state_a[1] < 0 or state_a[1] >= self.dimensions[1]):
                raise ValueError("Wall " + str((state_a, state_b)) + " cannot exist because " + str(state_a) + " is not a coordinate in the grid, which has dimensions " + str(self.dimensions)) 
            if  (state_b[0] < 0 or state_b[0] >= self.dimensions[0] or 
                state_b[1] < 0 or state_b[1] >= self.dimensions[1]):
                raise ValueError("Wall " + str((state_a, state_b)) + " cannot exist because " + str(state_b) + " is not a coordinate in the grid, which has dimensions " + str(self.dimensions))
            if {abs(state_a[0] - state_b[0]), abs(state_a[1] - state_b[1])} != {0,1}:
                raise ValueError("Wall " + str((state_a, state_b)) + " cannot exist because they are not neighbouring coordinates.")

class CylinderGridWorld(SimpleGridWorld):
    """
    """

    def __init__(self, side_lengths, walls=None, start_pos=None, junction=True):

        super().__init__(side_lengths, walls)

        # Setting up the junction location so it can be used as the default start_pos
        # We delete it later if junction is False or None.
        if isinstance(junction, tuple):
            self.junction = junction
        else:
            self.junction = (side_lengths[0]-1, side_lengths[1]//2)

        if start_pos:
            self.pos = start_pos
            self.start_pos = start_pos
        else:
            self.pos = self.junction
            self.start_pos = self.junction

        if not junction:
            self.junction = None

        #No actions can go down in cylinder world!
        self.actions = tuple(a for a in product((-1,0),(-1,0,1)))

    def get_next_state(self, state, action):
        # actions look like tuples that you add to the state to get the new position:
        #   [-1,0] is left, [1,0] is right, [0, -1] is probably up? [0,1] is probably down
        ##TODO: consider using np.clip

        assert action in self.actions

        new_col = min(max(state[1]+action[1],0), self.dimensions[1]-1)
        
        new_row = state[0] + action[0]
        if new_row == -1:         #Fall off the top to get to the junction location
            if self.junction:
                return self.junction
            else:
                new_row = self.dimensions[0]-1
        elif new_row == self.dimensions[0]:
            new_row = 0           #Or fall off the bottom to get to the top

        return (new_row, new_col)

