import numpy as np

from environments import SimpleGridWorld
from visualization import plot_heatmap
from random import choice

def value_iteration(rfunc, vfunc, gamma=0.9):
  vf = vfunc
  rf = rfunc
  vf_prev = None
  num_rows = vf.shape[0]
  num_cols = vf.shape[1]
  while not np.array_equal(vf, vf_prev):
  #for t in range(0,10):
    vf_prev = vf
    vf = np.zeros((num_rows,num_cols))
    for x in range(0,num_rows):
      for y in range(0,num_cols):
        vlist = [rf[x,y]+gamma*vf_prev[x,y]]
        try:
          if x+1 < num_rows:
            vlist.append(rf[min(x+1,num_rows-1),y] + gamma*vf_prev[min(x+1,num_rows-1),y])
        except:
          pass
        try: 
          if x-1 > -1:
            vlist.append(rf[max(x-1,0),y] + gamma*vf_prev[max(x-1,0),y])
        except:
          pass          
        try:
          if y+1 < num_cols: 
            vlist.append(rf[x,min(y+1,num_cols-1)] + gamma*vf_prev[x,min(y+1,num_cols-1)])
        except:
          pass
        try: 
          if y-1 > -1:
            vlist.append(rf[x,max(y-1,0)] + gamma*vf_prev[x,max(y-1,0)])
        except:
          pass
        vf[x,y] = max(vlist)
  return vf

class GridworldTDLearner(object):
    """This is our base TD Learner class, without any curiosity included.

    Designed to function on a two-dimensional gridworld, all the usual "vectors"
    for a TD-learner, like the weights, eligibility trace, feature vectors, etc.
    are all the same shape as the gridworld.

    Attributes:
        side_lengths (tuple): the length of each side of the grid (rows, columns)
            e.g. (11, 11) for a square gridworld
        V (np.array): the current estimate of the value function for each grid 
            coordinate; analogous to the weight "vector" for TD learning
        e (np.array): the eligibility trace "vector"
        
        state (tuple): the current coordinates of the learner on the grid; this 
            stands in as the feature vector for TD learning
        next_state (tuple): the previous coordinates
        R (float): the current reward (received in the transition from state to 
            next_state)
        
        delta (float): 
        
        alpha (float): the alpha parameter (step size) used to update weights
        lam (float): the lambda parameter used for eligibility trace updates
    """

  
    def __init__(self, side_lengths):
        """Initialize a new GridworldTDLearner

        Args:
            side_lengths (tuple): the length of each side of the grid 
                (rows, columns); e.g. (11, 11) for a square gridworld,
                (11,50) for a wide gridworld
        """
        self.side_lengths = side_lengths
        self.V = None
        self.e = None
        self.state = None
        self.next_state = None
        self.R = 0
        self.delta = 0
        self.alpha = 0.01
        self.lam = 0.0
        self.reset(True)
      
    def reset(self, zero_weights=False):
        """ Reset or initiaize all "vectors" for a new episode

        Args:
            zero_weights (bool): True iff the weight "vector" should be reset to
                a zero "vector".
        """
        self.e = np.zeros([self.side_lengths[0],self.side_lengths[1]])
        if zero_weights:
          self.V = np.zeros([self.side_lengths[0],self.side_lengths[1]])
      
    def get_blank_vector(self):
        """ New zero vector of same shape as other "vectors"
        """
        return np.zeros([self.side_lengths[0],self.side_lengths[1]])
      
    def update(self, state, next_state, R, gamma):
        """Method to update the learning weights based on a state and stuff
        """
        self.state = state
        self.next_state = next_state
        self.R = R
        self.delta = self.R + gamma*(self.V[next_state]) - self.V[state]
        # self.e = gamma*self.lam*self.e + state
        self.V[state] += self.alpha*self.delta

class CuriousTDLearner(GridworldTDLearner):
    """

    Attributes:
        spawn_point (tuple): This is the permanent, hard-coded  home of the 
            bookstore, saved as the coordinates (row, col) that consistently 
            induce curiosity in the agent.
        target (None or tuple): If curiosity has been induced in the agent, then
            a target is generated, represented as the coordinates (row, col) 
            that the agent will direct its behaviour towards as long as it
            remains curious.
    """

    def __init__(self, side_lengths, num_actions=8, target_col=1):
        """
        """
        super().__init__(side_lengths)

        self.spawn_point = (int(side_lengths[1]/2), int(side_lengths[1]/2))
        self.target = None
        self.vcurious = np.zeros(side_lengths)
        self.rcurious = np.zeros(side_lengths)
        self.action_list = [(0,1),
                            (0,-1),
                            (1,0),
                            (-1,0),
                            (1,1),
                            (1,-1),
                            (-1,1),
                            (1,-1)]
        self.target_row = target_col
        self.model = SimpleGridWorld(side_lengths)


    def update(self, state, next_state, R, gamma):
        """
        """
        if self.vcurious is None:
            vcurious = self.get_blank_vector()
        else:
            vcurious = self.vcurious
        self.state = state
        self.next_state = next_state
        self.R = R
        self.delta = self.R + gamma*self.V[next_state] - (self.V[state]+vcurious[state])
        # self.e = gamma*self.lam*self.e + state
        self.V[state] += self.alpha*self.delta

        if self.check_spawn_point(next_state):
            pass
            # print("Bookstore? Agent at", next_state)
            # plot_heatmap(self.V, title="Permanent Value", target=self.target,
            #     spawn=self.spawn_point, start=self.model.start_pos, agent=next_state)
            # plot_heatmap(self.vcurious, title="Transient Value", target=self.target,
            #     spawn=self.spawn_point, start=self.model.start_pos, agent=next_state)
        return self.is_target(next_state)



    def get_action(self, state, epsilon=0):
        """
        """
        best_actions = []
        max_qval = -np.inf

        pertinent_v = self.vcurious if self.target is not None else self.V

        for action in self.action_list:
            result_state = self.model.get_next_state_walled(state, action)
            # requires that the reward is zero
            qval = pertinent_v[result_state]
            if qval > max_qval:
                best_actions = [action]
                max_qval = qval
            elif qval == max_qval:
                best_actions.append(action)
        chosen_action = choice(best_actions)

        # only behave epsilon-greedily if not curious (ie self.target is None)
        if (np.random.rand() < epsilon) and (self.target is None):
            chosen_action = choice(self.action_list)
        return chosen_action

    def check_spawn_point(self, pos):
        if np.array_equal(pos, self.spawn_point):
            if self.target == None:
                new_target_col = np.random.randint(0,self.side_lengths[1])
                self.target = (self.target_row, new_target_col)

                self.rcurious = np.full(self.side_lengths, -1)
                self.rcurious[self.target] = 0

                self.vcurious = self.model.value_iteration(self.rcurious, 
                                                            gamma=0.9)
                return True
        return False

    def is_target(self, pos):
        if np.array_equal(pos, self.target):
            self.target = None

            self.rcurious = np.zeros(self.side_lengths)

            self.vcurious = np.zeros(self.side_lengths)

            return True
        return False
