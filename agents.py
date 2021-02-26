import numpy as np

class GridworldTDLearner(object):
    """This is our base TD Learner class, without any curiosity included.

    Designed to function on a two-dimensional gridworld, all the usual "vectors"
    for a TD-learner, like the weights, eligibility trace, feature vectors, etc.
    are all the same shape as the gridworld.

    Attributes:
        side_lengths (tuple): the length of each side of the grid (width, height)
            e.g. (11, 11) for a square gridworld
        w (np.array): the weight "vector" for TD learning
        e (np.array): the eligibility trace "vector"
        xt (np.array): the current feature "vector" representing the state
        xtp1 (np.array): the previous feature "vector"
        delta (float): 
        R (float):
        alpha (float): 
        lam (float): the lambda parameter used for eligibility trace updates
    """

  
    def __init__(self, side_lengths):
        """
        """
        self.side_lengths = side_lengths
        self.w = None
        self.e = None
        self.xt = None
        self.xtp1 = None
        self.delta = 0
        self.R = 0
        self.alpha = 0.01
        self.lam = 0.0
        self.reset(True)
      
    def reset(self, zero_weights=False):
        """ Reset or initiaize all "vectors" for a new episode

        Args:
            zero_weights (np.array): 
        """
        self.e = np.zeros([self.side_lengths[0],self.side_lengths[1]])
        if zero_weights:
          self.w = np.zeros([self.side_lengths[0],self.side_lengths[1]])
      
    def get_blank_vector(self):
        """ New zero vector of same shape as other "vectors"
        """
        return np.zeros([self.side_lengths[0],self.side_lengths[1]])
      
    def update(self, xt, xtp1, R, gamma, vshort=None):
        """Method to update the learning weights based on a state and stuff
        """
        self.xt = xt
        self.xtp1 = xtp1
        self.R = R
        self.delta = self.R + gamma*(self.w[xtp1[0],xtp1[1]]) - self.w[xt[0],xt[1]]
        # self.e = gamma*self.lam*self.e + xt
        self.w[xt[0],xt[1]] += self.alpha*self.delta