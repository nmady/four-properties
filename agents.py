import numpy as np

from environments import SimpleGridWorld
from visualization import plot_heatmap

from itertools import product


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
        next_state (tuple): the next coordinates
        R (float): the current reward (received in the transition from state to 
            next_state)
        
        delta (float): 
        
        alpha (float): the alpha parameter (step size) used to update weights
        lam (float): the lambda parameter used for eligibility trace updates
    """

  
    def __init__(self, side_lengths: tuple):
        """Initialize a new GridworldTDLearner

        Args:
            side_lengths (tuple): the length of each side of the grid 
                (rows, columns); e.g. (11, 11) for a square gridworld,
                (11,50) for a wide gridworld
        Raises:
            AssertionError : if side_lengths is not a tuple
        """

        assert type(side_lengths) is tuple

        self.side_lengths = side_lengths
        self.V = None
        self.e = None
        self.state = None
        self.next_state = None
        self.reward = 0
        self.delta = 0
        self.alpha = 0.01
        self.lam = 0.0

        self.reset(True)
      

    def reset(self, zero_weights=False):
        """ Zero eligibility trace and/or value "vectors" for a new episode

        Args:
            zero_weights (bool): True iff the weight "vector" should be reset to
                a zero "vector".
        """
        self.e = np.zeros(self.side_lengths)
        if zero_weights:
          self.V = np.zeros(self.side_lengths)
      

    def update(self, state: tuple, next_state: tuple, reward, gamma):
        """TD update V based on the transition from state to next_state
        
        Also sets the internal record of the state, next_state, and reward

        Args:
            state (tuple), next_state (tuple), R (real number) : are as 
                documented as Attributes of GridWorldTDLearner
            gamma (float) : the discount factor, between 0 and 1 inclusive

        """

        assert self.V.shape == self.side_lengths

        self.state = state
        self.next_state = next_state
        self.reward = reward
        self.delta = self.reward + gamma*self.V[next_state] - self.V[state]
        # self.e = gamma*self.lam*self.e + state
        self.V[state] += self.alpha*self.delta


class CuriousTDLearner(GridworldTDLearner):
    """A GridworldTDLearner with simple specific curiosity demonstration.

    TODO: This probably needs more explanation

    Attributes:
        See GridworldTDLearner for base class attributes.
        curiosity_inducing_state (tuple): This is the permanent, hard-coded  home of the 
            "bookstore", saved as the coordinates (row, col) that consistently 
            induce curiosity in the agent.
        target (None or tuple): If curiosity has been induced in the agent, then
            a target is generated, represented as the coordinates (row, col) 
            that the agent will direct its behaviour towards as long as it
            remains curious. Otherwise set to None.
        vcurious (numpy array): A transient value function that directs the 
            learner towards the target (set through value iteration) when the 
            learner is curious, and is zero everywhere otherwise.
        rcurious (numpy array): A transient reward function that, when the 
            learner is curious, is set to -1 everywhere except the target, and 
            zero everywhere when the learner is not curious.
        target_row (int): All targets will be generated in the target_row of the
            gridworld.
        model (SimpleGridWorld): model of the world, used for action selection
            and value iteration to set vcurious
        
    """

    def __init__(self, side_lengths, rng=None, target_row=1, directed=True, voluntary=True, aversive=True, ceases=True, positive=False, decays=False, flip_update=False):
        """Initialize a new CuriousTDLearner


        """
        super().__init__(side_lengths)

        assert 0 <= target_row < side_lengths[0]

        self.directed = directed
        self.voluntary = voluntary
        self.aversive = aversive
        self.ceases = ceases
        self.positive = positive
        self.decays = decays
        self.flip_update = flip_update

        self.curiosity_inducing_state = (side_lengths[0]-6, side_lengths[1]//2)
        self.target = None
        self.vcurious = np.zeros(side_lengths)
        self.rcurious = np.zeros(side_lengths)
        self.target_row = target_row
        self.model = SimpleGridWorld(side_lengths)

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()


    def update(self, state, next_state, reward, gamma):
        """
        """

        self.state = state
        self.next_state = next_state
        self.reward = reward

        if self.flip_update:
            if self.vcurious is None:
                vcurious_at_next_state = 0
            else:
                vcurious_at_next_state = self.vcurious[next_state]
            self.delta = (self.reward + gamma*(self.V[next_state]+vcurious_at_next_state) - 
                self.V[state])
        else:
            if self.vcurious is None:
                vcurious_at_state = 0
            else:
                vcurious_at_state = self.vcurious[state]
            self.delta = (self.reward + gamma*self.V[next_state] - 
                (self.V[state]+vcurious_at_state))

        if not self.voluntary:
            self.delta = self.reward + gamma*self.V[next_state] - self.V[state]
        self.V[state] += self.alpha*self.delta

        if self.is_curiosity_inducing(next_state):
            pass


    def get_action(self, state, epsilon=0):
        """
        """
        best_actions = []
        max_qval = -np.inf

        pertinent_v = self.vcurious if self.target is not None else self.V

        # For ablation study removing directed behaviour (see also epsilon greedy below)
        if not self.directed:
            pertinent_v = self.V

        for action in self.model.actions:
            result_state = self.model.get_next_state(state, action)
            # requires that the reward is zero
            qval = pertinent_v[result_state]
            if qval > max_qval:
                best_actions = [action]
                max_qval = qval
            elif qval == max_qval:
                best_actions.append(action)
        rindex = self.rng.integers(low=0, high=len(best_actions))
        chosen_action = best_actions[rindex]

        # only behave epsilon-greedily if not curious (ie self.target is None)
        # (or if we're ablating directed behaviour...)
        if (self.rng.random() < epsilon) and ((self.target is None) or (self.directed is False)):
            rindex = self.rng.integers(low=0, high=len(self.model.actions))
            chosen_action = self.model.actions[rindex]

        return chosen_action


    def is_curiosity_inducing(self, state):
        """ If state induces curiosity, sets up curiosity target, reward, value

        Note that the curiosity_inducing_state only induces curiosity when the 
        learner isn't already curious!

        Args:
            state (tuple)

        Returns:
            bool
        """
        if state == self.curiosity_inducing_state and self.target == None:

            # Targets are not allowed to be right next to the wall!
            new_target_col = self.rng.integers(low=1,high=self.side_lengths[1]-1)
            
            self.target = (self.target_row, new_target_col)

            self.rcurious = np.full(self.side_lengths, -1)
            self.rcurious[self.target] = 0

            # Ablation study: removing aversive quality
            if not self.aversive:
                self.rcurious = np.zeros(self.side_lengths)

                # Testing to see how different things are when experience is of 
                # a positive reward upon satisfying curiosity.
                if self.positive:
                    self.rcurious[self.target] = 1

            self.vcurious = self.model.value_iteration(self.rcurious, 
                                                        gamma=0.9)
            return True
        return False


    def get_all_possible_targets(self):
        return [(self.target_row, col) for col in range(1,self.side_lengths[1]-1)]


    def is_target(self, state):
        """ If state is the target, zeros out curiosity reward and value

        Args:
            state (tuple)

        Returns:
            bool
        """

        decay_rate = 0.8
        eps = 0.1

        if (state == self.target):
            if self.ceases:
                self.target = None

                self.rcurious.fill(0)

                self.vcurious.fill(0)

                return True
            if self.decays:

                # check to see if rcurious is basically zero.
                if np.any(np.absolute(self.rcurious) < eps):

                    self.target = None

                self.rcurious = self.rcurious * decay_rate
                self.vcurious = self.model.value_iteration(self.rcurious, 
                                                        gamma=0.9)
        return False