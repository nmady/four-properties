from agents import CuriousTDLearner
from environments import SimpleGridWorld
from visualization import plot_heatmap
import numpy as np

def basic_timestep(world, learner, stepnum=None):
    if (type(stepnum) is int and stepnum%100==0):
        print("Step ", stepnum)
    world.visit(world.pos)
    action = learner.get_action(world.pos, epsilon=0.2)
    next_pos = world.get_next_state(world.pos, action)
    #assert next_pos[0] >= 0, "Agent pos[0] outside bounds of world: %i" % next_pos[0]
    reward = world.get_reward()
    learner.update(world.pos, next_pos, reward, gamma=0.9)
    if learner.is_target(next_pos):
        world.visit(next_pos)
        next_pos = world.start_pos
    world.pos = next_pos

def batch_run_experiment(trials=1,steps=1000, dimensions = (11,11)):
    for n in range(trials):
        learner, world = basic_experiment(steps, dimensions)
        postfix="trial"+str(n)+"_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
        plot_heatmap(learner.V, target=learner.target, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=world.next_pos, title="Value", cmap="viridis",display="Save",savepostfix=postfix)
        plot_heatmap(world.visit_array, title="Visits",display="Save",savepostfix=postfix)
        #print(learner.V)

def basic_experiment(steps=1000, dimensions = (11,11)):
    gridworld_dimensions = dimensions
    total_steps = steps
        
    world = SimpleGridWorld(gridworld_dimensions)
    learner = CuriousTDLearner(gridworld_dimensions)
        
    print("Start pos:", world.start_pos)
        
    for i in range(total_steps):
        basic_timestep(world, learner, stepnum=i)
    print("Run Finished")
    return learner, world

def linegraph_experiment():
    """ Collect value of the bookstore and average value over all possible targets at each timestep"""

    gridworld_dimensions = 11,11
    total_steps = 500

    world = SimpleGridWorld(gridworld_dimensions)
    learner = CuriousTDLearner(gridworld_dimensions)

    inducer_over_time = np.zeros(total_steps)
    avg_target_over_time = np.zeros(total_steps)
    all_targets = learner.get_all_possible_targets()

    print("Start pos:", world.start_pos)
    for i in range(total_steps):
        basic_timestep(world, learner, stepnum=i)
        
        inducer_over_time[i] = learner.V[learner.curiosity_inducing_state]
        avg_target_over_time[i] = np.mean(np.array([learner.V[t] for t in all_targets]))


    print("Run Finished")
    plot_heatmap(learner.V, target=learner.target, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=world.pos, title="Value", cmap="viridis",display="Save")
    plot_heatmap(world.visit_array, title="Visits",display="Save")

if __name__=="__main__":
    linegraph_experiment()
    batch_run_experiment(trials=2,steps=1000, dimensions = (11,11))

