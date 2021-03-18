from agents import CuriousTDLearner
from environments import SimpleGridWorld
from visualization import plot_heatmap
import random
import numpy as np

def basic_timestep(world, learner, stepnum=None):
    if (type(stepnum) is int and stepnum%100==0):
        print(".",end="")
    world.visit(world.pos)
    action = learner.get_action(world.pos, epsilon=0.2)
    world.next_pos = world.get_next_state(world.pos, action)
    #assert next_pos[0] >= 0, "Agent pos[0] outside bounds of world: %i" % next_pos[0]
    reward = world.get_reward()
    learner.update(world.pos, world.next_pos, reward, gamma=0.9)
    if learner.is_target(world.next_pos):
        world.visit(world.next_pos)
        world.next_pos = world.start_pos
    world.pos = world.next_pos

def batch_run_experiment(trials=1,steps=1000, dimensions = (11,11)):
    value_stacked = None
    visit_stacked = None
    for n in range(trials):
        print("\nTrial",n)
        random.seed(n)
        learner, world, inducer_over_time, avg_target_over_time = basic_experiment(steps, dimensions)
        postfix="trial"+str(n)+"_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
        plot_heatmap(learner.V, target=learner.target, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=world.next_pos, title="Value", cmap="viridis",display="Save",savepostfix=postfix)
        plot_heatmap(world.visit_array, title="Visits",display="Save",savepostfix=postfix)

        if value_stacked is None:
            value_stacked = [learner.V]
        else:
            value_stacked.append(learner.V)

        if visit_stacked is None:
            visit_stacked = [world.visit_array]
        else:
            visit_stacked.append(world.visit_array)
            
        print(inducer_over_time.mean(),avg_target_over_time.mean())

    postfix="stackedMean_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    plot_heatmap((np.array(value_stacked)).mean(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=None, title="Value", cmap="viridis",display="Save",savepostfix=postfix)
    postfix="stackedStd_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    plot_heatmap((np.array(value_stacked)).std(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=None, title="Value", cmap="viridis",display="Save",savepostfix=postfix)
    postfix="stackedMean_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)

    plot_heatmap((np.array(visit_stacked)).mean(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=None, title="Visits", display="Save",savepostfix=postfix)
    postfix="stackedStd_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    plot_heatmap((np.array(visit_stacked)).std(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=None, title="Visits", display="Save",savepostfix=postfix)
        

def basic_experiment(steps=1000, dimensions = (11,11)):
    gridworld_dimensions = dimensions
    total_steps = steps
        
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
        
    print("Trial Finished")
    return learner, world, inducer_over_time, avg_target_over_time

if __name__=="__main__":
    batch_run_experiment(trials=5,steps=500, dimensions = (11,11))


