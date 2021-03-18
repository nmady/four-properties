from agents import CuriousTDLearner
from environments import SimpleGridWorld
from visualization import plot_heatmap

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
        if i%100==0:
            print("Step ",i)
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
    print("Run Finished")
    return learner, world

if __name__=="__main__":
    batch_run_experiment(trials=2,steps=1000, dimensions = (11,11))

