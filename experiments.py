from agents import CuriousTDLearner
from environments import SimpleGridWorld
from visualization import plot_heatmap


def basic_experiment():
	gridworld_dimensions = 11,11
	total_steps = 500

	world = SimpleGridWorld(gridworld_dimensions)
	learner = CuriousTDLearner(gridworld_dimensions)

	print("Start pos:", world.start_pos)

	for i in range(total_steps):
		if i%100==0:
			print("Step ",i)
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
	print("Run Finished")
	plot_heatmap(learner.V, target=learner.target, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=next_pos, title="Value", cmap="viridis",display="Save")
	plot_heatmap(world.visit_array, title="Visits",display="Save")
	print(world.visit_array)

if __name__=="__main__":
	basic_experiment()

