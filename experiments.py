from agents import CuriousTDLearner
from environments import SimpleGridWorld
from visualization import plot_heatmap


def basic_experiment():
	gridworld_dimensions = 11,11
	total_steps = 5000

	world = SimpleGridWorld(gridworld_dimensions)
	learner = CuriousTDLearner(gridworld_dimensions)

	print("Start pos:", world.start_pos)

	for i in range(total_steps):
		world.visit(world.pos)
		action = learner.get_action(world.pos, epsilon=0.2)
		next_pos = world.get_next_state_walled(world.pos, action)
		R = world.get_reward()
		satisfied = learner.update(world.pos, next_pos, R, gamma=0.9)
		if satisfied:
			world.visit(next_pos)
			next_pos = world.start_pos
		world.pos = next_pos
	plot_heatmap(learner.V, target=learner.target, spawn=learner.spawn_point,start=world.start_pos, agent=next_pos, title="Value", cmap="viridis")
	plot_heatmap(world.visit_array, title="Visits")
	print(world.visit_array)

if __name__=="__main__":
	basic_experiment()