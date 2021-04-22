from agents import CuriousTDLearner, GridworldTDLearner
from environments import SimpleGridWorld
from visualization import plot_heatmap, plot_lineplot
import typer
from enum import Enum
import random
import numpy as np


def basic_timestep(world, learner, stepnum=None):
    if (type(stepnum) is int and stepnum%100==0):
        print(".",end="",flush=True)
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


def batch_run_experiment(
                trials=1,
                steps=1000, 
                dimensions = (11,11), 
                learner_type=CuriousTDLearner, 
                directed=True, voluntary=True, aversive=True, ceases=True,
                positive=False, decays=False
):
    value_stacked = None
    visit_stacked = None

    ablation_postfix = ""
    if not directed:
        ablation_postfix += "_no_directed"
    if not voluntary:
        ablation_postfix += "_no_voluntary"
    if not aversive:
        ablation_postfix += "_no_aversive"
    if not ceases:
        ablation_postfix += "_no_ceases"
    if positive:
        ablation_postfix += "_yes_positive"
    if decays:
        ablation_postfix += "_yes_decays"

    for n in range(trials):
        print("\nTrial",n)
        random.seed(n)
        learner, world, inducer_over_time, avg_target_over_time = basic_experiment(steps, dimensions, learner_type=learner_type, directed=directed, voluntary=voluntary, aversive=aversive, ceases=ceases, positive=positive, decays=decays)
        postfix = "_"+str(dimensions[0])+"_"+str(dimensions[1])
        postfix += "_steps"+str(steps)+"_trial"+str(n)
        postfix += ablation_postfix
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
        print("Max at bookstore:", inducer_over_time.max(), "Max at targets:", avg_target_over_time.max())
        plot_lineplot(range(steps), inducer_over_time, 
            title="Inducer Value", 
            xlabel="Time", ylabel="Value of the Bookstore", 
            display="Save",
            savepostfix=postfix)

    postfix="stackedMean_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    postfix += ablation_postfix
    plot_heatmap((np.array(value_stacked)).mean(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=None, title="Value", cmap="viridis",display="Save",savepostfix=postfix)
    postfix="stackedStd_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    postfix += ablation_postfix
    plot_heatmap((np.array(value_stacked)).std(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=None, title="Value", cmap="viridis",display="Save",savepostfix=postfix)
    
    postfix="stackedMean_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    postfix += ablation_postfix
    plot_heatmap((np.array(visit_stacked)).mean(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=None, title="Visits", display="Save",savepostfix=postfix)
    postfix="stackedStd_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    postfix += ablation_postfix
    plot_heatmap((np.array(visit_stacked)).std(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=None, title="Visits", display="Save",savepostfix=postfix)
        

def basic_experiment(steps=1000, dimensions = (11,11), learner_type=CuriousTDLearner, directed=True, voluntary=True, aversive=True, ceases=True, positive=False, decays=False):
    gridworld_dimensions = dimensions
    total_steps = steps
        
    world = SimpleGridWorld(gridworld_dimensions)
    learner = learner_type(gridworld_dimensions, directed=directed, voluntary=voluntary, aversive=aversive, ceases=ceases, positive=positive, decays=decays)

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


class LearnerType(str, Enum):
    CuriousTDLearner = "CuriousTDLearner"
    GridworldTDLearner = "GridworldTDLearner"


def main(
    trials: int = typer.Option(1, help="Number of desired trials."),
    steps: int = typer.Option(200, help="Number of steps in each trial."),
    width: int = typer.Option(11, help="Width of gridworld."),
    height: int = typer.Option(11, help="Height of gridworld."),
    learner_type: LearnerType = typer.Option(LearnerType.CuriousTDLearner, help="Type of learner to experiment with"),
    directed: bool = typer.Option(True, help="Set to False to ablate directed behaviour."),
    voluntary: bool = typer.Option(True, help="Set to False to ablate voluntary."),
    aversive: bool = typer.Option(True, help="Set to False to ablate aversive quality."),
    ceases: bool = typer.Option(True, help="Set to False to ablate ceases when satisfied."),
    positive: bool = typer.Option(False, help="Set to True to make satisfying curiosity rewarding. The parameter 'aversive' must be set to False."),
    decays: bool = typer.Option(False, help="Set to True to make satisfying curiosity a slow decay. The parameter 'ceases' must be set to False.")
):
    """
    Run a batch of experiments.
    """

    if learner_type == LearnerType.CuriousTDLearner:
        l_type = CuriousTDLearner
    elif learner_type == LearnerType.StateCuriosityLearner:
        l_type = StateCuriosityLearner
    elif learner_type == LearnerType.GridworldTDLearner:
        l_type = GridworldTDLearner
    assert not (aversive and positive)
    assert not (ceases and decays)
    batch_run_experiment(
        trials=trials, steps=steps,
        dimensions=(height, width), learner_type=l_type,
        directed=directed, voluntary=voluntary,
        aversive=aversive,
        ceases=ceases, positive=positive, decays=decays
    )


if __name__ == "__main__":
    typer.run(main)
