from agents import CuriousTDLearner, GridworldTDLearner
from environments import SimpleGridWorld
from visualization import plot_heatmap, plot_lineplot, plot_lineplot_data
import typer
from enum import Enum
import numpy as np
import pandas as pd


def basic_timestep(world, learner, stepnum=None):
    if (type(stepnum) is int and stepnum%100==0):
        print(".", end="", flush=True)
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
                figsize=None,
                learner_type=CuriousTDLearner, 
                directed=True, voluntary=True, aversive=True, ceases=True,
                positive=False, decays=False,
                flip_update=False,
                reward_bonus=False
):
    
    # These "_stacked" variables will hold data for plotting aggregate data over
    # the batch of experiments. The first two are for heatmaps, which the second
    # two are for lineplots.
    value_stacked = None
    visit_stacked = None
    inducer_stacked = None
    df_stacked = None
    value_df = pd.DataFrame({"trial":[], "value":[], "type":[]})

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

    rng = np.random.default_rng(2021)

    for n in range(trials):
        print("\nTrial",n)
        learner, world, inducer_over_time, targets_over_time = basic_experiment(steps, dimensions, rng=rng, learner_type=learner_type, directed=directed, voluntary=voluntary, aversive=aversive, ceases=ceases, positive=positive, decays=decays, flip_update=flip_update, reward_bonus=reward_bonus)
        postfix = "_"+str(dimensions[0])+"_"+str(dimensions[1])
        postfix += "_steps"+str(steps)+"_trial"+str(n)
        postfix += ablation_postfix
        plot_heatmap(learner.V, target=learner.target, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=world.next_pos, title="Value", cmap="bwr_r", vmin=-(steps//500), vmax=steps//500, figsize=figsize, display="Save",savepostfix=postfix)
        plot_heatmap(world.visit_array, title="Visits", cmap="bone", vmin=0, vmax=steps//10, figsize=figsize, display="Save",savepostfix=postfix)

        if value_stacked is None:
            value_stacked = [learner.V]
        else:
            value_stacked.append(learner.V)

        if visit_stacked is None:
            visit_stacked = [world.visit_array]
        else:
            visit_stacked.append(world.visit_array)

        df_inducer = pd.DataFrame({"Value":inducer_over_time, "Time":range(steps)})
        df_inducer["Trial"] = n
        df_inducer["Type"] = "Curiosity-inducing State"
        df_target_frame = []
        for target_num, vector in enumerate(targets_over_time.transpose()):
            df_target_temp = pd.DataFrame({"Value":vector, "Time":range(steps)})
            df_target_temp["Target"] = target_num
            df_target_frame.append(df_target_temp)
        df_targets = pd.concat(df_target_frame)
        df_targets["Trial"] = n
        df_targets["Type"] = "Target"
        if df_stacked is None:
            df_stacked = [df_inducer, df_targets]
        else:
            df_stacked.append(df_inducer)
            df_stacked.append(df_targets)
        if inducer_stacked is None:
            inducer_stacked = [inducer_over_time]
        else:
            inducer_stacked.append(inducer_over_time)
            
        print("Mean at bookstore:", inducer_over_time.mean(), 
            "Mean over targets:", targets_over_time.mean())
        print("Max at bookstore:", inducer_over_time.max(),
            "Max over targets:", targets_over_time.max())
        plot_lineplot(range(steps), inducer_over_time, 
            title="Value of Curosity-inducing State", 
            xlabel="Time", ylabel="Value", 
            display="Save",
            savepostfix=postfix)

    postfix="stackedMean_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    postfix += ablation_postfix
    plot_heatmap((np.array(value_stacked)).mean(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=None, title="Value", cmap="bwr_r", vmin=-(steps//500), vmax=steps//500, figsize=figsize, display="Save",savepostfix=postfix)
    postfix="stackedStd_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    postfix += ablation_postfix
    plot_heatmap((np.array(value_stacked)).std(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=None, title="Value", cmap="viridis",figsize=figsize, display="Save",savepostfix=postfix)
    
    postfix="stackedMean_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    postfix += ablation_postfix
    plot_heatmap((np.array(visit_stacked)).mean(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos,  cmap="bone", vmin=0, vmax=steps//10, agent=None, title="Visits", figsize=figsize, display="Save",savepostfix=postfix)
    postfix="stackedStd_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    postfix += ablation_postfix
    plot_heatmap((np.array(visit_stacked)).std(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos,  cmap="viridis", agent=None, title="Visits", figsize=figsize, display="Save",savepostfix=postfix)
        
    postfix="stackedLineplot_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    postfix += ablation_postfix
    plot_lineplot_data(pd.concat(df_stacked, sort=False),
        title="Value over Time", 
        xlabel="Time", ylabel="Value", 
        display="Save",
        savepostfix=postfix)


def basic_experiment(steps=1000, dimensions = (11,11), rng=None, learner_type=CuriousTDLearner, directed=True, voluntary=True, aversive=True, ceases=True, positive=False, decays=False, flip_update=False, reward_bonus=False):
    gridworld_dimensions = dimensions
    total_steps = steps
        
    world = SimpleGridWorld(gridworld_dimensions)
    learner = learner_type(gridworld_dimensions, rng=rng, directed=directed, voluntary=voluntary, aversive=aversive, ceases=ceases, positive=positive, decays=decays, flip_update=flip_update, reward_bonus=reward_bonus)

    inducer_over_time = np.zeros(total_steps)
    all_targets = learner.get_all_possible_targets()
    targets_over_time = np.zeros((total_steps, len(all_targets)))

    print("Start pos:", world.start_pos)
        
    for i in range(total_steps):
        basic_timestep(world, learner, stepnum=i)
        inducer_over_time[i] = learner.V[learner.curiosity_inducing_state]
        targets_over_time[i] = [learner.V[t] for t in all_targets]
        
    print("Trial Finished")
    return learner, world, inducer_over_time, targets_over_time

class LearnerType(str, Enum):
    CuriousTDLearner = "CuriousTDLearner"
    GridworldTDLearner = "GridworldTDLearner"


def main(
    trials: int = typer.Option(1, help="Number of desired trials."),
    steps: int = typer.Option(200, help="Number of steps in each trial."),
    width: int = typer.Option(11, help="Width of gridworld."),
    height: int = typer.Option(11, help="Height of gridworld."),
    figwidth: float = typer.Option(None, help="Width of heatmap figures in inches"),
    figheight: float = typer.Option(None, help="Height of heatmap figure in inches"),
    learner_type: LearnerType = typer.Option(LearnerType.CuriousTDLearner, help="Type of learner to experiment with"),
    directed: bool = typer.Option(True, help="Set to False to ablate directed behaviour."),
    voluntary: bool = typer.Option(True, help="Set to False to ablate voluntary."),
    aversive: bool = typer.Option(True, help="Set to False to ablate aversive quality."),
    ceases: bool = typer.Option(True, help="Set to False to ablate ceases when satisfied."),
    positive: bool = typer.Option(False, help="Set to True to make the target have positive value/reward. The parameter 'aversive' must be set to False."),
    decays: bool = typer.Option(False, help="Set to True to make satisfying curiosity a slow decay. The parameter 'ceases' must be set to False."),
    flip_update: bool = typer.Option(False, help="Set to True to add instead of subtract the curiosity-value in the ."),
    reward_bonus: bool = typer.Option(False, help="Set to True to add a bonus to the reward.")
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
    if ((figwidth is not None and figheight is None) or 
        (figheight is not None and figwidth is None)):
        raise ValueError(
            "We need both figwidth and figheight to set figsize."
            )
    if figwidth is None and figheight is None:
        figsize = None
    else:
        figsize = (figwidth, figheight)
    batch_run_experiment(
        trials=trials, steps=steps,
        dimensions=(height, width), 
        figsize=figsize,
        learner_type=l_type,
        directed=directed, voluntary=voluntary,
        aversive=aversive,
        ceases=ceases, positive=positive, decays=decays,
        flip_update=flip_update,
        reward_bonus=reward_bonus
    )

if __name__ == "__main__":
    typer.run(main)
