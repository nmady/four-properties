from agents import CuriousTDLearner, GridworldTDLearner
from environments import SimpleGridWorld
from visualization import (
    plot_heatmap, plot_final_heatmap, plot_interim_heatmap,
    plot_lineplot, plot_lineplot_data)
import typer
from enum import Enum
import numpy as np
import pandas as pd
import os
import signal

def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    with open("./output/num_target_visits.csv", "a") as num_target_visits_f:
        num_target_visits_f.write(",Aborted!\n")
    exit(0)

signal.signal(signal.SIGINT, keyboardInterruptHandler)

def basic_timestep(
        world, 
        learner, 
        stepnum=None, 
        steps=None, 
        savepostfix="",
        animation=False):
    if (type(stepnum) is int and stepnum%100==0):
        print(".", end="", flush=True)
    world.visit(world.pos)

    if animation:
        if steps is None:
            plot_interim_heatmap(
                learner.V, stepnum, 
                target=learner.target, 
                spawn=learner.curiosity_inducing_state, 
                start=world.start_pos, agent=world.pos, 
                title="Value", 
                cmap="bwr_r", savepostfix=savepostfix
                )
        else:
            vmax = steps//500 if steps > 500 else steps/500
            plot_interim_heatmap(
                learner.V, stepnum, 
                target=learner.target, 
                spawn=learner.curiosity_inducing_state, 
                start=world.start_pos, 
                agent=world.pos, 
                vmin=-vmax, vmax=vmax, 
                title="Value", 
                cmap="bwr_r", savepostfix=savepostfix
                )

    action = learner.get_action(world.pos, epsilon=0.2)
    world.next_pos = world.get_next_state(world.pos, action)
    reward = world.get_reward()
    learner.update(world.pos, world.next_pos, reward, gamma=0.9)
    if learner.is_target(world.next_pos):
        world.visit(world.next_pos)
        world.next_pos = world.start_pos
    world.pos = world.next_pos

def get_ablation_postfix(**kwargs):
    postfix = ""
    for key, value in kwargs.items():
        if key == "directed":
            if not value:
                postfix += "_no_directed"
        elif key == "voluntary":
            if not value:
                postfix += "_no_voluntary"
        elif key == "aversive":
            if not value:
                postfix += "_no_aversive"
        elif key == "ceases":
            if not value:
                postfix += "_no_ceases"
        elif key == "positive":
            if value:
                postfix += "_yes_positive"
        elif key == "decays":
            if value:
                postfix += "_yes_decays"
        elif key == "flip_update":
            if value:
                postfix += "_yes_flip_update"
        elif key == "reward_bonus":
            if value:
                postfix += "_yes_reward_bonus"
        else:
            raise ValueError("Key " + key + " is not recognized.")
    return postfix

def batch_run_experiment(
        trials=1,
        steps=1000, 
        dimensions = (11,11), 
        figsize=None,
        learner_type=CuriousTDLearner, 
        animation=False,
        lineplot=True,
        **kwargs):
    
    # These "_stacked" variables will hold data for plotting aggregate data over
    # the batch of experiments. The first two are for heatmaps, which the second
    # two are for lineplots.
    value_stacked = None
    visit_stacked = None
    inducer_stacked = None
    df_stacked = None
    value_df = pd.DataFrame({"trial":[], "value":[], "type":[]})

    setup_info = (str(dimensions[0]) + "_" + str(dimensions[1]) 
                  + "_steps" + str(steps))
    ablation_postfix = get_ablation_postfix(**kwargs)

    if not os.path.exists("./output/num_target_visits.csv"):
        os.system("touch ./output/num_target_visits.csv")
    with open("./output/num_target_visits.csv", "a") as num_target_visits_f:
        num_target_visits_f.write(setup_info + ablation_postfix + ",")

    rng = np.random.default_rng(2021)

    for n in range(trials):
        print("\nTrial",n)
        postfix = "_" + setup_info + "_trial" + str(n) + ablation_postfix

        anim_postfix = str(postfix)

        learner, world, inducer_over_time, targets_over_time = basic_experiment(
            steps, dimensions, 
            rng=rng, learner_type=learner_type, 
            savepostfix=anim_postfix, 
            animation=animation,
            **kwargs
            )

        with open("./output/num_target_visits.csv", "a") as num_target_visits_f:
            num_target_visits_f.write(str(learner.num_target_visits) + ",")
        
        plot_final_heatmap(
            learner.V, 
            target=learner.target, spawn=learner.curiosity_inducing_state, 
            start=world.start_pos, agent=world.next_pos, 
            title="Value", cmap="bwr_r", vmin=-(steps//500), vmax=steps//500, 
            figsize=figsize, display="Save",savepostfix=postfix)
        plot_final_heatmap(
            world.visit_array, 
            title="Visits", cmap="bone", vmin=0, vmax=steps//10, 
            figsize=figsize, display="Save",savepostfix=postfix)

        
        if value_stacked is None:
            value_stacked = [learner.V]
        else:
            value_stacked.append(learner.V)

        if visit_stacked is None:
            visit_stacked = [world.visit_array]
        else:
            visit_stacked.append(world.visit_array)
    
        if lineplot:
        
            df_inducer = pd.DataFrame(
                {"Value":inducer_over_time, "Time":range(steps)}
                )
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
        
        if lineplot:
            plot_lineplot(range(steps), inducer_over_time, 
                title="Value of Curosity-inducing State", 
                xlabel="Time", ylabel="Value", 
                display="Save",
                savepostfix=postfix)

    with open("./output/num_target_visits.csv", "a") as num_target_visits_f:
        num_target_visits_f.write("\n")

    postfix = ("stackedMean_" + str(dimensions[0]) + "_" + str(dimensions[1])
        + "_steps"+str(steps))
    postfix += ablation_postfix
    plot_final_heatmap((np.array(value_stacked)).mean(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=None, title="Value", cmap="bwr_r", vmin=-(steps//500), vmax=steps//500, figsize=figsize, display="Save",savepostfix=postfix)

    postfix="stackedStd_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    postfix += ablation_postfix
    plot_final_heatmap((np.array(value_stacked)).std(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos, agent=None, title="Value", cmap="viridis",figsize=figsize, display="Save",savepostfix=postfix)

    postfix="stackedMean_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    postfix += ablation_postfix
    plot_final_heatmap((np.array(visit_stacked)).mean(axis=0), target=None, spawn=learner.curiosity_inducing_state, start=world.start_pos,  cmap="bone", vmin=0, vmax=steps//10, agent=None, title="Visits", figsize=figsize, display="Save",savepostfix=postfix)
    
    postfix="stackedStd_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
    postfix += ablation_postfix
    plot_final_heatmap((np.array(visit_stacked)).std(axis=0), target=None, spawn=learner.curiosity_inducing_state,start=world.start_pos,  cmap="viridis", agent=None, title="Visits", figsize=figsize, display="Save",savepostfix=postfix)

    if lineplot:
        postfix="stackedLineplot_"+str(dimensions[0])+"_"+str(dimensions[1])+"_steps"+str(steps)
        postfix += ablation_postfix
        plot_lineplot_data(pd.concat(df_stacked, sort=False),
            title="Value over Time", 
            xlabel="Time", ylabel="Value", 
            display="Save",
            savepostfix=postfix)

    if animation:
        os.system("ffmpeg -hide_banner -loglevel error -i ./output/Value_" 
                  + anim_postfix + "/%d.png ./output/" + anim_postfix + ".avi")


def basic_experiment(
        steps=1000, 
        dimensions = (11,11), 
        rng=None, learner_type=CuriousTDLearner, 
        savepostfix="",
        animation=False,
        **kwargs):
    gridworld_dimensions = dimensions
    total_steps = steps
        
    world = SimpleGridWorld(gridworld_dimensions)
    learner = learner_type(gridworld_dimensions, rng=rng, **kwargs)

    inducer_over_time = np.zeros(total_steps)
    all_targets = learner.get_all_possible_targets()
    targets_over_time = np.zeros((total_steps, len(all_targets)))

    print("Start pos:", world.start_pos)
        
    for i in range(total_steps):
        basic_timestep(world, learner, stepnum=i, 
                       steps=total_steps, savepostfix=savepostfix,
                       animation=animation)
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
        figwidth: float = typer.Option(None, 
            help="Width of heatmap figures in inches"),
        figheight: float = typer.Option(None, 
            help="Height of heatmap figure in inches"),
        learner_type: LearnerType = typer.Option(LearnerType.CuriousTDLearner, 
            help="Type of learner to experiment with"),
        directed: bool = typer.Option(True, 
            help="Set to False to ablate directed behaviour."),
        voluntary: bool = typer.Option(True, 
            help="Set to False to ablate voluntary."),
        aversive: bool = typer.Option(True, 
            help="Set to False to ablate aversive quality."),
        ceases: bool = typer.Option(True, 
            help="Set to False to ablate ceases when satisfied."),
        positive: bool = typer.Option(False, 
            help="Set to True to make the target have positive value/reward. " +
                "The parameter 'aversive' must be set to False."),
        decays: bool = typer.Option(False, 
            help="Set to True to make satisfying curiosity a slow decay. " + 
                "The parameter 'ceases' must be set to False."),
        flip_update: bool = typer.Option(False, 
            help="Set to True to add instead of subtract the " + 
                "curiosity value in the TD update."),
        reward_bonus: bool = typer.Option(False, 
            help="Set to True to add a bonus to the reward."),
        animation: bool = typer.Option(True, 
            help="If True, save plots of the value function with " + 
                "agent position at each timestep and output video animation."),
        lineplot: bool = typer.Option(True, 
            help="If True, output lineplots. (Slow.)")):
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
        animation=animation, lineplot=lineplot,
        directed=directed, voluntary=voluntary,
        aversive=aversive, ceases=ceases, 
        positive=positive, decays=decays,
        flip_update=flip_update,
        reward_bonus=reward_bonus
    )

if __name__ == "__main__":
    typer.run(main)
