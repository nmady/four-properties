from agents import CuriousTDLearner, GridworldTDLearner
from environments import SimpleGridWorld, CylinderGridWorld
from visualization import (
    plot_heatmap, plot_final_heatmap, plot_interim_heatmap,
    save_lineplot, plot_both_value_heatmaps)
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import typer
from enum import Enum
import numpy as np
import pandas as pd
import os

def get_curiosity_vmax(learner, gamma):
    max = 0
    for target in learner.get_all_possible_targets():
        rcurious = learner.get_new_rcurious(target)
        interim_max = np.max(np.abs(learner.model.value_iteration(rcurious, 
                                                        gamma=gamma)))
        if interim_max > max:
            max = interim_max
    return max


def basic_timestep(
        world, 
        learner, 
        gamma,
        stepnum=None, 
        steps=None, 
        savepostfix="",
        animation=False,
        figure2=False,
        scaling_constant=3,
        teleport=True):
    if (type(stepnum) is int and stepnum%100==0):
        print(".", end="", flush=True)
    world.visit(world.pos)

    learner.is_curiosity_inducing(world.pos, gamma)

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
            # plot_interim_heatmap(
            #     learner.V, stepnum, 
            #     target=learner.target, 
            #     spawn=learner.curiosity_inducing_state, 
            #     start=world.start_pos, 
            #     agent=world.pos, 
            #     vmin=-vmax, vmax=vmax, 
            #     title="Value", 
            #     cmap="bwr_r", savepostfix=savepostfix
            #     )
            plot_both_value_heatmaps(learner.V, learner.vcurious, stepnum, 
                title=["Learned Value Function", "Curiosity Value Function"],
                target=learner.target, 
                spawn=learner.curiosity_inducing_state, 
                start=world.start_pos, agent=world.pos, 
                vmin=-vmax, vmax=vmax,
                cmap="bwr_r", savepostfix=savepostfix
                )

    action = learner.get_action(world.pos, epsilon=0.2)
    
    # debugging code to create a nice file of value function printouts
    with open('output/' + savepostfix.replace(" ", "_") + '.txt', 'a') as f:
        print("t="+str(stepnum), action, file=f)
        np.set_printoptions(floatmode='maxprec_equal', 
            linewidth=100000, 
            threshold=sys.maxsize)
        print(learner.V, file=f)
        print(learner.vcurious, file=f)

    if world.pos == learner.curiosity_inducing_state and figure2:
        plot_final_heatmap(learner.rcurious,
            target=learner.target,
            spawn=learner.curiosity_inducing_state,
            start=world.start_pos,
            agent=world.pos,
            linewidths=0.05,
            linecolor='#AAAAAA22',
            vmin=-10, vmax=10,
            cmap="bwr_r", 
            xticklabels=False,
            savepostfix="figure2a_Rcurious"+str(stepnum)+savepostfix, 
            scaling_constant=scaling_constant, 
            display="Save")
        plot_final_heatmap(learner.rcurious,
            target=learner.target,
            spawn=learner.curiosity_inducing_state,
            start=world.start_pos,
            agent=world.pos,
            linewidths=0.05,
            linecolor='#AAAAAA22',
            vmin=-10, vmax=10,
            cmap="bwr_r", 
            xticklabels=False,
            yticklabels=False,
            savepostfix="figure2d_Rcurious"+str(stepnum)+savepostfix, 
            scaling_constant=scaling_constant, 
            display="Save")
        plot_final_heatmap(learner.vcurious,
            target=learner.target,
            spawn=learner.curiosity_inducing_state,
            start=world.start_pos,
            agent=world.pos,
            linewidths=0.05,
            linecolor='#AAAAAA22',
            vmin=-10, vmax=10,
            cmap="bwr_r", 
            xticklabels=False,
            savepostfix="figure2a_Vcurious"+str(stepnum)+savepostfix, 
            scaling_constant=scaling_constant, 
            display="Save")
        plot_final_heatmap(learner.vcurious,
            target=learner.target,
            spawn=learner.curiosity_inducing_state,
            start=world.start_pos,
            agent=world.pos,
            linewidths=0.05,
            linecolor='#AAAAAA22',
            vmin=-10, vmax=10,
            cmap="bwr_r", 
            xticklabels=False,
            yticklabels=False,
            savepostfix="figure2d_Vcurious"+str(stepnum)+savepostfix, 
            scaling_constant=scaling_constant, 
            display="Save")
        plot_final_heatmap(learner.V,
            target=learner.target,
            spawn=learner.curiosity_inducing_state,
            start=world.start_pos,
            agent=world.pos,
            linewidths=0.05,
            linecolor='#AAAAAA22',
            vmin=-0.035, vmax=0.035,
            xticklabels=False,
            cmap="bwr_r", savepostfix="figure2a_V"+str(stepnum)+savepostfix, 
            scaling_constant=scaling_constant, 
            display="Save")
        plot_final_heatmap(learner.V,
            target=learner.target,
            spawn=learner.curiosity_inducing_state,
            start=world.start_pos,
            agent=world.pos,
            linewidths=0.05,
            linecolor='#AAAAAA22',
            vmin=-0.035, vmax=0.035,
            yticklabels=False,
            xticklabels=False,
            cmap="bwr_r", savepostfix="figure2d_V"+str(stepnum)+savepostfix, 
            scaling_constant=scaling_constant, 
            display="Save")
        plot_final_heatmap(world.visit_array,
            target=learner.target,
            spawn=learner.curiosity_inducing_state,
            start=world.start_pos,
            agent=world.pos,
            linewidths=0.05,
            linecolor='#AAAAAA22',
            vmin=0, vmax=2,
            cmap="bone", savepostfix="figure2a_visits"+str(stepnum)+savepostfix,
            scaling_constant=scaling_constant, 
            count=True,
            display="Save")
        plot_final_heatmap(world.visit_array,
            target=learner.target,
            spawn=learner.curiosity_inducing_state,
            start=world.start_pos,
            agent=world.pos,
            linewidths=0.05,
            linecolor='#AAAAAA22',
            vmin=0, vmax=2,
            yticklabels=False,
            count=True,
            cmap="bone", savepostfix="figure2d_visits"+str(stepnum)+savepostfix,
            scaling_constant=scaling_constant, 
            display="Save")

    if world.pos == world.funnel_point and figure2:
        plot_final_heatmap(learner.rcurious,
            target=learner.target,
            spawn=learner.curiosity_inducing_state,
            start=world.start_pos,
            agent=world.pos,
            linewidths=0.05,
            linecolor='#AAAAAA22',
            vmin=-10, vmax=10,
            cmap="bwr_r", 
            yticklabels=False,
            xticklabels=False,
            savepostfix="figure2c_Rcurious"+str(stepnum)+savepostfix, 
            scaling_constant=scaling_constant, 
            display="Save")
        plot_final_heatmap(learner.vcurious,
            target=learner.target,
            spawn=learner.curiosity_inducing_state,
            start=world.start_pos,
            agent=world.pos,
            linewidths=0.05,
            linecolor='#AAAAAA22',
            vmin=-10, vmax=10,
            cmap="bwr_r", 
            yticklabels=False,
            xticklabels=False,
            savepostfix="figure2c_Vcurious"+str(stepnum)+savepostfix, 
            scaling_constant=scaling_constant, 
            display="Save")
        plot_final_heatmap(learner.V,
            target=learner.target,
            spawn=learner.curiosity_inducing_state,
            start=world.start_pos,
            agent=world.pos,
            linewidths=0.05,
            linecolor='#AAAAAA22',
            vmin=-0.035, vmax=0.035,
            yticklabels=False,
            xticklabels=False,
            cmap="bwr_r", savepostfix="figure2c_V"+str(stepnum)+savepostfix, 
            scaling_constant=scaling_constant, 
            display="Save")
        plot_final_heatmap(world.visit_array,
            target=learner.target,
            spawn=learner.curiosity_inducing_state,
            start=world.start_pos,
            agent=world.pos,
            linewidths=0.05,
            linecolor='#AAAAAA22',
            vmin=0, vmax=2,
            yticklabels=False,
            count=True,
            cmap="bone", savepostfix="figure2c_visits"+str(stepnum)+savepostfix,
            scaling_constant=scaling_constant, 
            display="Save")

    world.next_pos = world.get_next_state(world.pos, action)
    reward = world.get_reward()
    learner.update(world.pos, world.next_pos, reward, gamma=gamma)
    
    if learner.is_target(world.next_pos):
        if figure2:
            plot_final_heatmap(learner.rcurious,
                target=learner.target,
                spawn=learner.curiosity_inducing_state,
                start=world.start_pos,
                agent=world.next_pos,
                linewidths=0.05,
                linecolor='#AAAAAA22',
                vmin=-10, vmax=10,
                yticklabels=False,
                xticklabels=False,
                cmap="bwr_r", 
                savepostfix="figure2b_Rcurious"+str(stepnum)+savepostfix, 
                scaling_constant=scaling_constant, display="Save")
            plot_final_heatmap(learner.vcurious,
                target=learner.target,
                spawn=learner.curiosity_inducing_state,
                start=world.start_pos,
                agent=world.next_pos,
                linewidths=0.05,
                linecolor='#AAAAAA22',
                vmin=-10, vmax=10,
                yticklabels=False,
                xticklabels=False,
                cmap="bwr_r", 
                savepostfix="figure2b_Vcurious"+str(stepnum)+savepostfix, 
                scaling_constant=scaling_constant, display="Save")
            plot_final_heatmap(learner.V,
                target=learner.target,
                spawn=learner.curiosity_inducing_state,
                start=world.start_pos,
                agent=world.next_pos,
                linewidths=0.05,
                linecolor='#AAAAAA22',
                vmin=-0.035, vmax=0.035,
                yticklabels=False,
                xticklabels=False,
                cmap="bwr_r", savepostfix="figure2b_V"+str(stepnum)+savepostfix, 
                scaling_constant=scaling_constant, display="Save")
            temp_visits = world.visit_array
            temp_visits[world.next_pos] += 1
            plot_final_heatmap(temp_visits,
                target=learner.target,
                spawn=learner.curiosity_inducing_state,
                start=world.start_pos,
                agent=world.next_pos,
                linewidths=0.05,
                linecolor='#AAAAAA22',
                vmin=0, vmax=2,
                yticklabels=False,
                cmap="bone", 
                count=True,
                savepostfix="figure2b_visits"+str(stepnum)+savepostfix, 
                scaling_constant=scaling_constant, display="Save")
        if teleport:
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
        elif key == "cylinder":
            if not value:
                postfix += "_no_cylinder"
        elif key == "teleport":
            if value:
                postfix += "_yes_teleport"
        else:
            raise ValueError("Key " + key + " is not recognized.")
    return postfix

def batch_run_experiment(
        num_target_visits_path,
        trials=1,
        steps=1000, 
        dimensions = (11,11), 
        target_count='all',
        curiosity_inducing_state = (5,5),
        scaling_constant=3,
        learner_type=CuriousTDLearner, 
        animation=False,
        lineplot=True,
        figure2=False,
        **kwargs):
    
    gamma = 0.9

    # These "_stacked" variables will hold data for plotting aggregate data over
    # the batch of experiments. The first two are for heatmaps, which the second
    # two are for lineplots.
    value_stacked = None
    visit_stacked = None
    inducer_stacked = None
    df_stacked = None
    value_df = pd.DataFrame({"trial":[], "value":[], "type":[]})

    setup_info = (str(dimensions[0]) + "_" + str(dimensions[1]) 
                  + "_trials" + str(trials)
                  + "_steps" + str(steps)
                  + "_bookstore" + str(curiosity_inducing_state[0]) 
                  + "_" + str(curiosity_inducing_state[1]))
    ablation_postfix = get_ablation_postfix(**kwargs)

    if not os.path.exists(num_target_visits_path):
        os.system("touch " + num_target_visits_path)
    with open(num_target_visits_path, "a") as num_target_visits_f:
        num_target_visits_f.write(setup_info + ablation_postfix + target_count + ",")

    rng = np.random.default_rng(2021)

    for n in range(trials):
        print("\nTrial",n)
        postfix = "_" + setup_info + "_trial" + str(n) + ablation_postfix

        anim_postfix = str(postfix)

        learner, world, inducer_over_time, targets_over_time = basic_experiment(
            gamma,
            steps, dimensions,
            curiosity_inducing_state=curiosity_inducing_state,
            rng=rng, learner_type=learner_type, 
            savepostfix=anim_postfix, 
            animation=animation,
            figure2=figure2,
            scaling_constant=scaling_constant,
            **kwargs
            )

        with open(num_target_visits_path, "a") as num_target_visits_f:
            if target_count == 'new':
                num_target_visits_f.write(str(learner.num_new_target_visits) + ",")
            elif target_count == 'old':
                num_target_visits_f.write(str(learner.num_old_target_visits) + ',')
            elif target_count == 'all':
                num_target_visits_f.write(str(learner.num_new_target_visits 
                    + learner.num_old_target_visits) + ',')
            else:
                raise ValueError("target-count must be 'new', 'old', or 'all', not " + str(target_count))
        
        plot_final_heatmap(
            learner.V, 
            target=learner.target, spawn=learner.curiosity_inducing_state, 
            start=world.start_pos, agent=world.next_pos, 
            title="Value (single trial)", cmap="bwr_r", vmin=-(steps//500), vmax=steps//500, 
            scaling_constant=scaling_constant,
            linewidths=0.05,
            linecolor='#AAAAAA22',
            display="Save",savepostfix=postfix)
        plot_final_heatmap(
            world.visit_array, 
            title="Visits (single trial)", cmap="bone", vmin=0, vmax=steps//10,
            scaling_constant=scaling_constant, 
            linewidths=0.05,
            linecolor='#AAAAAA22',
            count=True,
            display="Save",savepostfix=postfix)
        
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
            df_inducer["Type"] = "Curiosity-inducing location"
            df_inducer["Target"] = -1
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
            ax = sns.lineplot(x=range(steps), y=inducer_over_time)
            save_lineplot(ax, 
                title="Value of Curosity-inducing location", 
                xlabel="Time", ylabel="Value", 
                display="Save",
                savepostfix=postfix + "lineplot")

    with open(num_target_visits_path, "a") as num_target_visits_f:
        num_target_visits_f.write("\n")

    # Compute max edge length for normalization
    maxlen = max(dimensions)

    postfix = "_" + setup_info + ablation_postfix

    plot_final_heatmap((np.array(value_stacked)).mean(axis=0), 
        target=None, 
        spawn=learner.curiosity_inducing_state,
        start=world.start_pos, 
        agent=None, 
        title="Value (mean)", 
        cmap="bwr_r", 
        vmin=-(steps//500), vmax=steps//500,
        scaling_constant=scaling_constant,
        linewidths=0.05,
        linecolor='#AAAAAA22', 
        display="Save",savepostfix="stackedMean_" + postfix)

    plot_final_heatmap((np.array(value_stacked)).mean(axis=0), 
        target=None, 
        spawn=learner.curiosity_inducing_state,
        start=world.start_pos, 
        agent=None, 
        title="Value (mean)", 
        cmap="bwr_r", 
        scaling_constant=scaling_constant,
        linewidths=0.05,
        linecolor='#AAAAAA22', 
        display="Save",savepostfix="stackedMeanAutoscale_" + postfix)

    plot_final_heatmap((np.array(value_stacked)).std(axis=0), 
        target=None, 
        spawn=learner.curiosity_inducing_state,
        start=world.start_pos, 
        agent=None, 
        title="Value (standard deviation)", 
        cmap="viridis",
        scaling_constant=scaling_constant,
        linewidths=0.05,
        linecolor='#AAAAAA22',
        display="Save",savepostfix="stackedStd_" + postfix)

    plot_final_heatmap((np.array(visit_stacked)).mean(axis=0), 
        target=None, 
        spawn=learner.curiosity_inducing_state, 
        start=world.start_pos,  
        cmap="bone", 
        vmin=0, vmax=steps//10, 
        agent=None, 
        title="Visits (mean)",
        scaling_constant=scaling_constant,
        linewidths=0.05,
        linecolor='#AAAAAA22', 
        count=True,
        display="Save",savepostfix="stackedMean_" + postfix)
    
    plot_final_heatmap((np.array(visit_stacked)).mean(axis=0), 
        target=None, 
        spawn=learner.curiosity_inducing_state, 
        start=world.start_pos,  
        cmap="bone", 
        vmin=0, vmax=steps//maxlen, 
        agent=None, title="Visits (normalized)", 
        scaling_constant=scaling_constant,
        linewidths=0.05,
        linecolor='#AAAAAA22',  
        display="Save",savepostfix="normStackedMean_" + postfix)

    plot_final_heatmap((np.array(visit_stacked)).mean(axis=0), 
        target=None, 
        spawn=learner.curiosity_inducing_state, 
        start=world.start_pos,  
        cmap="bone", 
        agent=None, title="Visits (mean)", 
        scaling_constant=scaling_constant,
        linewidths=0.05,
        linecolor='#AAAAAA22',
        display="Save",savepostfix="autoscaleStackedMean_" + postfix)

    plot_final_heatmap(np.median(np.array(visit_stacked), axis=0), 
        target=None, 
        spawn=learner.curiosity_inducing_state, 
        start=world.start_pos,  
        cmap="bone", 
        agent=None, title="Visits (median)", 
        scaling_constant=scaling_constant,
        linewidths=0.05,
        linecolor='#AAAAAA22',  
        count=True,
        display="Save",savepostfix="autoscaleStackedMedian_" + postfix)
    
    plot_final_heatmap((np.array(visit_stacked)).std(axis=0), 
        target=None, 
        spawn=learner.curiosity_inducing_state,
        start=world.start_pos,  
        cmap="viridis", 
        agent=None, 
        title="Visits (standard deviation)", 
        scaling_constant=scaling_constant,
        linewidths=0.05,
        linecolor='#AAAAAA22',  
        display="Save",savepostfix="stackedStd_" + postfix)

    if lineplot:
        df = pd.concat(df_stacked, sort=False)
        fig1, ax1 = plt.subplots(figsize=(6,4), dpi=200)
        sns.lineplot(data=df, y="Value", x="Time", hue="Type", 
            palette=sns.color_palette("colorblind")[:2], 
            ci='sd', 
            linewidth=1,
            ax=ax1)
        save_lineplot(ax1,
            title="Learned Value vs. Time", 
            xlabel="Time", ylabel="Learned Value", 
            display="Save",
            savepostfix="stackedLineplot_" + postfix)
        df["NewIndex"] = (df["Trial"].astype('str') 
            + df["Type"].astype('str') 
            + df["Target"].astype('str'))
        fig2, ax2 = plt.subplots(figsize=(6,4), dpi=200)
        ax2 = sns.lineplot(data=df[df["Type"] == "Curiosity-inducing location"], 
            palette=[sns.color_palette("colorblind")[0]], 
            linewidth=0.5, y="Value", x="Time", hue="Type", units="NewIndex", estimator=None, alpha=0.36, ax=ax2)
        sns.lineplot(data=df[df["Type"] == "Target"], 
            palette=[sns.color_palette("colorblind")[1]], 
            linewidth=0.5, y="Value", x="Time", hue="Type", units="NewIndex", estimator=None, alpha=0.04, ax=ax2)
        save_lineplot(ax2,
            title="Learned Value vs. Time", 
            xlabel="Time", ylabel="Learned Value", 
            display="Save",
            savepostfix="stackedLineplot_" + postfix + "multiline")

    if animation:
        os.system("ffmpeg -hide_banner -loglevel error -i ./output/Learned_Value_FunctionCuriosity_Value_Function_"
                  + anim_postfix + "/%d.png -vcodec ffv1 ./output/" + anim_postfix + ".avi")


def basic_experiment(
        gamma,
        steps=1000, 
        dimensions = (11,11), 
        curiosity_inducing_state = (5,5),
        rng=None, learner_type=CuriousTDLearner, 
        savepostfix="",
        animation=False,
        figure2=False,
        scaling_constant=3,
        **kwargs):
    gridworld_dimensions = dimensions
    total_steps = steps
    
    teleport = kwargs.pop('teleport')
    cylinder = kwargs.pop('cylinder')
    world_type = CylinderGridWorld if cylinder else SimpleGridWorld

    if curiosity_inducing_state is None:
        start_pos = (gridworld_dimensions[0]//2, gridworld_dimensions[1]//2)
    else:
        start_pos = curiosity_inducing_state

    world = world_type(gridworld_dimensions, 
        start_pos=start_pos)

    learner = learner_type(gridworld_dimensions, 
        curiosity_inducing_state=curiosity_inducing_state, 
        model_class=world_type, rng=rng, **kwargs)

    inducer_over_time = np.zeros(total_steps)
    all_targets = learner.get_all_possible_targets()
    targets_over_time = np.zeros((total_steps, len(all_targets)))

    print("Start pos:", world.start_pos)
        
    for i in range(total_steps):
        basic_timestep(world, learner, gamma, stepnum=i, 
                       steps=total_steps, savepostfix=savepostfix,
                       scaling_constant=scaling_constant,
                       animation=animation, teleport=teleport, figure2=figure2)
        inducer_over_time[i] = learner.V[learner.curiosity_inducing_state]
        targets_over_time[i] = [learner.V[t] for t in all_targets]
        
    print("Trial Finished")
    return learner, world, inducer_over_time, targets_over_time

class LearnerType(str, Enum):
    CuriousTDLearner = "CuriousTDLearner"
    GridworldTDLearner = "GridworldTDLearner"

class TargetCountType(str, Enum):
    new_target_visits = "new"
    old_target_visits = "old"
    all_target_visits = "all"


def main(
        trials: int = typer.Option(1, help="Number of desired trials."),
        steps: int = typer.Option(200, help="Number of steps in each trial."),
        width: int = typer.Option(11, help="Width of gridworld."),
        height: int = typer.Option(11, help="Height of gridworld."),
        bookstore_row: int = typer.Option(None, help="Row of the curiosity-inducing location on the grid.  [default: height-6]"),
        bookstore_col: int = typer.Option(None, help="Column of the curiosity-inducing location on the grid.  [default: width//2]"),
        scaling_constant: float = typer.Option(3, 
            help=""),
        csv_output: str = typer.Option("./output/num_target_visits.csv",
            help="File name to append the count of number of target visits."),
        learner_type: LearnerType = typer.Option(LearnerType.CuriousTDLearner, 
            help="Type of learner to experiment with"),
        target_count: TargetCountType = typer.Option(TargetCountType.all_target_visits, 
            help="Count target visits only when the target is new, old, or both."),
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
        cylinder: bool = typer.Option(True, 
            help="Connect the top and bottom of the gridworld and don't" +
                " allow downward actions."),
        teleport: bool = typer.Option(False,
            help="If True, the agent teleports whenever its curiosity ceases."),
        animation: bool = typer.Option(True, 
            help="If True, save plots of the value function with " + 
                "agent position at each timestep and output video animation."),
        lineplot: bool = typer.Option(False, 
            help="If True, output lineplots. (Slow.)"),
        figure2: bool = typer.Option(False,
            help="In True, save heatmaps when the agent visits the " + 
                "curiosity-inducing state or target.")):
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

    if bookstore_row is None:
        bookstore_row = height-6
    if bookstore_col is None:
        bookstore_col = width//2

    try:
        batch_run_experiment(
            csv_output,
            trials=trials, steps=steps,
            dimensions=(height, width), 
            target_count=target_count,
            curiosity_inducing_state=(bookstore_row, bookstore_col),
            scaling_constant=scaling_constant,
            learner_type=l_type,
            animation=animation, lineplot=lineplot, figure2=figure2,
            directed=directed, voluntary=voluntary,
            aversive=aversive, ceases=ceases, 
            positive=positive, decays=decays,
            flip_update=flip_update,
            reward_bonus=reward_bonus,
            cylinder=cylinder, teleport=teleport
    )   
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt has been caught. Cleaning up...")
        with open(csv_output, "a") as num_target_visits_f:
            num_target_visits_f.write(",Aborted!\n")
        raise e

    

if __name__ == "__main__":
    typer.run(main)
