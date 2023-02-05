# four-properties

```
 Usage: experiments.py [OPTIONS]                                                  
                                                                                  
 Run a batch of experiments.                                                      
                                                                                  
╭─ Options ──────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                    │
╰────────────────────────────────────────────────────────────────────────────────╯
╭─ Experiment Setup ─────────────────────────────────────────────────────────────╮
│ --trials              INTEGER                      Number of desired trials.   │
│                                                    [default: 1]                │
│ --steps               INTEGER                      Number of steps in each     │
│                                                    trial.                      │
│                                                    [default: 200]              │
│ --learner-type        [CuriousTDLearner|Gridworld  Type of learner to          │
│                       TDLearner]                   experiment with             │
│                                                    [default:                   │
│                                                    LearnerType.CuriousTDLearn… │
╰────────────────────────────────────────────────────────────────────────────────╯
╭─ Environment Configuration ────────────────────────────────────────────────────╮
│ --width                             INTEGER  Width of gridworld. [default: 11] │
│ --height                            INTEGER  Height of gridworld.              │
│                                              [default: 11]                     │
│ --bookstore-row                     INTEGER  Row of the curiosity-inducing     │
│                                              location on the grid.  [default:  │
│                                              height-6]                         │
│ --bookstore-col                     INTEGER  Column of the curiosity-inducing  │
│                                              location on the grid.  [default:  │
│                                              width//2]                         │
│ --cylinder         --no-cylinder             Connect the top and bottom of the │
│                                              gridworld and don't allow         │
│                                              downward actions.                 │
│                                              [default: cylinder]               │
│ --junction         --no-junction             Set to False for a true cylinder  │
│                                              with no junction location.        │
│                                              [default: junction]               │
│ --teleport         --no-teleport             If True, the agent teleports      │
│                                              whenever its curiosity ceases.    │
│                                              [default: no-teleport]            │
╰────────────────────────────────────────────────────────────────────────────────╯
╭─ Ablation Options ─────────────────────────────────────────────────────────────╮
│ --directed        --no-directed          Set to False to ablate directed       │
│                                          behaviour.                            │
│                                          [default: directed]                   │
│ --voluntary       --no-voluntary         Set to False to ablate voluntary      │
│                                          exposure.                             │
│                                          [default: voluntary]                  │
│ --aversive        --no-aversive          Set to False to ablate aversive       │
│                                          quality.                              │
│                                          [default: aversive]                   │
│ --ceases          --no-ceases            Set to False to ablate ceases when    │
│                                          satisfied.                            │
│                                          [default: ceases]                     │
│ --positive        --no-positive          Set to True to make the target have   │
│                                          positive value/reward. The parameter  │
│                                          'aversive' must be set to False.      │
│                                          [default: no-positive]                │
│ --decays          --no-decays            Set to True to make satisfying        │
│                                          curiosity a slow decay. The parameter │
│                                          'ceases' must be set to False.        │
│                                          [default: no-decays]                  │
│ --flip-update     --no-flip-update       Set to True to add instead of         │
│                                          subtract the curiosity value in the   │
│                                          TD update.                            │
│                                          [default: no-flip-update]             │
│ --reward-bonus    --no-reward-bonus      Set to True to add a bonus to the     │
│                                          reward.                               │
│                                          [default: no-reward-bonus]            │
╰────────────────────────────────────────────────────────────────────────────────╯
╭─ Figure Configuration ─────────────────────────────────────────────────────────╮
│ --logscale               --no-logscale                  Set to True to set the │
│                                                         heatmaps on a          │
│                                                         logscale.              │
│                                                         [default: no-logscale] │
│ --show-heatmap-yticks    --no-show-heatmap-y…           Set to false to hide   │
│                                                         the yticks on final    │
│                                                         heatmaps.              │
│                                                         [default:              │
│                                                         show-heatmap-yticks]   │
│ --show-heatmap-xticks    --no-show-heatmap-x…           Set to false to hide   │
│                                                         the yticks on final    │
│                                                         heatmaps.              │
│                                                         [default:              │
│                                                         show-heatmap-xticks]   │
│ --value-vmax                                     FLOAT  For setting the scale  │
│                                                         on the colorbars for   │
│                                                         value heatmaps.        │
│                                                         [default: 10.0]        │
│ --visit-vmax                                     FLOAT  For setting the scale  │
│                                                         on the colorbars for   │
│                                                         visit heatmaps.        │
│                                                         [default: 500]         │
│ --lineplot-height                                FLOAT  Height in inches for   │
│                                                         the lineplots.         │
│                                                         [default: 4.0]         │
│ --lineplot-width                                 FLOAT  Width in inches for    │
│                                                         the lineplots.         │
│                                                         [default: 6.0]         │
│ --lineplot-ymax                                  FLOAT  The height of the      │
│                                                         y-axis in the          │
│                                                         lineplots.             │
│                                                         [default: None]        │
│ --scaling-constant                               FLOAT  [default: 3]           │
│ --animation              --no-animation                 If True, save plots of │
│                                                         the value function     │
│                                                         with agent position at │
│                                                         each timestep and      │
│                                                         output video           │
│                                                         animation.             │
│                                                         [default: animation]   │
│ --lineplot               --no-lineplot                  If True, output        │
│                                                         lineplots. (Slow.)     │
│                                                         [default: no-lineplot] │
│ --figure2                --no-figure2                   In True, save heatmaps │
│                                                         when the agent visits  │
│                                                         the curiosity-inducing │
│                                                         state or target.       │
│                                                         [default: no-figure2]  │
╰────────────────────────────────────────────────────────────────────────────────╯
╭─ Text Output Configuration ────────────────────────────────────────────────────╮
│ --csv-output          TEXT           File name to append the count of number   │
│                                      of target visits.                         │
│                                      [default: ./output/num_target_visits.csv] │
│ --target-count        [new|old|all]  Count target visits only when the target  │
│                                      is new, old, or both.                     │
│                                      [default:                                 │
│                                      TargetCountType.all_target_visits]        │
╰────────────────────────────────────────────────────────────────────────────────╯
```
