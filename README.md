# four-properties

```
$python experiments.py --help
Usage: experiments.py [OPTIONS]

  Run a batch of experiments.

Options:
  --trials INTEGER                Number of desired trials.  [default: 1]
  --steps INTEGER                 Number of steps in each trial.  [default:
                                  200]

  --width INTEGER                 Width of gridworld.  [default: 11]
  --height INTEGER                Height of gridworld.  [default: 11]
  --figwidth FLOAT                Width of heatmap figures in inches
  --figheight FLOAT               Height of heatmap figure in inches
  --csv-output TEXT               File name to append the count of number of
                                  target visits.  [default:
                                  ./output/num_target_visits.csv]

  --learner-type [CuriousTDLearner|GridworldTDLearner]
                                  Type of learner to experiment with
                                  [default: CuriousTDLearner]

  --directed / --no-directed      Set to False to ablate directed behaviour.
                                  [default: True]

  --voluntary / --no-voluntary    Set to False to ablate voluntary.  [default:
                                  True]

  --aversive / --no-aversive      Set to False to ablate aversive quality.
                                  [default: True]

  --ceases / --no-ceases          Set to False to ablate ceases when
                                  satisfied.  [default: True]

  --positive / --no-positive      Set to True to make the target have positive
                                  value/reward. The parameter 'aversive' must
                                  be set to False.  [default: False]

  --decays / --no-decays          Set to True to make satisfying curiosity a
                                  slow decay. The parameter 'ceases' must be
                                  set to False.  [default: False]

  --flip-update / --no-flip-update
                                  Set to True to add instead of subtract the
                                  curiosity value in the TD update.  [default:
                                  False]

  --reward-bonus / --no-reward-bonus
                                  Set to True to add a bonus to the reward.
                                  [default: False]

  --cylinder / --no-cylinder      Connect the top and bottom of the gridworld
                                  and don't allow downward actions.  [default:
                                  True]

  --teleport / --no-teleport      If True, the agent teleports whenever its
                                  curiosity ceases.  [default: False]

  --animation / --no-animation    If True, save plots of the value function
                                  with agent position at each timestep and
                                  output video animation.  [default: True]

  --lineplot / --no-lineplot      If True, output lineplots. (Slow.)
                                  [default: True]

  --install-completion            Install completion for the current shell.
  --show-completion               Show completion for the current shell, to
                                  copy it or customize the installation.

  --help                          Show this message and exit.
```
