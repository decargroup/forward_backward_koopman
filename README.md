# Forward-Backward Extended DMD with an Asymptotic Stability Constraint

Companion code for Forward-Backward Extended DMD with an Asymptotic Stability
Constraint

## Installation

To clone the repository and its
[submodule](https://github.com/ramvasudevan/soft-robot-koopman), which contains
the soft robot dataset, run
```sh
$ git clone --recurse-submodules git@github.com:decargroup/forward_backward_koopman.git
```

## Generating data

    * Duffing oscillator: python preprocess.py preprocessing=nl_msd
    * Soft robot: python preprocess.py preprocessing=soft_robot
    * Soft robot +: python preprocess.py --multirun preprocessing=soft_robot preprocessing.data.noise=0,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.4,0.6,0.8,1

## Fitting models

    * Duffing oscillator: python main.py --multirun lifting_functions@pykoop_pipeline=nl_msd_poly2_centers10 regressors@pykoop_pipeline=EDMD,EDMD-AS,FBEDMD,FBEDMD-AS robot=nl_msd
    * Soft robot: python main.py --multirun lifting_functions@pykoop_pipeline=soft_robot_poly2_centers10 regressors@pykoop_pipeline=EDMD,EDMD-AS,FBEDMD,FBEDMD-AS robot=soft_robot
    * Soft robot +: python main.py --multirun lifting_functions@pykoop_pipeline=soft_robot_poly2_centers10 regressors@pykoop_pipeline=EDMD,EDMD-AS,FBEDMD,FBEDMD-AS variance=0,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.4,0.6,0.8,1 robot=soft_robot

## Generating plots

    * Duffing oscillator: python plot.py plotting@what_to_plot=nl_msd_plots
    * Soft robot: python plot.py plotting@what_to_plot=soft_robot_plots
    * Soft robot +: python plot.py plotting@what_to_plot=frob_err_plot