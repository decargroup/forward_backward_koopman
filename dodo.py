import pathlib
import numpy as np

WORKING_DIR = pathlib.Path(__file__).parent.resolve()

BUILD_DIR = WORKING_DIR.joinpath('build')

BUILD_DIRS = {
    dir: BUILD_DIR.joinpath(dir)
    for dir in [
        'figures',
        'outputs',
        'multirun',
        'preprocessed_data',
        'pykoop_objects',
    ]
}

CONFIG_DIR = WORKING_DIR.joinpath('config')

CONFIG_DIRS = {
    dir: CONFIG_DIR.joinpath(dir)
    for dir in [
        'lifting_functions',
        'regressor',
        'plotting',
        'preprocessing',
    ]
}


def task_generate__data():
    for robot in ['nl_msd', 'soft_robot']:
        yield {
            f'name': f'preprocessing {robot}',
            f'targets': [f'build/outputs/{robot}'],
            f'actions': [f'python preprocess.py preprocessing={robot}'],
        }


def task_generate__data__variances():
    variances = [
        0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04,
        0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1
    ]
    for variance in variances:
        variance = str(variance)
        yield {
            f'name':
            f'preprocessing soft_robot with variance {variance}',
            f'targets': [f'build/outputs/soft_robot/variance_{variance}'],
            f'actions': [
                f'python preprocess.py preprocessing=soft_robot preprocessing.data.noise={variance}'
            ],
        }


def task_fit__predict():
    for robot in ['nl_msd', 'soft_robot']:
        lifting_function = f'{robot}_poly2_centers10'
        print(lifting_function)
        for regressor in ['EDMD', 'EDMD-AS', 'FBEDMD', 'FBEDMD-AS']:
            yield {
                f'name':
                f'fitting {robot} with {regressor} using {lifting_function}',
                f'targets': [
                    f'build/pykoop_objects/{robot}/kp_{regressor}_{robot}_{lifting_function}.bin'
                ],
                f'actions': [
                    f'python main.py robot={robot} regressors@pykoop_pipeline={regressor} lifting_functions@pykoop_pipeline={lifting_function}'
                ],
            }


def task_fit__predict__variances():
    variances = [
        0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04,
        0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1
    ]
    for variance in variances:
        variance = str(variance)
        for regressor in ['EDMD', 'EDMD-AS', 'FBEDMD', 'FBEDMD-AS']:
            yield {
                f'name':
                f'fitting soft_robot with {regressor} using soft_robot_poly2_centers10 and variance {variance}',
                f'targets': [
                    f'build/pykoop_objects/soft_robot/variance_{variance}/kp_{regressor}_soft_robot_poly2_centers10.bin'
                ],
                f'actions': [
                    f'python main.py robot=soft_robot regressors@pykoop_pipeline={regressor} lifting_functions@pykoop_pipeline=soft_robot_poly2_centers10 variance={variance}'
                ],
            }


def task_generate__plots():
    for plot_type in ['nl_msd_plots', 'soft_robot_plots']:
        yield {
            f'name': f'generating {plot_type}',
            f'targets': [f'build/figures/{plot_type}/poly2_centers10.png'],
            f'actions': [f'python plot.py plotting@what_to_plot={plot_type}'],
        }


def task_generate__frobenius__plot():
    yield {
        f'name': f'generating frobenius plot',
        f'targets': [f'build/figures/frobenius_plot.png'],
        f'actions': [f'python plot.py plotting@what_to_plot=frob_err_plot'],
    }
