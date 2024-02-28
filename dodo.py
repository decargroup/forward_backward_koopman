import numpy as np
import os


def task_generate_data():

    for robot in ['nl_msd', 'soft_robot']:

        path = f'build/doit_outputs/preprocessed_data/{robot}'
        os.makedirs(os.path.dirname(path), exist_ok=True)

        yield {
            f'name':
            f'preprocessing {robot}',
            f'file_dep': [
                f'config/preprocessing/{robot}.yaml',
                f'config/preprocessing/{robot}.yaml',
            ],
            f'targets': [path],
            f'actions':
            [f'python preprocess.py preprocessing={robot} > %(targets)s'],
        }


def task_generate_data_variances():
    variances = [
        0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04,
        0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1
    ]
    for variance in variances:
        variance = str(variance)

        path = f'build/doit_outputs/preprocessed_data/soft_robot_variance_{variance}'
        os.makedirs(os.path.dirname(path), exist_ok=True)

        yield {
            f'name':
            f'preprocessing soft_robot with variance {variance}',
            f'file_dep': [
                f'config/preprocessing/soft_robot.yaml',
            ],
            f'targets': [path],
            f'actions': [
                f'python preprocess.py preprocessing=soft_robot preprocessing.data.noise={variance} > %(targets)s'
            ],
        }


def task_fit_predict():
    for lifting_function in [
            'nl_msd_poly2_centers10', 'soft_robot_poly2_centers10'
    ]:
        if lifting_function == 'nl_msd_poly2_centers10':
            robot = 'nl_msd'
        else:
            robot = 'soft_robot'

        for regressor in ['EDMD', 'EDMD-AS', 'FBEDMD', 'FBEDMD-AS']:

            path = f'build/doit_outputs/fit/{lifting_function}_{regressor}'
            os.makedirs(os.path.dirname(path), exist_ok=True)

            yield {
                f'name':
                f'fitting {lifting_function} with {regressor}',
                f'file_dep': [
                    f'config/lifting_functions/{lifting_function}.yaml',
                    f'config/regressors/{regressor}.yaml',
                ],
                f'targets': [path],
                f'actions': [
                    f'python main.py robot={robot} regressors@pykoop_pipeline={regressor} lifting_functions@pykoop_pipeline={lifting_function} > %(targets)s'
                ],
            }


def task_fit_predict_variances():
    variances = [
        0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04,
        0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1
    ]
    for variance in variances:
        variance = str(variance)
        for regressor in ['EDMD', 'EDMD-AS', 'FBEDMD', 'FBEDMD-AS']:

            path = f'build/doit_outputs/fit/{regressor}_{variance}'
            os.makedirs(os.path.dirname(path), exist_ok=True)

            yield {
                f'name':
                f'fitting soft_robot with {regressor} using soft_robot_poly2_centers10 and variance {variance}',
                f'file_dep': [f'main.py', f'build/preprocessed_data'],
                f'targets': [path],
                f'actions': [
                    f'python main.py robot=soft_robot regressors@pykoop_pipeline={regressor} lifting_functions@pykoop_pipeline=soft_robot_poly2_centers10 variance={variance} > %(targets)s'
                ],
            }


def task_generate_plots():
    for plot_type in ['nl_msd_plots', 'soft_robot_plots']:

        path = f'build/doit_outputs/plot/{plot_type}'
        os.makedirs(os.path.dirname(path), exist_ok=True)

        yield {
            f'name':
            f'generating {plot_type}',
            f'file_dep':
            [f'plot.py', f'build/pykoop_objects', f'build/preprocessed_data'],
            f'targets': [path],
            f'actions': [
                f'python plot.py plotting@what_to_plot={plot_type} > %(targets)s'
            ],
        }


def task_generate_frobenius_plot():

    path = f'build/doit_outputs/plot/frob_err'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    yield {
        f'name':
        f'generating frobenius plot',
        f'file_dep':
        [f'plot.py', f'build/pykoop_objects', f'build/preprocessed_data'],
        f'targets': [path],
        f'actions':
        [f'python plot.py plotting@what_to_plot=frob_err_plot > %(targets)s'],
    }
