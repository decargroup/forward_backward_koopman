import numpy as np
import pathlib

# Directory containing ``dodo.py``
WD = pathlib.Path(__file__).parent.resolve()


def task_generate_data():
    """Preprocess data without added noise."""
    for robot in ['nl_msd']:
        hydra_path = WD.joinpath(
            f'build/hydra_outputs/preprocessed_data/{robot}')
        yield {
            'name':
            f'preprocessing {robot}',
            'targets': [
                WD.joinpath(
                    f'build/preprocessed_data/{robot}/variance_0.01_norm_params.bin'
                ),
            ],
            'actions': [
                ('python preprocess.py '
                 f'preprocessing={robot} '
                 f'hydra.run.dir={hydra_path}'),
            ],
            'uptodate': [True],
        }


def task_generate_data_variances():
    """Preprocess soft robot data with added noise."""
    variances = [
        0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04,
        0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1
    ]
    for variance in variances:
        variance = str(variance)
        hydra_path = WD.joinpath(
            f'build/hydra_outputs/preprocessed_data/soft_robot_variance_{variance}'
        )
        yield {
            'name':
            f'preprocessing soft_robot with variance {variance}',
            'targets': [
                WD.joinpath(
                    f'build/preprocessed_data/soft_robot/variance_{variance}_norm_params.bin'
                ),
            ],
            'actions': [
                ('python preprocess.py preprocessing=soft_robot '
                 f'preprocessing.data.noise={variance} '
                 f'hydra.run.dir={hydra_path}'),
            ],
            'uptodate': [True],
        }


def task_fit_predict():
    """Fit Koopman models and run predictions with no added noise."""
    # Select robot
    lifting_function = 'nl_msd_poly2_centers10'
    robot = 'nl_msd'
    # Iterate over regressors
    for regressor in ['EDMD', 'EDMD-AS', 'FBEDMD', 'FBEDMD-AS']:
        hydra_path = WD.joinpath(
            f'build/hydra_outputs/fit/{lifting_function}_{regressor}')
        yield {
            'name':
            f'fitting {lifting_function} with {regressor}',
            'file_dep': [
                WD.joinpath(
                    f'build/preprocessed_data/{robot}/variance_0.01_norm_params.bin'
                ),
            ],
            'targets': [
                WD.joinpath(
                    f'build/pykoop_objects/{robot}/variance_0.01/kp_{regressor}_{lifting_function}.bin'
                ),
            ],
            'actions': [
                ('python main.py '
                 f'robot={robot} '
                 f'regressors@pykoop_pipeline={regressor} '
                 f'lifting_functions@pykoop_pipeline={lifting_function} '
                 f'hydra.run.dir={hydra_path}'),
            ],
        }


def task_fit_predict_variances():
    """Fit Koopman models and run predictions with added noise."""
    variances = [
        0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04,
        0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1
    ]
    for variance in variances:
        variance = str(variance)
        lifting_function = 'soft_robot_poly2_centers10'
        for regressor in ['EDMD', 'EDMD-AS', 'FBEDMD', 'FBEDMD-AS']:
            hydra_path = WD.joinpath(
                f'build/hydra_outputs/fit/{regressor}_{variance}')
            yield {
                'name':
                f'fitting soft_robot with {regressor} using soft_robot_poly2_centers10 and variance {variance}',
                'file_dep': [
                    WD.joinpath(
                        f'build/preprocessed_data/soft_robot/variance_{variance}_norm_params.bin'
                    ),
                ],
                'targets': [
                    WD.joinpath(
                        f'build/pykoop_objects/soft_robot/variance_{variance}/kp_{regressor}_{lifting_function}.bin'
                    ),
                ],
                'actions':
                [('python main.py robot=soft_robot '
                  f'regressors@pykoop_pipeline={regressor} '
                  'lifting_functions@pykoop_pipeline=soft_robot_poly2_centers10 '
                  f'variance={variance} '
                  f'hydra.run.dir={hydra_path}')],
            }


def task_generate_plots():
    """Generate plots."""
    for plot_type in ['nl_msd_plots', 'soft_robot_plots']:
        if plot_type == 'nl_msd_plots':
            targets = [
                WD.joinpath('build/figures/paper/nl_msd_polar.pdf'),
                WD.joinpath(
                    'build/figures/paper/nl_msd_summary_trajectory.pdf'),
                WD.joinpath('build/figures/paper/nl_msd_trajectory_err.pdf'),
            ]
        else:
            targets = [
                WD.joinpath('build/figures/paper/soft_robot_error_bars.pdf'),
                WD.joinpath('build/figures/paper/soft_robot_polar.pdf'),
                WD.joinpath(
                    'build/figures/paper/soft_robot_trajectory_err.pdf'),
            ]
        hydra_path = WD.joinpath(f'build/hydra_outputs/plot/{plot_type}')
        yield {
            'name':
            f'generating {plot_type}',
            'targets':
            targets,
            'task_dep': ['fit_predict', 'fit_predict_variances'],
            'actions': [
                ('python plot.py '
                 f'plotting@what_to_plot={plot_type} '
                 f'hydra.run.dir={hydra_path}'),
            ],
            'uptodate': [True],
        }


def task_generate_frobenius_plot():
    """Generate Frobenius norm plot."""
    hydra_path = WD.joinpath(f'build/hydra_outputs/plot/frob_err')
    yield {
        'name':
        f'generating frobenius plot',
        'targets': [
            WD.joinpath('build/figures/paper/frob_norm_sqrd.pdf'),
        ],
        'task_dep': ['fit_predict', 'fit_predict_variances'],
        'actions': [
            ('python plot.py plotting@what_to_plot=frob_err_plot '
             f'hydra.run.dir={hydra_path}'),
        ],
        'uptodate': [True],
    }
