import numpy as np
from typing import Any, Dict, Tuple
from matplotlib import pyplot as plt
import scipy
import hydra
import omegaconf
import pickle
import utilities
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import os


class PlotPaper():

    def __init__(self, regressors: list, lifting_functions, robot, variance,
                 delay, n, val):
        self.regressors = regressors
        self.lifting_functions = lifting_functions
        self.robot = robot
        self.variance = variance
        self.delay = delay
        self.n = n
        self.val = val

    def plot(self, **kwargs):
        x_pred = {}
        koop_matrices = {}
        for regressor in self.regressors:
            with open(
                    "build/pykoop_objects/{}/variance_{}/kp_{}_{}_{}.bin".
                    format(self.robot, self.variance, regressor, self.robot,
                           self.lifting_functions), "rb") as f:
                kp = pickle.load(f)
            x_pred[regressor] = kp.x_pred
            koop_matrices[regressor] = kp.regressor_.coef_.T

        # Get the true data
        with open(
                "build/preprocessed_data/{}/variance_{}.bin".format(
                    self.robot, self.variance), "rb") as f:
            data = pickle.load(f)

        true_data = data.pykoop_dict_true['X_valid']
        noisy_data = data.pykoop_dict['X_train']

        # Load normalization parameters
        with open(
                "build/preprocessed_data/{}/variance_{}_norm_params.bin".
                format(self.robot, self.variance), "rb") as f:
            norm_params = pickle.load(f)

        path = "build/figures/paper"

        os.makedirs(os.path.dirname(path + "/_.png"), exist_ok=True)

        utilities.plot_rms_and_avg_error_paper(x_pred,
                                               true_data,
                                               path,
                                               norm_params,
                                               self.robot,
                                               n=self.n,
                                               val=self.val,
                                               **kwargs)

        utilities.plot_trajectory_error_paper(x_pred,
                                              true_data,
                                              path,
                                              norm_params,
                                              self.robot,
                                              n=self.n,
                                              val=self.val,
                                              **kwargs)
        utilities.plot_polar(koop_matrices, path, self.robot, **kwargs)
        if self.robot == 'nl_msd':
            utilities.summary_fig(x_pred,
                                  true_data,
                                  path,
                                  norm_params,
                                  self.robot,
                                  n=self.n,
                                  val=self.val,
                                  **kwargs)
        # utilities.print_koop_matrices(koop_matrices, **kwargs)


class PlotFrobErr():

    def __init__(self, regressors: list, lifting_functions, robot, variance,
                 delay, variance_lvl):
        self.regressors = regressors
        self.lifting_functions = lifting_functions
        self.robot = robot
        self.variance = variance
        self.delay = delay
        self.variance_lvl = variance_lvl

    def plot(self, **kwargs):
        frob_error_U = {}
        frob_error_A = {}
        frob_error_B = {}
        koop_matrices_true = {}
        snr = np.zeros(len(self.variance_lvl))

        for regressor in self.regressors:
            with open(
                    "build/pykoop_objects/{}/variance_{}/kp_{}_{}_{}.bin".
                    format(self.robot, 0, regressor, self.robot,
                           self.lifting_functions), "rb") as f:
                kp_true = pickle.load(f)
            koop_matrices_true[regressor] = kp_true.regressor_.coef_.T
        k = 0
        # Compute errors
        for var in self.variance_lvl:
            for regressor in self.regressors:
                with open(
                        "build/pykoop_objects/{}/variance_{}/kp_{}_{}_{}.bin".
                        format(self.robot, var, regressor, self.robot,
                               self.lifting_functions), "rb") as f:
                    kp = pickle.load(f)
                koop_matrix = kp.regressor_.coef_.T

                frob_error_U[regressor] = np.append(
                    frob_error_U[regressor],
                    scipy.linalg.norm(
                        koop_matrices_true[regressor] - koop_matrix, 'fro')
                    / scipy.linalg.norm(koop_matrices_true[regressor], 'fro')
                ) if k > 0 else scipy.linalg.norm(
                    koop_matrices_true[regressor]
                    - koop_matrix, 'fro') / scipy.linalg.norm(
                        koop_matrices_true[regressor], 'fro')
                frob_error_A[regressor] = np.append(
                    frob_error_A[regressor],
                    scipy.linalg.norm(
                        koop_matrices_true[regressor]
                        [:, :koop_matrices_true[regressor].shape[0]]
                        - koop_matrix[:, :koop_matrix.shape[0]], 'fro')
                    / scipy.linalg.norm(koop_matrices_true[regressor], 'fro')
                ) if k > 0 else scipy.linalg.norm(
                    koop_matrices_true[regressor]
                    [:, :koop_matrices_true[regressor].shape[0]]
                    - koop_matrix[:, :koop_matrix.shape[0]], 'fro'
                ) / scipy.linalg.norm(koop_matrices_true[regressor], 'fro')
                frob_error_B[regressor] = np.append(
                    frob_error_B[regressor],
                    scipy.linalg.norm(
                        koop_matrices_true[regressor]
                        [:, koop_matrices_true[regressor].shape[0]:]
                        - koop_matrix[:, koop_matrix.shape[0]:], 'fro')
                    / scipy.linalg.norm(koop_matrices_true[regressor], 'fro')
                ) if k > 0 else scipy.linalg.norm(
                    koop_matrices_true[regressor]
                    [:, koop_matrices_true[regressor].shape[0]:]
                    - koop_matrix[:, koop_matrix.shape[0]:], 'fro'
                ) / scipy.linalg.norm(koop_matrices_true[regressor], 'fro')

            with open(
                    "build/preprocessed_data/{}/variance_{}_snr.bin".format(
                        self.robot, var), "rb") as f:
                snr[k] = pickle.load(f)
            k = k + 1

        frob_error = {'U': frob_error_U, 'A': frob_error_A, 'B': frob_error_B}

        path = 'build/figures/paper/'
        os.makedirs(os.path.dirname(path), exist_ok=True)

        utilities.plot_frob_err(frob_error, self.variance_lvl, snr, path,
                                **kwargs)


@hydra.main(config_path="config",
            config_name="default_plot_config",
            version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:

    my_plt = hydra.utils.instantiate(cfg.what_to_plot, _convert_='all')

    my_plt.plot(**cfg.figure_rcparams)


if __name__ == '__main__':
    main()
