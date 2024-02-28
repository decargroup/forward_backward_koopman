import glob
import pandas as pd
import numpy as np
import scipy.io
import utilities
import hydra
import omegaconf
import pickle
import pykoop
import os
from random import seed
from random import randint


class PreprocessSoftRobot:

    def __init__(self,
                 val_num: int = 2,
                 noise: float = 0.0,
                 normalize: bool = False,
                 delay: int = 0,
                 seed: int = 3,
                 train_eps_rmv: int = 0,
                 n_rmv_front: int = 0,
                 n_rmv_back: int = 0):

        self.val_num = val_num
        self.noise = noise
        self.normalize = normalize
        self.delay = delay
        self.seed = seed
        self.train_eps_rmv = train_eps_rmv
        self.n_rmv_front = n_rmv_front
        self.n_rmv_back = n_rmv_back
        self.noise = noise

    def preprocess(self, path: str):

        pykoop_dict = {}
        pykoop_dict_true = {}

        pykoop_dict_cut = {}

        mat = scipy.io.loadmat(path)

        train = mat['train']
        valid = mat['val']
        data_dict = {'train': train, 'valid': valid}

        n_inputs = train[0, 0][0, 0]['u'].shape[1]
        n_states = train[0, 0][0, 0]['x'].shape[1]
        t = {}
        x1 = {}
        x2 = {}
        u1 = {}
        u2 = {}
        u3 = {}

        # Generate discretized noise covariance matrix
        Q = utilities.Q_gen(n_states=n_states,
                            add_noise=[0, 1],
                            noise=self.noise)

        # Get dict data
        for tag, epi in data_dict.items():
            for i in range(epi.shape[1]):
                if i != 0:
                    t[tag] = np.concatenate((t[tag], epi[0, i][0, 0]['t']),
                                            axis=0)
                    x1[tag] = np.concatenate(
                        (x1[tag], epi[0, i][0, 0]['x'][:, 0].reshape(-1, 1)),
                        axis=0)
                    x2[tag] = np.concatenate(
                        (x2[tag], epi[0, i][0, 0]['x'][:, 1].reshape(-1, 1)),
                        axis=0)
                    u1[tag] = np.concatenate(
                        (u1[tag], epi[0, i][0, 0]['u'][:, 0].reshape(-1, 1)),
                        axis=0)
                    u2[tag] = np.concatenate(
                        (u2[tag], epi[0, i][0, 0]['u'][:, 1].reshape(-1, 1)),
                        axis=0)
                    u3[tag] = np.concatenate(
                        (u3[tag], epi[0, i][0, 0]['u'][:, 2].reshape(-1, 1)),
                        axis=0)
                else:
                    t[tag] = epi[0, i][0, 0]['t']
                    x1[tag] = epi[0, i][0, 0]['x'][:, 0].reshape(-1, 1)
                    x2[tag] = epi[0, i][0, 0]['x'][:, 1].reshape(-1, 1)
                    u1[tag] = epi[0, i][0, 0]['u'][:, 0].reshape(-1, 1)
                    u2[tag] = epi[0, i][0, 0]['u'][:, 1].reshape(-1, 1)
                    u3[tag] = epi[0, i][0, 0]['u'][:, 2].reshape(-1, 1)

        eps_dict = {
            'train': np.zeros((t['train'].shape)),
            'valid': np.zeros((t['valid'].shape))
        }

        for tag in eps_dict.keys():
            ep = -1
            for i in range(t[tag].shape[0]):
                if t[tag][i, 0] == t[tag][0, 0]: ep = ep + 1
                eps_dict[tag][i, 0] = ep

        for tag in eps_dict.keys():
            pykoop_dict['X_{}'.format(tag)] = np.concatenate(
                (eps_dict[tag], x1[tag], x2[tag], u1[tag], u2[tag], u3[tag]),
                axis=1)

        # Preprocess data by removing samples
        if self.n_rmv_front != 0 or self.n_rmv_back != 0:
            for i in range(int(np.max(pykoop_dict['X_valid'][:, 0])) + 1):
                temp_val = pykoop_dict['X_valid'][
                    pykoop_dict['X_valid'][:, 0] == i, :][self.n_rmv_front:, :]
                pykoop_dict_cut['X_valid'] = np.concatenate(
                    (pykoop_dict_cut['X_valid'],
                     temp_val), axis=0) if i != 0 else temp_val
            pykoop_dict['X_valid'] = pykoop_dict_cut['X_valid']

        pykoop_dict_true['X_train'] = pykoop_dict['X_train']
        pykoop_dict_true['X_valid'] = pykoop_dict['X_valid']

        # Add noise to the data
        if self.noise != 0:
            for i in range(int(np.max(pykoop_dict['X_train'][:, 0]))):
                x1_temp = pykoop_dict['X_train'][
                    pykoop_dict['X_train'][:, 0] == i, 1]
                x2_temp = pykoop_dict['X_train'][
                    pykoop_dict['X_train'][:, 0] == i, 2]
                x_temp, snr = utilities.add_noise(np.block([[x1_temp],
                                                            [x2_temp]]),
                                                  Q,
                                                  0,
                                                  1,
                                                  seed=self.seed)
                pykoop_dict['X_train'][pykoop_dict['X_train'][:, 0] == i,
                                       1] = x_temp[:, 0]
                pykoop_dict['X_train'][pykoop_dict['X_train'][:, 0] == i,
                                       2] = x_temp[:, 1]
                self.seed = self.seed + 1

            path = "build/preprocessed_data/soft_robot/variance_{}_snr.bin".format(
                self.noise)

            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "wb") as f:
                pickle.dump(snr, f)

        # Normalize the data
        if self.normalize:
            max_state_1 = np.max(
                np.abs(
                    np.concatenate((pykoop_dict['X_train'][:, 1],
                                    pykoop_dict['X_valid'][:, 1]),
                                   axis=0)))
            max_state_2 = np.max(
                np.abs(
                    np.concatenate((pykoop_dict['X_train'][:, 2],
                                    pykoop_dict['X_valid'][:, 2]),
                                   axis=0)))
            max_input = np.max((np.abs(
                np.concatenate(
                    (pykoop_dict['X_train'][:, 3], pykoop_dict['X_valid'][:,
                                                                          3]),
                    axis=0)),
                                np.abs(
                                    np.concatenate(
                                        (pykoop_dict['X_train'][:, 4],
                                         pykoop_dict['X_valid'][:, 4]),
                                        axis=0)),
                                np.abs(
                                    np.concatenate(
                                        (pykoop_dict['X_train'][:, 5],
                                         pykoop_dict['X_valid'][:, 5]),
                                        axis=0))))

            pykoop_dict['X_train'][:,
                                   1] = pykoop_dict['X_train'][:,
                                                               1] / max_state_1
            pykoop_dict['X_train'][:,
                                   2] = pykoop_dict['X_train'][:,
                                                               2] / max_state_2
            pykoop_dict['X_valid'][:,
                                   1] = pykoop_dict['X_valid'][:,
                                                               1] / max_state_1
            pykoop_dict['X_valid'][:,
                                   2] = pykoop_dict['X_valid'][:,
                                                               2] / max_state_2
            pykoop_dict['X_train'][:,
                                   3] = pykoop_dict['X_train'][:,
                                                               3] / max_input
            pykoop_dict['X_train'][:,
                                   4] = pykoop_dict['X_train'][:,
                                                               4] / max_input
            pykoop_dict['X_valid'][:,
                                   3] = pykoop_dict['X_valid'][:,
                                                               3] / max_input
            pykoop_dict['X_valid'][:,
                                   4] = pykoop_dict['X_valid'][:,
                                                               4] / max_input
            pykoop_dict['X_train'][:,
                                   5] = pykoop_dict['X_train'][:,
                                                               5] / max_input
            pykoop_dict['X_valid'][:,
                                   5] = pykoop_dict['X_valid'][:,
                                                               5] / max_input

            # Save the normalizing parameters
            norm_coef = np.array([max_state_1, max_state_2, max_input])

            path = "build/preprocessed_data/soft_robot/variance_{}_norm_params.bin".format(
                self.noise)

            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "wb") as f:
                pickle.dump(norm_coef, f)

        x0_valid = np.zeros(
            (self.val_num, n_states + 1)) if self.delay == 0 else np.zeros(
                (self.val_num * (self.delay + 1), n_states + 1))

        for i in range(self.val_num):

            x0_valid[i * (self.delay + 1):i * (self.delay + 1) + self.delay
                     + 1, :] = pykoop_dict['X_valid'][
                         pykoop_dict['X_valid'][:, 0] == i, :3][0:self.delay
                                                                + 1, :]

        u_valid = np.concatenate([
            pykoop_dict['X_valid'][:, [0]], pykoop_dict['X_valid'][:,
                                                                   -n_inputs:]
        ],
                                 axis=1)

        self.pykoop_dict = dict(X_train=pykoop_dict['X_train'],
                                X_valid=pykoop_dict['X_valid'],
                                n_inputs=n_inputs,
                                x0_valid=x0_valid,
                                u_valid=u_valid)
        self.pykoop_dict_true = dict(X_train=pykoop_dict_true['X_train'],
                                     X_valid=pykoop_dict_true['X_valid'],
                                     n_inputs=n_inputs,
                                     x0_valid=x0_valid,
                                     u_valid=u_valid)

        path = "build/preprocessed_data/soft_robot/variance_{}.bin".format(
            self.noise)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)


class PreprocessNonlinearMassSpringDamper:

    def __init__(
        self,
        noise: float = 0.0,
        normalize: bool = False,
        delay: int = 0,
        seed: int = 1,
        m: int = 0,
        k: int = 0,
        k2: int = 0,
        c: int = 0,
        t0: int = 0,
        t1: int = 10,
        dt: float = 0.01,
        x0_0: int = -5,
        x0_1: int = 5,
        n_inputs: int = 1,
        n_eps_train: int = 20,
        n_eps_valid: int = 2,
        u_min: int = -2,
        u_max: int = 2,
        cutoff_freq: int = 1,
        filter_order: int = 2,
    ):

        self.noise = noise
        self.normalize = normalize
        self.delay = delay
        self.seed = seed
        self.noise = noise
        self.m = m
        self.k = k
        self.k2 = k2
        self.c = c
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.x0_0 = x0_0
        self.x0_1 = x0_1
        self.n_inputs = n_inputs
        self.n_eps_train = n_eps_train
        self.n_eps_valid = n_eps_valid
        self.u_min = u_min
        self.u_max = u_max
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order

    def preprocess(self, path: str):

        pykoop_dict = {}
        pykoop_dict_true = {}

        t = np.arange(self.t0, self.t1, self.dt).reshape(-1, 1)

        n_states = 2
        Q = utilities.Q_gen(n_states=n_states,
                            add_noise=[0, 1],
                            noise=self.noise)

        rng = np.random.default_rng(self.seed)
        x0 = rng.uniform(self.x0_0, self.x0_1,
                         (self.n_eps_train + self.n_eps_valid, n_states))
        n_eps = {'X_train': self.n_eps_train, 'X_valid': self.n_eps_valid}

        for tag, n_eps in n_eps.items():
            for ep in range(n_eps):
                x = np.zeros((t.shape[0], n_states))
                x[0, :] = x0[ep, :]
                u = pykoop.random_input((self.t0, self.t1),
                                        self.dt,
                                        self.u_min,
                                        self.u_max,
                                        self.cutoff_freq,
                                        order=self.filter_order,
                                        rng=rng,
                                        output='array').reshape(-1, 1)

                for i in range(t.shape[0] - 1):
                    x[[i + 1], :] = x[[i], :] + self.dt * self.ode(
                        x[i, :], u[i, :]).T

                X = np.concatenate((np.ones((t.shape[0], 1)) * ep, x, u),
                                   axis=1)

                pykoop_dict[tag] = np.concatenate(
                    (pykoop_dict[tag], X), axis=0) if ep != 0 else X

        pykoop_dict_true['X_train'] = pykoop_dict['X_train']
        pykoop_dict_true['X_valid'] = pykoop_dict['X_valid']

        # Add noise to the data
        if self.noise != 0.0:
            for i in range(int(np.max(pykoop_dict['X_train'][:, 0]))):
                x1_temp = pykoop_dict['X_train'][
                    pykoop_dict['X_train'][:, 0] == i, 1]
                x2_temp = pykoop_dict['X_train'][
                    pykoop_dict['X_train'][:, 0] == i, 2]
                x_temp, snr = utilities.add_noise(np.block([[x1_temp],
                                                            [x2_temp]]),
                                                  Q,
                                                  0,
                                                  1,
                                                  seed=self.seed)
                pykoop_dict['X_train'][pykoop_dict['X_train'][:, 0] == i,
                                       1] = x_temp[:, 0]
                pykoop_dict['X_train'][pykoop_dict['X_train'][:, 0] == i,
                                       2] = x_temp[:, 1]
                self.seed = self.seed + 1

            path = "build/preprocessed_data/nl_msd/variance_{}_snr.bin".format(
                self.noise)

            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "wb") as f:
                pickle.dump(snr, f)

        # Normalize the data
        if self.normalize:
            max_state_1 = np.max(
                np.abs(
                    np.concatenate((pykoop_dict['X_train'][:, 1],
                                    pykoop_dict['X_valid'][:, 1]),
                                   axis=0)), )
            max_state_2 = np.max(
                np.abs(
                    np.concatenate((pykoop_dict['X_train'][:, 2],
                                    pykoop_dict['X_valid'][:, 2]),
                                   axis=0)), )
            max_input = np.max([
                np.abs(
                    np.concatenate((pykoop_dict['X_train'][:, 3],
                                    pykoop_dict['X_valid'][:, 3]),
                                   axis=0))
            ])
            pykoop_dict['X_train'][:,
                                   1] = pykoop_dict['X_train'][:,
                                                               1] / max_state_1
            pykoop_dict['X_train'][:,
                                   2] = pykoop_dict['X_train'][:,
                                                               2] / max_state_2
            pykoop_dict['X_valid'][:,
                                   1] = pykoop_dict['X_valid'][:,
                                                               1] / max_state_1
            pykoop_dict['X_valid'][:,
                                   2] = pykoop_dict['X_valid'][:,
                                                               2] / max_state_2
            pykoop_dict['X_train'][:,
                                   3] = pykoop_dict['X_train'][:,
                                                               3] / max_input
            pykoop_dict['X_valid'][:,
                                   3] = pykoop_dict['X_valid'][:,
                                                               3] / max_input

            # Save the normalizing parameters
            norm_coef = np.array([max_state_1, max_state_2, max_input])
            path = "build/preprocessed_data/nl_msd/variance_{}_norm_params.bin".format(
                self.noise)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(norm_coef, f)

        x0_valid = np.zeros(
            (self.n_eps_valid, n_states + 1)) if self.delay == 0 else np.zeros(
                (self.n_eps_valid * (self.delay + 1), n_states + 1))

        for i in range(self.n_eps_valid):

            x0_valid[i * (self.delay + 1):i * (self.delay + 1) + self.delay
                     + 1, :] = pykoop_dict['X_valid'][
                         pykoop_dict['X_valid'][:, 0] == i, :3][0:self.delay
                                                                + 1, :]

        u_valid = np.concatenate([
            pykoop_dict['X_valid'][:, [0]],
            pykoop_dict['X_valid'][:, -self.n_inputs:]
        ],
                                 axis=1)

        self.pykoop_dict = dict(X_train=pykoop_dict['X_train'],
                                X_valid=pykoop_dict['X_valid'],
                                n_inputs=self.n_inputs,
                                x0_valid=x0_valid,
                                u_valid=u_valid)
        self.pykoop_dict_true = dict(X_train=pykoop_dict_true['X_train'],
                                     X_valid=pykoop_dict_true['X_valid'],
                                     n_inputs=self.n_inputs,
                                     x0_valid=x0_valid,
                                     u_valid=u_valid)

        path = "build/preprocessed_data/nl_msd/variance_{}.bin".format(
            self.noise)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def ode(self, x, u):
        return np.array([[
            x[1]
        ], [(u[0] - self.c * x[1] - self.k * x[0] - self.k2 * x[0]**3) / self.m
            ]])


@hydra.main(config_path="config",
            config_name="default_preprocess_config",
            version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:

    data = hydra.utils.instantiate(cfg.preprocessing.data, _convert_='all')

    data.preprocess(cfg['preprocessing']['path'])


if __name__ == '__main__':
    main()
