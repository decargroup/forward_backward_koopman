import numpy as np
import scipy
from numpy import random
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict
from pykoop.lmi_regressors import LmiRegressor
from pykoop import lmi_regressors
from typing import Any, Dict, Optional
from pykoop import tsvd
import logging
import tempfile
import joblib
import picos
import scipy.signal
import sklearn.base
import os
from random import randint
import pykoop
import shutil
from pathlib import Path
import scipy.io as sio
import numpy as np

polite_stop = False

# Create logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# Create temporary cache directory for memoized computations
_cachedir = tempfile.TemporaryDirectory(prefix='pykoop_')
log.info(f'Temporary directory created at `{_cachedir.name}`')
memory = joblib.Memory(_cachedir.name, verbose=0)

color_dict = {
    'EDMD': (0.90, 0.60, 0.00),
    'EDMD-AS': (0.80, 0.40, 0.00),
    'FBEDMD': (0.35, 0.70, 0.90),
    'FBEDMD-AS': (0.00, 0.45, 0.70)
}
color_list = [(0.00, 0.45, 0.70), (0.90, 0.60, 0.00), (0.00, 0.60, 0.50),
              (0.35, 0.70, 0.90), (0.60, 0.60, 0.60)]
linestyle_dict = {
    'EDMD': '-',
    'EDMD-AS': '--',
    'FBEDMD': '-',
    'FBEDMD-AS': '--'
}


def Q_gen(n_states: int = 2,
          add_noise: list = [0, 1],
          noise: int = 0) -> np.ndarray:

    Q = np.zeros((n_states, n_states))

    for i in add_noise:
        Q[i, i] = noise

    return Q


def add_noise(data, Q, mu, n, seed: int = 3) -> np.ndarray:

    v = np.zeros((Q.shape[0], data.shape[1]))

    e, u = np.linalg.eig(Q)

    rng = random.default_rng(seed=seed)

    r_noise = rng.normal(0, 1, (data.shape[0], data.shape[1])) + mu

    v = np.diag(e)**0.5 @ r_noise

    noise = u @ v

    Data = noise + data

    n = data.shape[1]

    s_mean = np.mean(data)
    n_mean = np.mean(noise)
    s_pow = np.sum((data - s_mean)**2) / n
    n_pow = np.sum((noise - n_mean)**2) / n
    snr = 10 * np.log10(s_pow / n_pow)

    print('Avg signal power: {}, avg noise power: {}, SNR: {}'.format(
        s_pow, n_pow, snr))

    return Data.T, snr


def regressor_profilers(regressors, func_name="pykoop.koopman_pipeline.fit", **kwargs) -> None:
    lifting_levels = ["10", "25", "50"]
    all_times = {regressor: [] for regressor in regressors}
    all_memories = {regressor: [] for regressor in regressors}

    for regressor in regressors:
        regressor_dir = Path(f"build/stats/{regressor}")
        regressor_times = []
        regressor_memories = []

        dat_files = sorted(regressor_dir.glob("*.dat"))
        if len(dat_files) < 3:
            print(f"⚠️ Not enough .dat files found for {regressor}")
            continue

        for i, lifting_label in enumerate(lifting_levels):
            dat_path = dat_files[i]

            times = []
            max_mems = []

            with dat_path.open() as f:
                lines = f.readlines()
                num_lines = len(lines)

                idx = 0
                current_max_mem = 0
                current_start_time = None
                current_end_time = None

                while idx < num_lines:
                    line = lines[idx]

                    if line.startswith("CMDLINE"):
                        # Save the previous run if it exists
                        if current_start_time is not None and current_end_time is not None:
                            duration = current_end_time - current_start_time
                            times.append(duration)
                            max_mems.append(current_max_mem)

                        # Start a new run → reset memory
                        current_max_mem = 0
                        current_start_time = None
                        current_end_time = None

                        # CASE 1: Check if previous line is FUNC (belongs to this run)
                        if idx > 0 and lines[idx - 1].startswith("FUNC") and func_name in lines[idx - 1]:
                            func_line = lines[idx - 1].strip().split()
                            if len(func_line) >= 6:
                                current_start_time = float(func_line[3])
                                current_end_time = float(func_line[5])
                        else:
                            # CASE 2: Look ahead for FUNC for this run
                            idx_search = idx + 1
                            while idx_search < num_lines:
                                next_line = lines[idx_search]
                                if next_line.startswith("CMDLINE"):
                                    break  # No FUNC found before next CMDLINE
                                if next_line.startswith("FUNC") and func_name in next_line:
                                    func_line = next_line.strip().split()
                                    if len(func_line) >= 6:
                                        current_start_time = float(func_line[3])
                                        current_end_time = float(func_line[5])
                                    break
                                idx_search += 1

                    elif line.startswith("MEM"):
                        parts = line.strip().split()
                        if len(parts) < 3:
                            idx += 1
                            continue
                        mem_value = float(parts[1])
                        current_max_mem = max(current_max_mem, mem_value)

                    idx += 1

                # Save the last run if the file ends without another CMDLINE
                if current_start_time is not None and current_end_time is not None:
                    duration = current_end_time - current_start_time
                    times.append(duration)
                    max_mems.append(current_max_mem)

            regressor_times.append(times)
            regressor_memories.append(max_mems)

        all_times[regressor] = regressor_times
        all_memories[regressor] = regressor_memories


    plt.rcParams.update(**kwargs)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)

        plt.rcParams.update({
        'font.family': 'serif',
        })

    fig, axs = plt.subplots(3, 2, figsize=(5.3348, 5.5), sharex=True)
    lifting_levels = ["normal", "25_lift", "50_lift"]

    for row, lifting_label in enumerate(lifting_levels):
        time_data = []
        time_labels = []
        mem_data = []
        mem_labels = []

        # Collect data for this lifting level across all regressors
        for regressor in regressors:
            time_samples = all_times[regressor][row]
            mem_samples = all_memories[regressor][row]

            time_data.append(time_samples)
            time_labels.append(regressor)

            mem_data.append(mem_samples)
            mem_labels.append(regressor)

        # Plot Time Boxplot (remove labels from here)
        box_time = axs[row, 0].boxplot(
            time_data,
            showfliers=False,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            meanprops={'color': 'black', 'linestyle': '-', 'linewidth': 1.4}
        )

        # Apply colors to boxes
        for patch, regressor in zip(box_time['boxes'], regressors):
            patch.set_facecolor(color_dict[regressor])

        # Remove median lines
        for median in box_time['medians']:
            median.set_visible(False)

        # Prepare LaTeX formatted x-tick labels (handle dashes correctly)
        # latex_labels = [r"$\mathrm{" + reg.replace("_", r"\_").replace("-", r"\text{-}") + "}$" for reg in time_labels]

        axs[row, 0].set_xticks(range(1, len(time_labels) + 1))
        # axs[row, 0].set_xticklabels(latex_labels, rotation=40)
        axs[row, 0].set_xticklabels(time_labels, rotation=40)

        axs[row, 0].set_ylabel(r"$\mathrm{Time\ (s)}$")
        axs[row, 0].grid(True, axis='y')

        # Plot Memory Boxplot (remove labels from here)
        box_mem = axs[row, 1].boxplot(
            mem_data,
            showfliers=False,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            meanprops={'color': 'black', 'linestyle': '-', 'linewidth': 1.4}
        )

        # Apply colors to boxes
        for patch, regressor in zip(box_mem['boxes'], regressors):
            patch.set_facecolor(color_dict[regressor])

        # Remove median lines
        for median in box_mem['medians']:
            median.set_visible(False)

        # Prepare LaTeX formatted x-tick labels (handle dashes correctly)
        # latex_labels = [r"$\mathrm{" + reg.replace("_", r"\_").replace("-", r"\text{-}") + "}$" for reg in mem_labels]

        axs[row, 1].set_xticks(range(1, len(mem_labels) + 1))
        axs[row, 1].set_xticklabels(mem_labels, rotation=40)

        axs[row, 1].set_ylabel(r"$\mathrm{Memory\ (MB)}$")
        axs[row, 1].grid(True, axis='y')

    # Final layout and save
    plt.tight_layout()
    plt.savefig("build/stats/computation_memory_comparison.png")
    plt.savefig("build/figures/paper/computation_memory_comparison.pdf")
    plt.close()

    print("✅ Computation and memory subplot saved to build/stats/computation_memory_comparison.png")

    # # Create boxplots for times
    # plt.figure(figsize=(12, 7))
    # positions = []
    # data = []
    # labels = []
    # pos_offset = 0

    # for regressor in regressors:
    #     times_lists = all_times[regressor]
    #     for j, times in enumerate(times_lists):
    #         pos = pos_offset + j + 1
    #         positions.append(pos)
    #         data.append(times)
    #         labels.append(f"{regressor}\n{lifting_levels[j]}")
    #     pos_offset += len(lifting_levels) + 1

    # box = plt.boxplot(data, positions=positions, showfliers=False, patch_artist=True)

    # # Apply regressor colors
    # box_index = 0
    # for regressor in regressors:
    #     regressor_color = color_dict[regressor]
    #     for _ in lifting_levels:
    #         box['boxes'][box_index].set_facecolor(regressor_color)
    #         box_index += 1

    # # plt.title("Computation Time Comparison")
    # plt.ylabel("Time (s)")
    # plt.xticks(positions, labels, rotation=45)
    # plt.grid(True, axis='y')
    # plt.tight_layout()
    # plt.savefig("build/stats/computation_time_boxplot.png")
    # plt.close()
    # print("✅ Computation time boxplot saved to build/stats/computation_time_boxplot.png")


    # # Create boxplots for memory
    # plt.figure(figsize=(12, 7))
    # positions = []
    # data = []
    # labels = []
    # pos_offset = 0

    # for regressor in regressors:
    #     mem_lists = all_memories[regressor]
    #     for j, mems in enumerate(mem_lists):
    #         pos = pos_offset + j + 1
    #         positions.append(pos)
    #         data.append(mems)
    #         labels.append(f"{regressor}\n{lifting_levels[j]}")
    #     pos_offset += len(lifting_levels) + 1

    # box = plt.boxplot(data, positions=positions, showfliers=False, patch_artist=True)

    # # Apply regressor colors
    # box_index = 0
    # for regressor in regressors:
    #     regressor_color = color_dict[regressor]
    #     for _ in lifting_levels:
    #         box['boxes'][box_index].set_facecolor(regressor_color)
    #         box_index += 1

    # # plt.title("Maximum Memory Usage Comparison")
    # plt.ylabel("Memory (MB)")
    # plt.xticks(positions, labels, rotation=45)
    # plt.grid(True, axis='y')
    # plt.tight_layout()
    # plt.savefig("build/stats/memory_usage_boxplot.png")
    # plt.close()
    # print("✅ Memory usage boxplot saved to build/stats/memory_usage_boxplot.png")







def plot_rms_and_avg_error_paper(
    val_data: Dict[str, np.ndarray],
    true_val_data: np.ndarray,
    path: str,
    norm_params: np.ndarray,
    robot: str,
    val: int = 0,
    n: int = None,
    n_bins: int = 20,
    **kwargs,
) -> None:

    plt.rcParams.update(**kwargs)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)

    fig, ax = plt.subplots(
        1,
        2,
        constrained_layout=True,
        figsize=(5.3348, 2.5),
        sharey=True,
    )
    i = 0

    conv_param = 2.54 if robot == 'soft_robot' else 1

    if robot == 'soft_robot':
        if n is None:
            n = val_data[list(val_data.keys())[0]].shape[0]

        n_states = val_data[list(val_data.keys())[0]].shape[1] - 1
        n_val_eps = int(np.max(true_val_data[:, 0]) + 1)

        error_metrics = [r'Root-mean-square error', r'Mean error']
        for metric in error_metrics:
            avg_error = {}
            error = {}
            ep_val = {}
            for tag in val_data.keys():
                if tag == 'EDMD' or tag == 'FBEDMD':
                    continue
                avg_error[tag] = 0
            for val in range(n_val_eps):
                ep_val['{}'.format(val)] = {}
            max_error = 0
            regressors = [r'EDMD-AS', r'FBEDMD-AS']
            for ep in range(n_val_eps):

                for tag, data in val_data.items():
                    if tag == r'EDMD' or tag == r'FBEDMD':
                        continue
                    if metric == r'Root-mean-square error':
                        temp = np.sqrt(
                            np.sum(
                                (data[data[:, 0] == ep, 1:][:n] *
                                 (conv_param * norm_params[0])
                                 - true_val_data[true_val_data[:, 0] == ep, 1:
                                                 (n_states + 1)][:n] *
                                 (conv_param * norm_params[0]))**2) / n)
                    elif metric == r'Mean error':
                        temp = np.abs(
                            np.sum(data[data[:, 0] == ep, 1:][:n] *
                                   (conv_param * norm_params[0])
                                   - true_val_data[true_val_data[:, 0] == ep,
                                                   1:(n_states + 1)][:n] *
                                   (conv_param * norm_params[0])) / n)
                    max_error = temp if temp > max_error else max_error
                    error[tag] = temp
                    avg_error[tag] += temp / n_val_eps
                ep_val['{}'.format(ep)] = list(error.values())

            for tag in val_data.keys():
                if tag == r'EDMD' or tag == r'FBEDMD':
                    continue
                print('Average {} for {}: {}'.format(metric, tag,
                                                     avg_error[tag]))

            x_ticks = np.arange(0, len(error.keys()), 1)
            y_ticks = np.arange(0, max_error, max_error / 10)
            width = 0.1
            multiplier = 0

            for ep_tag, data in ep_val.items():
                offset = width * multiplier
                rects = ax[i].bar(x_ticks + offset,
                                  np.abs(data),
                                  width,
                                  edgecolor='k',
                                  linewidth=0.5,
                                  color=color_list[int(ep_tag)],
                                  label=r'test ep. {}'.format(int(ep_tag) + 1),
                                  zorder=3)
                multiplier += 1

            offset = width * multiplier
            rects = ax[i].bar(x_ticks + offset,
                              np.abs(list(avg_error.values())),
                              width,
                              edgecolor='k',
                              linewidth=0.5,
                              hatch='//',
                              color=color_list[int(ep_tag) + 1],
                              label='average',
                              zorder=3)

            if robot == 'soft_robot':
                ax[i].set_ylabel('{}'.format(metric) + r' (cm)')
            else:
                ax[i].set_ylabel('{}'.format(metric) + r' (mm)')
            if metric == r'Root-mean-square error':
                ax[i].set_ylim(0, 1.2)
                ax[i].set_yticks(np.linspace(0, 1.2, 7))

            ax[i].grid(axis='y', linestyle='--', zorder=0)
            i = +1

        ax[1].set_xticks(x_ticks + width * 2.0, regressors)
        ax[0].set_xticks(x_ticks + width * 2.0, regressors)
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, 0.03),
            loc='upper center',
            ncol=3,
        )

        fig.text(0.034, 1.015, r'$(a)$')
        fig.text(0.54, 1.015, r'$(b)$')

        fig.savefig('build/figures/paper/{}_error_bars.pdf'.format(robot),
                    bbox_inches='tight')
        fig.savefig('build/figures/paper/{}_error_bars.png'.format(robot),
                    bbox_inches='tight')
        fig.tight_layout()


def plot_trajectory_error_paper(
    val_data: Dict[str, np.ndarray],
    true_val_data: np.ndarray,
    path: str,
    norm_params: np.ndarray,
    robot: str,
    val: int = 2,
    n: int = None,
    **kwargs,
) -> None:

    if n is None:
        n = val_data[list(val_data.keys())[0]].shape[0]

    n_val_eps = int(np.max(true_val_data[:, 0]) + 1)

    plt.rcParams.update(**kwargs)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)

    fig_dict = {}

    conv_param = 2.54 if robot == 'soft_robot' else 1

    for i in range(n_val_eps):
        fig, axs = plt.subplot_mosaic(
            [['x1_err', 'traj'], ['x2_err', 'traj']],
            figsize=(5.3348, 2.2),
            layout='constrained',
        )
        max_y = 0
        min_y = 0
        max_x = 0
        min_x = 0

        ax = axs['traj']

        # Plot trajectory
        for tag, data in val_data.items():

            if np.isnan(data[data[:, 0] == i, 1][:n]).any() or np.isnan(
                    data[data[:, 0] == i, 2][:n]).any():
                print('NaN detected in {}.'.format(tag))
                continue
            else:
                ax.plot(data[data[:, 0] == i, 1][:n] *
                        (conv_param * norm_params[0]),
                        data[data[:, 0] == i, 2][:n] *
                        (conv_param * norm_params[1]),
                        label=tag,
                        color=color_dict[tag],
                        linestyle=linestyle_dict[tag])
            temp_x = data[data[:, 0] == i,
                          1][:n] * (conv_param * norm_params[0])
            temp_y = data[data[:, 0] == i,
                          2][:n] * (conv_param * norm_params[1])
            max_x = np.max(temp_x) if (np.max(temp_x) > max_x) else max_x
            min_x = np.min(temp_x) if (np.min(temp_x) < min_x
                                       and np.min(temp_x) < 0) else min_x
            max_y = np.max(temp_y) if (np.max(temp_y) > max_y) else max_y
            min_y = np.min(temp_y) if (np.min(temp_y) < min_y
                                       and np.min(temp_y) < 0) else min_y

        ax.plot(true_val_data[true_val_data[:, 0] == i, 1][:n] *
                (conv_param * norm_params[0]),
                true_val_data[true_val_data[:, 0] == i, 2][:n] *
                (conv_param * norm_params[1]),
                label='Ground truth',
                color='k',
                linestyle=':',
                zorder=5)

        if i == val:

            ax.scatter(true_val_data[true_val_data[:, 0] == i, 1][0] *
                       (conv_param * norm_params[0]),
                       true_val_data[true_val_data[:, 0] == i, 2][0] *
                       (conv_param * norm_params[1]),
                       marker='x',
                       s=25,
                       color='k',
                       zorder=4)

            ax.scatter(true_val_data[true_val_data[:, 0] == i, 1][n - 1] *
                       (conv_param * norm_params[0]),
                       true_val_data[true_val_data[:, 0] == i, 2][n - 1] *
                       (conv_param * norm_params[1]),
                       marker='x',
                       s=25,
                       color='k',
                       zorder=4)

        if robot == 'soft_robot':
            ax.set(ylabel=r'$x_{}$ (cm)'.format(2))
            ax.set(xlabel=r'$x_{}$ (cm)'.format(1))
        else:
            ax.set(ylabel=r'$x_{}$ (mm/s)'.format(2))
            ax.set(xlabel=r'$x_{}$ (mm)'.format(1))
        # ax.set_ylim(min_y - np.abs(min_y / 10), max_y + np.abs(max_y / 10))
        # ax.set_xlim(min_x - np.abs(min_x / 10), max_x + np.abs(max_x / 10))
        if robot == 'soft_robot':
            ax.set_ylim(0, 20)
            ax.set_xlim(0, 20)
            ax.set_xticks(np.linspace(0, 20, 11))
            ax.set_yticks(np.linspace(0, 20, 11))
        elif robot == 'nl_msd':
            ax.set_ylim(-4, 6)
            ax.set_xlim(-4, 6)
            ax.set_xticks(np.linspace(-4, 6, 6))
            ax.set_yticks(np.linspace(-4, 6, 6))

        ax.set_aspect('equal')

        ax.grid(linestyle='--')

        # Plot prediction errors
        ax = [axs['x1_err'], axs['x2_err']]

        t = np.linspace(0, n * 0.01, n) if robot == 'nl_msd' else np.linspace(
            0, n * 0.0829, n)

        conv_param = 2.54 if robot == 'soft_robot' else 1
        for tag, data in val_data.items():

            for k in range(2):

                true_temp = true_val_data[true_val_data[:, 0] == i,
                                          k + 1][:n] * (conv_param
                                                        * norm_params[k])
                pred_temp = data[data[:, 0] == i,
                                 k + 1][:n] * (conv_param * norm_params[k])

                if np.isnan(pred_temp).any():
                    print('NaN detected in {} state {}'.format(tag, k))
                    continue
                else:

                    if i == val:
                        ax[k].plot(
                            t,
                            pred_temp - true_temp,
                            color=color_dict[tag],
                            label=tag,
                            linestyle=linestyle_dict[tag],
                        )
                    else:
                        ax[k].plot(
                            pred_temp - true_temp,
                            color=color_dict[tag],
                            label=tag,
                        )

                if robot == 'soft_robot':
                    ax[k].set(ylabel='$\Delta x_{}$ (cm)'.format(k + 1))
                else:
                    if k == 0:
                        ax[k].set(ylabel='$\Delta x_{}$ (mm)'.format(k + 1))
                    else: 
                        ax[k].set(ylabel='$\Delta x_{}$ (mm/s)'.format(k + 1))


        if robot == 'nl_msd':
            ax[0].set_ylim(-3, 3)
            ax[0].set_yticks(np.linspace(-3, 3, 7))
            ax[1].set_ylim(-3, 3)
            ax[1].set_yticks(np.linspace(-3, 3, 7))
            ax[1].set_xlim(0, n * 0.01)
            ax[1].set_xticks(np.linspace(0, 20, 11))
            fig.text(0.73, 0.88, r'$t = 0$' + ' s')
            fig.text(0.745, 0.305, r'$t = 21$' + ' s')
        else:
            ax[0].set_ylim(-3, 3)
            ax[0].set_yticks(np.linspace(-3, 3, 7))
            ax[1].set_ylim(-3, 3)
            ax[1].set_yticks(np.linspace(-3, 3, 7))
            ax[1].set_xlim(0, n * 0.0829)
            ax[1].set_xticks(np.arange(0, 26, 5))
            fig.text(0.64, 0.55, r'$t = 0$' + ' s')
            fig.text(0.74, 0.51, r'$t = 24.5$' + ' s')
        ax[1].set(xlabel=r'$t$' + ' (s)')
        ax[0].grid(linestyle='--')
        ax[1].grid(linestyle='--')
        ax[0].tick_params(axis='y', which='major', pad=2)

        ax[0].sharex(ax[1])
        ax[0].tick_params(labelbottom=False)

        handles, labels = axs['traj'].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, 0.03),
            loc='upper center',
            ncol=3,
        )
        fig.text(0.034, 1.015, r'$(a)$')
        fig.text(0.56, 1.015, r'$(b)$')
        fig_dict["F_{}".format(i)] = fig

    fig_dict['F_{}'.format(val)].savefig(
        'build/figures/paper/{}_trajectory_err.png'.format(robot),
        bbox_inches='tight')

    fig_dict['F_{}'.format(val)].savefig(
        'build/figures/paper/{}_trajectory_err.pdf'.format(robot),
        bbox_inches='tight')


def plot_frob_err(frob_error: Dict[str, np.ndarray], variances: np.ndarray,
                  snr: np.ndarray, path: str, **kwargs) -> None:

    plt.rcParams.update(**kwargs)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)
    i = 0

    fig, axs = plt.subplots(1, 3, layout='constrained', figsize=(5.3348, 2))
    # for (k,v), (k2,v2) in zip(d.items(), d2.items()):
    for (matrix, err), ax in zip(frob_error.items(), range(len(axs))):
        for tag, data in err.items():
            axs[ax].plot(snr,
                         data,
                         color=color_dict[tag],
                         label=tag,
                         linestyle=linestyle_dict[tag])

        if usetex == True:
            axs[ax].set(
                ylabel=
                r'$\frac{{\left\|\mathbf{{{}}}_\mathrm{{true}} - \bf{{{}}}_\mathrm{{approx}}\right\|_\ensuremath{{\mathsf{{F}}}}}}{{\left\|\mathbf{{{}}}_\mathrm{{true}}\right\|_\ensuremath{{\mathsf{{F}}}}}}$'
                .format(matrix, matrix, matrix))
        else:
            axs[ax].set(ylabel='rel. frob. norm of {}'.format(matrix))
        axs[ax].grid(linestyle='--')
        axs[ax].set_xlim(np.min(snr), np.max(snr))
        axs[ax].set_xticks(np.arange(10, 60, 10))
        axs[ax].set(xlabel=r'SNR')

    axs[0].set_yticks(np.linspace(0, 2.5, 6))
    axs[1].sharey(axs[0])
    axs[1].tick_params(labelleft=False)
    axs[2].set_yticks(np.linspace(0, 0.20, 6))

    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 0.04),
        loc='upper center',
        ncol=4,
    )

    fig.savefig('{}/frob_norm_sqrd.png'.format(path), bbox_inches='tight')
    fig.savefig('{}/frob_norm_sqrd.pdf'.format(path), bbox_inches='tight')


def plot_polar(koop_matrices: Dict[str, np.ndarray], path: str, robot: str,
               **kwargs):

    plt.rcParams.update(**kwargs)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)

    p_theta, p = koop_matrices[list(koop_matrices.keys())[0]].shape
    n_inputs = p - p_theta

    fig = plt.figure(figsize=(5.3348, 3.5), constrained_layout=True)
    ax = plt.subplot(projection='polar')
    ax.set_xlabel(r'$\mathrm{Re}(\lambda)$')
    ax.set_ylabel(r'$\mathrm{Im}(\lambda)$', labelpad=30)

    theta_min = -12
    theta_max = 12

    axin = ax.inset_axes([1.05, -0.3, 0.7, 0.7], projection='polar')

    axes = [ax, axin]
    sub_eig = 0
    sup_eig = 2
    max_eig = 0
    i = 0

    koop_matrices_new = {}
    koop_matrices_new['EDMD'] = koop_matrices['EDMD']
    koop_matrices_new['EDMD-AS'] = koop_matrices['EDMD-AS']
    koop_matrices_new['FBEDMD'] = koop_matrices['FBEDMD']
    koop_matrices_new['FBEDMD-AS'] = koop_matrices['FBEDMD-AS']

    zorder = {}
    zorder['EDMD'] = 7
    zorder['EDMD-AS'] = 8
    zorder['FBEDMD'] = 5
    zorder['FBEDMD-AS'] = 6

    markers = ['s', 'D', 'v', 'o'] + ['o'] * (len(koop_matrices.keys()) - 4)

    # plot eigenvalues
    for tag, U in koop_matrices_new.items():
        eigv = scipy.linalg.eig(U[:, :-n_inputs])[0]
        marker = markers.pop()

        for axx in axes:
            axx.scatter(np.angle(eigv),
                        np.absolute(eigv),
                        zorder=zorder[tag],
                        marker=marker,
                        s=60,
                        color=color_dict[tag],
                        linewidths=0.5,
                        edgecolors='w',
                        label=tag)
        i += 1

    unit_angles = np.linspace(0, 2 * np.pi, 100)
    unit_radius = np.ones_like(unit_angles)
    axin.plot(unit_angles, unit_radius, color='k', linestyle='--', linewidth=1)

    axin.set_thetamin(theta_min)
    axin.set_thetamax(theta_max)

    sub = 0.999
    sup = 1.0001

    sub = float("{:.5f}".format(sub))
    sup = float("{:.5f}".format(sup))
    axin.set_rticks([sub - (1 - sub), sub, 1.0, sup])
    axin.set_rmin(sub)
    axin.set_rmax(sup)
    axin.set_xticks([
        theta_min * np.pi / 180, theta_min / 2 * np.pi / 180, 0,
        theta_max / 2 * np.pi / 180, theta_max * np.pi / 180
    ])

    ax.plot(unit_angles, unit_radius, color='k', linestyle='--', linewidth=1)

    ax.set_rlim(0, 1.5)
    ax.set_rticks([0, 0.5, 1, 1.5])
    ax.grid(linestyle='--')
    axin.set_rticks([sub, sup])
    axin.set_rmin(sub)
    axin.set_rmax(sup)
    axin.grid(linestyle='--')

    # Create lines linking border to zoomed plot
    axin.annotate(
        '',
        xy=(0, sup),
        xycoords=ax.transData,
        xytext=(theta_max * np.pi / 180, sup),
        textcoords=axin.transData,
        arrowprops={
            'arrowstyle': '-',
            'color': 'k',
            'shrinkA': 0,
            'shrinkB': 0,
        },
    )

    axin.annotate(
        '',
        xy=(0, sub),
        xycoords=ax.transData,
        xytext=(0, sub),
        textcoords=axin.transData,
        arrowprops={
            'arrowstyle': '-',
            'color': 'k',
            'shrinkA': 0,
            'shrinkB': 0,
        },
    )

    axin.set_zorder(15)

    ax.legend(
        bbox_to_anchor=(0.74, -0.2),
        loc='upper center',
        ncol=4,
    )

    fig.savefig('build/figures/paper/{}_polar.pdf'.format(robot))
    fig.savefig('build/figures/paper/{}_polar.png'.format(robot))


def summary_fig(
    val_data: Dict[str, np.ndarray],
    true_val_data: np.ndarray,
    path: str,
    norm_params: np.ndarray,
    robot: str,
    val: int = 2,
    n: int = None,
    **kwargs,
) -> None:

    if n is None:
        n = val_data[list(val_data.keys())[0]].shape[0]

    n_val_eps = int(np.max(true_val_data[:, 0]) + 1)

    plt.rcParams.update(**kwargs)
    plt.rc('font', size=12)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)

    fig_dict = {}

    conv_param = 2.54 if robot == 'soft_robot' else 1

    for i in range(n_val_eps):
        fig, ax = plt.subplots(
            figsize=(5, 2.2),
            layout='constrained',
        )
        max_y = 0
        min_y = 0
        max_x = 0
        min_x = 0

        ax.plot(true_val_data[true_val_data[:, 0] == i, 1][:(n - 1000)] *
                (conv_param * norm_params[0]),
                true_val_data[true_val_data[:, 0] == i, 2][:(n - 1000)] *
                (conv_param * norm_params[1]),
                label='Ground truth',
                color='k',
                linestyle=':',
                zorder=5,
                linewidth=2.5)

        for tag, data in val_data.items():

            if np.isnan(data[data[:, 0] == i,
                             1][:(n - 1000)]).any() or np.isnan(
                                 data[data[:, 0] == i, 2][:(n - 1000)]).any():
                print('NaN detected in {}.'.format(tag))
                continue
            else:
                if tag == 'EDMD' or tag == 'FBEDMD':
                    continue
                else:
                    if tag == "EDMD-AS":
                        ax.plot(data[data[:, 0] == i, 1][:(n - 1000)] *
                                (conv_param * norm_params[0]),
                                data[data[:, 0] == i, 2][:(n - 1000)] *
                                (conv_param * norm_params[1]),
                                label=tag + " (biased)",
                                color=color_list[2],
                                linestyle=linestyle_dict['EDMD'],
                                zorder=4,
                                linewidth=2.5)
                    else:
                        ax.plot(data[data[:, 0] == i, 1][:(n - 1000)] *
                                (conv_param * norm_params[0]),
                                data[data[:, 0] == i, 2][:(n - 1000)] *
                                (conv_param * norm_params[1]),
                                label=tag + " (unbiased)",
                                color=color_list[1],
                                linestyle=linestyle_dict['EDMD'],
                                zorder=3,
                                linewidth=2.5)
            temp_x = data[data[:, 0] == i,
                          1][:n] * (conv_param * norm_params[0])
            temp_y = data[data[:, 0] == i,
                          2][:n] * (conv_param * norm_params[1])
            max_x = np.max(temp_x) if (np.max(temp_x) > max_x) else max_x
            min_x = np.min(temp_x) if (np.min(temp_x) < min_x
                                       and np.min(temp_x) < 0) else min_x
            max_y = np.max(temp_y) if (np.max(temp_y) > max_y) else max_y
            min_y = np.min(temp_y) if (np.min(temp_y) < min_y
                                       and np.min(temp_y) < 0) else min_y

        if robot == 'soft_robot':
            ax.set(ylabel=r'$x_{}$ (cm)'.format(2))
            ax.set(xlabel=r'$x_{}$ (cm)'.format(1))
        else:
            ax.set(ylabel=r'$x_{}$'.format(2))
            ax.set(xlabel=r'$x_{}$'.format(1))
        if robot == 'soft_robot':
            ax.set_ylim(0, 20)
            ax.set_xlim(0, 20)
            ax.set_xticks(np.linspace(0, 20, 11))
            ax.set_yticks(np.linspace(0, 20, 11))

        ax.set_aspect('equal')

        ax.tick_params(labelleft=False, labelbottom=False)
        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       top=False,
                       labelbottom=False,
                       left=False,
                       labelleft=False)

        fig.text(0.49, 0.8, r'$\mathbf{U}_\mathrm{f}$')
        fig.text(0.595, 0.8, r'$\tilde{\mathbf{U}}$')

        handles, labels = ax.get_legend_handles_labels()

        fig_dict["F_{}".format(i)] = fig

    fig_dict['F_{}'.format(val)].savefig(
        'build/figures/paper/{}_summary_trajectory.png'.format(robot),
        bbox_inches='tight')

    fig_dict['F_{}'.format(val)].savefig(
        'build/figures/paper/{}_summary_trajectory.pdf'.format(robot),
        bbox_inches='tight')


def print_koop_matrices(koop_matrices: Dict[str, np.ndarray], **kwargs):

    n_inputs = koop_matrices[list(
        koop_matrices.keys())[0]].shape[1] - koop_matrices[list(
            koop_matrices.keys())[0]].shape[0]

    for tag, U in koop_matrices.items():
        print('Eigenvalues of A_{}: '.format(tag),
              scipy.linalg.eig(U[:, :-n_inputs])[0])


class LmiEdmd(LmiRegressor):
    """LMI-based EDMD with regularization. Inspired from the pykoop package.

    """

    def __init__(
        self,
        alpha: float = 0,
        ratio: float = 1,
        reg_method: str = 'tikhonov',
        inv_method: str = 'svd',
        tsvd: Optional[tsvd.Tsvd] = None,
        square_norm: bool = False,
        picos_eps: float = 0,
        solver_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Instantiate :class:`LmiEdmd`. Inspired from the pykoop package.

        """
        self.alpha = alpha
        self.ratio = ratio
        self.reg_method = reg_method
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.square_norm = square_norm
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Compute regularization coefficients
        if self.reg_method == 'tikhonov':
            self.alpha_tikhonov_ = self.alpha
            self.alpha_other_ = 0.0
        else:
            self.alpha_tikhonov_ = self.alpha * (1.0 - self.ratio)
            self.alpha_other_ = self.alpha * self.ratio
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Form optimization problem. Regularization coefficients must be scaled
        # because of how G and H are defined.
        q = X_unshifted.shape[0]
        problem = self._create_base_problem(X_unshifted, X_shifted,
                                            self.alpha_tikhonov_ / q,
                                            self.inv_method, self.tsvd_,
                                            self.picos_eps)
        if self.reg_method == 'twonorm':
            problem = lmi_regressors._add_twonorm(problem,
                                                  problem.variables['U'],
                                                  self.alpha_other_ / q,
                                                  self.square_norm,
                                                  self.picos_eps)
        elif self.reg_method == 'nuclear':
            problem = lmi_regressors._add_nuclear(problem,
                                                  problem.variables['U'],
                                                  self.alpha_other_ / q,
                                                  self.square_norm,
                                                  self.picos_eps)
        # Solve optimization problem
        problem.solve(**self.solver_params_)
        # Save solution status
        self.solution_status_ = problem.last_solution.claimedStatus
        # Extract solution from ``Problem`` object
        coef = self._extract_solution(problem)
        return coef

    def _validate_parameters(self) -> None:
        # Check regularization methods
        valid_reg_methods = ['tikhonov', 'twonorm', 'nuclear']
        if self.reg_method not in valid_reg_methods:
            raise ValueError('`reg_method` must be one of '
                             f'{valid_reg_methods}.')
        # Check ratio
        if (self.ratio <= 0) or (self.ratio > 1):
            raise ValueError('`ratio` must be positive and less than one.')

    @staticmethod
    def _create_base_problem(
        X_unshifted: np.ndarray,
        X_shifted: np.ndarray,
        alpha_tikhonov: float,
        inv_method: str,
        tsvd: tsvd.Tsvd,
        picos_eps: float,
    ) -> picos.Problem:
        """Create optimization problem."""
        # Validate ``alpha``
        if alpha_tikhonov < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``inv_method``
        valid_inv_methods = [
            'inv', 'pinv', 'eig', 'ldl', 'chol', 'sqrt', 'svd'
        ]
        if inv_method not in valid_inv_methods:
            raise ValueError('`inv_method` must be one of '
                             f'{valid_inv_methods}.')
        # Validate ``picos_eps``
        if picos_eps < 0:
            raise ValueError('Parameter `picos_eps` must be positive or zero.')
        var = np.var(X_shifted, axis=0)
        W = np.diag(1 / var)
        W = np.eye(W.shape[0])
        # Compute ``G`` and ``H``. ``alpha_tikhonov`` must already be scaled
        # by ``q`` if applicable.
        c, G, H, _ = lmi_regressors._calc_c_G_H(X_unshifted, X_shifted @ W.T,
                                                alpha_tikhonov)
        # Optimization problem
        problem = picos.Problem()
        # Constants
        G_T = picos.Constant('G^T', G.T)
        W = picos.Constant('W', W)
        q = X_shifted.shape[0]
        gamma = 10
        # Variables
        U = picos.RealVariable('U', (G.shape[0], H.shape[0]))
        Z = picos.SymmetricVariable('Z', (G.shape[0], G.shape[0]))
        # W = picos.RealVariable('W', (G.shape[0], G.shape[0]))
        v = picos.RealVariable('v', 1)
        Q = picos.SymmetricVariable('Q', (G.shape[1], G.shape[1]))
        # Constraints
        problem.add_constraint(Z >> picos_eps)
        # problem.add_constraint(Q >> picos_eps)
        # Choose method to handle inverse of H
        if inv_method == 'inv':
            H_inv = picos.Constant('H^-1', lmi_regressors._calc_Hinv(H))
            problem.add_constraint(
                picos.block([
                    [Z, U],
                    [U.T, H_inv],
                ]) >> picos_eps)
        elif inv_method == 'pinv':
            H_inv = picos.Constant('H^+', lmi_regressors._calc_Hpinv(H))
            problem.add_constraint(
                picos.block([
                    [Z, U],
                    [U.T, H_inv],
                ]) >> picos_eps)
        elif inv_method == 'eig':
            VsqrtLmb = picos.Constant('(V Lambda^(1/2))',
                                      lmi_regressors._calc_VsqrtLmb(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * VsqrtLmb],
                    [VsqrtLmb.T * U.T, 'I'],
                ]) >> picos_eps)
        elif inv_method == 'ldl':
            LsqrtD = picos.Constant('(L D^(1/2))',
                                    lmi_regressors._calc_LsqrtD(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * LsqrtD],
                    [LsqrtD.T * U.T, 'I'],
                ]) >> picos_eps)
        elif inv_method == 'chol':
            L = picos.Constant('L', lmi_regressors._calc_L(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * L],
                    [L.T * U.T, 'I'],
                ]) >> picos_eps)
        elif inv_method == 'sqrt':
            sqrtH = picos.Constant('sqrt(H)', lmi_regressors._calc_sqrtH(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * sqrtH],
                    [sqrtH.T * U.T, 'I'],
                ]) >> picos_eps)
        elif inv_method == 'svd':
            QSig = picos.Constant(
                'Q Sigma',
                lmi_regressors._calc_QSig(X_unshifted, alpha_tikhonov, tsvd))
            problem.add_constraint(
                picos.block([
                    [Z, W * U * QSig],
                    [QSig.T * U.T * W.T, 'I'],
                ]) >> picos_eps)
        else:
            # Should never, ever get here.
            assert False
        # Set objective
        obj = c - 2 * picos.trace(W * U * G_T * W.T) + picos.trace(Z)
        problem.set_objective('min', obj)
        return problem

    @staticmethod
    def _create_new_forw_base_problem(
        # Variablesm(
        X_unshifted: np.ndarray,
        X_shifted: np.ndarray,
        alpha_tikhonov: float,
        spectral_radius: float,
        picos_eps: float,
        new_eps: np.ndarray,
    ) -> picos.Problem:
        """Create optimization problem."""
        # Validate ``alpha``
        if alpha_tikhonov < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``inv_method``

        # Validate ``picos_eps``
        if picos_eps < 0:
            raise ValueError('Parameter `picos_eps` must be positive or zero.')
        # Compute ``G`` and ``H``. ``alpha_tikhonov`` must already be scaled
        # by ``q`` if applicable.
        # c, G, H, _ = lmi_regressors._calc_c_G_H(X_unshifted, X_shifted, alpha_tikhonov)
        # G = picos.Constant('G', G)
        # H = picos.Constant('H', H)

        # Optimization problem
        problem = picos.Problem()

        # Constants
        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi = X_unshifted.T
        Theta_plus = X_shifted.T
        n_inputs = Psi.shape[0] - Theta_plus.shape[0]

        _G = 1 / q * Theta_plus @ Psi.T
        # _G = _G + np.eye(_G.shape[0])*1e-5
        _H = 1 / q * Psi @ Psi.T
        # _H = _H + np.eye(_H.shape[0])*1e-5
        H_inv = scipy.linalg.pinv(_H)
        U_edmd = scipy.linalg.lstsq(_H.T, _G.T)[0].T
        # _R = scipy.linalg.cholesky(_Hb, lower=True)
        W = picos.Constant(
            'W',
            picos.block([[np.eye(p_theta),
                          np.zeros((p_theta, n_inputs))],
                         [np.zeros((n_inputs, p_theta)),
                          np.eye(n_inputs)]]))
        rho = picos.Constant('rho', spectral_radius)
        alpha = picos.Constant('alpha', 1)

        # Variables
        P = picos.SymmetricVariable('P', p_theta)
        B = picos.RealVariable('B', (_G.shape[0], _H.shape[0] - _G.shape[0]))
        K = picos.RealVariable('K', (p_theta, p_theta))
        P_tilde = picos.block(
            [[P, np.zeros((P.shape[0], H_inv.shape[0] - P.shape[0]))],
             [
                 np.zeros((H_inv.shape[0] - P.shape[0], P.shape[0])),
                 alpha * np.eye(H_inv.shape[0] - P.shape[0])
             ]])
        gamma = picos.RealVariable('gamma_f', 1)
        Z = picos.SymmetricVariable('Z', _H.shape[0])

        # Constraints for forward dynamics
        # problem.add_constraint(P >> picos_eps)
        problem.add_constraint(P >> new_eps)
        problem.add_constraint(
            picos.block([
                [rho * P, K],
                [K.T, rho * P],
            ]) >> picos_eps)
        problem.add_constraint(picos.trace(Z) << 1)
        problem.add_constraint(Z >> picos_eps)
        problem.add_constraint(
            picos.block([[Z, (U_edmd * P_tilde - picos.block([[K, B]])).T],
                         [(U_edmd * P_tilde - picos.block([[K, B]])), gamma
                          * np.eye(p_theta)]]) >> picos_eps)

        obj = gamma
        problem.set_objective('min', obj)
        return problem

    @staticmethod
    def _create_new_avg_base_problem(
        # Variablesm(
        X_unshifted: np.ndarray,
        X_shifted: np.ndarray,
        alpha_tikhonov: float,
        spectral_radius: float,
        picos_eps: float,
        new_eps: np.ndarray,
    ) -> picos.Problem:
        """Create optimization problem."""
        # Validate ``alpha``
        if alpha_tikhonov < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``inv_method``

        # Validate ``picos_eps``
        if picos_eps < 0:
            raise ValueError('Parameter `picos_eps` must be positive or zero.')
        # Compute ``G`` and ``H``. ``alpha_tikhonov`` must already be scaled
        # by ``q`` if applicable.
        # c, G, H, _ = lmi_regressors._calc_c_G_H(X_unshifted, X_shifted, alpha_tikhonov)
        # G = picos.Constant('G', G)
        # H = picos.Constant('H', H)

        # Optimization problem
        problem = picos.Problem()

        # Constants
        # var = np.var(X_shifted, axis=0)
        # W = np.diag(1 / var)
        W = np.eye(X_shifted.shape[1])
        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi_f = X_unshifted.T
        Theta_plus_f = X_shifted.T
        n_inputs = Psi_f.shape[0] - Theta_plus_f.shape[0]
        Psi_b = np.zeros(Psi_f.shape)
        Theta_plus_b = np.zeros(Theta_plus_f.shape)
        Psi_b[:-n_inputs, :] = Theta_plus_f
        Psi_b[-n_inputs:, :] = Psi_f[-n_inputs:, :]
        Theta_plus_b = Psi_f[:-n_inputs, :]

        _Gf = 1 / q * Theta_plus_f @ Psi_f.T
        # _Gf = _Gf + np.eye(_Gf.shape[0])*1e-5
        _Hf = 1 / q * Psi_f @ Psi_f.T
        # _Hf = _Hf + np.eye(_Hf.shape[0])*1e-5
        H_inv_f = scipy.linalg.pinv(_Hf)
        U_edmd_f = scipy.linalg.lstsq(_Hf.T, _Gf.T)[0].T
        _Gb = 1 / q * Theta_plus_b @ Psi_b.T
        _Hb = 1 / q * Psi_b @ Psi_b.T
        H_inv_b = scipy.linalg.pinv(_Hb)
        U_edmd_b = scipy.linalg.lstsq(_Hb.T, _Gb.T)[0].T
        # _R = scipy.linalg.cholesky(_Hb, lower=True)
        W = picos.Constant('W', W)
        rho = picos.Constant('rho', spectral_radius)
        alpha = picos.Constant('alpha', 1)

        # Variables
        P = picos.SymmetricVariable('P', p_theta)
        Bf = picos.RealVariable('Bf',
                                (_Gf.shape[0], _Hf.shape[0] - _Gf.shape[0]))
        Kf = picos.RealVariable('Kf', (p_theta, p_theta))
        Bb = picos.RealVariable('Bb',
                                (_Gb.shape[0], _Hb.shape[0] - _Gb.shape[0]))
        Kb = picos.RealVariable('Kb', (p_theta, p_theta))
        P_tilde = picos.block(
            [[P, np.zeros((P.shape[0], H_inv_f.shape[0] - P.shape[0]))],
             [
                 np.zeros((H_inv_f.shape[0] - P.shape[0], P.shape[0])),
                 alpha * np.eye(H_inv_f.shape[0] - P.shape[0])
             ]])
        gamma_f = picos.RealVariable('gamma_f', 1)
        gamma_b = picos.RealVariable('gamma_b', 1)
        Zf = picos.SymmetricVariable('Zf', _Hf.shape[0])
        Zb = picos.SymmetricVariable('Zb', _Hb.shape[0])
        Qf = picos.SymmetricVariable('Qf', _Gf.shape[0])

        # Constraints for forward dynamics
        # problem.add_constraint(P >> picos_eps)
        problem.add_constraint(P >> new_eps)
        problem.add_constraint(
            picos.block([
                [rho * P, Kf],
                [Kf.T, rho * P],
            ]) >> picos_eps)
        problem.add_constraint(picos.trace(Zf) << 1)
        problem.add_constraint(Zf >> picos_eps)
        problem.add_constraint(
            picos.block([
                [Zf, (W * U_edmd_f * P_tilde - W * picos.block([[Kf, Bf]])).T],
                [(W * U_edmd_f * P_tilde
                  - W * picos.block([[Kf, Bf]])), gamma_f * np.eye(p_theta)]
            ]) >> picos_eps)

        # Constraints for backward dynamics
        problem.add_constraint(rho * Kb + rho * Kb.T - 2 * P >> picos_eps)
        problem.add_constraint(picos.trace(Zb) << 1)
        problem.add_constraint(Zb >> picos_eps)
        problem.add_constraint(
            picos.block([
                [Zb, (W * U_edmd_b * P_tilde - W * picos.block([[Kb, Bb]])).T],
                [(W * U_edmd_b * P_tilde
                  - W * picos.block([[Kb, Bb]])), gamma_b * np.eye(p_theta)]
            ]) >> picos_eps)

        obj = gamma_f + gamma_b
        problem.set_objective('min', obj)
        return problem


class LmiEdmdSpectralRadiusConstrForw(LmiRegressor):
    """LMI-based EDMD with spectral radius constraint. Inspired by the pykoop package.
    """

    def __init__(self,
                 spectral_radius: float = 1.0,
                 new_eps: np.ndarray = 1e-2,
                 max_iter: int = 100,
                 iter_atol: float = 1e-8,
                 iter_rtol: float = 0,
                 alpha: float = 0,
                 inv_method: str = 'svd',
                 tsvd: tsvd.Tsvd = None,
                 picos_eps: float = 1e-5,
                 P: np.ndarray = None,
                 solver_params: Dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiEdmdSpectralRadiusConstr`. Inspired by the pykoop package.

        """
        self.spectral_radius = spectral_radius
        self.new_eps = new_eps
        self.max_iter = max_iter
        self.iter_atol = iter_atol
        self.iter_rtol = iter_rtol
        self.alpha = alpha
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.picos_eps = picos_eps
        self.solver_params = solver_params
        self.P = P
        # self.cost = np.zeros((self.new_eps.shape))
        # self.new_rad = np.zeros((self.new_eps.shape))

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Make initial guesses and iterate
        P = np.eye(p_theta)
        # Set scope of other variables
        U = np.zeros((p_theta, p))
        self.objective_log_ = []

        # Forw

        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi = X_unshifted.T
        Theta_plus = X_shifted.T
        n_inputs = Psi.shape[0] - Theta_plus.shape[0]
        new_eps_og = np.linalg.norm(Psi.T @ scipy.linalg.pinv(Psi @ Psi.T), 2)

        _G = 1 / q * Theta_plus @ Psi.T
        _H = 1 / q * Psi @ Psi.T
        H_inv = scipy.linalg.pinv(_H)
        U_edmd = scipy.linalg.lstsq(_H.T, _G.T)[0].T

        if self.new_eps == -1:
            self.new_eps = new_eps_og

        # self.solver_params_ = {
        #     'solver': 'mosek',
        #     'primals': True,
        #     'duals': True,
        #     'dualize': True,
        #     'abs_bnb_opt_tol': None,
        #     'abs_dual_fsb_tol': 1e-9,
        #     'abs_ipm_opt_tol': 1e-9,
        #     'abs_prim_fsb_tol': 1e-9,
        #     'integrality_tol': None,
        #     'markowitz_tol': None,
        #     'rel_bnb_opt_tol': None,
        #     'rel_dual_fsb_tol': 1e-9,
        #     'rel_ipm_opt_tol': 1e-9,
        #     'rel_prim_fsb_tol': 1e-9,
        # }

        problem = LmiEdmd._create_new_forw_base_problem(
            X_unshifted, X_shifted, self.alpha, self.spectral_radius,
            self.picos_eps, self.new_eps)

        problem.solve(**self.solver_params_, verbose=True)

        K = np.array(problem.get_valued_variable('K'), ndmin=2)
        B = np.array(problem.get_valued_variable('B'), ndmin=2)
        P = np.array(problem.get_valued_variable('P'),
                     ndmin=2) if self.P is None else self.P

        A = scipy.linalg.solve(P.T, K.T).T

        coef = np.block([[A, B]])

        return coef.T

    def _validate_parameters(self) -> None:
        # Check spectral radius
        if self.spectral_radius <= 0:
            raise ValueError('`spectral_radius` must be positive.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_atol < 0:
            raise ValueError('`iter_atol` must be positive or zero.')
        if self.iter_rtol < 0:
            raise ValueError('`iter_rtol` must be positive or zero.')


class LmiEdmdSpectralRadiusConstrAvg(LmiRegressor):
    """LMI-based EDMD with spectral radius constraint for forward and backward. Inspired by the pykoop package. 
    """

    def __init__(self,
                 spectral_radius: float = 1.0,
                 new_eps: np.ndarray = 1e-2,
                 max_iter: int = 100,
                 iter_atol: float = 1e-6,
                 iter_rtol: float = 0,
                 alpha: float = 0,
                 inv_method: str = 'svd',
                 tsvd: tsvd.Tsvd = None,
                 picos_eps: float = 1e-5,
                 solver_params: Dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiEdmdSpectralRadiusConstrAvg`. Inspired by the pykoop package.
        """
        self.spectral_radius = spectral_radius
        self.new_eps = new_eps
        self.max_iter = max_iter
        self.iter_atol = iter_atol
        self.iter_rtol = iter_rtol
        self.alpha = alpha
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Make initial guesses and iterate
        P = np.eye(p_theta)
        # Set scope of other variables
        U = np.zeros((p_theta, p))
        self.objective_log_ = []

        # Avg
        var = np.var(X_shifted, axis=0)
        W = np.diag(1 / var)
        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi_f = X_unshifted.T
        Theta_plus_f = X_shifted.T
        n_inputs = Psi_f.shape[0] - Theta_plus_f.shape[0]
        Psi_b = np.zeros(Psi_f.shape)
        Psi_b[:-(n_inputs), :] = np.flip(Theta_plus_f, axis=1)
        Psi_b[-(n_inputs):, :-1] = np.flip(Psi_f[-(n_inputs):, :-1], axis=1)
        Theta_plus_b = np.flip(Psi_f[:-(n_inputs), :], axis=1)
        new_eps_og = np.linalg.norm(
            Psi_f.T @ scipy.linalg.pinv(Psi_f @ Psi_f.T), 2)
        # new_eps_og = 0.001

        _Gf = 1 / q * Theta_plus_f @ Psi_f.T
        _Hf = 1 / q * Psi_f @ Psi_f.T
        H_inv_f = scipy.linalg.pinv(_Hf)
        U_edmd_f = scipy.linalg.lstsq(_Hf.T, _Gf.T)[0].T
        _Gb = 1 / q * Theta_plus_b @ Psi_b.T
        _Hb = 1 / q * Psi_b @ Psi_b.T
        H_inv_b = scipy.linalg.pinv(_Hb)
        U_edmd_b = scipy.linalg.lstsq(_Hb.T, _Gb.T)[0].T

        if self.new_eps == -1:
            self.new_eps = new_eps_og

        # self.solver_params_ = {
        #     'solver': 'mosek',
        #     'primals': True,
        #     'duals': False,
        #     'dualize': False,
        #     'abs_bnb_opt_tol': None,
        #     'abs_dual_fsb_tol': 1e-9,
        #     'abs_ipm_opt_tol': 1e-9,
        #     'abs_prim_fsb_tol': 1e-9,
        #     'integrality_tol': None,
        #     'markowitz_tol': None,
        #     'rel_bnb_opt_tol': None,
        #     'rel_dual_fsb_tol': 1e-9,
        #     'rel_ipm_opt_tol': 1e-9,
        #     'rel_prim_fsb_tol': 1e-9,
        # }

        problem = LmiEdmd._create_new_avg_base_problem(X_unshifted, X_shifted,
                                                       self.alpha,
                                                       self.spectral_radius,
                                                       self.picos_eps,
                                                       self.new_eps)

        problem.solve(**self.solver_params_, verbose=True)

        P = np.array(problem.get_valued_variable('P'), ndmin=2)

        Kf = np.array(problem.get_valued_variable('Kf'), ndmin=2)
        Af = scipy.linalg.solve(P.T, Kf.T).T
        Bf = np.array(problem.get_valued_variable('Bf'), ndmin=2)

        Kb = np.array(problem.get_valued_variable('Kb'), ndmin=2)
        Bb = np.array(problem.get_valued_variable('Bb'), ndmin=2)
        Ab = scipy.linalg.solve(P.T, Kb.T).T

        P_tilde = np.block(
            [[P, np.zeros((P.shape[0], H_inv_f.shape[0] - P.shape[0]))],
             [
                 np.zeros((H_inv_f.shape[0] - P.shape[0], P.shape[0])),
                 np.eye(H_inv_f.shape[0] - P.shape[0])
             ]])

        A_squared = scipy.linalg.lstsq(Ab.T, Af.T)[0].T
        Afb = scipy.linalg.lstsq(Ab, np.eye(Ab.shape[0]))[0]
        Bfb = -Afb @ Bb
        temp = Bf + Af @ Bfb

        A = scipy.linalg.sqrtm(A_squared).real
        B = scipy.linalg.lstsq((np.eye(A.shape[0]) + A), temp)[0]

        return np.hstack([A, B]).T

    def _validate_parameters(self) -> None:
        # Check spectral radius
        if self.spectral_radius <= 0:
            raise ValueError('`spectral_radius` must be positive.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_atol < 0:
            raise ValueError('`iter_atol` must be positive or zero.')
        if self.iter_rtol < 0:
            raise ValueError('`iter_rtol` must be positive or zero.')


class LmiEdmdSpectralRadiusConstrAvgNoAS(LmiRegressor):
    """LMI-based EDMD with no spectral radius constraint for forward and backward. Inspired by the pykoop package.

   
    """

    def __init__(self,
                 spectral_radius: float = 1.0,
                 new_eps: np.ndarray = 1e-2,
                 max_iter: int = 100,
                 iter_atol: float = 1e-12,
                 iter_rtol: float = 0,
                 alpha: float = 0,
                 inv_method: str = 'svd',
                 tsvd: tsvd.Tsvd = None,
                 picos_eps: float = 1e-5,
                 solver_params: Dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiEdmdSpectralRadiusConstrAvgNoAS'. Inspired by the pykoop package.

        """
        self.spectral_radius = spectral_radius
        self.new_eps = new_eps
        self.max_iter = max_iter
        self.iter_atol = iter_atol
        self.iter_rtol = iter_rtol
        self.alpha = alpha
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Make initial guesses and iterate
        P = np.eye(p_theta)
        # Set scope of other variables
        U = np.zeros((p_theta, p))
        self.objective_log_ = []

        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi_f = X_unshifted.T
        Theta_plus_f = X_shifted.T
        n_inputs = Psi_f.shape[0] - Theta_plus_f.shape[0]
        Psi_b = np.zeros(Psi_f.shape)
        Psi_b[:-(n_inputs), :] = np.flip(Theta_plus_f, axis=1)
        Psi_b[-(n_inputs):, :-1] = np.flip(Psi_f[-(n_inputs):, :-1], axis=1)
        Theta_plus_b = np.flip(Psi_f[:-(n_inputs), :], axis=1)

        # self.solver_params_ = {
        #     'solver': 'mosek',
        #     'primals': True,
        #     'duals': True,
        #     'dualize': True,
        #     'abs_bnb_opt_tol': None,
        #     'abs_dual_fsb_tol': 1e-9,
        #     'abs_ipm_opt_tol': 1e-9,
        #     'abs_prim_fsb_tol': 1e-9,
        #     'integrality_tol': None,
        #     'markowitz_tol': None,
        #     'rel_bnb_opt_tol': None,
        #     'rel_dual_fsb_tol': 1e-9,
        #     'rel_ipm_opt_tol': 1e-9,
        #     'rel_prim_fsb_tol': 1e-9,
        # }

        kp_f = pykoop.Edmd(alpha=0)
        kp_f.fit(Psi_f.T, Theta_plus_f.T)
        Af = kp_f.coef_.T[:, :p_theta]
        Bf = kp_f.coef_.T[:, p_theta:]

        kp_b = pykoop.Edmd(alpha=0)
        kp_b.fit(Psi_b.T, Theta_plus_b.T)
        Ab = kp_b.coef_.T[:, :p_theta]
        Bb = kp_b.coef_.T[:, p_theta:]

        Afb = scipy.linalg.lstsq(Ab, np.eye(Ab.shape[0]))[0]
        Bfb = -Afb @ Bb
        A_squared = scipy.linalg.lstsq(Ab.T, Af.T)[0].T
        temp = Bf + Af @ Bfb

        A = scipy.linalg.sqrtm(A_squared).real
        B = scipy.linalg.lstsq((np.eye(A.shape[0]) + A), temp)[0]

        return np.hstack([A, B]).T
    


    def _validate_parameters(self) -> None:
        # Check spectral radius
        if self.spectral_radius <= 0:
            raise ValueError('`spectral_radius` must be positive.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_atol < 0:
            raise ValueError('`iter_atol` must be positive or zero.')
        if self.iter_rtol < 0:
            raise ValueError('`iter_rtol` must be positive or zero.')
