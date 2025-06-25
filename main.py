import numpy as np
import hydra
import omegaconf
import pickle
from pprint import pprint
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import logging
import os
from random import randint
import config
# from memory_profiler import profile
import time
# from scalene import scalene_profiler
from pathlib import Path

@hydra.main(config_path="config",
            config_name="default_kp_config",
            version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:

    # Format the config file
    kp = hydra.utils.instantiate(cfg.pykoop_pipeline, _convert_='all')
    hydra_cfg = HydraConfig.get()
    # Get parameters and create folders
    if cfg.robot == 'nl_msd':
        path = "build/pykoop_objects/{}/variance_{}/kp_{}_{}.bin".format(
            cfg.robot, cfg.variance,
            hydra_cfg.runtime.choices['regressors@pykoop_pipeline'],
            hydra_cfg.runtime.choices['lifting_functions@pykoop_pipeline'])
    elif cfg.robot == 'soft_robot':
        path = "build/pykoop_objects/{}/variance_{}/kp_{}_{}.bin".format(
            cfg.robot, cfg.variance,
            hydra_cfg.runtime.choices['regressors@pykoop_pipeline'],
            hydra_cfg.runtime.choices['lifting_functions@pykoop_pipeline'])
    os.makedirs(os.path.dirname(path), exist_ok=True)

    hydra_cfg = HydraConfig.get()

    # Get preprocessed data
    with open(
            "build/preprocessed_data/{}/variance_{}.bin".format(
                cfg.robot, cfg.variance), "rb") as f:
        data = pickle.load(f)

    # Train model 
    # profile(kp.fit)(data.pykoop_dict['X_train'],
    #        n_inputs=data.pykoop_dict['n_inputs'],
    #        episode_feature=True)

    # Train model
    # if cfg.profiler == False:
    #     kp.fit(data.pykoop_dict['X_train'],
    #        n_inputs=data.pykoop_dict['n_inputs'],
    #        episode_feature=True)
    
    # Analyse computation cost using profiler
    kp.fit(data.pykoop_dict['X_train'],
        n_inputs=data.pykoop_dict['n_inputs'],
        episode_feature=True)
    
    # profile(kp.fit)(data.pykoop_dict['X_train'],
    #     n_inputs=data.pykoop_dict['n_inputs'],
    #     episode_feature=True)

    with open(path, "wb") as f:
        data_dump = pickle.dump(kp, f)

    # Predict. Note that pedictions are only good at low noise.
    if cfg.profiler == False:
        if cfg.variance < 0.1:
            kp.x_pred = kp.predict_trajectory(
                data.pykoop_dict['x0_valid'],
                data.pykoop_dict['u_valid'],
                relift_state=True,
                return_lifted=False,
            )
  

    with open(path, "wb") as f:
        data_dump = pickle.dump(kp, f)
    
    # stats_file = Path("stats")

    # last_func_line = None
    # with stats_file.open() as f:
    #     for line in f:
    #         if line.startswith("FUNC") and "pykoop.koopman_pipeline.fit" in line:
    #             last_func_line = line.strip()

    # if last_func_line is None:
    #     print("No matching FUNC line found.")
    # else:
    #     parts = last_func_line.split()
    #     if len(parts) < 6:
    #         print("Malformed FUNC line.")
    #     else:
    #         start_mem = float(parts[2])
    #         start_time = float(parts[3])
    #         end_mem = float(parts[4])
    #         end_time = float(parts[5])
    #         duration = end_time - start_time
    #         mem_diff = end_mem - start_mem
    #         print(f"Function duration: {duration:.4f} seconds")
    #         print(f"Memory change: {mem_diff:+.2f} MB (start: {start_mem:.2f}, end: {end_mem:.2f})")


if __name__ == '__main__':
    main()
