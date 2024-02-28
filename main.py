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
    kp.fit(data.pykoop_dict['X_train'],
           n_inputs=data.pykoop_dict['n_inputs'],
           episode_feature=True)

    with open(path, "wb") as f:
        data_dump = pickle.dump(kp, f)

    # Predict
    kp.x_pred = kp.predict_trajectory(
        data.pykoop_dict['x0_valid'],
        data.pykoop_dict['u_valid'],
        relift_state=True,
        return_lifted=False,
    )

    with open(path, "wb") as f:
        data_dump = pickle.dump(kp, f)


if __name__ == '__main__':
    main()
