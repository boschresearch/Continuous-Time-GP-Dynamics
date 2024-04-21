# Copyright (c) 2024 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Nicholas Tagliapietra, Katharina Ensinger (katharina.ensinger@bosch.com)


import argparse
import logging
import os
import time

import torch
from configs.get_configs import get_configs, save_config
from factories.data_factory import DataFactory
from factories.dynamicgp_factory import DynamicGPFactory
from train.training_functions import train_dynamicGP
from utils.train_utils import log_runtime, set_seed
from tune_lengthscale import tune_lengthscale

def main_loop(cfg_path: str,
              experiment_dir: str,
              seed: int,
              device: torch.device = torch.device("cpu")) -> None:
    """Loads the correct data and models from config objects, then train the
    models, make predictions and save everything.

    Args:
        cfg_path (str): Path to the config file
        experiment_dir (str): Directory of the experiment
        seed (int): Random seed
        device (torch.device): Device on which to train the model
    """

    cfg = get_configs(cfg_path)

    seed_path = os.path.join(experiment_dir, f"{seed}")
    if not os.path.exists(seed_path):
        os.makedirs(seed_path)

    save_config(cfg, experiment_dir)

    logging.info("Started training for seed {}".format(seed))
    start_cpu_time = time.process_time()
    start_time = time.asctime()

    data_cfg = cfg.data_config

    set_seed(0)
    (dataset_train,
     train_list,
     eval_list,
     traj_mean,
     traj_std) = DataFactory.build(data_cfg)
    set_seed(seed)

    if cfg.dynamicgp_config.pretrain_gpy:
        cfg = tune_lengthscale(seed, cfg)
        save_config(cfg, seed_path, config_name="config_gpy.json")

    gp = DynamicGPFactory.build(cfg.dynamicgp_config, dataset_train)

    # TRAINING
    train_cfg = cfg.train_config

    try:  # A singular kernel may rise a RuntimeError
        gp_trained = train_dynamicGP(gp,
                                     train_cfg,
                                     seed_path,
                                     device)
        gp_trained.to("cpu")
        
    except Exception as err:
        log_runtime(start_time,
                    start_cpu_time,
                    False,
                    seed_path,
                    exception=err)
        return

    log_runtime(start_time, start_cpu_time, True, seed_path)

    # NOTE: DATA IS SAVED AS NORMALIZED.
    torch.save(train_list,
               os.path.join(seed_path, "train_trajectories.pt"))
    torch.save(eval_list,
               os.path.join(seed_path, "eval_trajectories.pt"))
    torch.save((traj_mean, traj_std),
               os.path.join(seed_path, "normalization_constants.pt"))
    torch.save(gp_trained.state_dict(),
               os.path.join(seed_path, "final_model.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        metavar="Config path",
                        type=str,
                        nargs=1,
                        help="Absolute path containing the config file")
    parser.add_argument('--experiment_dir',
                        metavar="Experiment directory",
                        type=str,
                        nargs=1,
                        help="Absolute path where to store results"])
    parser.add_argument('--seed',
                        metavar="Seed to be used",
                        type=int,
                        nargs=1,
                        help="Seed to use for every training loop")
    
    args = parser.parse_args()
    args_dict = vars(args)

    config_path = os.path.abspath(args_dict["config_path"][0])
    experiment_dir = os.path.abspath(args_dict["experiment_dir"][0])
    seed = args_dict["seed"][0]

    if not os.path.exists(config_path):
        raise AssertionError(f"Config path {config_path} does not exist")
    if not isinstance(seed, int):
        raise AssertionError("Seed has to be an integer")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    main_loop(config_path, experiment_dir, seed, device=device)
