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
import os
from typing import Optional

import torch
import numpy as np
from configs.data_config import MocapConfig
from configs.get_configs import get_configs
from data.datasets.dynamics_dataset import DynamicsDataset
from utils.metrics import evaluate_trajectories
from utils.mocap_utils import mocap_project_list
from utils.numerical_schemes import get_scheme
from utils.prediction_utils import get_method_list
from utils.train_utils import get_filename_prefix, load_model
from utils.validation_utils import build_mocap_test


def eval_metrics_dynamicgp(exp_dir: str) -> None:
    """Evaluate metrics for prediction coming from a dynamicgp model. This
    function loads the config file and evaluate the predictions for every
    integrator and every prediction method.

    Args:
        exp_dir (str): _description_

    Raises:
        ValueError: _description_
    """
    cfg_path = os.path.join(exp_dir, "config.json")
    cfg = get_configs(os.path.join(cfg_path))

    prediction_cfg = cfg.prediction_config
    prediction_methods = get_method_list(prediction_cfg)

    pred_cfg = cfg.prediction_config
    integrator_cfg_list = pred_cfg.integrator_config
    method = cfg.data_config.method
    dataset_cfg = cfg.data_config.dataset_config

    # Load Latent-to-Original space projection in case of MoCap
    mocap = False
    if isinstance(dataset_cfg, MocapConfig):
        projector, mocap_eval, _ = build_mocap_test(dataset_cfg)
        mocap = True

    seed_list = os.listdir(exp_dir)

    for seed in seed_list:
        if not seed.isdigit():
            continue

        seed_path = os.path.join(exp_dir, seed)

        # Load DynamicGP from saved data if necessary.
        if mocap is False:
            # If this is not Mocap, initialize the gp and extract the
            # observation noise in order to evaluate the NLL and log-density
            (traj_mean,
             traj_std) = torch.load(os.path.join(seed_path,
                                                 "normalization_constants.pt"))
            # Load data
            train_list = torch.load(os.path.join(seed_path,
                                                 "train_trajectories.pt"))
            # Initialize DynamicsDataset
            if method == "multistep":
                scheme = get_scheme(cfg.data_config.integration_rule)
            elif method == "taylor":
                scheme = None
            else:
                raise ValueError("Unknown method")

            dataset_train = DynamicsDataset(train_list,
                                            method,
                                            scheme=scheme)
            gp = load_model(cfg.dynamicgp_config,
                            dataset_train,
                            seed_path,
                            "final_model.pt")
            obs_noise = gp.get_obs_noise().to("cpu")
        else:
            obs_noise = None
            traj_std = None

        for pred_method in prediction_methods:
            for integrator_cfg in integrator_cfg_list:
                prefix = get_filename_prefix(pred_method, integrator_cfg)

                prediction_filename = os.path.join(seed_path,
                                                   f"{prefix}_prediction.pt")
                prediction = torch.load(prediction_filename)
                mean_traj = prediction["mean_trajectory"]

                if pred_method == "ds" and not mocap:
                    pred_var = prediction["var_trajectory"]
                else:
                    pred_var = None

                if not mocap:
                    # In case of synthetic data, just load the normal
                    # evaluation trajectories.
                    eval_filename = os.path.join(seed_path,
                                                 f"{prefix}_unnorm_eval_trajectories.pt")
                    eval_traj = torch.load(eval_filename)
                else:
                    # Since Mocap uses PCA, the ground truth is the raw 50
                    # dimensional data, not the PCA data projected back in
                    # the original space.
                    eval_traj = mocap_eval.copy()
                    mean_traj = mocap_project_list(mean_traj, projector)

                metrics = evaluate_trajectories(eval_traj,
                                                mean_traj,
                                                obs_noise=obs_noise,
                                                traj_std=traj_std,
                                                pred_var_list=pred_var)

                torch.save(metrics, os.path.join(seed_path,
                                                 prefix + "_metrics.pt"))

def main(data_dir: str, experiment_names: Optional[str] = None) -> None:
    """This function scans all the experiment directories in data_dir, loads
    the trajectories and config files, and generate the plots.
    Sometimes the main code is executed on a cluster, so the environment where
    that code is executed may not have all the LaTex and other fancy packages
    needed for generating the plots. In any case, this function could be useful
    because it permits to mantain the main environment less heavy and as
    vanilla as possible.

    Args:
        data_dir (str): Directory to the experiment data.
        experiment_names (str): Name of the specific experiments.
    """

    dir_list = []
    if experiment_names is None:
        dir_list = os.listdir(data_dir)
        for i in range(len(dir_list)):
            dir_list[i] = os.path.join(data_dir, dir_list[i])
    else:
        for experiment in experiment_names:
            dir_list.append(os.path.join(data_dir, experiment))

    for exp_dir in dir_list:
        # Check if exp_dir is a directory and not a file
        if os.path.isfile(exp_dir):
            continue
        eval_metrics_dynamicgp(exp_dir)
    return


if __name__ == "__main__":
    # This function should parse the path of the experiments, and the name of
    # every experiment folder for which the plots have to be generated
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        metavar="Data Path",
                        type=str,
                        nargs=1,
                        help="provide path to experiments")
    parser.add_argument('--experiment',
                        metavar="experiment",
                        type=str,
                        nargs="+",
                        default=None,
                        help="Name of the specific experiment folders")

    args = parser.parse_args()
    args_dict = vars(args)

    data_path = os.path.abspath(args_dict["data_path"][0])
    if args_dict["experiment"] is not None:
        experiment_names = os.path.abspath(args_dict["experiment"][0])
    else:
        experiment_names = None

    if not os.path.exists(data_path):
        raise AssertionError(f"Config path {data_path} does not exist")

    main(data_path, experiment_names)

