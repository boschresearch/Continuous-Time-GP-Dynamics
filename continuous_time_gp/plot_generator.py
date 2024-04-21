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
from configs.get_configs import get_configs
from configs.data_config import MocapConfig
from train.create_training_data import un_normalize_list
from utils.mocap_utils import mocap_project_list
from utils.plots import generate_plots, mocap_observation_plots, loss_plot, plot_latents_3d
from utils.train_utils import get_filename_prefix
from utils.validation_utils import build_mocap_test
from typing import List, Tuple
from torch import Tensor


def adjust_length(eval_list: List[Tuple[Tensor, Tensor]],
                  pred_list: List[Tuple[Tensor, Tensor]],
                  pred_var_list: List[Tuple[Tensor, Tensor]]
                  ) -> tuple[List[Tuple[Tensor, Tensor]],
                             List[Tuple[Tensor, Tensor]]]:
    """Sometimes during integration it is possible to get predicted
    trajectories that are shorter than the ground truth by 1 point.
    Thins function checks when this happens, and eventually discard the one
    datapoint in one of the trajectories (At the end).

    Args:
        eval_list (List[Tuple[Tensor, Tensor]]): List of test trajectories
        pred_list (List[Tuple[Tensor, Tensor]]): List of predicted trajectories

    Returns:
        tuple[List[Tuple[Tensor, Tensor]], List[Tuple[Tensor, Tensor]]]:
        Adjusted trajectories.
    """
    for i in range(len(eval_list)):
        l_eval = eval_list[i][0].shape[0]
        l_pred = pred_list[i][0].shape[0]

        if l_eval > l_pred:
            eval_list[i] = (eval_list[i][0][:l_pred, :],
                            eval_list[i][1][:l_pred])
        elif l_pred > l_eval:
            pred_list[i] = (pred_list[i][0][:l_eval, :],
                            pred_list[i][1][:l_eval])
            pred_var_list[i] = pred_var_list[i][:l_eval, :]

    return eval_list, pred_list, pred_var_list


def plot_experiment(cfg, exp_dir) -> None:
    prediction_config = cfg.prediction_config
    prediction_methods = []
    if prediction_config.ds_pred:
        prediction_methods.append("ds")
    if prediction_config.mean_pred:
        prediction_methods.append("mean")

    integrator_cfg_list = prediction_config.integrator_config

    seed_list = os.listdir(exp_dir)
    plots_settings = {}

    for seed in seed_list:
        if not seed.isdigit():
            continue

        seed_path = os.path.join(exp_dir, seed)

        loss_path = os.path.join(seed_path, "loss.pt")
        if os.path.exists(loss_path):
            loss_history = torch.load(loss_path)
            loss_plot(loss_history, seed_path)

        train_filename = os.path.join(seed_path,
                                      "train_trajectories.pt")
        (traj_mean,
            traj_std) = torch.load(os.path.join(seed_path,
                                                "normalization_constants.pt"))

        norm_train_traj = torch.load(train_filename)
        train_traj = un_normalize_list(norm_train_traj,
                                       traj_mean,
                                       traj_std)

        for method in prediction_methods:
            for integrator_cfg in integrator_cfg_list:
                prefix = get_filename_prefix(method, integrator_cfg)
                eval_filename = os.path.join(seed_path,
                                             f"{prefix}_unnorm_eval_trajectories.pt")
                eval_traj = torch.load(eval_filename)

                prediction_filename = os.path.join(seed_path,
                                                   f"{prefix}_prediction.pt")

                prediction = torch.load(prediction_filename)
                mean_pred = prediction["mean_trajectory"]
                generate_plots(mean_pred,
                               prediction["var_trajectory"],
                               seed_path,
                               prefix,
                               # The first eval_trajectory is the longest one covering the entire domain
                               eval_traj=eval_traj,
                               train_set=train_traj[0],
                               **plots_settings
                               )

    return


def plot_mocap(cfg, exp_dir) -> None:
    prediction_cfg = cfg.prediction_config
    data_cfg = cfg.data_config
    prediction_methods = []
    if prediction_cfg.ds_pred:
        prediction_methods.append("ds")
    if prediction_cfg.mean_pred:
        prediction_methods.append("mean")

    integrator_cfg_list = prediction_cfg.integrator_config

    seed_list = os.listdir(exp_dir)
    plots_settings = {"mocap": True}
    projector, eval_list, eval_list_pca = build_mocap_test(
        data_cfg.dataset_config)

    for seed in seed_list:
        if not seed.isdigit():
            continue

        seed_path = os.path.join(exp_dir, seed)

        loss_path = os.path.join(seed_path, "loss.pt")
        if os.path.exists(loss_path):
            loss_history = torch.load(loss_path)
            loss_plot(loss_history, seed_path)

        for method in prediction_methods:
            for integrator_cfg in integrator_cfg_list:
                prefix = get_filename_prefix(method, integrator_cfg)

                prediction_filename = os.path.join(seed_path,
                                                   f"{prefix}_prediction.pt")

                prediction = torch.load(prediction_filename)
                mean_pred = prediction["mean_trajectory"]
                var_pred = prediction["var_trajectory"]
                (eval_list_pca,
                 mean_pred,
                 var_pred) = adjust_length(eval_list_pca,
                                           mean_pred,
                                           var_pred)
                generate_plots(mean_pred,
                               var_pred,
                               seed_path,
                               prefix,
                               # The first eval_trajectory is the longest one covering the entire domain
                               eval_traj=eval_list_pca,
                               # train_set=train_traj[0],
                               **plots_settings
                               )

                mean_pred = mocap_project_list(mean_pred,
                                               projector)
                (eval_list, mean_pred, var_pred) = adjust_length(eval_list,
                                                                 mean_pred,
                                                                 var_pred)
                mocap_observation_plots(mean_pred,
                                        eval_list,
                                        seed_path,
                                        f"{prefix}_observations")


def main(data_dir: str, experiment_names: Optional[str] = None) -> None:
    """This function scans all the experiment directories in data_dir, loads
    the trajectories and config files, and generate the plots.

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
        if "gpode" in exp_dir:
            continue

        config_dir = os.path.join(exp_dir, "config.json")
        cfg = get_configs(os.path.join(config_dir))
        if isinstance(cfg.data_config.dataset_config, MocapConfig):
            plot_mocap(cfg, exp_dir)
        else:
            plot_experiment(cfg, exp_dir)

    return


if __name__ == "__main__":
    # This function should parse the path of the experiments, and the name of
    # every experiment folder for which the plots have to be generated
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        metavar="Data Path",
                        type=str,
                        nargs=1,
                        help="Absolute path containing the experiment data")
    parser.add_argument('--experiment',
                        metavar="experiment",
                        type=str,
                        nargs=1,
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
