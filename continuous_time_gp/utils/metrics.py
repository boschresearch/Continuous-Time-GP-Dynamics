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

"""Toolbox for calculating distances between trajectories of dynamical systems
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas
import torch
import numpy as np
from configs.global_config import GlobalConfig
from configs.integrator_config import (MultistepIntegratorConfig,
                                       TaylorIntegratorConfig,
                                       TrajectorySamplerConfig)
from models.dynamicgp import DynamicGP
from torch import Tensor
from torch.nn import GaussianNLLLoss
from train.create_training_data import extract_timelines
from utils.plots import generate_local_l2_plot
from utils.prediction_utils import get_method_list
from torch.distributions.normal import Normal
# REPORT GENERATION


def search_metrics_dynamicgp(root_dir,
                             method,
                             integration_rule) -> Tuple[List[List[Tensor]],
                                                        List[List[Tensor]],
                                                        List[List[Tensor]],
                                                        List[Tensor]]:
    rmse_list = []
    loc_rmse_list = []
    mse_list = []
    timelines = []
    for dirname in os.listdir(root_dir):
        if dirname.isdigit():  # If it is a seed
            subdir = os.path.join(root_dir, dirname)
            filename_metrics = f"{integration_rule}_{method}_metrics.pt"
            filename_eval = f"{integration_rule}_{method}_prediction.pt"

            if filename_metrics in os.listdir(subdir):
                metrics = torch.load(os.path.join(subdir, filename_metrics))
                rmse_list.append(metrics["rmse"])
                mse_list.append(metrics["mse"])
                loc_rmse_list.append(metrics["local_rmse"])
                # Load eval metrics to extract the timelines
            if filename_eval in os.listdir(subdir):
                # TODO: Clean this code
                path = os.path.join(subdir, filename_eval)
                eval_trajectories = torch.load(path)
                timelines = extract_timelines(
                    eval_trajectories["mean_trajectory"])
    return (rmse_list,
            mse_list,
            loc_rmse_list,
            timelines)


def search_metrics(root_dir: str,
                   method: str,
                   integration_rule: str) -> Tuple[List[List[Tensor]],
                                                   List[List[Tensor]],
                                                   List[List[Tensor]],
                                                   List[Tensor]]:
    """Given an experiment directory, search for all the metrics.pt files
    inside of every folder with integer name (assumed to be the random seed
    of an experiment).

    Args:
        root_dir (str): Root directory of an experiment.
        method (str): Prediction method
        integration_rule (str): Integration rule

    Returns:
        Tuple[List[List[Tensor]],List[List[Tensor]]]: List with every metrics
        evaluated for every experiment and every predicted trajectory.
    """

    if "gpode" in root_dir:
        return search_metrics_gpode(root_dir)
    else:
        return search_metrics_dynamicgp(root_dir, method, integration_rule)


def compute_global_metrics(rmse_list: List[List[Tensor]],
                           mse_list: List[List[Tensor]],
                           loc_rmse_list: List[List[Tensor]]
                           ) -> Dict[str, Any]:
    """Given the euclidean (local and global), calculate the average and standard
    error for every experiment.

    Args:
        euclidean_err_list (List[List[Tensor]]): Global euclidean distances.
        loc_euclidean_err_list (List[List[Tensor]]): Local euclidean distances.

    Returns:
        Dict[str,Union[Tensor,List[Tensor]]]: Dictionary containing average and
        standard error for all metrics and for every experiment.
    """
    n_eval_traj = len(rmse_list)
    rmse_list = torch.as_tensor(rmse_list, dtype=torch.float64)
    mse_list = torch.as_tensor(mse_list, dtype=torch.float64)
    n_eval_traj = rmse_list.shape[1]

    loc_mean_rmse = []
    loc_std_rmse = []
    # Different trajectories may have different length, so we cannot transform
    # the local errors in a multiD-tensor, so it has to be done list by list
    for j in range(n_eval_traj):
        local_err = []

        for k in range(len(loc_rmse_list)):
            local_err.append(loc_rmse_list[k][j].clone().detach())

        local_err = torch.stack(local_err, 0)
        loc_mean_rmse.append(torch.mean(local_err, 0))
        loc_std_rmse.append(torch.sqrt(torch.var(local_err, 0)))

    report = {
        "mean_rmse": torch.mean(rmse_list, 0),
        "std_rmse": torch.sqrt(torch.var(rmse_list, 0)),
        "mean_mse": torch.mean(mse_list, 0),
        "std_mse": torch.sqrt(torch.var(mse_list, 0)),
        "local_mean_rmse": loc_mean_rmse,
        "local_std_rmse": loc_std_rmse
    }

    return report


def save_csv_point(root_dir: str, report) -> None:
    pred_integration_rule = report["integrator"]
    pred_method = report["method"]
    train_integrator = report["train_method"]
    exp_name = report["experiment_name"]

    # Convert names to upper case without underscores
    train_integrator = str.upper(train_integrator).replace("_", "")
    pred_integration_rule = str.upper(pred_integration_rule).replace("_", "")

    parent_dir = os.path.dirname(root_dir)

    for i in range(len(report["mean_rmse"])):
        if str.lower(pred_integration_rule) == "rk45":
            filename = f"{pred_method}_rk45_metrics_{i}.csv"
        else:
            filename = f"{pred_method}_metrics_{i}.csv"

        csv_path = os.path.join(parent_dir, filename)

        if not os.path.isfile(csv_path):
            with open(csv_path, "w") as file:
                file.write(
                    "Experiment Name; Train Integrator; Prediction Integrator; RMSE; MSE")

        mean_rmse = report["mean_rmse"][i]
        std_rmse = report["std_rmse"][i]
        mean_mse = report["mean_mse"][i]
        std_mse = report["std_mse"][i]
        rmse = f"{mean_rmse:.6f}({std_rmse:.6f})"
        mse = f"{mean_mse:.6f}({std_mse:.6f})"



        data_format = ["Experiment Name",
                       "Train Integrator",
                       "Prediction Integrator",
                       "RMSE",
                       "MSE"]

        data = {"Experiment Name": [exp_name],
                "Train Integrator": [train_integrator],
                "Prediction Integrator": [pred_integration_rule],
                "RMSE": [rmse],
                "MSE": [mse]}
        new_df = pandas.DataFrame(data, columns=data_format)
        new_df.to_csv(csv_path, mode='a', sep=";", header=False, index=False)
    return


def generate_report(root_dir: str,
                    method: str,
                    integration_rule: str,
                    train_method: str) -> Dict:
    """Given the root directory of an experiment, prediction method, and
    integration rule, generate the report containing average and standard error
    of every metric for a specific prediction method and integration rule.

    Args:
        root_dir (str): Root directory.
        method (str): Prediction method
        integration_rule (str): Integration rule
    """

    # Take into account of the integration_rule
    (rmse_list,
     mse_list,
     loc_rmse_list,
     timelines) = search_metrics(root_dir, method, integration_rule)

    report = compute_global_metrics(rmse_list,
                                    mse_list,
                                    loc_rmse_list)

    # Extract experiment name
    exp_name = root_dir.split(os.sep)[-1]
    report.update({"experiment_name": exp_name})
    report.update({"method": method})
    report.update({"integrator": integration_rule})
    report.update({"train_method": train_method})
    save_csv_point(root_dir, report)

    # Generate plots
    loc_mean_rmse = report["local_mean_rmse"]
    loc_std_rmse = report["local_std_rmse"]
    for i in range(len(loc_mean_rmse)):
        if len(loc_mean_rmse) != len(timelines):  # TODO: Clean this code
            continue
        loc_mean_rmse[i] = loc_mean_rmse[i].reshape(-1, 1)
        loc_std_rmse[i] = loc_std_rmse[i].reshape(-1, 1)

        filename = f"{integration_rule}_{method}_loc_euclidean_{i}"
        generate_local_l2_plot(loc_mean_rmse[i],
                               loc_std_rmse[i],
                               timelines[i],
                               root_dir,
                               filename)

    return report


def save_reports_dynamicgp(root_dir: str,
                           cfg: GlobalConfig,
                           print_metrics: bool = True) -> None:
    """Generate a report for every prediction method and integration rule.

    Args:
        root_dir (str): Root directory of the experiment
        cfg (PredictionConfig): Configuration for prediction
    """

    pred_cfg = cfg.prediction_config
    integrator_cfg = pred_cfg.integrator_config

    # Extract list of integration rules
    integrator_list = []
    for int_cfg in integrator_cfg:
        if isinstance(int_cfg, MultistepIntegratorConfig):
            integrator_list.append(int_cfg.integration_rule)
        elif isinstance(int_cfg, TrajectorySamplerConfig):
            integrator_list.append("rk45")
        elif isinstance(int_cfg, TaylorIntegratorConfig):
            name = f"taylor_{str(int_cfg.order)}"
            integrator_list.append(name)
        else:
            raise ValueError("Unknown Integrator")

    # Extract list of training methods
    train_method = "[]"
    match cfg.data_config.method:
        case "multistep":
            train_method = cfg.data_config.integration_rule
        case "taylor":
            order = str(cfg.dynamicgp_config.gp_config.order)
            train_method = f"taylor_{order}"

    # Extract list of methods
    method_list = get_method_list(pred_cfg)

    # Loop over integration rules and methods, to generate reports
    for integrator in integrator_list:
        for pred_method in method_list:
            filename = f"{integrator}_{pred_method}_report.pt"
            report = generate_report(root_dir,
                                     pred_method,
                                     integrator,
                                     train_method)
            if print_metrics is True:
                print_report(report)
            torch.save(report, os.path.join(root_dir, filename))

    # Generate runtime report
    runtime_report = generate_runtime_report(root_dir)
    torch.save(runtime_report,
               os.path.join(root_dir, "runtime_report.pt"))

    return


def generate_runtime_report(root_dir: str) -> Dict[str, Any]:
    """Read the runtime.pt files for every seed folder, and merge them in a
    general global runtime report

    Args:
        root_dir (str): Root directory

    Returns:
        Dict[str, Any]: Runtime report for every seed
    """
    runtime = {}
    for dirname in os.listdir(root_dir):
        if dirname.isdigit():
            subdir = os.path.join(root_dir, dirname)
            if "runtime.pt" in os.listdir(subdir):
                runtime[dirname] = torch.load(
                    os.path.join(subdir, "runtime.pt"))
    return runtime


def print_report(report: Dict) -> None:
    method, integrator = report["method"], report["integrator"]
    print(f"Metrics for {integrator} with {method}")

    metrics = ["mean_rmse", "std_rmse"]
    for metric in metrics:
        if metric in report:
            num_metrics = report[metric].shape[0]
            for i in range(num_metrics):
                print(f"{metric}_{i}: {report[metric][i]:.6f}")
    return


# METHODS FOR METRICS EVALUATION

def adjust_length(eval_list: List[Tuple[Tensor, Tensor]],
                  pred_list: List[Tuple[Tensor, Tensor]]
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

    return eval_list, pred_list


def evaluate_trajectories(eval_list: List[Tuple[Tensor, Tensor]],
                          pred_list: List[Tuple[Tensor, Tensor]],
                          obs_noise: Optional[DynamicGP] = None,
                          traj_std: Optional[Tensor] = None,
                          pred_var_list: Optional[Tensor] = None
                          ) -> Dict[str, List[Tensor]]:
    """Given lists containing predicted and original trajectories, calculate
    the distance between those trajectories by using euclidean distances.

    Args:
        prediction_list (List[Tuple[Tensor, Tensor]]): List of predictions
        original_list (List[Tuple[Tensor, Tensor]]): List of predictions
        obs_noise (Tensor) : Tensor with each dimension's observation noise.
        traj_std (Tensor): Normalization factor for the trajectories

    Returns:
        Tuple[Tensor, Tensor]: Tuple with list of metrics evaluated
        for every pair of trajectories.

    Raises:
        ValueError: A different number of predicted and original trajectories
        has been provided.
        ValueError: Shape mismatch between two trajectories.
    """
    if len(pred_list) != len(eval_list):
        raise ValueError("List of predictions and Original trajectories have \
            the same length")

    # Adjust trajectory length
    (eval_list, pred_list) = adjust_length(eval_list, pred_list)

    rmse_list = []
    mse_list = []
    local_rmse_list = []
    for i in range(len(pred_list)):
        original_trajectory = eval_list[i][0]
        pred_trajectory = pred_list[i][0]

        if pred_trajectory.shape != original_trajectory.shape:
            raise ValueError(
                f"Trajectories at position {i} have not the same shape!")

        mse = eval_mse(original_trajectory,
                       pred_trajectory)
        mse_list.append(mse)
        rmse_list.append(torch.sqrt(mse))

        local_rmse_list.append(eval_mse(original_trajectory,
                                        pred_trajectory,
                                        local=True))

    metrics_result = {
        "rmse": rmse_list,
        "local_rmse": local_rmse_list,
        "mse": mse_list
    }

    return metrics_result


# METRICS


def eval_mse(trajectory1: Tensor,
             trajectory2: Tensor,
             local: bool = False) -> Tensor:
    """Return the L2 distance between trajectory1 and trajectory 2,
     sampled on the same timeline. If local == True, evaluate the local error
     at time t_i

    Args:
        trajectory1 (Tensor): Trajectory 1
        trajectory2 (Tensor): Trajectory 2
        local (local): Evaluate local squared error (SE). Defaults to False.

    Raises:
        RuntimeError: The dimensions of the trajectories do not match.

    Returns:
        Tensor: L2 distance between trajectory 1 and trajectory 2.
    """

    if trajectory1.size() != trajectory1.size():
        raise RuntimeError("Tensor dimensions must agree!")
    if local:
        trajectory1, trajectory2 = trajectory1.T, trajectory2.T

    trajectory1, trajectory2 = trajectory1.squeeze(), trajectory2.squeeze(),
    n = trajectory1.shape[0]
    d = trajectory1.shape[1]

    sqr_traj = torch.square(trajectory1 - trajectory2)

    if local is True:
        return torch.sum(sqr_traj, axis=0)
    else:
        return torch.sum(sqr_traj)/(n*d)



