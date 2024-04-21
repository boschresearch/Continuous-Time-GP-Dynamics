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

import os
import sys
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np

from data.dynamical_models import (add_noise, generate_timeline,
                                   generate_trajectory)
from torch import Tensor
from utils.numerical_schemes import NumericalScheme


def generate_training_data(dataset_name: str,
                           y0: Tensor,
                           t_endpoints: Union[Tuple, List],
                           dt: Tensor,
                           bound: Tensor,
                           noise_std: Tensor) -> Tuple[Tensor, ...]:
    """Generate a trajectory from the dataset and save it, or just load it if
    it has been already generated.

    Args:
        dataset_name (str): Name of the model to simulate
        y0 (Tensor): Initial conditions.
        t_endpoints (List): Initial and final time of the simulation.
        dt (Tensor): Reference dt.
        bound (Tensor): Error variation in percentage for the timesteps.
        noise_std (Tensor): Noise of the training trajectory.

    Returns:
        Tuple[Tensor]: Noiseless and noisy trajectory, plus timeline.
    """
    root = sys.path[0]
    filename = f"dt_{dt}_bound_{bound}_endpoints_{t_endpoints[0]}_" + \
        f"{t_endpoints[1]}_noise_{noise_std}"
    raw_path = os.path.join(root, "data", "raw", f"{dataset_name}")

    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
    file_path = os.path.join(raw_path, filename)

    if os.path.isfile(file_path + ".npz"):
        file_dict = np.load(os.path.join(raw_path, filename + ".npz"))
        noiseless_traj = torch.as_tensor(file_dict["noiseless_trajectory"])
        noisy_traj = torch.as_tensor(file_dict["noisy_trajectory"])
        timeline = torch.as_tensor(file_dict["timeline"])
    else:
        timeline = generate_timeline(t_endpoints, dt, bound)
        noiseless_traj = generate_trajectory(dataset_name, y0, timeline)
        noisy_traj = add_noise(noiseless_traj, noise_std)
        np.savez(file_path,
                 noiseless_trajectory=noiseless_traj,
                 noisy_trajectory=noisy_traj,
                 timeline=timeline)

    return noiseless_traj, noisy_traj, timeline


def create_training_data_multistep(x: Tensor,
                                   timeline: Tensor,
                                   scheme: NumericalScheme
                                   ) -> Tuple[Tuple[Tensor],
                                              Tensor,
                                              Tensor,
                                              Tensor]:
    """Given a tensor x containing datapoints in its rows, this function
    builds a dataset for a specified integration rule.

    Args:
        x (Tensor): Tensor containing the generated training datapoints
        timeline (float): Timeline of the simulated data.
        scheme (NumericalScheme): Function that generates the coefficients

    Returns:
        Tuple[Tuple[Tensor], Tensor, Tensor, Tensor]: Training inputs, targets,
        timesteps, epsilon matrix
    """
    data_dimensions = x.shape[1]
    n_total_patterns = x.shape[0]

    alpha, beta = scheme.build_coeffs(timeline=timeline)
    nsteps = scheme.k
    if nsteps == 0:
        nsteps = 1

    h = timeline[1:] - timeline[:-1]

    x_train = [x[i:(n_total_patterns - nsteps + i), :].double()
               for i in range(nsteps + 1)]
    Y = torch.zeros(n_total_patterns - nsteps, x.shape[1], dtype=torch.float64)
    epsilon = torch.zeros(n_total_patterns - nsteps, dtype=torch.float64)

    alpha = alpha[:(n_total_patterns - nsteps + 1), :]
    beta = beta[:(n_total_patterns - nsteps + 1), :]

    for i in range(nsteps + 1):
        a = alpha[:, i].reshape(-1, 1).repeat(1, data_dimensions)
        Y = Y + x_train[i].mul(a)
        # OLD. This computation of epsilon is wrong. 
        # It does not consider the noise correlation.
        #epsilon = epsilon + torch.square(alpha[:, i])

    h_inv = (1/h[(nsteps - 1):]).reshape(-1, 1).repeat(1, data_dimensions)
    Y = torch.mul(Y, h_inv)
    h = h.reshape(-1,1)[(nsteps - 1):, :]
    hh_inv = 1/torch.mul(h,h.T)
    A = torch.zeros(n_total_patterns - nsteps, n_total_patterns)
    for i in range(nsteps +1):
        matrix_component = torch.diagflat(alpha[:,i])
        A[:, i:n_total_patterns-nsteps+i]+=matrix_component[:,:]
    AtA = torch.matmul(A,A.T)
    epsilon = torch.mul(hh_inv, AtA)
    timeline = timeline[:(n_total_patterns - nsteps)]

    return tuple(x_train), Y, timeline, epsilon


def create_training_data_taylor(x: Tensor,
                                timeline: Tensor
                                ) -> Tuple[Tuple[Tensor],
                                           Tensor,
                                           Tensor,
                                           Tensor]:
    """Given a tensor x containing datapoints in its rows, this function
    builds a dataset for training using Taylor expansion.

    Args:
        x (Tensor): Tensor containing the generated training datapoints
        in its rows
        timeline (float): Timeline of the simulated data.
    Returns:
        Tuple[Tuple[Tensor], Tensor, Tensor, Tensor]: Training inputs, targets,
        timesteps, epsilon matrix
    """
    n_total_patterns = x.shape[0]

    h = timeline[1:] - timeline[:-1]

    x_list = [x[0:(n_total_patterns - 1), :].double()]
    y = (x[1:, :] - x[0:-1, :]).div(h[:, None])
    h = h.reshape(-1,1)
    hh_inv = 1/torch.mul(h,h.T)
    epsilon = 2 * torch.eye(n_total_patterns - 1, dtype=torch.float64)
    extra_diagonal = (-1) * torch.diagflat(torch.ones(n_total_patterns - 2, 1), offset = 1)
    epsilon = epsilon + extra_diagonal + extra_diagonal.T
    epsilon = torch.mul(hh_inv, epsilon)

    return tuple(x_list), y, timeline, epsilon


def list_mean(traj_list: List[Tuple[Tensor, Tensor]]) -> Tensor:
    """Given a list of trajectories modeled as tensors, calculate the mean
    trajectory

    Args:
        traj_list (List[List[Tensor]]): List of trajectories as torch
        tensors

    Returns:
        Tuple[List[Tensor], List[Tensor]]: Mean trajectory
    """
    dynamics = extract_trajectories(traj_list)

    traj_mean = torch.mean(dynamics, dim=0)
    return traj_mean


def list_var(traj_list: List[Tuple[Tensor, Tensor]]) -> Tensor:
    """Given a list of trajectories, calculate the variance around the mean
    trajectory point by point

    Args:
        traj_list (List[Tensor]): List of trajectories described as torch
        tensors

    Returns:
        Tensor: Tensor with same length of the trajectories
    """
    dynamics = extract_trajectories(traj_list)

    # Since we are using the unbiased variance, the variance of a single
    # trajectory would be Nan because of the division by zero (there is a
    # division by n-1).
    if dynamics.shape[0] == 1:
        return torch.zeros(dynamics.size()).squeeze()

    trajectory_var = torch.var(dynamics, dim=0, unbiased=True)
    return trajectory_var


def extract_trajectories(traj_list: List[Tuple[Tensor, Tensor]]) -> Tensor:
    """Given a list of (trajectory,timeline) pairs, extract the trajectories 
    and stack them on a pytorch tensor

    Args:
        traj_list (List[Tuple[Tensor, Tensor]]): List of (trajectory,timeline) 
        pairs

    Raises:
        RuntimeError: Shape mismatch between trajectories

    Returns:
        Tensor: Trajectories stacked on a torch Tensor
    """
    dynamics = []
    # Extract the trajectory tensors without the timelines
    [dynamics.append(traj_list[k][0]) for k in range(len(traj_list))]

    if not all(dynamics[0].shape == tensor.shape for tensor in dynamics):
        raise RuntimeError("Shape mismatch between trajectories")
    return torch.stack(dynamics)


def extract_timelines(traj_list: List[Tuple[Tensor, Tensor]]) -> List[Tensor]:
    """Given a list of (trajectory,timeline) pairs, extract the trajectories 
    and stack them on a pytorch tensor

    Args:
        traj_list (List[Tuple[Tensor, Tensor]]): List of (trajectory,timeline) 
        pairs

    Raises:
        RuntimeError: Shape mismatch between trajectories

    Returns:
        Tensor: Trajectories stacked on a torch Tensor
    """
    timelines = []
    # Extract the trajectory tensors without the timelines
    [timelines.append(traj_list[k][1]) for k in range(len(traj_list))]

    return timelines


def normalize_data(x: Tensor,
                   mean_x: Optional[Tensor] = None,
                   std_x: Optional[Tensor] = None
                   ) -> tuple[Tensor, Tensor, Tensor]:
    """Apply a simple normalization to the input data x, and returns the
    normalized data x_norm, and the two parameters x_mean and std_x

    Args:
        x (Tensor): Input data to be normalized.

    Returns:
        tuple[Tensor, Tensor, Tensor]: Normalized data, mean and standard
        deviation.
    """
    if mean_x is None:
        mean_x = torch.mean(x, dim=0)
    if std_x is None:
        std_x = torch.std(x, dim=0)
    x_norm = (x - mean_x).div(std_x)

    return x_norm, mean_x, std_x


def un_normalize_data(x_norm: Tensor,
                      mean_x: Tensor,
                      std_x: Tensor) -> Tensor:
    """Apply the inverse transformation of the normalize_data function, given
    the normalized data x and the two parameters mean_x and std_x.

    Args:
        x_norm (Tensor): Normalized data
        mean_x (Tensor): Mean of the un-normalized data.
        std_x (Tensor): Standard deviation of the un-normalized data.

    Returns:
        Tensor: Original matrix x.
    """
    x = (x_norm * std_x) + mean_x
    return x


def un_normalize_dict(trajectory_dict: Dict[str, List[Tensor]],
                      mean_x: Tensor,
                      std_x: Tensor) -> Dict:
    """Given a dictionary containing trajectories, return a new dictionary
    containing the un-normalized trajectories

    Args:
        trajectory_dict (Dict): Dictionary containing trajectories
        containing "Trajectory"
        mean_x (Tensor): Mean to use for un-normalization.
        std_x (Tensor): Std to use for un-normalization.

    Returns:
        Dict: Dictionary identical the original, but with un-normalized
        trajectories.
    """
    new_trajectory_dict = deepcopy(trajectory_dict)
    for key in new_trajectory_dict:
        if isinstance(key, str):
            if key.find("Trajectory") != -1:
                new_trajectory_dict[key][0] = un_normalize_data(
                    new_trajectory_dict[key][0], mean_x, std_x)

    return new_trajectory_dict


def un_normalize_list(trajectory_list: List[Tuple[Tensor, Tensor]],
                      mean_x: Tensor,
                      std_x: Tensor) -> List[Tuple[Tensor, Tensor]]:
    """Given a dictionary containing trajectories, return a new dictionary
    containing the un-normalized trajectories

    Args:
        trajectory_list (List[Tuple[Tensor,Tensor]]): Dictionary containing
        trajectories on keys containing "Trajectory"
        mean_x (Tensor): Mean to use for un-normalization.
        std_x (Tensor): Std to use for un-normalization.

    Returns:
        List[Tuple[Tensor,Tensor]]: List identical the original, but with
        un-normalized trajectories.
    """
    new_trajectory_list = []
    for i in range(len(trajectory_list)):
        new_trajectory_list.append([un_normalize_data(
            trajectory_list[i][0], mean_x, std_x), trajectory_list[i][1]])

    return new_trajectory_list
