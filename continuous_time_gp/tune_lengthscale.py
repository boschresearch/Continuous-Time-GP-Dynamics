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
from typing import List, Tuple

import GPy
import torch
import numpy as np
from configs.get_configs import get_configs, save_config
from factories.data_factory import DataFactory
from GPy.core.gp import GP
from torch import Tensor
from configs.global_config import GlobalConfig
from utils.train_utils import set_seed

"""
Pretraining heavily relies on GPy. For the license details, we refer to the 
3rd-party-licenses.txt
"""

def predict(eval_tuple: Tuple[Tensor, Tensor], models: List[GP]) -> Tensor:
    Y = eval_tuple[0].numpy()
    timeline = eval_tuple[1].numpy()

    h = timeline[1:] - timeline[0:-1]
    pred = torch.zeros(500, Y.shape[1]).numpy()
    pred[0, :] = Y[0, :]
    data_dimensions = Y.shape[1]
    for i in range(1, 500):
        for k in range(data_dimensions):
            pred[i, k] = pred[i - 1, k] + \
                models[k].predict(
                    pred[i-1, :].reshape(1, -1))[0] * h[i - 1]

    return torch.as_tensor(pred)


def train_gpy(X: Tensor, Y: Tensor) -> Tuple[GP, Tensor]:
    """Trains a simple GPy GP model with the Explicit Euler method, and returns
    the lengthscales. 

    Args:
        X (Tensor): Train inputs
        Y (Tensor): Train targets

    Returns:
        Tensor: Lengthscale of the trained model
    """
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    kernel = GPy.kern.RBF(input_dim=X.shape[1],
                          variance=1.,
                          lengthscale=1.,
                          ARD=True)
    model = GPy.models.GPRegression(X.numpy(), Y.numpy(), kernel)
    model.optimize(messages=True)

    return torch.as_tensor(model.rbf.lengthscale), model.rbf.variance


def tune_lengthscale(seed, config) -> GlobalConfig:
    """Given a seed and a config, pretrain a gpy model and return an update
    config with the optimized lengthscales.

    Args:
        seed (_type_): _description_
        config (_type_): _description_

    Returns:
        GlobalConfig: _description_
    """
    data_config = config.data_config
    gp_config = config.dynamicgp_config.gp_config

    set_seed(0)

    (_, train_list, *_) = DataFactory.build(data_config)

    # Even if there can be multiple trajectories, use the first one
    train_tuple = train_list[0]
    trajectory = train_tuple[0]
    timeline = train_tuple[1]
    h = (timeline[1:] - timeline[0:-1]).reshape(-1, 1)

    Y = (trajectory[1:, :] - trajectory[0:-1]).div(h)
    X = trajectory[0:-1]

    dims = Y.shape[1]

    rbf_config = gp_config.kernel_config.basekernel_config
    for component in range(dims):
        set_seed(seed)
        lengthscale, kernel_std = train_gpy(X, Y[:, component])
        rbf_config.lengthscale[component][0] = lengthscale.tolist()

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        metavar="Config path",
                        type=str,
                        nargs=1,
                        help="Absolute path containing the config file")

    parser.add_argument('--seed',
                        metavar="Seed",
                        type=int,
                        nargs=1,
                        help="Random Seed")
    args = parser.parse_args()
    args_dict = vars(args)

    config_path = os.path.abspath(args_dict["config_path"][0])
    seed = args_dict["config_path"][0]

    if not os.path.exists(config_path):
        raise AssertionError(f"Config path {config_path} does not exist")

    config = get_configs(config_path)
    config = tune_lengthscale(train_gpy, seed, config)

    dir = os.path.dirname(config_path)
    filename = "".join(os.path.basename(
        config_path).split(".")[:-1]) + "_gpy.json"
    save_config(config, dir, config_name=filename)
