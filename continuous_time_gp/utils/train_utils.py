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

import torch
import numpy as np

import logging
import os
import time
from typing import List, Optional, Mapping, Any
from models.abstractgp import AbstractGP
from configs.dynamicgp_config import DynamicGPConfig
from configs.integrator_config import (IntegratorConfig,
                                       MultistepIntegratorConfig,
                                       TaylorIntegratorConfig,
                                       TrajectorySamplerConfig)
from data.datasets.dynamics_dataset import DynamicsDataset
from factories.dynamicgp_factory import DynamicGPFactory
from models.dynamicgp import DynamicGP
from configs.gp_config import MultistepGPConfig, TaylorGPConfig, ExactTaylorGPConfig
from copy import deepcopy
from factories.gp_factory import TaylorGPFactory, MultistepGPFactory, ExactTaylorGPFactory
from factories.basekernel_factory import RBFKernelFactory


def set_seed(seed: int) -> None:
    """Set the seed for torch and numpy.

    Args:
        seed (int): Random seed to be set
    """
    if not isinstance(seed, int):
        raise TypeError("Seed is not an integer")

    torch.manual_seed(seed)
    np.random.default_rng(seed)
    return


def load_abstract_gp(state_dict: Optional[Mapping[str, Any]]) -> AbstractGP:
    """Initialize AbstractGP from state_dict

    Args:
        state_dict (Optional[Mapping[str, Any]], optional): _description_. Defaults to None.
        f (Optional[DynamicGP], optional): _description_. Defaults to None.

    Raises:
        RuntimeError: _description_

    Returns:
        DynamicGP: _description_
    """
    cfg = deepcopy(state_dict["cfg"])
    dataset = deepcopy(state_dict["dataset"])
    component = deepcopy(state_dict["component"])
    dimensions = dataset.data_dimension

    # Initialize GP and load state dict
    if isinstance(cfg, MultistepGPConfig):
        gp = MultistepGPFactory.build(cfg, dataset, component)
    elif isinstance(cfg, TaylorGPConfig):
        gp = TaylorGPFactory.build(cfg, dataset, component)
    elif isinstance(cfg, ExactTaylorGPConfig):
        basekernel_config = cfg.kernel_config.basekernel_config
        rbf_kernel_list = []
        for i in range(dimensions):
            rbf_i = RBFKernelFactory.build(basekernel_config, i, None)
            rbf_kernel_list.append(rbf_i)
        rbf_kernel_list = torch.nn.ModuleList(rbf_kernel_list)
        gp = ExactTaylorGPFactory.build(cfg, dataset, component,
                                                 rbf_kernel_list)
    else:
        RuntimeError("Invalid GP Config")

    gp.load_state_dict(state_dict)
    return gp


def load_final_model(dynamicgp_cfg: DynamicGPConfig,
                     dataset_train: DynamicsDataset,
                     seed_path: str) -> DynamicGP:
    """Load the model from a certain seed with the highest number of epochs.

    Args:
        dynamicgp_config (DynamicGPConfig): Config for the DynamicGP
        dataset_train (DynamicsDataset): Dataset with training data
        seed_path (str): Path to the seed folder

    Returns:
        DynamicGP: DynamicGP object for the loaded model
    """
    gp = DynamicGPFactory.build(dynamicgp_cfg, dataset_train)
    models_dir = os.path.join(seed_path, "training")
    models_list = os.listdir(models_dir)
    # This should be refactored in an utility function
    for i, dir in enumerate(models_list):
        path = os.path.join(models_dir, dir)
        filenames = os.listdir(path)
        pt_files = [file for file in filenames if file.endswith(".pt")]
        int_names = [int(file.split(".")[0]) for file in pt_files]
        max_int = max(int_names)

        model_path = os.path.join(path, str(max_int) + ".pt")
        gp.models[i].load_state_dict(torch.load(model_path),
                                     strict=False)
    return gp


def load_model(dynamicgp_cfg: DynamicGPConfig,
               dataset_train: DynamicsDataset,
               seed_path: str,
               filename: str) -> DynamicGP:
    """Load a model from a certain seed path where each AbstractGP is saved 
    with a specific filename. 

    Args:
        dynamicgp_config (DynamicGPConfig): Config for the DynamicGP
        dataset_train (DynamicsDataset): Dataset with training data
        seed_path (str): Path to the seed folder
        filename (str): Filename of the model

    Returns:
        DynamicGP: DynamicGP object for the loaded model
    """
    gp = DynamicGPFactory.build(dynamicgp_cfg, dataset_train)
    model_path = os.path.join(seed_path, filename)
    gp.load_state_dict(torch.load(model_path),
                       strict=False)
    return gp


def parse_config_filenames(cfg_filename_list: List[str]) -> List[str]:
    """Given a list of strings with the name of .json files, eliminate ".json"
    from each string, returning only the filename

    Args:
        config_filename_list (List[str]): List of filenames with its ".json"
        extension

    Returns:
        List[str]: List with string filenames without their extension.
    """
    if len(cfg_filename_list) > 0:
        filenames = [s.replace(".json", "") for s in cfg_filename_list]
    else:
        filenames = ["default"]
    return filenames


def get_filename_prefix(pred_method: str, cfg: IntegratorConfig) -> str:
    """Given prediction method and integrator config, create the filename

    Args:
        pred_method (str): Prediction method
        cfg (IntegratorConfig): Integrator config

    Returns:
        str: Filename
    """
    filename = ""

    if cfg.__class__ == MultistepIntegratorConfig:
        filename = cfg.integration_rule + f"_{pred_method}"
    elif cfg.__class__ == TrajectorySamplerConfig:
        filename = f"rk45_{pred_method}"
    elif cfg.__class__ == TaylorIntegratorConfig:
        filename = f"taylor_{str(cfg.order)}_{pred_method}"
    else:
        raise ValueError("Unknown Integrator")

    return filename


def log_runtime(start_time: float,
                start_cpu_time: float,
                success: bool,
                seed_path: str,
                exception: Optional[Exception] = None) -> None:
    """Log training data into the runtime dictionary, log runtime informations,
    and save everything.

    Args:
        start_time (float): Start time
        start_cpu_time (float): Start cpu time
        success (bool): Success of training
        seed_path (str): Seed path
        exception (Optional[Exception], optional): Raise exception, if any.
        Defaults to None.
    """
    seed_path_split = os.path.split(seed_path)
    seed = seed_path_split[-1]

    if success is False:
        err_string = type(exception).__name__
        logging.warning(f"{err_string} was raised for seed {seed}. \
                Training has been terminated.")
    else:
        err_string = None
        logging.info("Ended training for seed {}.".format(seed))

    runtime = {"start_time": start_time,
                "success": False,
                "error": err_string,
                "cpu_time": time.process_time() - start_cpu_time,
                "end_time": time.asctime()
                }
    torch.save(runtime, os.path.join(seed_path, "runtime.pt"))
    return
