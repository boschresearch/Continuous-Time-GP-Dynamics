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
from typing import List, Tuple, Union
from copy import deepcopy
import torch
import numpy as np
from configs.data_config import (DataConfig, MocapConfig, SimulationConfig,
                                 SyntheticDataConfig, TestConfig,
                                 ExternalDataConfig)
from data.datasets.dynamics_dataset import DynamicsDataset
from data.datasets.mocap import MocapDataset
from data.datasplitter import DataSplitter
from factories.factory import Factory
from torch import Tensor
from train.create_training_data import generate_training_data, normalize_data
from utils.numerical_schemes import get_scheme
from configs.data_config import MethodEnum


external_data_path = os.path.join(sys.path[0], "data", "raw", "external_data")


class DataFactory(Factory):
    """Class Responsible for initializing training and evaluation set, plus the
    creation of the DynamicsDataset used for training.
    """
    @staticmethod
    def build(cfg: DataConfig) -> Tuple[DynamicsDataset,
                                        List[Tuple[Tensor, Tensor]],
                                        List[Tuple[Tensor, Tensor]],
                                        Tensor,
                                        Tensor]:

        dataset_cfg = cfg.dataset_config

        if isinstance(dataset_cfg, MocapConfig):
            (dataset_train,
             train_list,
             eval_list,
             mean,
             std) = DataFactory.build_mocap(cfg)

        elif isinstance(dataset_cfg, SyntheticDataConfig):
            (dataset_train,
             train_list,
             eval_list,
             mean,
             std) = DataFactory.build_synthetic(cfg)

        elif isinstance(dataset_cfg, ExternalDataConfig):
            (dataset_train,
             train_list,
             eval_list,
             mean,
             std) = DataFactory.build_external(cfg)

        else:
            raise NotImplementedError("Unknown data configuration")

        return dataset_train, train_list, eval_list, mean, std

    @staticmethod
    def build_trajectory(config: Union[SimulationConfig, TestConfig],
                         dataset_name: str
                         ) -> Tuple[Tensor,
                                    Tensor,
                                    Tensor]:
        """Generate or load the data with the specifications indicated in the
        config object, using the model indicated in dataset_name.

        Args:
            config (Union[SimulationConfig, TestConfig]): config object.
            dataset_name (str): Name of model

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Noiseless 
            trajectory, noisy trajectory, timeline.
        """
        y0 = torch.tensor(config.y0, dtype=torch.float64).reshape(1, -1)
        t_endpoints = config.t_endpoints
        dt = torch.tensor(config.dt, dtype=torch.float64)
        bound = torch.tensor(config.bound, dtype=torch.float64)
        noise_std = torch.tensor(config.noise_std, dtype=torch.float64)

        (noiseless_traj,
         noisy_traj,
         timeline) = generate_training_data(dataset_name,
                                            y0,
                                            t_endpoints,
                                            dt,
                                            bound,
                                            noise_std)

        return noiseless_traj, noisy_traj, timeline

    @staticmethod
    def build_mocap(cfg: DataConfig) -> Tuple[DynamicsDataset,
                                              List[Tuple[Tensor, Tensor]],
                                              List[Tuple[Tensor, Tensor]],
                                              Tensor,
                                              Tensor]:
        """Builds the MoCap data and adapts its format to this framework.
        """
        dataset_cfg = cfg.dataset_config

        data_path = os.path.join(sys.path[0], "data", "raw", "mocap")
        data_pca = MocapDataset(data_path=data_path,
                                subject=dataset_cfg.subject_num,
                                pca_components=dataset_cfg.pca_components,
                                data_normalize=dataset_cfg.data_normalize,
                                pca_normalize=dataset_cfg.pca_normalize,
                                dt=dataset_cfg.dt,
                                seqlen=dataset_cfg.seqlen)

        train_list = []
        for i in range(data_pca.trn.ys.shape[0]):
            traj = torch.as_tensor(data_pca.trn.ys[i, :, :].squeeze()).double()
            timeline = torch.as_tensor(data_pca.trn.ts).double()
            train_list.append((traj,
                               timeline))

        eval_list = []
        for i in range(data_pca.tst.ys.shape[0]):
            traj = torch.as_tensor(data_pca.tst.ys[i, :, :].squeeze()).double()
            timeline = torch.as_tensor(data_pca.tst.ts).double()
            eval_list.append((traj,
                              timeline))

        mean = torch.tensor(data_pca.pca_normalize.mean,
                            dtype=torch.float64).squeeze()
        std = torch.tensor(data_pca.pca_normalize.std,
                           dtype=torch.float64).squeeze()

        dataset_train = DataFactory.build_dataset(train_list,
                                                  cfg.integration_rule)

        return dataset_train, train_list, eval_list, mean, std

    @staticmethod
    def build_synthetic(cfg: DataConfig) -> Tuple[DynamicsDataset,
                                                  List[Tuple[Tensor, Tensor]],
                                                  List[Tuple[Tensor, Tensor]],
                                                  Tensor,
                                                  Tensor]:
        """Build a dataset out from the synthetic data described in the config
        object.

        Args:
            cfg (DataConfig): Configuration for dataset.

        Returns:
            Tuple[DynamicsDataset, List[Tuple[Tensor, Tensor]], 
            List[Tuple[Tensor, Tensor]], Tensor, Tensor]: Dataset, train and
            evaluation list, mean, std.
        """
        dataset_cfg = cfg.dataset_config

        # Simulate train Data
        simulation_config = dataset_cfg.simulation_config
        dataset_name = dataset_cfg.model

        (noiseless_train,
         noisy_train,
         timeline_train) = DataFactory.build_trajectory(simulation_config,
                                                        dataset_name)
        test_config = dataset_cfg.test_config
        (noiseless_test,
         noisy_test,
         timeline_test
         ) = DataFactory.build_trajectory(test_config, dataset_name)

        noiseless_train, mean, std = normalize_data(noiseless_train)
        noisy_train, _, _ = normalize_data(noisy_train, mean, std)
        noiseless_test, _, _ = normalize_data(noiseless_test, mean, std)

        # Split Data
        split_config = dataset_cfg.split_config
        splitter = DataSplitter(split_config.a_train,
                                split_config.b_train,
                                split_config.a_eval,
                                split_config.b_eval)
        train_list, eval_list = splitter.extract_dataset(
            noisy_train, noiseless_test, timeline_train, timeline_test)

        dataset_train = DataFactory.build_dataset(train_list,
                                                  cfg.integration_rule)

        return dataset_train, train_list, eval_list, mean, std

    @staticmethod
    def build_external(cfg: DataConfig) -> Tuple[DynamicsDataset,
                                                 List[Tuple[Tensor, Tensor]],
                                                 List[Tuple[Tensor, Tensor]],
                                                 Tensor,
                                                 Tensor]:
        """Build a dataset out from the external data described in the config
        object.

        Args:
            cfg (DataConfig): Configuration for dataset.

        Returns:
            Tuple[DynamicsDataset, List[Tuple[Tensor, Tensor]], 
            List[Tuple[Tensor, Tensor]], Tensor, Tensor]: Dataset, train and
            evaluation list, mean, std.
        """
        dataset_cfg = cfg.dataset_config

        # Load trajectory
        dataset_name = dataset_cfg.model
        data = np.load(os.path.join(external_data_path,
                                    dataset_name + ".npz"),
                       allow_pickle=True)
        noisy_traj = torch.as_tensor(data["noisy_trajectory"])
        timeline = torch.as_tensor(data["timeline"].T)

        # Normalize trajectory
        norm_traj, mean, std = normalize_data(noisy_traj)

        # Split Data
        split_config = dataset_cfg.split_config
        splitter = DataSplitter(split_config.a_train,
                                split_config.b_train,
                                split_config.a_eval,
                                split_config.b_eval)
        train_list, eval_list = splitter.extract_dataset(
            deepcopy(norm_traj), deepcopy(norm_traj), deepcopy(timeline), deepcopy(timeline))
        
        dataset_train = DataFactory.build_dataset(train_list,
                                                  cfg.integration_rule)

        return dataset_train, train_list, eval_list, mean, std

    @staticmethod
    def build_dataset(train_list: List[Tuple[Tensor, Tensor]],
                      integration_rule: str
                      ) -> DynamicsDataset:

        if integration_rule == "":
            dataset = DynamicsDataset(train_list, MethodEnum.Taylor)
        else:
            scheme = get_scheme(integration_rule)
            dataset = DynamicsDataset(train_list,
                                      MethodEnum.Multistep,
                                      scheme=scheme)
        return dataset
