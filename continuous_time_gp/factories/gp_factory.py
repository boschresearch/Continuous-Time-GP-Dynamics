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
from configs.gp_config import MultistepGPConfig, TaylorGPConfig
from data.datasets.dynamics_dataset import DynamicsDataset
from factories.kernel_factory import (MultiStepKernelFactory,
                                      TaylorKernelFactory)
from models.multistepgp import MultistepGP
from models.taylorgp import TaylorGP

from factories.kernel_factory import ExactTaylorKernelFactory, ExactTaylorKernelFactory


class MultistepGPFactory():
    @staticmethod
    def build(config: MultistepGPConfig,
              dataset_train: DynamicsDataset,
              component: int,
              ) -> MultistepGP:
        """Initialized a MultistepGP as described in the config object.

        Args:
            config (MultistepGPConfig): Configuration object
            dataset_train (DynamicsDataset): Initialized dataset
            component (int): Which component of the data the GP is learning.

        Returns:
            MultistepGP: MultistepGP as in the configs.
        """
        # MultistepKernel initialization
        kernel_config = config.kernel_config
        act_func = config.act_func
        init_obs_noise = torch.tensor(config.init_obs_noise[component],
                                      dtype=torch.float64)
        noise_mode = config.noise_mode
        kernel = MultiStepKernelFactory.build(kernel_config,
                                              dataset_train.beta,
                                              component)
        gp = MultistepGP(dataset_train,
                         kernel,
                         component,
                         act_func,
                         init_obs_noise,
                         noise_mode)
        gp.cfg = config
        return gp


class TaylorGPFactory():
    @staticmethod
    def build(config: TaylorGPConfig,
              dataset_train: DynamicsDataset,
              component: int,
              ) -> TaylorGP:
        """Initialized a TaylorGP as described in the config object.

        Args:
            config (TaylorGPConfig): Configuration object
            dataset_train (DynamicsDataset): Initialized dataset
            component (int): Which component of the data the GP is learning.

        Returns:
            TaylorGP: TaylorGP as in the configs.
        """
        kernel_config = config.kernel_config
        act_func = config.act_func
        init_obs_noise = torch.tensor(config.init_obs_noise[component],
                                      dtype=torch.float64)
        noise_mode = config.noise_mode
        order = config.order

        # Stack h s
        h = torch.tensor([], dtype=torch.float64)
        for i in range(len(dataset_train.timelines)):
            timeline = dataset_train.timelines[i]
            h = torch.cat((h, timeline[1:] - timeline[:-1]), 0)

        kernel = TaylorKernelFactory.build(kernel_config,
                                           h,
                                           component)

        gp = TaylorGP(dataset_train,
                      kernel,
                      component,
                      act_func,
                      init_obs_noise,
                      noise_mode,
                      order)
        gp.cfg = config
        return gp

class ExactTaylorGPFactory():
    @staticmethod
    def build(config: TaylorGPConfig,
              dataset_train: DynamicsDataset,
              component: int,
              rbf_list: list,
              ) -> TaylorGP:
        """Initialized a TaylorGP as described in the config object.

        Args:
            config (TaylorGPConfig): Configuration object
            dataset_train (DynamicsDataset): Initialized dataset
            component (int): Which component of the data the GP is learning.

        Returns:
            TaylorGP: TaylorGP as in the configs.
        """
        kernel_config = config.kernel_config
        act_func = config.act_func
        init_obs_noise = torch.tensor(config.init_obs_noise[component],
                                      dtype=torch.float64)
        noise_mode = config.noise_mode
        order = config.order

        # Stack h s
        h = torch.tensor([], dtype=torch.float64)
        for i in range(len(dataset_train.timelines)):
            timeline = dataset_train.timelines[i]
            h = torch.cat((h, timeline[1:] - timeline[:-1]), 0)

        kernel = ExactTaylorKernelFactory.build(kernel_config,
                                           h,
                                           component,
                                           rbf_list)

        gp = TaylorGP(dataset_train,
                      kernel,
                      component,
                      act_func,
                      init_obs_noise,
                      noise_mode,
                      order)
        gp.cfg = config
        return gp