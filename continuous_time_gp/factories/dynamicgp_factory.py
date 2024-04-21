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

from configs.dynamicgp_config import DynamicGPConfig
from configs.gp_config import MultistepGPConfig, TaylorGPConfig, ExactTaylorGPConfig
from data.datasets.dynamics_dataset import DynamicsDataset
from factories.gp_factory import MultistepGPFactory, TaylorGPFactory
from models.dynamicgp import DynamicGP
import torch
from factories.basekernel_factory import RBFKernelFactory
from factories.gp_factory import ExactTaylorGPFactory


class DynamicGPFactory():
    @staticmethod
    def build(config: DynamicGPConfig,
              dataset_train: DynamicsDataset) -> DynamicGP:
        """Initialize a DynamicGP from the config.

        Args:
            config (DynamicGPConfig):Configuration object
            dataset_train (DynamicsDataset): Initialized dataset.

        Raises:
            RuntimeError: Missing parameter for multistep case
            NotImplementedError: Invalid config file

        Returns:
            DynamicGP: DynamicGP as described in the config object.
        """
        
        gp_config = config.gp_config
        dimensions = dataset_train.data_dimension

        # Models list initialization
        model_list = []
        if isinstance(gp_config, TaylorGPConfig):
            for i in range(dimensions):
                base_gp = TaylorGPFactory.build(gp_config, dataset_train, i)
                model_list.append(base_gp)
        elif isinstance(gp_config, ExactTaylorGPConfig):
            basekernel_config = config.gp_config.kernel_config.basekernel_config
            rbf_kernel_list = []
            for i in range(dimensions):
                rbf_i = RBFKernelFactory.build(basekernel_config, i, 0)
                rbf_kernel_list.append(rbf_i)
            rbf_kernel_list = torch.nn.ModuleList(rbf_kernel_list)
            for d in range(dimensions):
                base_gp =ExactTaylorGPFactory.build(gp_config, dataset_train, d,
                                              rbf_kernel_list)
                model_list.append(base_gp)
        elif isinstance(gp_config, MultistepGPConfig):
            if dataset_train.beta is None:
                raise RuntimeError("Missing beta parameter.")
            for i in range(dimensions):
                base_gp = MultistepGPFactory.build(
                    gp_config, dataset_train, i)
                model_list.append(base_gp)
        else:
            raise NotImplementedError
        gp = DynamicGP(model_list)
        gp.cfg = config
        return gp
