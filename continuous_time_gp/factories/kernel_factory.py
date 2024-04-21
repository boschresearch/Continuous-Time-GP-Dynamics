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
from torch import Tensor

from configs.kernel_config import MultistepKernelConfig, TaylorKernelConfig
from factories.basekernel_factory import RBFKernelFactory
from factories.factory import Factory
from models.kernels.multistepkernel import MultistepKernel
from models.kernels.taylorkernel import TaylorKernel
import torch
from models.kernels.ode_kernel import FirstOrderTaylorRBF, SecondOrderTaylorRBF


class TaylorKernelFactory(Factory):
    @staticmethod
    def build(config: TaylorKernelConfig,
              h: Tensor,
              component: int) -> TaylorKernel:
        # Basekernels initialization
        rbf_config = config.basekernel_config
        kernel_list = []
        for i in range(config.order):
            kernel_list.append(RBFKernelFactory.build(rbf_config,
                                                      component,
                                                      i))
        # TaylorKernel initialization
        order = config.order
        ker = TaylorKernel(h, order, kernel_list)
        return ker

class ExactTaylorKernelFactory(Factory):
    @staticmethod
    def build(config: TaylorKernelConfig,
              h: Tensor,
              component: int,
              rbf_list: torch.nn.ModuleList) -> TaylorKernel:
        rbf_list = torch.nn.ModuleList(rbf_list)
        order = config.order
        if order == 2:
            first_order_dim_i = FirstOrderTaylorRBF(rbf_list=rbf_list,evaluated_dimension=component)
            taylor_order_i = [rbf_list[component], first_order_dim_i]
            ker = TaylorKernel(h, 2, taylor_order_i)
        elif order == 3:
            first_order_dim_i = FirstOrderTaylorRBF(rbf_list=rbf_list,evaluated_dimension=component)
            second_order_dim_i = SecondOrderTaylorRBF(rbf_list=rbf_list,evaluated_dimension=component)
            taylor_order_i = [rbf_list[component], first_order_dim_i, second_order_dim_i]
            ker = TaylorKernel(h, 3, taylor_order_i)
        else:
            raise RuntimeError("Order not implemented for exact Taylor")
        return ker

class MultiStepKernelFactory(Factory):
    @staticmethod
    def build(config: MultistepKernelConfig,
              beta: Tensor,
              component: int) -> MultistepKernel:

        basekernel = RBFKernelFactory.build(config.basekernel_config,
                                            component)
        ker = MultistepKernel(beta, basekernel)
        return ker
