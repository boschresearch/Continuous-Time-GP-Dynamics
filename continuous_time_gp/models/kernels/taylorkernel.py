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


import math
from typing import List, Tuple

import torch
from torch import Tensor, einsum, matmul

from .basekernel import RBFKernel
from .customkernel import CustomKernel


class TaylorKernel(CustomKernel):
    """Kernel responsible for learning the components of a Taylor series
    expansion up to order k.
    """
    rbf_list: torch.nn.ModuleList
    H: Tensor
    num_patterns: int

    def __init__(self,
                 h: Tensor,
                 order: int,
                 rbf_list: List[RBFKernel]) -> None:
        super().__init__()

        self.order = order
        self.num_patterns = h.shape[0]

        self.rbf_list = torch.nn.ModuleList(rbf_list)

        self.register_buffer("H", torch.zeros(self.order,
                                              self.num_patterns,
                                              self.num_patterns,
                                              dtype=torch.float64))

        for i in range(1, self.order + 1):
            hn = torch.pow(h, i - 1).div(math.factorial(i))
            self.H[i-1, :, :] = torch.diag(hn)

    def forward(self,
                x_1: Tuple[Tensor, ...],
                x_2: Tuple[Tensor, ...]) -> Tensor:

        if (len(x_1) > 1) or (len(x_2) > 1):
            raise ValueError("The training set should be a unique tensor!")
        x1, x2 = x_1[0], x_2[0]

        rbf_kernels = torch.zeros((self.order, x1.shape[0], x2.shape[0]),
                                  dtype=torch.float64,
                                  device=x1.device)
        for i in range(self.order):
            rbf_kernels[i, :, :] = self.rbf_list[i](x1, x2)

        return einsum("bij,bjk -> ik", [self.H, matmul(rbf_kernels, self.H)])

    def cross_kernel(self,
                     x_eval: Tensor,
                     x_train: Tuple[Tensor, ...]) -> List[Tensor]:

        if len(x_train) > 1:
            raise ValueError("The training set should be a unique tensor!")

        components = []
        for i in range(len(self.rbf_list)):
            components.append(self.cross_kernel_component(x_eval, x_train, i))
        return components

    def cross_kernel_component(self,
                               x_eval: Tensor,
                               x_train: Tuple[Tensor, ...],
                               component: int) -> Tensor:
        """Given x_eval, evaluate the i-th term of K(x_eval, x_list).

        Args:
            x_eval (Tensor): Tensor at which the crossKernel has to evaluated.
            x_list (Tuple[Tensor, ...]): List of training data.
            component (int) : For which term of the series to evaluate the 
            Cross kernel
        Raises:
            ValueError: The training list contains more than one element

        Returns:
            Tensor: Resulting cross-Kernel.
        """
        if len(x_train) > 1:
            raise ValueError("The training set should be a unique tensor!")
        return matmul((self.rbf_list[component](x_eval, x_train[0])),
                      self.H[component, :, :])

    def rff_prior_train(self, x_train: Tuple[Tensor]) -> Tensor:
        """Evaluates RFF prior using Taylor series.

        Args:
            x_eval (Tuple[Tensor]): Point at time t_n at which to evaluate
            RFF prior.

        Raises:
            ValueError: Raised when the training list contains more than one
            element

        Returns:
            Tensor: Prior for the input training data.
        """
        if len(x_train) > 1:
            raise ValueError("The training set should be a unique tensor!")

        rff_priors = torch.zeros((self.order, x_train[0].shape[0], 1),
                                 dtype=torch.float64,
                                 device=x_train[0].device)
        for i in range(self.order):
            rff_priors[i, :, :] = self.rff_prior_component(x_train[0], i)
        return einsum("bij,bjk->ik", [self.H, rff_priors])

    def rff_prior_test(self, x_eval: Tensor, h: Tensor) -> Tensor:
        """Evaluates RFF prior using Taylor series.

        Args:
            x_eval (Tensor): Point at time t_n at which to evaluate RFF prior.
            h (Tensor): Stepsize at time t_n ( h_n = t_{n+1} - t_n ).

        Returns:
            Tensor: Prior for the input test data.
        """
        priors = torch.zeros(
            self.order, 1, dtype=torch.float64, device=x_eval.device)
        coeff = torch.zeros(
            self.order, 1, dtype=torch.float64, device=x_eval.device)

        for i in range(self.order):
            priors[i] = self.rff_prior_component(x_eval, i)
            coeff[i] = torch.pow(h, i)/math.factorial(i + 1)

        return matmul(coeff.T, priors)

    def rff_prior_component(self, x_eval: Tensor, component: int) -> Tensor:
        return self.rbf_list[component].rff_prior(x_eval)

    def resample_rff_weights(self) -> None:
        [self.rbf_list[i].resample_rff_weights()
         for i in range(len(self.rbf_list))]
        return

    def log_tensorboard(self, tb_writer, epoch: int, model_str: str) -> None:
        # Maybe I could ask to the rbf kernels to add its own hyperparameters,
        # so decoupling the code of this class from the RBFKernels?
        for i in range(len(self.rbf_list)):
            self.rbf_list[i].log_tensorboard(
                tb_writer, epoch, f"{model_str}/RBF_{i}")
        return

    def set_noise_par(self, value):
        [self.rbf_list[i].set_noise_par(value)
         for i in range(len(self.rbf_list))]
        return
