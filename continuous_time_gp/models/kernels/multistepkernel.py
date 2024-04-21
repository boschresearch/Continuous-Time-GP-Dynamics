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

from typing import Tuple

import torch
from torch import Tensor, addmm, matmul

from models.kernels.basekernel import BaseKernel
from models.kernels.customkernel import CustomKernel


class MultistepKernel(CustomKernel):

    beta: Tensor
    B: Tensor
    basekernel: BaseKernel
    nsteps: int
    num_patterns: int

    def __init__(self, beta: Tensor, basekernel: BaseKernel) -> None:
        super().__init__()

        self.beta = beta.clone()
        self.basekernel = basekernel

        self.nsteps = self.beta.shape[1] - 1
        self.num_patterns = self.beta.shape[0]

        self.register_buffer("B", torch.zeros(self.nsteps + 1,
                                              self.num_patterns,
                                              self.num_patterns,
                                              dtype=torch.float64))

        for i in range(self.nsteps + 1):
            self.B[i, :, :] = torch.diag(self.beta[:, i])

    def forward(self,
                X1_list: Tuple[Tensor, ...],
                X2_list: Tuple[Tensor, ...]) -> Tensor:

        basekernels = torch.zeros((self.nsteps + 1,
                                   self.nsteps + 1,
                                   self.num_patterns,
                                   self.num_patterns),
                                  dtype=torch.float64,
                                  device=X1_list[0].device)

        for i in range(self.nsteps + 1):
            for j in range(self.nsteps + 1):
                basekernels[i, j, :, :] = self.basekernel(
                    X1_list[i], X2_list[j])

        ker = torch.zeros(self.num_patterns, self.num_patterns,
                          dtype=torch.float64, device=X1_list[0].device)
        for i in range(self.nsteps + 1):
            for j in range(self.nsteps + 1):
                ker = addmm(ker, self.B[i, :, :], matmul(
                    basekernels[i, j, :, :], self.B[j, :, :]))
        return ker

    def cross_kernel(self,
                     x_eval: Tensor,
                     x_list: Tuple[Tensor, ...]) -> Tensor:

        cross_ker = torch.zeros(
            self.num_patterns, dtype=torch.float64, device=x_eval.device)
        for i in range(self.nsteps + 1):
            # B matrices are diagonal, so no need to transpose them
            cross_ker = addmm(cross_ker, self.basekernel(
                x_eval, x_list[i]), self.B[i, :, :])
        return cross_ker

    def rff_prior_test(self, x_eval: Tensor) -> Tensor:
        value = self.basekernel.rff_prior(x_eval)
        return value

    def rff_prior_train(self, x_train: Tuple[Tensor, ...]) -> Tensor:
        basekernel_prior = self.basekernel.rff_prior(x_train[0])
        prior = torch.zeros(basekernel_prior.shape)
        for i in range(self.nsteps + 1):
            basekernel_prior = self.basekernel.rff_prior(x_train[i])
            prior = prior + matmul(self.B[i], basekernel_prior)
        return prior

    def log_tensorboard(self, writer, epoch: int, model_str: str) -> None:
        self.basekernel.log_tensorboard(writer, epoch, f"{model_str}/RBF_0")
        return

    def resample_rff_weights(self) -> None:
        self.basekernel.resample_rff_weights()
        return

    def set_noise_par(self, value: torch.nn.Parameter) -> None:
        self.basekernel.set_noise_par(value)
        return
