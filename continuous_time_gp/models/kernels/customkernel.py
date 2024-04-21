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

from abc import ABC, abstractclassmethod
from typing import Tuple

import torch
from torch import Tensor


class CustomKernel(ABC, torch.nn.Module):

    @abstractclassmethod
    def forward(self,
                x1_list: Tuple[Tensor, ...],
                x2_list: Tuple[Tensor, ...]) -> Tensor:
        """Given two list containing the data, evaluate the Kernel.

        Args:
            X1_list (Tuple[Tensor, ...]): List of data x1
            X2_list (Tuple[Tensor, ...]): List of data x2

        Returns:
            Tensor: Evaluation of the kernel.
        """
        raise NotImplementedError

    @abstractclassmethod
    def cross_kernel(self, x_eval: Tensor, x_train: Tuple[Tensor]) -> Tensor:
        """Given x_eval, evaluate K(x_eval, x_list) with the Kernel.

        Args:
            x_eval (Tensor): Tensor at which the crossKernel has to evaluated.
            x_train (Tuple[Tensor]): List of training data.

        Returns:
            Tensor: Resulting cross-Kernel.
        """
        raise NotImplementedError

    @abstractclassmethod
    def rff_prior_test(self, x_eval: Tensor, *args) -> Tensor:
        raise NotImplementedError

    @abstractclassmethod
    def rff_prior_train(self, x_train: Tuple[Tensor, ...], *args) -> Tensor:
        raise NotImplementedError

    @abstractclassmethod
    def resample_rff_weights(self) -> None:
        raise NotImplementedError

    @abstractclassmethod
    def log_tensorboard(self, writer, epoch: int, model_str: str) -> None:
        raise NotImplementedError
