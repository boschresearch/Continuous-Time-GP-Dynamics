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

from typing import List, Union

import torch
from configs.gp_config import NoiseMode
from data.datasets.dynamics_dataset import DynamicsDataset
from models.abstractgp import AbstractGP
from models.kernels.taylorkernel import TaylorKernel
from torch import Tensor, matmul


class TaylorGP(AbstractGP):

    order: int
    kernel: TaylorKernel

    def __init__(self,
                 dataset: DynamicsDataset,
                 kernel: TaylorKernel,
                 component: int,
                 act_func: str,
                 init_obs_noise: Tensor,
                 noise_mode: NoiseMode,
                 order: int) -> None:

        AbstractGP.__init__(self,
                            dataset,
                            kernel,
                            component,
                            act_func,
                            init_obs_noise,
                            noise_mode)

        if isinstance(order, int):
            self.order = order
        else:
            raise ValueError("Invalid order for Taylor Expansion.")

    def posterior_mean(self,
                       x_eval: Tensor,
                       **kwargs) -> Union[Tensor, List[Tensor]]:

        Kmn = None
        Kmn_list = []
        if "index" in kwargs:
            index = kwargs["index"]
            Kmn = self.kernel.cross_kernel_component(x_eval,
                                                     self.dataset.data,
                                                     index)
        else:
            Kmn_list = self.kernel.cross_kernel(x_eval, self.dataset.data)

        Y = self.dataset.targets[:, self.component].reshape(-1, 1)

        if Kmn is not None:
            return matmul(Kmn, self.alpha(Y))
        else:
            return [matmul(Kmn_list[i],
                           self.alpha(Y)) for i in range(self.order)]

    def decoupled_sampling(self,
                           x_eval: Tensor,
                           **kwargs) -> Union[Tensor, List[Tensor]]:
        # Depending on the keyword parameters,
        # chose the right decoupled sampling function.

        if "index" in kwargs:
            index = kwargs["index"]
            kwargs.pop("index")
            return self.decoupled_sampling_component(x_eval, index, **kwargs)
        else:
            pred_list = []
            for index in range(self.order):
                pred_list.append(self.decoupled_sampling_component(x_eval,
                                                                   index, **kwargs))
            return pred_list

    def decoupled_sampling_component(self,
                                     x_eval: Tensor,
                                     index: int, **kwargs) -> Tensor:
        """Calculate a posterior prediction using decoupled sampling.

        Args:
            x_eval (Tensor): Set of points to be evaluated.

        Returns:
            Tensor: Posterior prediction made with decoupled sampling.
        """
        prior = self.kernel.rff_prior_component(x_eval, index)
        update = self.rff_update_component(x_eval, index, **kwargs)

        return prior + update

    def rff_update_component(self,
                             x_eval: Tensor,
                             index: int, **kwargs) -> Tensor:
        """Calculates the update term for decoupled sampling,
        but only for evaluating the i-th component.

        Args:
            x_eval (Tensor): Set of points to be evaluated.

        Returns:
            Tensor: Tensor with the decoupled sampling update term.
        """
        Y = self.dataset.targets[:, self.component].reshape(-1, 1)
        A = Y - self.kernel.rff_prior_train(self.dataset.data)
        if "ds_prior_noise" in kwargs:
            if kwargs["ds_prior_noise"]:
                A = A - self.ds_prior_noise
                print("check")
        V = torch.cholesky_solve(A, self.L)
        # I should take only the i-th component
        Kmn = self.kernel.cross_kernel_component(
            x_eval, self.dataset.data, index)

        return matmul(Kmn, V)
