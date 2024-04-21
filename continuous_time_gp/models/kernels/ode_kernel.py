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

from models.kernels.basekernel import RBFKernel, BaseKernel
from torch import Tensor, matmul
import torch
from typing import Tuple, Callable
from torch.distributions.multivariate_normal import MultivariateNormal
from abc import ABC, abstractmethod
import warnings
class InheritanceKernel(ABC, torch.nn.Module):

    rbf_list: list[BaseKernel]
    evaluated_dimension: int

    def __init__(self, rbf_list, evaluated_dimension) -> None:
        super().__init__()
        # The activation functions can be useful in a moltitude of kernels,
        # so they are initialized here in the mother class
        self.rbf_list = torch.nn.ModuleList(rbf_list)
        self.evaluated_dimension = evaluated_dimension

    @abstractmethod
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Evaluation of the Kernel between x1 and x2

        Args:
            x1 (Tensor): Tensor x1.
            x2 (Tensor): Tensor x2.
        """
        raise NotImplementedError

    @abstractmethod
    def cross_kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Cross-Kernel evaluated between x1 and x2

        Args:
            x1 (Tensor): Tensor x1.
            x2 (Tensor): Tensor x2.
        """
        raise NotImplementedError

    def set_noise_par(self, value: Tensor) -> None:
        self.noise_par = value
        return

    @abstractmethod
    def log_tensorboard(self,
                        tb_writer: Callable,
                        epoch: int,
                        model_str: str) -> None:
        """Given the tensorboard callback, save on tensorboard the data
        regarding the model trainable hyperparameters.

        Args:
            tb_writer (Callable): A SummaryWriter object used to save data on
             tensorboard.
            epoch (int): Epoch number.
            model_str (str): String identifying the model and the layout to
            use on tensorboard.
        """
        raise NotImplementedError

class FirstOrderTaylorRBF(InheritanceKernel):
    """Radial Basis Function Kernel.
    """
    rbf_list: list[RBFKernel]
    evaluated_dimension: int
    def __init__(self,
                 rbf_list, evaluated_dimension,
                 ) -> None:
        super().__init__(rbf_list, evaluated_dimension)

        self.lengthscale = self.rbf_list[evaluated_dimension].lengthscale
        self.sigma_k = self.rbf_list[evaluated_dimension].sigma_k
        self.num_fourier_feat = self.rbf_list[self.evaluated_dimension].num_fourier_feat
        self.act_func = self.rbf_list[self.evaluated_dimension].act_func
        self.data_dimensions = len(self.rbf_list)


    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # # we have 4 dimensions, (batch_dim_x1, batch_dim_x2, sum_j, sum_k)
        batch_x1 = x1.shape[0]
        batch_x2 = x2.shape[0]
        x1_num_patterns = x1.shape[1]
        x2_num_patterns = x2.shape[1]
        d = self.data_dimensions
        if x1_num_patterns != d:
              raise RuntimeError(
                  f"Expected {d} dimensions, but x1 \
                      has {x1_num_patterns}")
        if x2_num_patterns != d:
              raise RuntimeError(
                  f"Expected {d} dimensions, but x2 \
                      has {x2_num_patterns}")
        # we calculate the inner sum
        x1_k = x1.reshape([batch_x1, 1, 1, d])
        x2_k = x2.reshape([1, batch_x2, 1, d])
        l_i = self.rbf_list[self.evaluated_dimension].lengthscale.reshape([1, 1, 1, d])
        l_j = torch.cat([rbf.lengthscale.reshape(1, 1, 1, d) for rbf in self.rbf_list], dim=2)
        inner_summand = 0.5 * ((x1_k - x2_k)**2 * (1 / l_i**2 +1 / l_j**2))
        #inner_summand = 0.5 * ((x1_k - x2_k) * (1 / l_i+ 1 / l_j)**2)
        exp_sum = torch.exp(-torch.sum(inner_summand, dim=3))
        # # we calculate the outer sum
        sigma_j = torch.cat([rbf.sigma_k.reshape(1, 1, 1) for rbf in self.rbf_list], dim=2)
        l_ij_squared = self.rbf_list[self.evaluated_dimension].lengthscale.reshape([1, 1, d]) ** 2
        x1_j = x1.reshape([batch_x1, 1, d])
        x2_j = x2.reshape([1, batch_x2, d])
        outer_summand = (sigma_j / (l_ij_squared ** 2)) * (l_ij_squared - (x1_j - x2_j) ** 2) * exp_sum
        result = self.rbf_list[self.evaluated_dimension].sigma_k * torch.sum(outer_summand, dim=2)
        return result

    def lengthscale_kernel_multiplication(self, l_i: torch.Tensor,
                                          l_j: torch.Tensor):
        # multiplication of kernels, new lengthscale: 1/(1/l_i+1/l_j))
        return torch.div(1, torch.div(1, l_i) + torch.div(1, l_j))

    def cross_kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.forward(x1, x2)

    def sample_rff_weights(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        warnings.warn("Decoupled smapling not yet implemented for exact Taylor")
        return

    def resample_rff_weights(self) -> None:
#        warnings.warn("Decoupled smapling not yet implemented for exact Taylor")
        return

    def build_feature_map(self, x: Tensor) -> Tensor:
        warnings.warn("Decoupled smapling not yet implemented for exact Taylor")
        return

    def rff_prior(self, x_eval:Tensor) -> Tensor:
        warnings.warn("Decoupled smapling not yet implemented for exact Taylor")
        return

    def log_tensorboard(self,
                        tb_writer: Callable,
                        epoch: int,
                        model_str: str) -> None:
        pass


class SecondOrderTaylorRBF(InheritanceKernel):
    """Radial Basis Function Kernel.
    """
    rbf_list: list[RBFKernel]
    evaluated_dimension: int
    def __init__(self,
                 rbf_list, evaluated_dimension,
                 ) -> None:
        super().__init__(rbf_list, evaluated_dimension)
        self.lengthscale = self.rbf_list[evaluated_dimension].lengthscale
        self.sigma_k = self.rbf_list[evaluated_dimension].sigma_k
        self.num_fourier_feat = self.rbf_list[self.evaluated_dimension].num_fourier_feat
        self.act_func = self.rbf_list[self.evaluated_dimension].act_func
        self.data_dimensions = len(self.rbf_list)

    def inner_sum(self, x1:Tensor, x2:Tensor):
        d = self.data_dimensions
        batch_x1 = x1.shape[0]
        batch_x2 = x2.shape[0]
        x1_k = x1.reshape(batch_x1, 1, 1, 1, d)
        x2_k = x2.reshape(1, batch_x2, 1, 1, d)
        tilde_l_ijl_squared_inverse = (1/self.rbf_list[self.evaluated_dimension].lengthscale**2).reshape(1, 1, 1, 1, d)\
                      +torch.cat([(1/rbf.lengthscale.reshape(1, 1, 1, 1, d))**2 for rbf in self.rbf_list], dim = 2)\
                      +torch.cat([(1/rbf.lengthscale.reshape(1, 1, 1, 1, d))**2 for rbf in self.rbf_list], dim = 3)
        inner_summand = 0.5 * ((x1_k - x2_k)**2*tilde_l_ijl_squared_inverse)
        exp_sum = torch.exp(-torch.sum(inner_summand, dim=4))
        sigma_j = torch.cat(
            [rbf.sigma_k.reshape(1, 1, 1, 1) for rbf in self.rbf_list], dim=3)
        l_ij_squared = (self.rbf_list[self.evaluated_dimension].lengthscale.reshape(1, 1, 1, d))**2
        tilde_l_ij_l_squared_inverse = ((1/self.rbf_list[self.evaluated_dimension].lengthscale)**2).reshape(1, 1, d, 1)\
                      +torch.cat([(1/rbf.lengthscale.reshape(1, 1, d, 1))**2 for rbf in self.rbf_list], dim = 3)
        # batch1, batch2, l, j
        x1_j = x1.reshape(batch_x1, 1, 1, d)
        x2_j = x2.reshape(1, batch_x2, 1, d)
        x1_l = x1.reshape(batch_x1, 1, d, 1)
        x2_l = x2.reshape(1, batch_x2, d, 1)
        outer_summand = (sigma_j/ (l_ij_squared**2)) * (tilde_l_ij_l_squared_inverse**2)\
                        *(1 / tilde_l_ij_l_squared_inverse - (x1_l-x2_l)**2)*(l_ij_squared-(x1_j-x2_j)**2)*exp_sum
        current_diagonal_entries = torch.diagonal(outer_summand, dim1=2, dim2=3)
        desired_diagonal_entries = self.j_equals_l(x1, x2)
        mask = torch.diag_embed(desired_diagonal_entries - current_diagonal_entries , dim1 = 2)
        correct_solution = outer_summand + mask
        result = torch.sum(correct_solution, dim = 3)
        return result

    def j_equals_l(self, x1:Tensor, x2:Tensor):
        d = self.data_dimensions
        batch_x1 = x1.shape[0]
        batch_x2 = x2.shape[0]
        x1_k = x1.reshape(batch_x1, 1, 1, d)
        x2_k = x2.reshape(1, batch_x2, 1, d)
        tilde_l_ill_squared_inverse = (1/self.rbf_list[self.evaluated_dimension].lengthscale**2).reshape(1, 1, 1, d)\
                      +torch.cat([(1/rbf.lengthscale.reshape(1, 1, 1, d))**2 for rbf in self.rbf_list], dim = 2)\
                      +torch.cat([(1/rbf.lengthscale.reshape(1, 1, 1, d))**2 for rbf in self.rbf_list], dim = 2)
        inner_summand  = 0.5 * ((x1_k - x2_k)**2*tilde_l_ill_squared_inverse)
        exp_sum = torch.exp(-torch.sum(inner_summand, dim=3))
        sigma_l = torch.cat([rbf.sigma_k.reshape(1, 1, 1) for rbf in self.rbf_list], dim=2)
        l_il_squared = self.rbf_list[self.evaluated_dimension].lengthscale.reshape(1, 1, d) ** 2
        tilde_lil_squared_inverse = 1/self.rbf_list[self.evaluated_dimension].lengthscale.reshape(1, 1, 1, d) ** 2\
                                    +torch.cat([(1/rbf.lengthscale.reshape(1, 1, 1, d))**2 for rbf in self.rbf_list], dim = 2)
        tilde_lil_l_squared_inverse = torch.diagonal(tilde_lil_squared_inverse, dim1 = 2, dim2 = 3)
        x1_l = x1.reshape(batch_x1, 1, d)
        x2_l = x2.reshape(1, batch_x2, d)
        result = (sigma_l/l_il_squared**2)*(-(x1_l- x2_l)**2*(5*tilde_lil_l_squared_inverse+l_il_squared*tilde_lil_l_squared_inverse**2)\
                      +tilde_lil_l_squared_inverse**2*(x1_l- x2_l)**4+2+(l_il_squared*tilde_lil_l_squared_inverse))*exp_sum
        return result

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # # we have 4 dimensions, (batch_dim_x1, batch_dim_x2, sum_j, sum_k)
        sigma_i = self.rbf_list[self.evaluated_dimension].sigma_k
        correct_solution = self.inner_sum(x1, x2)
        result = sigma_i * torch.sum(correct_solution, dim = 2)
        return result

    def cross_kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.forward(x1, x2)

    def sample_rff_weights(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        warnings.warn("Decoupled smapling not yet implemented for exact Taylor")
        return

    def build_feature_map(self, x: Tensor) -> Tensor:
        warnings.warn("Decoupled smapling not yet implemented for exact Taylor")
        return

    def resample_rff_weights(self) -> None:
        warnings.warn("Decoupled smapling not yet implemented for exact Taylor")
        return

    def rff_prior(self, x_eval):
        warnings.warn("Decoupled smapling not yet implemented for exact Taylor")
        return

    def log_tensorboard(self,
                        tb_writer: Callable,
                        epoch: int,
                        model_str: str) -> None:
        pass