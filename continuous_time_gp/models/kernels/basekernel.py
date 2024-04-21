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

from abc import ABC, abstractmethod
from typing import Callable, Tuple

import torch
import torch.nn.utils.parametrize as P
from torch import Tensor, matmul
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import transforms


class BaseKernel(ABC, torch.nn.Module):

    data_dimensions: int
    act_func: str
    num_fourier_feat: int

    def __init__(self, act_func: str,
                 data_dimensions: int,
                 num_fourier_feat: int) -> None:
        super().__init__()

        # The activation functions can be useful in a moltitude of kernels,
        # so they are initialized here in the mother class
        self.act_func = act_func
        self.data_dimensions = data_dimensions
        self.num_fourier_feat = num_fourier_feat

        self.register_buffer("w", torch.tensor([], dtype=torch.float64))
        self.register_buffer("Omega", torch.tensor([], dtype=torch.float64))

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

    @abstractmethod
    def sample_rff_weights(self) -> Tuple[Tensor, Tensor]:
        """Generate the weights for random fourier features.
        w is sampled from to a gaussian standard distribution

        Returns:
            Tuple[Tensor, Tensor]: RFF weights
        """
        raise NotImplementedError

    @abstractmethod
    def build_feature_map(self, x: Tensor) -> Tensor:
        """Build the feature map Phi for the Random Fourier Features

        Args:
            x (Tensor): Matrix of training data x (Datapoints as row vectors)

        Returns:
            Tensor: Phi matrix with the random fourier features for the input x
        """
        raise NotImplementedError

    @abstractmethod
    def rff_prior(self, x_eval: Tensor) -> Tensor:
        """Calculates the prior term for decoupled sampling.

        Args:
            x_eval (Tensor): Set of points to evaluate the prior term

        Returns:
            Tensor: Prior term calculated using decoupled sampling
        """
        raise NotImplementedError

    def resample_rff_weights(self) -> None:
        """Regenerate the weights for Random Fourier Features, and store them
        on the class variables.
        """
        self.w, self.Omega = self.sample_rff_weights()
        return

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


class RBFKernel(BaseKernel):
    """Radial Basis Function Kernel.
    """
    ard: bool
    lengthscale: torch.nn.Parameter
    sigma_k: torch.nn.Parameter

    def __init__(self,
                 init_lengthscale: Tensor,
                 init_sigma_k: Tensor,
                 ard: bool,
                 act_func: str,
                 data_dimension: int,
                 num_fourier_feat: int) -> None:
        super().__init__(act_func, data_dimension, num_fourier_feat)

        if isinstance(ard, bool) is not True:
            raise TypeError("ARD is expected to be of Type bool.")
        self.ard = ard

        if isinstance(init_lengthscale, Tensor) is not True:
            raise TypeError("The initial lengthscale has to be a Tensor.")

        # TODO Should add some more type checks and error handling in here...

        self.ard_dims = self.data_dimensions if self.ard is True else 1
        lengthscale = init_lengthscale.double() + \
            init_lengthscale.double() * \
            torch.abs((torch.rand(self.ard_dims,
                                  dtype=torch.float64) - 1/2)*0.4)

        self.lengthscale = torch.nn.Parameter(init_lengthscale.double())

        parametrization = transforms.ActivationFunction(
            self.act_func, threshold=True)
        P.register_parametrization(self, "lengthscale", parametrization)

        sigma_k = init_sigma_k.double() + \
            torch.abs(torch.randn(1, dtype=torch.float64))
        self.sigma_k = torch.nn.Parameter(sigma_k)
        P.register_parametrization(self, "sigma_k", parametrization)

        self.resample_rff_weights()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Kernel function for the Squared Exponential Kernel (RBF) from
        two sets of datapoints x1 and x2.
        Supports ARD.

        Args:
            x1 (Tensor): First set of datapoints, listed as row-vectors.
            x2 (Tensor): Second set of datapoints, listed as row-vectors.

        Raises:
            RuntimeError: If the tensor with data does not pass a
            dimensionality check.

        Returns:
            Tensor: Tensor containing the Squared exponential kernel function
             between x1 and x2.
        """
        if not (isinstance(x1, Tensor) or isinstance(x1, Tensor)):
            raise TypeError("Inputs must be of torch.Tensor type")

        x1_num_patterns = x1.shape[1]
        x2_num_patterns = x2.shape[1]
        if x1_num_patterns != self.data_dimensions:
            raise RuntimeError(
                f"Expected {self.data_dimensions} dimensions, but x1 \
                    has {x1_num_patterns}")
        if x2_num_patterns != self.data_dimensions:
            raise RuntimeError(
                f"Expected {self.data_dimensions} dimensions, but x2 \
                    has {x2_num_patterns}")

        sqdist = torch.zeros((x1.shape[0], x2.shape[0]), dtype=torch.float64)
        sqdist = torch.cdist(torch.div(x1, self.lengthscale),
                             torch.div(x2, self.lengthscale))

        return self.sigma_k * torch.exp(-(0.5)*torch.square(sqdist))

    def cross_kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.forward(x1, x2)



    def sample_rff_weights(self) -> Tuple[Tensor, Tensor]:
        """Generate the weights for the linear combination of random
        fourier features.
        w is sampled from to a gaussian standard distribution.
        Omega is sampled according to a multivariate normal distribution
        with zero mean and covariance matrix equal to the inverse of the
        length scales^2.

        Returns:
            Tuple[Tensor, Tensor]: RFF weights
        """

        w = torch.randn(2*self.num_fourier_feat, 1, dtype=torch.float64)

        mu = torch.zeros(1, self.data_dimensions, dtype=torch.float64)
        cov = torch.div(torch.eye(self.data_dimensions,
                        dtype=torch.float64), torch.square(self.lengthscale))
        distrib = MultivariateNormal(mu, cov)
        Omega = torch.reshape(torch.squeeze(distrib.sample(torch.Size(
            (2*self.num_fourier_feat,)))), (-1, self.data_dimensions))

        return w, Omega

    def build_feature_map(self, x: Tensor) -> Tensor:

        Omega1 = self.Omega[0:self.num_fourier_feat, :]
        Omega2 = self.Omega[self.num_fourier_feat:(2*self.num_fourier_feat), :]
        COS_PHI = torch.cos(matmul(x, Omega1.T))
        SIN_PHI = torch.sin(matmul(x, Omega2.T))
        PHI = torch.cat((COS_PHI, SIN_PHI), dim=1)
        #par = torch.sqrt((self.noise_par**2) / self.num_fourier_feat)

        par = torch.sqrt((self.sigma_k) / self.num_fourier_feat)

        return PHI*par

    def rff_prior(self, x_eval: Tensor) -> Tensor:
        PHI = self.build_feature_map(x_eval)
        return matmul(PHI, self.w)

    def log_tensorboard(self,
                        tb_writer: Callable,
                        epoch: int,
                        model_str: str) -> None:

        tb_writer.add_scalar(f'{model_str}/Sigma_k',
                             self.sigma_k.item(),
                             epoch)

        for i in range(self.lengthscale.shape[0]):
            tb_writer.add_scalar(f'{model_str}/Lengthscale_{i}',
                                 self.lengthscale[i].item(),
                                 epoch)
        return
