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

from abc import ABC
from typing import Tuple, Mapping, Any

import torch
import torch.nn.utils.parametrize as P
from configs.gp_config import NoiseMode
from data.datasets.dynamics_dataset import DynamicsDataset
from models.kernels.customkernel import CustomKernel
from torch import Tensor, matmul
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import transforms
from configs.gp_config import GPConfig


def init_noise(mode: NoiseMode,
               init_obs_noise: Tensor,
               dataset_train: DynamicsDataset) -> Tuple[Tensor, bool]:
    """Configure the noise parameter with which to initialize the GP and choose
    if it is trainable or not.

    Args:
        config (MultistepGPConfig): Configuration of the GP
        dataset_train (DynamicsDataset): Dataset

    Raises:
        ValueError: Unknown initialization

    Returns:
        Tuple[Tensor, bool]: Noise parameter and flag indicating if it is
        trainable or not.
    """
    trainable_noise = True
    match mode:
        case NoiseMode.EXACT:
            noise_par = init_obs_noise
        case NoiseMode.EXACT_CORRELATED:
            noise_par = init_obs_noise
        case NoiseMode.AVG:
            epsilon = dataset_train.epsilon
            noise_par = init_obs_noise.div(torch.mean(epsilon))
        case NoiseMode.NOISELESS:
            noise_par = torch.tensor([0], dtype=torch.float64)
            trainable_noise = False
        case NoiseMode.DIRECT:
            noise_par = init_obs_noise
        case _:
            raise ValueError("Unknown noise mode")

    return torch.squeeze(noise_par), trainable_noise


class AbstractGP(torch.nn.Module, ABC):
    """Base abstract class for GP models."""

    act_func: str
    noise_par: torch.nn.Parameter
    dataset: DynamicsDataset
    kernel: CustomKernel
    component: int
    trainable_noise: bool
    noise_mode: NoiseMode
    cfg: GPConfig

    def __init__(self,
                 dataset: DynamicsDataset,
                 kernel: CustomKernel,
                 component: int,
                 act_func: str,
                 init_obs_noise: Tensor,
                 noise_mode: NoiseMode) -> None:
        super().__init__()

        if isinstance(act_func, str) is True:
            self.act_func = act_func
        else:
            raise RuntimeError("Missing Activation Function.")

        noise_par, trainable_noise = init_noise(noise_mode,
                                                init_obs_noise,
                                                dataset)
        self.trainable_noise = trainable_noise
        self.noise_mode = noise_mode

        if trainable_noise:
            if isinstance(noise_par, Tensor) is True:
                par = noise_par.clone()
                variation = (torch.rand(1, dtype=torch.float64) - 1/2)*0.4
                value = torch.nn.Parameter(par * (1 + variation))
                self.noise_par = value
            else:
                raise RuntimeError("Missing Noise std.")
        else:
            self.register_buffer("noise_par",
                                 torch.tensor(0.0,
                                              dtype=torch.float64),
                                 persistent=True)

        self.dataset = dataset
        self.component = component
        super(AbstractGP, self).add_module("kernel", kernel)

        # Parametrize weights using activation functions.
        parametrization = transforms.ActivationFunction(
            self.act_func, threshold=True)
        P.register_parametrization(self, "noise_par", parametrization)

        # Jitter terms
        self.register_buffer("jitter_posterior",
                             torch.tensor([0], dtype=torch.float64))
        self.register_buffer("jitter_cholesky", torch.tensor(
            [1e-10], dtype=torch.float64))

        self.register_buffer("ds_prior_noise",
                             torch.tensor([], dtype=torch.float64))
        self.resample_rff_weights()

        # Register Variables in buffer
        Var, L = self.variance_update(self.dataset.data)
        self.register_buffer("Var",
                             Var,
                             persistent=True)
        self.register_buffer("L",
                             L,
                             persistent=True)

    def forward(self, x_train: Tuple[Tensor, ...]) -> MultivariateNormal:
        """Forward function that, once given a set of datapoints x, evaluates
        its likelihood assuming a Gaussian distribution as likelihood.

        Args:
            x (Tensor): Tensor x of datapoints to evaluate
            Each datapoint is a row vector.

        Returns:
            MultivariateNormal: Likelihood distribution for every datapoint.
        """
        num_patterns = x_train[0].shape[0]
        self.Var, self.L = self.variance_update(x_train)
        mean = torch.zeros((1, num_patterns),
                           dtype=torch.float64, device=self.L.device)
        distrib = MultivariateNormal(mean, scale_tril=self.L)
        return distrib

    def variance_update(self,
                        x_train: Tuple[Tensor, ...]
                        ) -> tuple[Tensor, Tensor]:
        """Based on the training set memorized in self.x, calculate the Kernel
        matrix and its Cholesky factor.
        Adaptable to different integration rules.

        Returns:
            tuple[Tensor, Tensor]: Variance matrix Var(x_test) and its Cholesky
            factor
        """

        ker = self.kernel(x_train, x_train)

        num_patterns = x_train[0].shape[0]
        Var = torch.tensor([], dtype=torch.float64, device=ker.device)
        noise_matrix = self.data_noise_matrix(num_patterns,
                                              device=ker.device)
        jitter_matrix = self.jitter_cholesky * torch.eye(num_patterns,
                                                         dtype=torch.float64,
                                                         device=ker.device)
        Var = ker + noise_matrix + jitter_matrix
        L = torch.linalg.cholesky(Var)

        return Var, L

    def data_noise_matrix(self,
                          num_patterns: int,
                          device: torch.device = torch.device("cpu")
                          ) -> Tensor:
        """Generate the data noise matrix depending on the chosen mode.

        Args:
            device (Optional[torch.device]): Device of the matrix.
            Defaults to torch.device("cpu").

        Returns:
            Tensor: Observation noise matrix
        """
        match self.noise_mode:
            case NoiseMode.EXACT:
                # Extract diagonal
                diag = torch.diagonal(self.dataset.epsilon, offset = 0)
                # Initialize diagonal matrix
                noise_matrix = torch.diag(diag).to(device)
            case NoiseMode.EXACT_CORRELATED:
                # In this case, we keep all the noise correlations
                noise_matrix = self.dataset.epsilon.to(device)
            case NoiseMode.AVG:
                noise_matrix = torch.eye(num_patterns,
                                         dtype=torch.float64,
                                         device=device)
            case NoiseMode.DIRECT:
                noise_matrix = torch.eye(num_patterns,
                                         dtype=torch.float64,
                                         device=device)
            case NoiseMode.NOISELESS:
                return torch.tensor(0, dtype=torch.float64)

        return noise_matrix * self.noise_par

    def observation_noise(self) -> Tensor:
        """Generate the observation noise variance depending on the chosen mode

        Returns:
            Tensor: Observation noise variance
        """
        match self.noise_mode:
            case NoiseMode.EXACT:
                # The observation noise is already stored, so just return it
                return self.noise_par
            case NoiseMode.EXACT_CORRELATED:
                return self.noise_par     
            case NoiseMode.AVG:
                # The data noise is stored, so it has to be scaled
                scale = torch.mean(1/self.dataset.epsilon,
                                   dtype=torch.float64)
                return scale * self.noise_par
            case NoiseMode.NOISELESS:
                return torch.tensor(0, dtype=torch.float64)
            case NoiseMode.DIRECT:
                scale = torch.mean(1/self.dataset.epsilon,
                                   dtype=torch.float64)
                return scale * self.noise_par    
            case  _:
                raise RuntimeError("Unknown noise mode")

    def posterior_mean(self, x_eval: Tensor, **kwargs) -> Tensor:
        """Given a test set, evaluate the posterior mean at these x_eval points

        Args:
            x_eval (Tensor): Tensor containing the datapoints as row vectors.

        Returns:
            Tensor: Posterior mean of the function learned by the GP.
        """
        Kmn = self.kernel.cross_kernel(x_eval, self.dataset.data)
        return matmul(Kmn, self.alpha(self.dataset.targets[:, self.component]))

    def posterior_distribution(self, x_test: Tensor) -> MultivariateNormal:
        """Calculate the posterior distribution.

        Args:
            x_test (Tensor): The data at which to evaluate the posterior

        Returns:
            MultivariateNormal: Posterior distribution f(x_eval)|x,Y,x_eval)
        """
        posterior_mean = self.posterior_mean(x_test)
        posterior_cov = self.posterior_cov(x_test)
        distrib = MultivariateNormal(
            loc=posterior_mean, covariance_matrix=posterior_cov)
        return distrib

    def posterior_cov(self, x_eval: Tensor) -> Tensor:
        """Given a set of points, evaluate the posterior covariance matrix at
        these x_eval points.

        Args:
            x_eval (Tensor): Tensor containing the datapoints as row vectors.

        Returns:
            Tensor: Posterior covariance matrix of the function learned by the
            GP
        """
        self.eval()
        Kmn = self.kernel.cross_kernel(x_eval)

        V = torch.cholesky_solve(Kmn.T, self.L)
        K_ = self.kernel(x_eval, x_eval)
        posterior_cov = K_ - \
            matmul(Kmn, V) + self.jitter_posterior * torch.eye(K_.size()[0])

        return posterior_cov

    def alpha(self, y: Tensor) -> Tensor:
        """Efficient calculation of alpha = L * L.T * y used for the
        calculation of the posterior mean.

        Returns:
            Tensor: The alpha parameter
        """
        alpha = torch.cholesky_solve(y.reshape(-1, 1), self.L)
        return alpha

    def decoupled_sampling(self, x_eval: Tensor, **kwargs) -> Tensor:
        """Calculate a posterior prediction using decoupled sampling.

        Args:
            x_eval (Tensor): Set of points for which the posterior has to be
            evaluated.

        Returns:
            Tensor: Posterior prediction made with decoupled sampling.
        """

        prior = self.kernel.rff_prior_test(x_eval)
        update = self.rff_update(x_eval, **kwargs)

        return prior + update

    def rff_update(self,
                   x_eval: Tensor,
                   **kwargs) -> Tensor:
        """Calculates the update term for decoupled sampling.

        Args:
            x_eval (Tensor): Set of points for which the update term has to be
            evaluated.
            prior_noise (Optional[bool]): Add noise to the decoupled sampling
            prior term. Defaults to False.

        Returns:
            Tensor: Tensor with the update term calculated using decoupled
            sampling.
        """
        Y = self.dataset.targets[:, self.component].reshape(-1, 1)
        A = Y - self.kernel.rff_prior_train(self.dataset.data)

        if "ds_prior_noise" in kwargs:
            if kwargs["ds_prior_noise"]:
                A = A - self.ds_prior_noise

        V = torch.cholesky_solve(A, self.L)
        Kmn = self.kernel.cross_kernel(x_eval, self.dataset.data)

        return matmul(Kmn, V)

    def log_tensorboard(self, tb_writer, epoch: int, model_str: str) -> None:
        """Log on the tensorboard callback the Noise std, and calls the Kernel
        to write its hyperparameters.

        Args:
            tb_writer (_type_): Tensorboard Callback
            epoch (int): Epoch number.
            model_str (str): String with model name.
        """
        tb_writer.add_scalar(f"{model_str}/Noise_std",
                             self.noise_par.item(), epoch)
        self.kernel.log_tensorboard(tb_writer, epoch, model_str)
        return

    def resample_rff_weights(self) -> None:
        self.kernel.resample_rff_weights()

        # Here it is sampled the decoupled sampling prior noise, epsilon.
        Y = self.dataset.targets[:, self.component].reshape(-1, 1)
        data_noise_matrix = self.data_noise_matrix(Y.shape[0])
        mean = torch.zeros(Y.shape[0], dtype=torch.float64)
        distrib = MultivariateNormal(mean,
                                     covariance_matrix=data_noise_matrix)
        self.ds_prior_noise = distrib.sample().reshape(-1, 1)
        return

    def eval(self) -> None:
        super().eval()
        self.kernel.set_noise_par(self.noise_par)
        return

    def to(self, device: torch.device) -> None:
        super().to(device)
        self.dataset.to(device)
        return

    def load_state_dict(self,
                        state_dict: Mapping[str, Any],
                        strict: bool = True) -> None:
        for key in ["cfg", "dataset", "component"]:
            if key in state_dict:
                setattr(self, key, state_dict.pop(key))
        super().load_state_dict(state_dict, strict)
        return

    def state_dict(self,
                   destination=None,
                   prefix='',
                   keep_vars=False) -> Mapping[str, Any]:
        """ Overrides state_dict() to save also cfg and dataset"""
        original_dict = super().state_dict(destination=destination,
                                           prefix=prefix,
                                           keep_vars=keep_vars)
        original_dict[prefix+'cfg'] = self.cfg
        original_dict[prefix+'dataset'] = self.dataset
        original_dict[prefix+'component'] = self.component
        return original_dict
