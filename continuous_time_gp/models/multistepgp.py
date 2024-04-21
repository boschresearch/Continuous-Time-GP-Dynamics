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

from configs.gp_config import NoiseMode
from data.datasets.dynamics_dataset import DynamicsDataset
from models.abstractgp import AbstractGP
from models.kernels.customkernel import CustomKernel
from torch import Tensor


class MultistepGP(AbstractGP):
    """"Implementation of a Gaussian Process for learning the dynamics
    of different physical systems.
    This class learns the dynamics to be used for the numerical
    integration with different numerical schemes, for example
    Adam-Bashforth 2 or the Forward Euler method."""

    def __init__(self,
                 dataset: DynamicsDataset,
                 kernel: CustomKernel,
                 component: int,
                 act_func: str,
                 init_obs_noise: Tensor,
                 noise_mode: NoiseMode) -> None:

        AbstractGP.__init__(self,
                            dataset,
                            kernel,
                            component,
                            act_func,
                            init_obs_noise,
                            noise_mode)
