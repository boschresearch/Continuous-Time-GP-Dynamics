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

import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from models.abstractgp import AbstractGP
from models.multistepgp import MultistepGP
from models.taylorgp import TaylorGP

# LOSS FUNCTIONS


def marginalLogLikelihoodLoss(likelihood_distrib: MultivariateNormal,
                              y: Tensor) -> Tensor:
    """Calculated the marginal log-likelihood loss by evaluating the 
    log-probability of likelihood_distrib on y.
    marginalLogLikelihood(y) = log(p(y|x)).

    Args:
        likelihood_distrib (MultivariateNormal) : A multivariate Normal 
        distribution of the 
        torch.distributions.multivariate_normal.MultivariateNormal type.
        y (Tensor) : The data for which the likelihood have to be evaluated.

    Returns:
        Tensor : A Tensor containing the likelihood of the input data y
    """
    y = torch.unsqueeze(y, 0)
    return likelihood_distrib.log_prob(y)


def kernel_std_reg(gp: AbstractGP) -> Tensor:
    """Compute the sum of the square of every kernel_std for an AbstractGP
    instance.

    Args:
        gp (AbstractGP): AbstractGP concrete model.

    Returns:
        Tensor: Regularization term.
    """
    regularization = torch.empty(1, dtype=torch.float64)
    if isinstance(gp, MultistepGP):
        regularization = torch.square(gp.kernel.basekernel.sigma_k)
    elif isinstance(gp, TaylorGP):
        for k in range(len(gp.kernel.rbf_list)):
            regularization = regularization + \
                torch.square(gp.kernel.rbf_list[k].sigma_k)
    else:
        return ValueError("Unknown model")

    return regularization
