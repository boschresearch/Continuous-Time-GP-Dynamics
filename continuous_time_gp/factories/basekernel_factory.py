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

from typing import Optional

import torch
from configs.kernel_config import RBFKernelConfig
from models.kernels.basekernel import RBFKernel


class RBFKernelFactory():
    @staticmethod
    def build(config: RBFKernelConfig,
              component: int,
              order: Optional[int] = None) -> RBFKernel:

        if order is not None:
            lengthscale = torch.tensor(config.lengthscale[component][order],
                                       dtype=torch.float64)
        else:
            lengthscale = torch.tensor(config.lengthscale[component][0],
                                       dtype=torch.float64)

        data_dimension = lengthscale.shape[0]
        sigma_k = torch.tensor(config.sigma_k[component], dtype=torch.float64)
        kernel = RBFKernel(lengthscale,
                           sigma_k,
                           config.ard,
                           config.act_func,
                           data_dimension,
                           config.num_fourier_feat)
        return kernel
