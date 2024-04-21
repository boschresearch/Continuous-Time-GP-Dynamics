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

from typing import Any, List, Union

from configs.modelbase_config import ModelBaseConfig
from pydantic import Field

# BASE KERNELS


class RBFKernelConfig(ModelBaseConfig):
    num_fourier_feat: int = 256
    act_func: str = "log"
    lengthscale: List[List[List[float]]] = [[[1., 1.]], [[1., 1.]]]
    sigma_k: List[float] = [0.5, 0.5]
    ard: bool = True


# Later this may become a Union type, if additional BaseKernels are added
BaseKernelConfig = RBFKernelConfig

# CUSTOM KERNELS


class TaylorKernelConfig(ModelBaseConfig):
    TaylorKernelConfig: Any = ""
    order: int = 2
    basekernel_config: BaseKernelConfig = Field(
        default_factory=lambda: [RBFKernelConfig()])

class MultistepKernelConfig(ModelBaseConfig):
    MultistepKernelConfig: Any = ""
    basekernel_config: BaseKernelConfig = Field(
        default_factory=lambda: RBFKernelConfig())


KernelConfig = Union[TaylorKernelConfig, MultistepKernelConfig]
