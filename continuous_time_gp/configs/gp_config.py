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

from enum import Enum
from typing import Any, Union, List

from configs.kernel_config import MultistepKernelConfig, TaylorKernelConfig
from configs.modelbase_config import ModelBaseConfig
from pydantic import Field, validator



class NoiseMode(Enum):
    """Enumeration used to choose how the noise parameter in the config should
    be interpreted and consequently if it has to be rescaled during the
    initialization.

    DIRECT: The parameter in the config is copied directly inside of the GP, 
    and the model directly learns the data noise.
    AVG: It is scaled to the same order of magnitude of the data noise 
    #by an average factor.
    NOISELESS: Noise always equal to 0 and not trainable.
    EXACT: The noise is scaled to be the exact data noise Component-wise during
    training. Only the main diagonal is considered. The model stores the observation noise.
    EXACT_CORRELATED: All correlations are considered. The model stores the observation noise.
    """
    AVG = "average"
    NOISELESS = "noiseless"
    EXACT = "exact"
    EXACT_CORRELATED = "exact_correlated"
    DIRECT = "direct"



class TaylorGPConfig(ModelBaseConfig):
    TaylorGPConfig: Any = ""
    kernel_config: TaylorKernelConfig = Field(
        default_factory=lambda: TaylorKernelConfig())
    order: int = 2
    act_func: str = "log"
    init_obs_noise: List[float] = [0.5, 0.5, 0.5, 0.5, 0.5]
    noise_mode: NoiseMode = NoiseMode.AVG

    @validator("kernel_config")
    def is_taylorkernel(cls, ker) -> TaylorKernelConfig:
        if isinstance(ker, TaylorKernelConfig):
            return ker
        else:
            raise TypeError("TaylorGP has the wrong kernel")

class ExactTaylorGPConfig(ModelBaseConfig):
    ExactTaylorGPConfig: Any = ""
    kernel_config: TaylorKernelConfig = Field(
        default_factory=lambda: TaylorKernelConfig())
    order: int = 2
    act_func: str = "log"
    init_obs_noise: List[float] = [0.5, 0.5, 0.5, 0.5, 0.5]
    noise_mode: NoiseMode = NoiseMode.AVG

    @validator("kernel_config")
    def is_taylorkernel(cls, ker) -> TaylorKernelConfig:
        if isinstance(ker, TaylorKernelConfig):
            return ker
        else:
            raise TypeError("TaylorGP has the wrong kernel")

class MultistepGPConfig(ModelBaseConfig):
    MultistepGPConfig: Any = ""
    kernel_config: MultistepKernelConfig = Field(
        default_factory=lambda: MultistepKernelConfig())
    act_func: str = "log"
    init_obs_noise: List[float] = [0.5, 0.5, 0.5, 0.5, 0.5]
    noise_mode: NoiseMode = NoiseMode.AVG

    @validator("kernel_config")
    def is_multistepkernel(cls, ker) -> MultistepKernelConfig:
        if isinstance(ker, MultistepKernelConfig):
            return ker
        else:
            raise TypeError("MultistepGP has the wrong kernel")


GPConfig = Union[TaylorGPConfig, MultistepGPConfig, ExactTaylorGPConfig]
