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

# Authors: Nicholas Tagliapietra, Katharina Ensinger (katharina.ensinger@bosch.com)


from enum import Enum
from typing import Any, List, Tuple, Union

from configs.modelbase_config import ModelBaseConfig
from pydantic import Field, root_validator
from utils.numerical_schemes import get_scheme


class MethodEnum(str, Enum):
    Taylor = "taylor"
    Multistep = "multistep"


class SplitConfig(ModelBaseConfig):
    # Data Split default parameters
    a_train: int = 0
    b_train: int = -1  
    # split-percentage e.g. 50-50, 70-30 etc.
    a_eval: List = [0, 0]
    b_eval: List = [-1, 10]


class SimulationConfig(ModelBaseConfig):
    # Train trajectory simulation default parameters
    dt: float = 0.1
    t_endpoints: Tuple[float, float] = (0., 40.)
    # y0 should later be converted in a torch.Tensor inside of a factory.
    y0: List[float] = [0., 7.]
    bound: float = 0.
    noise_std: float = 0.1


class TestConfig(ModelBaseConfig):
    # Test trajectory default parameters
    dt: float = 0.1
    t_endpoints: Tuple[float, float] = (0., 40.)
    # y0 should later be converted in a torch.Tensor inside of a factory.
    y0: List[float] = [0., 7.]
    bound: float = 0.
    noise_std: float = 0.


# CONFIGS FOR DATASETS
class MocapConfig(ModelBaseConfig):
    """This class describe a config object for the CMU mocap dataset.
    """
    MocapConfig: Any = ""
    subject_num: str = "09"
    pca_components: int = 5
    data_normalize: bool = False
    pca_normalize: bool = True
    dt: float = 0.01
    seqlen: int = 100


class ExternalDataConfig(ModelBaseConfig):
    """This class describe a config object for external data, and
    handles the splitting for training and test data.
    """
    ExternalDataConfig: Any = ""
    model: str = ""
    split_config: SplitConfig = Field(default_factory=lambda: SplitConfig())


class SyntheticDataConfig(ModelBaseConfig):
    """This class describe a config object for synthetic generated data, and
    handles both the simulation of the mathematical model for training data
    and test data.
    """
    SyntheticDataConfig: Any = ""
    model: str = "van_der_pol"
    split_config: SplitConfig = Field(default_factory=lambda: SplitConfig())
    simulation_config: SimulationConfig = Field(
        default_factory=lambda: SimulationConfig())
    test_config: TestConfig = Field(
        default_factory=lambda: TestConfig())


DatasetConfig = Union[MocapConfig, SyntheticDataConfig, ExternalDataConfig]


class DataConfig(ModelBaseConfig):
    method: MethodEnum = MethodEnum.Taylor
    integration_rule: str = ""
    dataset_config: DatasetConfig = Field(
        default_factory=lambda: DatasetConfig())

    @root_validator
    def taylor_consistent(cls, values):
        """When Taylor is used, the integration rule must be empty.

        Raises: 
            ValueError: Integration rule has to be empty if Taylor is used.
        """

        method = values.get("method")
        integration_rule = values.get("integration_rule")
        if (method == MethodEnum.Taylor) and integration_rule != "":
            raise ValueError(
                "When using Taylor, no integration rule can be used.")
        return values

    @root_validator
    def multistep_consistent(cls, values):
        """Checks if the integration rule used has been implemented.

        Raises:
            NotImplementedError: The integration rule has not been implemented
        """
        method = values.get("method")
        integration_rule = values.get("integration_rule")
        if (method == MethodEnum.Multistep):
            get_scheme(integration_rule)
        return values
