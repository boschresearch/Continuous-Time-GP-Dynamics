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

from pydantic import BaseSettings, Field, root_validator

from configs.data_config import DataConfig, MethodEnum
from configs.dynamicgp_config import DynamicGPConfig
from configs.gp_config import MultistepGPConfig, TaylorGPConfig, ExactTaylorGPConfig
from configs.integrator_config import (MultistepIntegratorConfig,
                                       TaylorIntegratorConfig)
from configs.prediction_config import PredictionConfig
from configs.train_config import TrainConfig


class GlobalConfig(BaseSettings):
    dynamicgp_config: DynamicGPConfig = Field(
        default_factory=lambda: DynamicGPConfig())
    data_config: DataConfig = Field(default_factory=lambda: DataConfig())
    train_config: TrainConfig = Field(default_factory=lambda: TrainConfig())
    prediction_config: PredictionConfig = Field(
        default_factory=lambda: PredictionConfig())

    @root_validator
    def is_consistent(cls, configs):
        """ This validator checks if the config is consistent e.g. it is not
        mixing Taylor and Multistep components.

        """
        method = configs.get("data_config").method
        gp_config = configs.get("dynamicgp_config").gp_config
        integrator_config = configs.get("prediction_config").integrator_config

        taylor_integration = True
        multistep_integration = True
        # This check is passed if passed if we have only a TrajectorySampler,
        # as intended.
        for i in range(len(integrator_config)):
            if isinstance(integrator_config[i], TaylorIntegratorConfig):
                multistep_integration = multistep_integration and False
                taylor_integration = taylor_integration and True
            if isinstance(integrator_config[i], MultistepIntegratorConfig):
                multistep_integration = multistep_integration and True
                taylor_integration = taylor_integration and False

        is_taylor = (method == MethodEnum.Taylor) and (isinstance(
            gp_config, TaylorGPConfig) or isinstance(gp_config, ExactTaylorGPConfig))\
                    and taylor_integration

        is_multistep = (method == MethodEnum.Multistep) and isinstance(
            gp_config, MultistepGPConfig) and multistep_integration
        # It can be either Taylor or Multistep
        if is_taylor ^ is_multistep:
            return configs
        else:
            raise RuntimeError("Config is not consistent")

    class config:
        smart_union = True
