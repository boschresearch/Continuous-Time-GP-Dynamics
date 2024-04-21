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

from typing import List

from configs.integrator_config import IntegratorConfig, TaylorIntegratorConfig
from configs.modelbase_config import ModelBaseConfig
from pydantic import Field


class PredictionConfig(ModelBaseConfig):
    # Prediction default parameters
    # 2 flags to indicate how to make predictions is more than enough, even if
    # maybe it is not the most elegant solution.
    ds_pred: bool = True
    num_ds_trajectories: int = 5
    ds_prior_noise: bool = False
    mean_pred: bool = True
    integrator_config: List[IntegratorConfig] = Field(
        default_factory=lambda: [TaylorIntegratorConfig()])
