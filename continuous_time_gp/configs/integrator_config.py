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


from typing import Union

from configs.modelbase_config import ModelBaseConfig


class MultistepIntegratorConfig(ModelBaseConfig):
    MultistepIntegratorConfig = ""
    integration_rule: str = "ab_1"

    class config:
        smart_union = True


class TaylorIntegratorConfig(ModelBaseConfig):
    TaylorIntegratorConfig = ""
    order: int = 2

    class config:
        smart_union = True


class TrajectorySamplerConfig(ModelBaseConfig):
    TrajectorySamplerConfig = ""

    class config:
        smart_union = True


IntegratorConfig = Union[TrajectorySamplerConfig,
                         MultistepIntegratorConfig,
                         TaylorIntegratorConfig]
