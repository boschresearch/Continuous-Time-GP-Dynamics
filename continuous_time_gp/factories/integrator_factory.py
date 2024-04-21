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


from configs.integrator_config import (IntegratorConfig,
                                       MultistepIntegratorConfig,
                                       TaylorIntegratorConfig,
                                       TrajectorySamplerConfig)
from factories.factory import Factory
from models.integrator import (AbstractIntegrator, MultistepIntegrator,
                               TaylorIntegrator, TrajectorySampler)
from utils.numerical_schemes import get_scheme


class IntegratorFactory(Factory):
    @staticmethod
    def build(config: IntegratorConfig) -> AbstractIntegrator:
        """Given an IntegratorConfig object, initialize the correspondent 
        Integrator.

        Args:
            config (IntegratorConfig): Object containing the integrator 
            initial configuration.

        Raises:
            NotImplementedError: In the requested integrator has not been 
            implemented.

        Returns:
            AbstractIntegrator: Integrator as specified in the IntegratorConfig
        """
        if isinstance(config, TaylorIntegratorConfig):
            return TaylorIntegrator(config.order)

        elif isinstance(config, MultistepIntegratorConfig):
            scheme = get_scheme(config.integration_rule)
            return MultistepIntegrator.build(scheme)

        elif isinstance(config, TrajectorySamplerConfig):
            return TrajectorySampler()

        else:
            raise NotImplementedError
